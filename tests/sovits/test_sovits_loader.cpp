// tests/sovits/test_sovits_loader.cpp
//
// Loader contract + end-to-end forward control for the unified SoVITS GGUF.
//
// The synthetic model carries the FULL real tensor set with the exact shapes
// the runtime expects (parameterized only by gin_channels / generator initial
// channels / v2Pro presence). That makes it loadable AND forward-able, so the
// positive controls build and execute a graph for every block (ref_enc, RVQ
// decode, text encoder, flow, generator, ge_to512 / v2pro ge). Every negative
// case mutates exactly one thing away from that complete model.
//
// Torch-free: GGUFs are built directly with the ggml/gguf C API.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "gguf.h"
#include "test_backend.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#  define SV_TEST_GETPID() _getpid()
#else
#  include <unistd.h>
#  define SV_TEST_GETPID() ::getpid()
#endif

namespace {

const std::string kTestDir = SOVITS_TEST_DIR;

static const char * kRealV2     = "models/v2-sovits-f16.gguf";
static const char * kRealV2Pro  = "models/v2pro-sovits-f16.gguf";
static const char * kRealV2Plus = "models/v2proplus-sovits-f16.gguf";

static std::string tmp_path(const std::string & tag) {
    static std::atomic<int> counter{0};
    return ::testing::TempDir() + "sovits_ldr_" + std::to_string(SV_TEST_GETPID()) +
           "_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".gguf";
}

struct TempFile {
    std::string path;
    explicit TempFile(std::string p) : path(std::move(p)) {}
    ~TempFile() { if (!path.empty()) std::remove(path.c_str()); }
    TempFile(const TempFile &) = delete;
    TempFile & operator=(const TempFile &) = delete;
};

struct Backend {
    ggml_backend_t h = ggml_backend_cpu_init();
    ~Backend() { if (h) ggml_backend_free(h); }
    operator ggml_backend_t() const { return h; }
};

struct SovitsModel {
    gpt_sovits::sovits_model m{};
    ~SovitsModel() { gpt_sovits::sovits_model_free(m); }
};

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Synthetic GGUF construction -- complete architecture, real shapes.
// ---------------------------------------------------------------------------

struct TensorSpec {
    std::string          name;
    std::vector<int64_t> ne;
    ggml_type            type = GGML_TYPE_F32;
};
using Overrides = std::map<std::string, std::vector<uint8_t>>;

// Shared zero backing store, sized for the largest tensor
// (v2pro.sv_emb_w = 20480*1024*4 = 80 MiB). Zero weights are a valid model and
// keep every forward finite.
static const std::vector<uint8_t> & zero_buffer() {
    static const std::vector<uint8_t> buf(84u * 1024u * 1024u, 0);
    return buf;
}

struct ModelShape {
    int64_t gin      = 512;   // 512 (v2) / 1024 (v2Pro+)
    int64_t gen_init = 512;   // 512 (v2/v2Pro) / 768 (v2ProPlus)
    bool    v2pro    = false; // emit the v2pro.* module set
};

static int64_t up_kernel(int64_t gen_init, int stage) {
    static const int64_t k512[5] = {16, 16, 8, 2, 2};
    static const int64_t k768[5] = {20, 16, 8, 2, 2};
    return (gen_init == 768) ? k768[stage] : k512[stage];
}

// The complete SoVITS tensor set with the exact runtime shapes. `bad_init`
// (default 0) lets a negative case build a generator whose initial channels are
// not 512/768 without touching this layout code.
static std::vector<TensorSpec> required_specs(const ModelShape & s) {
    std::vector<TensorSpec> v;
    auto add = [&](std::string n, std::vector<int64_t> ne) {
        // Match the shipped f16 GGUFs: vectors stay f32, matrices/tensors f16.
        const ggml_type t = (ne.size() >= 2) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        v.push_back({std::move(n), std::move(ne), t});
    };

    // ---- ref_enc (MelStyleEncoder) ----
    add("ref_enc.spectral_1_w", {704, 128});
    add("ref_enc.spectral_1_b", {128});
    add("ref_enc.spectral_2_w", {128, 128});
    add("ref_enc.spectral_2_b", {128});
    for (int i = 0; i < 2; ++i) {
        add("ref_enc.temporal." + std::to_string(i) + ".conv_w", {5, 128, 256});
        add("ref_enc.temporal." + std::to_string(i) + ".conv_b", {256});
    }
    add("ref_enc.attention.qkv_w", {128, 384});
    add("ref_enc.attention.qkv_b", {384});
    add("ref_enc.attention.out_w", {128, 128});
    add("ref_enc.attention.out_b", {128});
    add("ref_enc.fc_w", {128, s.gin});
    add("ref_enc.fc_b", {s.gin});

    // ---- quantizer ----
    add("quantizer.codebook", {768, 1024});

    // ---- text encoder (enc_p) ----
    add("text_encoder_ssl.ssl_proj_w", {768, 192});
    add("text_encoder_ssl.ssl_proj_b", {192});
    const char * secs[3] = {"text_encoder_ssl", "text_encoder_text", "text_encoder_post"};
    const int    n_lay[3] = {3, 6, 3};
    for (int sec = 0; sec < 3; ++sec) {
        for (int i = 0; i < n_lay[sec]; ++i) {
            const std::string b = std::string(secs[sec]) + ".layers." + std::to_string(i);
            add(b + ".qkv_w", {192, 576});   add(b + ".qkv_b", {576});
            add(b + ".out_w", {192, 192});   add(b + ".out_b", {192});
            add(b + ".rel_k", {96, 9});      add(b + ".rel_v_t", {9, 96});
            add(b + ".ln1_w", {192});        add(b + ".ln1_b", {192});
            add(b + ".ffn_up_w", {576, 768});   add(b + ".ffn_up_b", {768});
            add(b + ".ffn_down_w", {2304, 192}); add(b + ".ffn_down_b", {192});
            add(b + ".ln2_w", {192});        add(b + ".ln2_b", {192});
        }
    }
    add("text_encoder_text.text_embedding", {192, 732});
    add("text_encoder_post.proj_w", {192, 384});
    add("text_encoder_post.proj_b", {384});
    add("text_encoder_mrte.ssl_fused_w", {192, 704});
    add("text_encoder_mrte.ssl_fused_b", {704});
    add("text_encoder_mrte.text_kv_w", {192, 1024});
    add("text_encoder_mrte.text_kv_b", {1024});
    add("text_encoder_mrte.attn_out_w", {512, 192});
    add("text_encoder_mrte.attn_out_b", {192});
    add("text_encoder_mrte.ge_out_w", {512, 192});
    add("text_encoder_mrte.ge_out_b", {192});

    // ---- flow (ResidualCouplingBlock) ----
    for (int L = 0; L < 4; ++L) {
        const std::string p = "flow.layers." + std::to_string(L) + ".";
        add(p + "pre_w", {96, 192});   add(p + "pre_b", {192});
        add(p + "post_w", {192, 96});  add(p + "post_b", {96});
        add(p + "enc.cond_w", {s.gin, 1536});
        add(p + "enc.cond_b", {1536});
        for (int j = 0; j < 4; ++j) {
            const std::string q = p + "enc." + std::to_string(j) + ".";
            add(q + "in_w", {960, 384}); add(q + "in_b", {384});
            if (j < 3) add(q + "rs_w", {192, 384});
            else       add(q + "rs_w", {192, 192});
            if (j < 3) add(q + "rs_b", {384});
            else       add(q + "rs_b", {192});
        }
    }

    // ---- generator (HiFi-GAN) ----
    const int64_t gi = s.gen_init;
    add("generator.conv_pre_w", {192 * 7, gi});
    add("generator.conv_pre_b", {gi});
    add("generator.cond_w", {s.gin, gi});
    add("generator.cond_b", {gi});
    add("generator.conv_post_w", {(gi >> 5) * 7, 1});
    static const int64_t res_k[3] = {3, 7, 11};
    for (int st = 0; st < 5; ++st) {
        const int64_t in_ch  = gi >> st;
        const int64_t out_ch = gi >> (st + 1);
        const std::string p = "generator.stages." + std::to_string(st) + ".";
        add(p + "up_w", {up_kernel(gi, st), out_ch, in_ch});
        add(p + "up_b", {out_ch});
        for (int br = 0; br < 3; ++br) {
            const std::string q = p + "resblocks." + std::to_string(br) + ".";
            for (int i = 0; i < 3; ++i) {
                add(q + "convs1." + std::to_string(i) + ".w", {out_ch * res_k[br], out_ch});
                add(q + "convs1." + std::to_string(i) + ".b", {out_ch});
                add(q + "convs2." + std::to_string(i) + ".w", {out_ch * res_k[br], out_ch});
                add(q + "convs2." + std::to_string(i) + ".b", {out_ch});
            }
        }
    }

    // ---- v2Pro module set ----
    if (s.v2pro) {
        add("v2pro.sv_emb_w", {20480, s.gin});
        add("v2pro.sv_emb_b", {s.gin});
        add("v2pro.ge_to512_w", {s.gin, 512});
        add("v2pro.ge_to512_b", {512});
        add("v2pro.prelu_w", {s.gin});
    }
    return v;
}

struct Meta {
    struct gguf_context * g;
    void valid(const ModelShape & s) {
        str("general.architecture", "sovits");
        str("sovits.block", "full");
        str("sovits.semantic_frame_rate", "25hz");
        bool_("sovits.v2pro", s.v2pro);
        u32("sovits.ref_enc.out_dim", static_cast<uint32_t>(s.gin));
        u32("sovits.generator.gin_dim", static_cast<uint32_t>(s.gin));
    }
    void str (const char * k, const char * v) { gguf_set_val_str(g, k, v); }
    void u32 (const char * k, uint32_t v)     { gguf_set_val_u32(g, k, v); }
    void bool_(const char * k, bool v)        { gguf_set_val_bool(g, k, v); }
    void remove(const char * k)               { gguf_remove_key(g, k); }
};

static bool write_model(const std::string & path,
                        const ModelShape & shape,
                        const std::function<void(Meta &)> & meta_cfg,
                        const std::function<void(std::vector<TensorSpec> &, Overrides &)> & tensor_cfg) {
    std::vector<TensorSpec> specs = required_specs(shape);
    Overrides ov;
    if (tensor_cfg) tensor_cfg(specs, ov);

    struct gguf_context * g = gguf_init_empty();
    Meta meta{g};
    meta.valid(shape);
    if (meta_cfg) meta_cfg(meta);

    struct ggml_init_params p = {
        /*.mem_size=*/ ggml_tensor_overhead() * (specs.size() + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc=*/ true,
    };
    struct ggml_context * c = ggml_init(p);
    for (const TensorSpec & s : specs) {
        ggml_tensor * t = ggml_new_tensor(c, s.type, static_cast<int>(s.ne.size()), s.ne.data());
        ggml_set_name(t, s.name.c_str());
        gguf_add_tensor(g, t);
        auto it = ov.find(s.name);
        gguf_set_tensor_data(g, s.name.c_str(),
                             it != ov.end() ? it->second.data() : zero_buffer().data());
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    ggml_free(c);
    gguf_free(g);
    return ok;
}

// ---------------------------------------------------------------------------
// Graph helpers (single-output block forward on the CPU backend).
// ---------------------------------------------------------------------------

struct GraphContext {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = nullptr;
    explicit GraphContext(size_t max_nodes) {
        const size_t sz = ggml_tensor_overhead() * max_nodes
                        + ggml_graph_overhead_custom(max_nodes, false);
        buf.resize(sz);
        struct ggml_init_params params = {sz, buf.data(), true};
        ctx = ggml_init(params);
    }
    ~GraphContext() { if (ctx) ggml_free(ctx); }
    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;
    operator struct ggml_context *() { return ctx; } // NOLINT
};

using Input = std::pair<struct ggml_tensor *, std::pair<const void *, size_t>>;

static bool graph_compute(ggml_backend_t backend, struct ggml_context * ctx,
                          struct ggml_tensor * out, size_t max_nodes,
                          const std::vector<Input> & inputs,
                          std::vector<float> & out_vals) {
    out = ggml_cont(ctx, out);
    ggml_set_output(out);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        return false;
    }
    for (const Input & in : inputs) {
        ggml_backend_tensor_set(in.first, in.second.first, 0, in.second.second);
    }
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        const size_t n = ggml_nelements(out);
        out_vals.assign(n, 0.0f);
        ggml_backend_tensor_get(out, out_vals.data(), 0, n * sizeof(float));
    }
    ggml_gallocr_free(alloc);
    return ok;
}

static std::vector<float> zeros(size_t n) { return std::vector<float>(n, 0.0f); }

// ===========================================================================
// Positive controls: load + run one graph per block.
// ===========================================================================

TEST(SovitsLoader, LoadsAndRunsCompleteSyntheticV2) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("valid_v2"));
    ASSERT_TRUE(write_model(f.path, ModelShape{512, 512, false}, nullptr, nullptr));

    SovitsModel model;
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, model.m, backend));
    EXPECT_FALSE(model.m.hparams.is_v2pro);
    EXPECT_EQ(model.m.hparams.gin_channels, 512);
    EXPECT_EQ(model.m.v2pro.sv_emb_w, nullptr);

    const int64_t gin = 512;

    // ref_enc: refer {704, T} -> ge {512}
    {
        GraphContext g(8192);
        auto refer = ggml_new_tensor_2d(g, GGML_TYPE_F32, 704, 4);
        ggml_set_input(refer);
        auto out = gpt_sovits::sovits_mel_style_encoder_block_forward(g, refer, model.m.ref_enc);
        ASSERT_NE(out, nullptr);
        auto data = zeros(704 * 4);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 8192,
            {{refer, {data.data(), data.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(gin));
    }
    // quantizer: codes {T} -> {768, T}
    {
        GraphContext g(4096);
        auto codes = ggml_new_tensor_1d(g, GGML_TYPE_I32, 3);
        ggml_set_input(codes);
        auto out = gpt_sovits::sovits_rvq_decode_block_forward(g, codes, model.m.quantizer);
        ASSERT_NE(out, nullptr);
        std::vector<int32_t> data = {0, 1, 2};
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 4096,
            {{codes, {data.data(), data.size() * sizeof(int32_t)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(768 * 3));
    }
    // text encoder: ssl {768,T}, text {T}, ge {512,1}
    {
        constexpr size_t kMaxNodes = 1u << 18;
        GraphContext g(kMaxNodes);
        auto ssl = ggml_new_tensor_2d(g, GGML_TYPE_F32, 768, 2);
        ggml_set_input(ssl);
        auto text = ggml_new_tensor_1d(g, GGML_TYPE_I32, 2);
        ggml_set_input(text);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, 512, 1);
        ggml_set_input(ge);
        auto enc = gpt_sovits::sovits_text_encoder_block_forward(g, ssl, text, ge, model.m.text_encoder);
        ASSERT_NE(enc.x, nullptr);
        auto sd = zeros(768 * 2); auto gd = zeros(512);
        std::vector<int32_t> td = {0, 1};
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, enc.x, kMaxNodes,
            {{ssl, {sd.data(), sd.size() * sizeof(float)}},
             {text, {td.data(), td.size() * sizeof(int32_t)}},
             {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(192 * 2));
    }
    // flow: z_p {192,T}, ge {gin,1} -> z {192,T}
    {
        GraphContext g(16384);
        auto zp = ggml_new_tensor_2d(g, GGML_TYPE_F32, 192, 3);
        ggml_set_input(zp);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge);
        auto out = gpt_sovits::sovits_flow_block_inverse_forward(g, zp, ge, model.m.flow);
        ASSERT_NE(out, nullptr);
        auto zd = zeros(192 * 3); auto gd = zeros(gin);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 16384,
            {{zp, {zd.data(), zd.size() * sizeof(float)}},
             {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(192 * 3));
    }
    // generator: z {192,T}, ge {gin,1} -> wav {1, T*640}
    {
        GraphContext g(65536);
        auto z = ggml_new_tensor_2d(g, GGML_TYPE_F32, 192, 1);
        ggml_set_input(z);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge);
        auto out = gpt_sovits::sovits_generator_block_forward(g, z, ge, model.m.generator);
        ASSERT_NE(out, nullptr);
        auto zd = zeros(192); auto gd = zeros(gin);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 65536,
            {{z, {zd.data(), zd.size() * sizeof(float)}},
             {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(640));
        for (float x : vals) EXPECT_TRUE(std::isfinite(x));
    }
}

TEST(SovitsLoader, LoadsAndRunsCompleteSyntheticV2Pro) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("valid_pro"));
    ASSERT_TRUE(write_model(f.path, ModelShape{1024, 512, true}, nullptr, nullptr));

    SovitsModel model;
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, model.m, backend));
    EXPECT_TRUE(model.m.hparams.is_v2pro);
    EXPECT_EQ(model.m.hparams.gin_channels, 1024);

    const int64_t gin = 1024;

    // v2pro ge: ge_raw {gin,1} + sv {20480} -> ge {gin,1}
    {
        GraphContext g(512);
        auto ge_raw = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge_raw);
        auto sv = ggml_new_tensor_1d(g, GGML_TYPE_F32, 20480);
        ggml_set_input(sv);
        auto out = gpt_sovits::sovits_v2pro_ge_forward(g, ge_raw, sv, model.m.v2pro);
        ASSERT_NE(out, nullptr);
        auto gd = zeros(gin); auto sd = zeros(20480);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 512,
            {{ge_raw, {gd.data(), gd.size() * sizeof(float)}},
             {sv, {sd.data(), sd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(gin));
    }
    // ge_to512: ge {gin,1} -> {512,1}
    {
        GraphContext g(256);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge);
        auto out = gpt_sovits::sovits_ge_to512_forward(g, ge, model.m.v2pro);
        ASSERT_NE(out, nullptr);
        auto gd = zeros(gin);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 256,
            {{ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(512));
    }
    // flow + generator with gin=1024 conditioning.
    {
        GraphContext g(16384);
        auto zp = ggml_new_tensor_2d(g, GGML_TYPE_F32, 192, 2);
        ggml_set_input(zp);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge);
        auto out = gpt_sovits::sovits_flow_block_inverse_forward(g, zp, ge, model.m.flow);
        ASSERT_NE(out, nullptr);
        auto zd = zeros(192 * 2); auto gd = zeros(gin);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 16384,
            {{zp, {zd.data(), zd.size() * sizeof(float)}},
             {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(192 * 2));
    }
    {
        GraphContext g(65536);
        auto z = ggml_new_tensor_2d(g, GGML_TYPE_F32, 192, 1);
        ggml_set_input(z);
        auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
        ggml_set_input(ge);
        auto out = gpt_sovits::sovits_generator_block_forward(g, z, ge, model.m.generator);
        ASSERT_NE(out, nullptr);
        auto zd = zeros(192); auto gd = zeros(gin);
        std::vector<float> vals;
        ASSERT_TRUE(graph_compute(backend, g, out, 65536,
            {{z, {zd.data(), zd.size() * sizeof(float)}},
             {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
        EXPECT_EQ(vals.size(), static_cast<size_t>(640));
    }
}

TEST(SovitsLoader, LoadsAndRunsCompleteSyntheticV2ProPlus) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("valid_plus"));
    // v2ProPlus: same gin=1024, generator initial=768 (first up kernel 20).
    ASSERT_TRUE(write_model(f.path, ModelShape{1024, 768, true}, nullptr, nullptr));

    SovitsModel model;
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, model.m, backend));
    EXPECT_TRUE(model.m.hparams.is_v2pro);
    EXPECT_EQ(model.m.hparams.gin_channels, 1024);
    EXPECT_EQ(model.m.generator.conv_pre.b->ne[0], 768);

    const int64_t gin = 1024;
    GraphContext g(65536);
    auto z = ggml_new_tensor_2d(g, GGML_TYPE_F32, 192, 1);
    ggml_set_input(z);
    auto ge = ggml_new_tensor_2d(g, GGML_TYPE_F32, gin, 1);
    ggml_set_input(ge);
    auto out = gpt_sovits::sovits_generator_block_forward(g, z, ge, model.m.generator);
    ASSERT_NE(out, nullptr);
    auto zd = zeros(192); auto gd = zeros(gin);
    std::vector<float> vals;
    ASSERT_TRUE(graph_compute(backend, g, out, 65536,
        {{z, {zd.data(), zd.size() * sizeof(float)}},
         {ge, {gd.data(), gd.size() * sizeof(float)}}}, vals));
    EXPECT_EQ(vals.size(), static_cast<size_t>(640));
}

TEST(SovitsLoader, LoadsRealModelsIfPresent) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    struct Expect { const char * rel; bool v2pro; int64_t gin; };
    const Expect cases[] = {
        {kRealV2,     false, 512},
        {kRealV2Pro,  true,  1024},
        {kRealV2Plus, true,  1024},
    };
    int loaded = 0;
    for (const Expect & e : cases) {
        const std::string p = kTestDir + e.rel;
        if (!file_exists(p)) continue;
        SCOPED_TRACE(e.rel);
        SovitsModel model;
        ASSERT_TRUE(gpt_sovits::sovits_model_load(p, model.m, backend));
        EXPECT_EQ(model.m.hparams.is_v2pro, e.v2pro);
        EXPECT_EQ(model.m.hparams.gin_channels, e.gin);
        ++loaded;
    }
    if (loaded == 0) GTEST_SKIP() << "no real SoVITS GGUFs present";
}

// Explicit optional coverage for every locally checked-in variant (the five v2
// quant levels plus the older f16 Pro). Each is skipped when absent, so CI
// without models is unaffected. This only loads the files -- it never writes
// or overwrites any GGUF artifact.
TEST(SovitsLoader, LoadsLocalVariantGgufsIfPresent) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    struct Variant { const char * rel; bool v2pro; int64_t gin; };
    const Variant variants[] = {
        {"models/v2-sovits-f32.gguf",    false, 512},
        {"models/v2-sovits-f16.gguf",    false, 512},
        {"models/v2-sovits-q8.gguf",     false, 512},
        {"models/v2-sovits-q5.gguf",     false, 512},
        {"models/v2-sovits-q4.gguf",     false, 512},
        {"models/v2pro-sovits-f16.gguf", true,  1024},
    };
    int loaded = 0;
    for (const Variant & v : variants) {
        const std::string p = kTestDir + v.rel;
        if (!file_exists(p)) continue;
        SCOPED_TRACE(v.rel);
        SovitsModel model;
        ASSERT_TRUE(gpt_sovits::sovits_model_load(p, model.m, backend));
        EXPECT_EQ(model.m.hparams.is_v2pro, v.v2pro);
        EXPECT_EQ(model.m.hparams.gin_channels, v.gin);
        ++loaded;
    }
    if (loaded == 0) GTEST_SKIP() << "no local SoVITS GGUF variants present";
}

// ===========================================================================
// Negative cases: exactly one mutation away from the complete model.
// ===========================================================================

struct Case {
    const char * name;
    ModelShape   shape;
    std::function<void(Meta &)> meta;
    std::function<void(std::vector<TensorSpec> &, Overrides &)> tensors;
};

static void set_ne(std::vector<TensorSpec> & s, const std::string & name,
                   std::vector<int64_t> ne) {
    for (TensorSpec & t : s) if (t.name == name) t.ne = std::move(ne);
}
static void drop(std::vector<TensorSpec> & s, const std::string & name) {
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i].name == name) { s.erase(s.begin() + static_cast<long>(i)); return; }
}

static std::vector<Case> negative_cases() {
    const ModelShape v2{512, 512, false};
    const ModelShape pro{1024, 512, true};
    return {
        // --- metadata typing / value ---
        {"wrong_v2pro_kv_type", v2,
         [](Meta & m) { m.str("sovits.v2pro", "yes"); }, nullptr},
        {"wrong_out_dim_kv_type", v2,
         [](Meta & m) { m.str("sovits.ref_enc.out_dim", "512"); }, nullptr},
        {"wrong_sfr_kv_type", v2,
         [](Meta & m) { m.u32("sovits.semantic_frame_rate", 25); }, nullptr},
        {"out_dim_kv_value_conflict", v2,
         [](Meta & m) { m.u32("sovits.ref_enc.out_dim", 1024); }, nullptr},
        {"v2pro_flag_without_tensors", v2,
         [](Meta & m) { m.bool_("sovits.v2pro", true); }, nullptr},
        {"v2pro_tensors_without_flag", pro,
         [](Meta & m) { m.bool_("sovits.v2pro", false); }, nullptr},

        // --- missing / partial tensors ---
        {"missing_ref_enc_tensor", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) { drop(s, "ref_enc.fc_b"); }},
        {"missing_generator_tensor", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) { drop(s, "generator.stages.4.up_w"); }},
        {"partial_v2pro", pro, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             drop(s, "v2pro.ge_to512_w");  // sv_emb still present -> partial
         }},

        // --- gin derivation / channel conflict ---
        {"non_v2pro_gin_1024", ModelShape{1024, 512, false}, nullptr, nullptr},
        {"flow_cond_gin_mismatch", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "flow.layers.2.enc.cond_w", {256, 1536});
         }},
        {"flow_cond_dim_mismatch", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "flow.layers.0.enc.cond_w", {512, 1024});
         }},
        {"flow_cond_bias_mismatch", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "flow.layers.1.enc.cond_b", {1024});
         }},
        {"flow_cond_trailing_dim", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "flow.layers.3.enc.cond_w", {512, 1536, 2});
         }},
        {"generator_cond_gin_mismatch", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.cond_w", {256, 512});
         }},

        // --- generator architecture ---
        {"generator_bad_init_channels", ModelShape{512, 384, false}, nullptr, nullptr},
        {"generator_pre_kernel", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.conv_pre_w", {192 * 5, 512});
         }},
        {"generator_pre_out", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.conv_pre_b", {384});
         }},
        {"generator_post_kernel", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.conv_post_w", {16 * 5, 1});
         }},
        {"generator_up_kernel", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.stages.0.up_w", {12, 256, 512});
         }},
        {"generator_up_kernel_plus_first", ModelShape{1024, 768, true}, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.stages.0.up_w", {16, 384, 768});  // must be 20
         }},
        {"generator_up_channels", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.stages.2.up_b", {48});   // must be gen_init>>3 = 64
         }},
        {"generator_resblock_kernel", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.stages.0.resblocks.0.convs1.0.w", {256 * 5, 256});  // k must be 3
         }},
        {"generator_resblock_channels", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "generator.stages.1.resblocks.1.convs2.2.b", {256});  // must be 128
         }},

        // --- fixed-architecture shape constraints ---
        {"ref_enc_fc_in_dim", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "ref_enc.fc_w", {64, 512});
         }},
        {"ref_enc_fc_trailing_dim", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "ref_enc.fc_w", {128, 512, 2});
         }},
        {"quantizer_wrong_shape", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "quantizer.codebook", {768, 512});
         }},
        {"ref_enc_spectral_wrong_shape", v2, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "ref_enc.spectral_1_w", {704, 64});
         }},

        // --- v2pro tensor shapes ---
        {"v2pro_ge_to512_wrong_out", pro, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "v2pro.ge_to512_w", {1024, 256});
         }},
        {"v2pro_sv_emb_in_dim", pro, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "v2pro.sv_emb_w", {10240, 1024});
         }},
        {"v2pro_prelu_width", pro, nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             set_ne(s, "v2pro.prelu_w", {512});
         }},
    };
}

TEST(SovitsLoader, RejectsSingleFieldMutations) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);

    // Control: the unmutated synthetic v2 model loads (proves the harness).
    {
        TempFile f(tmp_path("loop_control"));
        ASSERT_TRUE(write_model(f.path, ModelShape{512, 512, false}, nullptr, nullptr));
        gpt_sovits::sovits_model m{};
        ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, m, backend));
        gpt_sovits::sovits_model_free(m);
    }

    for (const Case & c : negative_cases()) {
        SCOPED_TRACE(c.name);
        TempFile f(tmp_path(c.name));
        ASSERT_TRUE(write_model(f.path, c.shape, c.meta, c.tensors));

        gpt_sovits::sovits_model m{};
        const bool ok = gpt_sovits::sovits_model_load(f.path, m, backend);
        EXPECT_FALSE(ok) << "mutation '" << c.name << "' was not rejected";
        EXPECT_EQ(m.ctx_w, nullptr);
        EXPECT_EQ(m.buf_w, nullptr);
        gpt_sovits::sovits_model_free(m);
    }
}

// ===========================================================================
// Standalone loader contracts
// ===========================================================================

TEST(SovitsLoader, MissingFileReturnsFalse) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    gpt_sovits::sovits_model m{};
    EXPECT_FALSE(gpt_sovits::sovits_model_load(kTestDir + "models/not-a-model.gguf", m, backend));
    gpt_sovits::sovits_model_free(m);
}

TEST(SovitsLoader, NullBackendReturnsFalse) {
    gpt_sovits::sovits_model m{};
    EXPECT_FALSE(gpt_sovits::sovits_model_load(kTestDir + std::string(kRealV2), m, nullptr));
    EXPECT_FALSE(gpt_sovits::sovits_model_load("/nonexistent/sovits.gguf", m, nullptr));
}

TEST(SovitsLoader, ReloadSameModelSucceeds) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("reload"));
    ASSERT_TRUE(write_model(f.path, ModelShape{512, 512, false}, nullptr, nullptr));

    SovitsModel model;
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, model.m, backend));
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, model.m, backend));
    EXPECT_NE(model.m.ref_enc.fc_w, nullptr);
}

TEST(SovitsLoader, FreeIsIdempotent) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("free"));
    ASSERT_TRUE(write_model(f.path, ModelShape{512, 512, false}, nullptr, nullptr));

    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(f.path, m, backend));
    gpt_sovits::sovits_model_free(m);
    gpt_sovits::sovits_model_free(m);
    EXPECT_EQ(m.ctx_w, nullptr);
    EXPECT_EQ(m.buf_w, nullptr);
}

TEST(SovitsLoader, FreeUnloadedModelIsSafe) {
    gpt_sovits::sovits_model m{};
    gpt_sovits::sovits_model_free(m);
    gpt_sovits::sovits_model_free(m);
    SUCCEED();
}

} // namespace
