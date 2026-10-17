// tests/sovits/test_sv_loader.cpp
//
// Loader contract for the standalone ERes2NetV2 SV GGUF.
//
// A complete synthetic model (correct shapes for all 842 required tensors,
// valid metadata, zero conv/bn weights and unit BN running_var) is the positive
// control: it must load. Every negative case mutates exactly one thing from that
// control, so a false result is attributable to that single change.

#include <gtest/gtest.h>

#include "gpt_sovits/sv.h"

#include "ggml.h"
#include "gguf.h"
#include "test_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
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
const std::string kModel   = kTestDir + "models/sv-eres2netv2-f32.gguf";

static std::string tmp_path(const std::string & tag) {
    static std::atomic<int> counter{0};
    return ::testing::TempDir() + "sv_ldr_" + std::to_string(SV_TEST_GETPID()) + "_" +
           tag + "_" + std::to_string(counter.fetch_add(1)) + ".gguf";
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

struct SvModel { gpt_sovits::sv_model m{}; ~SvModel() { gpt_sovits::sv_model_free(m); } };

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Synthetic GGUF construction
// ---------------------------------------------------------------------------

struct TensorSpec {
    std::string          name;
    std::vector<int64_t> ne;
    ggml_type            type = GGML_TYPE_F32;
};
using Overrides = std::map<std::string, std::vector<uint8_t>>;

// Shared backing data: all weights are zero; every BN running_var is 1.
static const std::vector<uint8_t> & zero_buffer() {
    static const std::vector<uint8_t> buf(80u * 1024u * 1024u, 0);  // >= largest tensor
    return buf;
}
static const std::vector<uint8_t> & ones_buffer() {
    static const std::vector<uint8_t> buf = [] {
        std::vector<uint8_t> b(2048 * sizeof(float), 0);
        const float one = 1.0f;
        for (size_t i = 0; i < b.size(); i += sizeof(float)) std::memcpy(&b[i], &one, sizeof(float));
        return b;
    }();
    return buf;
}

static std::vector<uint8_t> f32_fill(int64_t n, float v) {
    std::vector<uint8_t> b(static_cast<size_t>(n) * sizeof(float));
    for (int64_t i = 0; i < n; ++i) std::memcpy(&b[static_cast<size_t>(i) * sizeof(float)], &v, sizeof(float));
    return b;
}

static const void * tensor_data(const TensorSpec & s, const Overrides & ov) {
    auto it = ov.find(s.name);
    if (it != ov.end()) return it->second.data();
    static const std::string kRN = ".running_var";
    if (s.name.size() >= kRN.size() &&
        s.name.compare(s.name.size() - kRN.size(), kRN.size(), kRN) == 0) {
        return ones_buffer().data();
    }
    return zero_buffer().data();
}

// The complete required tensor set for the fixed architecture
// (base_width=24, scale=4, expansion=4). Matches the real GGUF's 842 names and
// shapes exactly.
static std::vector<TensorSpec> required_specs() {
    std::vector<TensorSpec> v;
    auto bn = [&](const std::string & p, int64_t c) {
        for (const char * k : {"weight", "bias", "running_mean", "running_var"})
            v.push_back({p + "." + k, {c}});
    };
    v.push_back({"sv.conv1.weight", {3, 3, 1, 64}});
    bn("sv.bn1", 64);

    struct Layer { int L; int blocks; int64_t in0, c1, c3; int red; };
    const Layer layers[] = {
        {1, 3,   64,  96,  256,  0},
        {2, 4,  256, 192,  512,  0},
        {3, 6,  512, 384, 1024, 24},
        {4, 3, 1024, 768, 2048, 48},
    };
    for (const Layer & l : layers) {
        const int64_t width = l.c1 / 4;
        for (int b = 0; b < l.blocks; ++b) {
            const std::string p = "sv.layer" + std::to_string(l.L) + "." + std::to_string(b);
            const int64_t cin = (b == 0) ? l.in0 : l.c3;
            v.push_back({p + ".conv1.weight", {1, 1, cin, l.c1}});
            bn(p + ".bn1", l.c1);
            for (int i = 0; i < 4; ++i) {
                v.push_back({p + ".convs." + std::to_string(i) + ".weight", {3, 3, width, width}});
                bn(p + ".bns." + std::to_string(i), width);
            }
            v.push_back({p + ".conv3.weight", {1, 1, width * 4, l.c3}});
            bn(p + ".bn3", l.c3);
            if (b == 0) {
                v.push_back({p + ".shortcut.0.weight", {1, 1, l.in0, l.c3}});
                bn(p + ".shortcut.1", l.c3);
            }
            if (l.red > 0) {
                for (int j = 0; j < 3; ++j) {
                    const std::string fp = p + ".fuse_models." + std::to_string(j);
                    v.push_back({fp + ".local_att.0.weight", {1, 1, 2 * width, l.red}});
                    v.push_back({fp + ".local_att.0.bias",   {l.red}});
                    bn(fp + ".local_att.1", l.red);
                    v.push_back({fp + ".local_att.3.weight", {1, 1, l.red, width}});
                    v.push_back({fp + ".local_att.3.bias",   {width}});
                    bn(fp + ".local_att.4", width);
                }
            }
        }
    }
    v.push_back({"sv.layer3_ds.weight", {3, 3, 1024, 2048}});
    v.push_back({"sv.fuse34.local_att.0.weight", {1, 1, 4096, 512}});
    v.push_back({"sv.fuse34.local_att.0.bias",   {512}});
    bn("sv.fuse34.local_att.1", 512);
    v.push_back({"sv.fuse34.local_att.3.weight", {1, 1, 512, 2048}});
    v.push_back({"sv.fuse34.local_att.3.bias",   {2048}});
    bn("sv.fuse34.local_att.4", 2048);
    return v;
}

struct Meta {
    struct gguf_context * g;
    void valid() {
        str("general.architecture", "sovits-sv");
        str("sv.arch", "ERes2NetV2");
        u32("sv.base_width", 24);
        u32("sv.scale", 4);
        u32("sv.expansion", 4);
        u32("sv.feat_dim", 80);
        u32("sv.out_dim", 20480);
    }
    void str(const char * k, const char * v) { gguf_set_val_str(g, k, v); }
    void u32(const char * k, uint32_t v)     { gguf_set_val_u32(g, k, v); }
    void remove(const char * k)              { gguf_remove_key(g, k); }
};

// Write a complete synthetic GGUF: valid metadata plus `specs` (correct shapes
// unless a case mutates one). Data comes from shared zero/one buffers via
// no_alloc tensor descriptors, so nothing large is materialized.
static bool write_model(const std::string & path,
                        const std::function<void(Meta &)> & meta_cfg,
                        const std::function<void(std::vector<TensorSpec> &, Overrides &)> & tensor_cfg) {
    std::vector<TensorSpec> specs = required_specs();
    Overrides ov;
    if (tensor_cfg) tensor_cfg(specs, ov);

    struct gguf_context * g = gguf_init_empty();
    Meta meta{g};
    meta.valid();
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
        gguf_set_tensor_data(g, s.name.c_str(), tensor_data(s, ov));
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    ggml_free(c);
    gguf_free(g);
    return ok;
}

struct ModelState {
    ggml_backend_t backend;
    struct ggml_context * ctx_w, * ctx_bn;
    ggml_backend_buffer_t buf_w, buf_bn;
    gpt_sovits::sv_hparams hp;
};
static ModelState snapshot(const gpt_sovits::sv_model & m) {
    return {m.backend, m.ctx_w, m.ctx_bn, m.buf_w, m.buf_bn, m.hparams};
}
static void expect_unchanged(const gpt_sovits::sv_model & m, const ModelState & s) {
    EXPECT_EQ(m.backend, s.backend);
    EXPECT_EQ(m.ctx_w, s.ctx_w);
    EXPECT_EQ(m.ctx_bn, s.ctx_bn);
    EXPECT_EQ(m.buf_w, s.buf_w);
    EXPECT_EQ(m.buf_bn, s.buf_bn);
    EXPECT_EQ(m.hparams.base_width, s.hp.base_width);
    EXPECT_EQ(m.hparams.scale,      s.hp.scale);
    EXPECT_EQ(m.hparams.expansion,  s.hp.expansion);
    EXPECT_EQ(m.hparams.feat_dim,   s.hp.feat_dim);
    EXPECT_EQ(m.hparams.out_dim,    s.hp.out_dim);
}

// ===========================================================================
// Positive control: the complete synthetic model must load.
// ===========================================================================
TEST(SVLoader, LoadsCompleteSyntheticModel) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("valid"));
    ASSERT_TRUE(write_model(f.path, nullptr, nullptr));

    SvModel model;
    ASSERT_TRUE(gpt_sovits::sv_model_load(f.path, model.m, backend));
    EXPECT_EQ(model.m.hparams.base_width, 24);
    EXPECT_EQ(model.m.hparams.out_dim, 20480);
    EXPECT_EQ(model.m.backend, backend.h);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.conv1.weight"), nullptr);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.bn1.scale"), nullptr);
}

// ===========================================================================
// Negative cases: each differs from the control by exactly one thing.
// ===========================================================================
struct Case {
    const char * name;
    std::function<void(Meta &)> meta;
    std::function<void(std::vector<TensorSpec> &, Overrides &)> tensors;
};

static std::vector<Case> negative_cases() {
    return {
        {"wrong_general_architecture",
         [](Meta & m) { m.str("general.architecture", "sovits"); }, nullptr},
        {"wrong_sv_architecture",
         [](Meta & m) { m.str("sv.arch", "ERes2Net"); }, nullptr},
        {"missing_general_architecture",
         [](Meta & m) { m.remove("general.architecture"); }, nullptr},
        {"missing_sv_architecture",
         [](Meta & m) { m.remove("sv.arch"); }, nullptr},
        {"missing_u32_metadata",
         [](Meta & m) { m.remove("sv.out_dim"); }, nullptr},
        {"wrong_u32_metadata_type",
         [](Meta & m) { m.str("sv.base_width", "24"); }, nullptr},
        {"wrong_u32_metadata_value",
         [](Meta & m) { m.u32("sv.base_width", 16); }, nullptr},
        {"missing_tensor",
         nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             for (size_t i = 0; i < s.size(); ++i)
                 if (s[i].name == "sv.layer3_ds.weight") { s.erase(s.begin() + static_cast<long>(i)); break; }
         }},
        {"missing_bn_tensor",
         nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             for (size_t i = 0; i < s.size(); ++i)
                 if (s[i].name == "sv.bn1.running_var") { s.erase(s.begin() + static_cast<long>(i)); break; }
         }},
        {"extra_derived_constant",
         nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             // An extra "derived" name could shadow the folded BN tensor; the
             // loader must require exactly the 842 required tensors.
             s.push_back({"sv.bn1.scale", {63}});
         }},
        {"wrong_tensor_shape",
         nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             for (TensorSpec & t : s)
                 if (t.name == "sv.conv1.weight") t.ne = {3, 3, 1, 32};
         }},
        {"bn_wrong_dtype",
         nullptr,
         [](std::vector<TensorSpec> & s, Overrides &) {
             for (TensorSpec & t : s)
                 if (t.name == "sv.bn1.weight") t.type = GGML_TYPE_F16;
         }},
        {"bn_negative_variance",
         nullptr,
         [](std::vector<TensorSpec> &, Overrides & ov) {
             ov["sv.bn1.running_var"] = f32_fill(64, -1.0f);
         }},
        {"bn_nan_weight",
         nullptr,
         [](std::vector<TensorSpec> &, Overrides & ov) {
             ov["sv.bn1.weight"] = f32_fill(64, std::numeric_limits<float>::quiet_NaN());
         }},
    };
}

TEST(SVLoader, RejectsSingleFieldMutations) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);

    // Control: the unmutated synthetic model loads (proves the harness).
    {
        TempFile f(tmp_path("loop_control"));
        ASSERT_TRUE(write_model(f.path, nullptr, nullptr));
        gpt_sovits::sv_model m{};
        ASSERT_TRUE(gpt_sovits::sv_model_load(f.path, m, backend));
        gpt_sovits::sv_model_free(m);
    }

    for (const Case & c : negative_cases()) {
        SCOPED_TRACE(c.name);
        TempFile f(tmp_path(c.name));
        ASSERT_TRUE(write_model(f.path, c.meta, c.tensors));

        gpt_sovits::sv_model m{};
        const bool ok = gpt_sovits::sv_model_load(f.path, m, backend);
        EXPECT_FALSE(ok) << "mutation '" << c.name << "' was not rejected";
        gpt_sovits::sv_model_free(m);
    }
}

// ===========================================================================
// Standalone loader contracts
// ===========================================================================

TEST(SVLoader, MissingFileReturnsFalse) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    gpt_sovits::sv_model m{};
    EXPECT_FALSE(gpt_sovits::sv_model_load(kTestDir + "models/not-a-model.gguf", m, backend));
    gpt_sovits::sv_model_free(m);
}

TEST(SVLoader, NullBackendReturnsFalse) {
    gpt_sovits::sv_model m{};
    EXPECT_FALSE(gpt_sovits::sv_model_load(kModel, m, nullptr));
    EXPECT_FALSE(gpt_sovits::sv_model_load("/nonexistent/sv.gguf", m, nullptr));
}

TEST(SVLoader, ModelUnchangedOnFailure) {
    Backend sentinel;
    Backend other;
    ASSERT_NE(sentinel.h, nullptr);
    ASSERT_NE(other.h, nullptr);

    gpt_sovits::sv_model m{};
    m.backend            = sentinel.h;
    m.hparams.base_width = 7;
    m.hparams.scale      = 3;
    m.hparams.expansion  = 2;
    m.hparams.feat_dim   = 11;
    m.hparams.out_dim    = 13;
    const ModelState before = snapshot(m);

    // A valid GGUF with a wrong architecture and no tensors: fails early.
    TempFile f(tmp_path("unchanged_fail"));
    ASSERT_TRUE(write_model(f.path, [](Meta & meta) { meta.str("general.architecture", "bogus"); },
                            [](std::vector<TensorSpec> & s, Overrides &) { s.clear(); }));

    EXPECT_FALSE(gpt_sovits::sv_model_load(f.path, m, other));
    expect_unchanged(m, before);
}

// A failed reload must leave the previously loaded model intact and usable.
// The bad file is itself a complete, correct-shape model whose only fault is a
// negative BN running_var, so the failure lands late (during BN folding). The
// old model's weights must still be readable and unchanged afterwards.
TEST(SVLoader, FailedLoadPreservesLoadedModel) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);

    TempFile good(tmp_path("preserve_good"));
    ASSERT_TRUE(write_model(good.path, nullptr, nullptr));
    TempFile bad(tmp_path("preserve_bad"));
    ASSERT_TRUE(write_model(bad.path, nullptr,
                            [](std::vector<TensorSpec> &, Overrides & ov) {
                                ov["sv.bn1.running_var"] = f32_fill(64, -1.0f);
                            }));

    SvModel model;
    ASSERT_TRUE(gpt_sovits::sv_model_load(good.path, model.m, backend));
    const ModelState before = snapshot(model.m);

    EXPECT_FALSE(gpt_sovits::sv_model_load(bad.path, model.m, backend));
    expect_unchanged(model.m, before);

    // The preserved model's weights are still readable and hold their original
    // (zero) values.
    const struct ggml_tensor * w = gpt_sovits::sv_model_tensor(model.m, "sv.conv1.weight");
    ASSERT_NE(w, nullptr);
    std::vector<float> vals(8, -1.0f);
    ggml_backend_tensor_get(w, vals.data(), 0, vals.size() * sizeof(float));
    for (float v : vals) EXPECT_EQ(v, 0.0f);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.bn1.scale"), nullptr);
}

TEST(SVLoader, ReloadSameModelSucceeds) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("reload"));
    ASSERT_TRUE(write_model(f.path, nullptr, nullptr));

    SvModel model;
    ASSERT_TRUE(gpt_sovits::sv_model_load(f.path, model.m, backend));
    ASSERT_TRUE(gpt_sovits::sv_model_load(f.path, model.m, backend));
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.conv1.weight"), nullptr);
}

TEST(SVLoader, FreeIsIdempotent) {
    Backend backend;
    ASSERT_NE(backend.h, nullptr);
    TempFile f(tmp_path("free"));
    ASSERT_TRUE(write_model(f.path, nullptr, nullptr));

    gpt_sovits::sv_model m{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(f.path, m, backend));
    gpt_sovits::sv_model_free(m);
    gpt_sovits::sv_model_free(m);
    EXPECT_EQ(m.ctx_w, nullptr);
    EXPECT_EQ(m.buf_w, nullptr);
}

TEST(SVLoader, FreeUnloadedModelIsSafe) {
    gpt_sovits::sv_model m{};
    gpt_sovits::sv_model_free(m);
    gpt_sovits::sv_model_free(m);
    SUCCEED();
}

// Real model, when present locally: end-to-end shape + tensor lookup.
TEST(SVLoader, LoadRealModelIfPresent) {
    if (!file_exists(kModel)) {
        GTEST_SKIP() << "large SV model not present: " << kModel;
    }
    Backend backend;
    ASSERT_NE(backend.h, nullptr);

    SvModel model;
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModel, model.m, backend));
    EXPECT_EQ(model.m.hparams.base_width, 24);
    EXPECT_EQ(model.m.hparams.scale, 4);
    EXPECT_EQ(model.m.hparams.expansion, 4);
    EXPECT_EQ(model.m.hparams.feat_dim, 80);
    EXPECT_EQ(model.m.hparams.out_dim, 20480);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.conv1.weight"), nullptr);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.layer3_ds.weight"), nullptr);
    EXPECT_NE(gpt_sovits::sv_model_tensor(model.m, "sv.bn1.scale"), nullptr);
    EXPECT_EQ(gpt_sovits::sv_model_tensor(model.m, "sv.nonexistent"), nullptr);
}

} // namespace
