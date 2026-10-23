// tests/t2s/test_t2s_generate.cpp
//
// Tests for the single-request synchronous T2S generator
// (t2s_session_generate).
//
// This file deliberately uses tiny CPU-built synthetic models so the request
// contract, EOS handling, slot lifecycle, reproducibility and compute-failure
// recovery can be tested without external assets.  The weights are either
// explicitly zeroed or set to simple direction vectors, making the expected
// next token exact.
//
// Integration tests against the real GGUF model and cross-checks against a
// low-level runner are kept in a separate file.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "test_backend.h"
#include "../../third_party/ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// RAII helpers
// ---------------------------------------------------------------------------

struct BackendGuard {
    ggml_backend_t backend = nullptr;
    ~BackendGuard() {
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

struct ModelGuard {
    gpt_sovits::t2s_model model;
    ~ModelGuard() { gpt_sovits::t2s_model_free(model); }
};

struct SessionGuard {
    gpt_sovits::t2s_session session;
    ~SessionGuard() { gpt_sovits::t2s_session_free(session); }
};

// ---------------------------------------------------------------------------
// Synthetic model construction
// ---------------------------------------------------------------------------

struct SynthSpec {
    uint32_t d_model        = 8;
    uint32_t n_head         = 2;
    uint32_t n_layer        = 1;
    uint32_t linear_units   = 16;
    uint32_t vocab          = 4;
    uint32_t phoneme_vocab  = 4;
};

// HuBERT features are always 768-wide; the extract-latent weights must match
// this even in synthetic models used with compute_ref_emb.
static constexpr int64_t kHubertDim = 768;

static void set_tensor_f32(ggml_tensor * t, const std::vector<float> & vals) {
    ASSERT_NE(t, nullptr);
    ASSERT_EQ((size_t) ggml_nelements(t), vals.size());
    ggml_backend_tensor_set(t, vals.data(), 0, vals.size() * sizeof(float));
}

static void set_tensor_scalar(ggml_tensor * t, float v) {
    set_tensor_f32(t, std::vector<float>{v});
}

// Overwrite one column of a 2-D F32 tensor {rows, cols}.  Column `col` is
// contiguous: elements [col * rows, (col + 1) * rows).
static void set_tensor_column(ggml_tensor * t, int64_t col,
                              const std::vector<float> & vals) {
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(ggml_n_dims(t), 2);
    const int64_t rows = t->ne[0];
    ASSERT_EQ((int64_t) vals.size(), rows);
    ASSERT_GE(col, 0);
    ASSERT_LT(col, t->ne[1]);

    std::vector<float> buf((size_t) ggml_nelements(t), 0.0f);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    std::copy(vals.begin(), vals.end(), buf.begin() + (size_t) (col * rows));
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
}

// A zero-mean unit-ish direction vector: e_idx - mean(1).
static std::vector<float> zero_mean_dir(int64_t d, int64_t idx) {
    std::vector<float> v((size_t) d, -1.0f / (float) d);
    if (idx >= 0 && idx < d) {
        v[(size_t) idx] += 1.0f;
    }
    return v;
}

static std::vector<float> scale_vec(const std::vector<float> & v, float s) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = v[i] * s;
    return out;
}

static std::vector<float> make_zeros(size_t n) {
    return std::vector<float>(n, 0.0f);
}

// Explicitly zero every weight.  The GGUF loader normally provides initialized
// data; here the tensors are freshly allocated, so nothing may be left
// uninitialized.
static void create_synthetic_model(const SynthSpec & spec,
                                   ggml_backend_t backend,
                                   gpt_sovits::t2s_model & model) {
    ASSERT_NE(backend, nullptr);
    ASSERT_GE(spec.vocab, 2u);
    ASSERT_GT(spec.n_layer, 0u);
    ASSERT_GT(spec.n_head, 0u);
    ASSERT_EQ(spec.d_model % spec.n_head, 0u);

    gpt_sovits::t2s_hparams hp;
    hp.embedding_dim      = spec.d_model;
    hp.hidden_dim         = spec.d_model;
    hp.n_head             = spec.n_head;
    hp.linear_units       = spec.linear_units;
    hp.n_layer            = spec.n_layer;
    hp.vocab_size         = spec.vocab;
    hp.phoneme_vocab_size = spec.phoneme_vocab;
    hp.eos                = spec.vocab - 1;
    hp.inter_channels     = (uint32_t) kHubertDim;

    model.hparams = hp;
    model.backend = backend;

    const int64_t d  = (int64_t) spec.d_model;
    const int64_t d3 = 3 * d;
    const int64_t df = (int64_t) spec.linear_units;
    const int64_t v  = (int64_t) spec.vocab;
    const int64_t pv = (int64_t) spec.phoneme_vocab;
    const int64_t nc = v - 1;  // codebook covers every non-EOS token

    const size_t n_tensors = (size_t) (10 + 12 * spec.n_layer + 4);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model.ctx_w = ggml_init(params);
    ASSERT_NE(model.ctx_w, nullptr);
    ggml_context * ctx = model.ctx_w;

    auto & el = model.weights.extract_latent;
    // ggml_conv_1d requires an F16 kernel weight.  The buffer is cleared to
    // zero below, and zero is a valid F16 bit pattern.
    el.ssl_proj_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, kHubertDim, kHubertDim);
    el.ssl_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHubertDim);
    el.codebook   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHubertDim, nc);

    auto & em = model.weights.embed;
    em.text_embedding  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, pv);
    em.bert_proj_w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, d);
    em.bert_proj_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
    em.text_pos_alpha  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    em.audio_embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, v);
    em.audio_pos_alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    model.weights.attention.resize(spec.n_layer);
    for (uint32_t i = 0; i < spec.n_layer; i++) {
        auto & w = model.weights.attention[i];
        w.qkv_w      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d3);
        w.qkv_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d3);
        w.out_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
        w.out_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        w.ln1_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        w.ln1_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        w.ffn_up_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, df);
        w.ffn_up_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, df);
        w.ffn_down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, df, d);
        w.ffn_down_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        w.ln2_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        w.ln2_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
    }

    model.weights.lm_head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, v);

    model.buf_w = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ASSERT_NE(model.buf_w, nullptr);

    ggml_backend_buffer_clear(model.buf_w, 0);
}

// One-layer block that is exactly y = LN(LN(x)): QKV/attention/FFN are zero,
// so the last token's output depends only on its own input vector.  Tests pick
// the next token by choosing the input direction and LM-head column.
static void setup_ln_only(gpt_sovits::t2s_model & model) {
    const int64_t d = model.hparams.hidden_dim;
    std::vector<float> ones((size_t) d, 1.0f);
    std::vector<float> zeros((size_t) d, 0.0f);
    for (auto & layer : model.weights.attention) {
        set_tensor_f32(layer.ln1_w, ones);
        set_tensor_f32(layer.ln1_b, zeros);
        set_tensor_f32(layer.ln2_w, ones);
        set_tensor_f32(layer.ln2_b, zeros);
    }
    set_tensor_scalar(model.weights.embed.text_pos_alpha, 0.0f);
    set_tensor_scalar(model.weights.embed.audio_pos_alpha, 0.0f);
}

// Always predict token `a`: every audio embedding is u_A and lm_head column
// `a` is u_A, so prefill and every decode step produce A.
static void setup_constant_token(gpt_sovits::t2s_model & model, int32_t a) {
    setup_ln_only(model);
    const int64_t d = model.hparams.hidden_dim;
    const int64_t v = model.hparams.vocab_size;
    const std::vector<float> u = zero_mean_dir(d, 0);
    for (int64_t t = 0; t < v; t++) {
        set_tensor_column(model.weights.embed.audio_embedding, t, u);
    }
    set_tensor_column(model.weights.lm_head_w, a, u);
}

static std::vector<float> ref_audio_ending_in(int64_t d, int64_t T_prompt,
                                              const std::vector<float> & last) {
    std::vector<float> v((size_t) (d * T_prompt), 0.0f);
    std::copy(last.begin(), last.end(), v.begin() + (size_t) ((T_prompt - 1) * d));
    return v;
}

// KV caches are uninitialized at session init.  Masked positions are supposed
// not to contribute, but zeroing them keeps the synthetic tests deterministic
// on every backend.
static void zero_session_kv_caches(gpt_sovits::t2s_session & session) {
    for (size_t i = 0; i < session.k_caches.size(); i++) {
        ggml_backend_tensor_memset(session.k_caches[i], 0, 0,
                                   ggml_nbytes(session.k_caches[i]));
        ggml_backend_tensor_memset(session.v_caches[i], 0, 0,
                                   ggml_nbytes(session.v_caches[i]));
    }
}

// Install a cached reference embedding in the same persistent form that
// compute_ref_emb produces.  ctx_ref/buf_ref are owned by the session.
static void install_cached_ref(gpt_sovits::t2s_session & session,
                               const gpt_sovits::t2s_model & model,
                               int64_t T_ref, int64_t T_prompt,
                               const std::vector<float> & text,
                               const std::vector<float> & audio) {
    const int64_t d = model.hparams.hidden_dim;
    ASSERT_EQ((int64_t) text.size(), d * T_ref);
    ASSERT_EQ((int64_t) audio.size(), d * T_prompt);
    ASSERT_GT(T_ref, 0);
    ASSERT_GT(T_prompt, 0);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 3,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    session.ctx_ref = ggml_init(params);
    ASSERT_NE(session.ctx_ref, nullptr);

    session.ref_text_emb  = ggml_new_tensor_2d(session.ctx_ref, GGML_TYPE_F32, d, T_ref);
    session.ref_audio_emb = ggml_new_tensor_2d(session.ctx_ref, GGML_TYPE_F32, d, T_prompt);

    session.buf_ref = ggml_backend_alloc_ctx_tensors(session.ctx_ref, session.backend);
    ASSERT_NE(session.buf_ref, nullptr);

    set_tensor_f32(session.ref_text_emb, text);
    set_tensor_f32(session.ref_audio_emb, audio);

    session.ref_T_ref    = T_ref;
    session.ref_T_prompt = T_prompt;
    session.ref_model    = &model;
}

static gpt_sovits::t2s_sampler_config make_sampler_config(int top_k = 1,
                                                          float rep_penalty = 1.0f,
                                                          float top_p = 1.0f,
                                                          float temperature = 1.0f) {
    gpt_sovits::t2s_sampler_config cfg;
    cfg.top_k              = top_k;
    cfg.top_p              = top_p;
    cfg.temperature        = temperature;
    cfg.repetition_penalty = rep_penalty;
    return cfg;
}

static void init_one_slot_session(gpt_sovits::t2s_session & session,
                                  const gpt_sovits::t2s_model & model,
                                  uint32_t slot_size,
                                  const gpt_sovits::t2s_sampler_config & sampler_cfg) {
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, model.backend,
                                             1, slot_size, sampler_cfg));
    zero_session_kv_caches(session);
}

struct Rig {
    ggml_backend_t backend = nullptr;
    gpt_sovits::t2s_model model;
    gpt_sovits::t2s_session session;

    ~Rig() {
        gpt_sovits::t2s_session_free(session);
        gpt_sovits::t2s_model_free(model);
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

static void init_rig(Rig & rig, const SynthSpec & spec, uint32_t slot_size,
                     const gpt_sovits::t2s_sampler_config & sampler_cfg) {
    rig.backend = ggml_backend_cpu_init();
    ASSERT_NE(rig.backend, nullptr);
    create_synthetic_model(spec, rig.backend, rig.model);
    init_one_slot_session(rig.session, rig.model, slot_size, sampler_cfg);
}

// Standard synthetic reference: T_ref=2, T_prompt=2, zero ref text.
static void install_zero_ref(Rig & rig, const std::vector<float> & audio) {
    const int64_t d = rig.model.hparams.hidden_dim;
    install_cached_ref(rig.session, rig.model, 2, 2,
                       make_zeros((size_t) (d * 2)), audio);
}

static gpt_sovits::t2s_generate_config gen_cfg(int32_t max_new_tokens,
                                               uint32_t seed = 42) {
    gpt_sovits::t2s_generate_config c;
    c.max_new_tokens = max_new_tokens;
    c.seed = seed;
    return c;
}

static gpt_sovits::t2s_generate_result make_sentinel_result() {
    gpt_sovits::t2s_generate_result r;
    r.codes = {12345, -42};
    r.stop_reason = gpt_sovits::t2s_stop_reason::eos;
    return r;
}

static void expect_result_unchanged(const gpt_sovits::t2s_generate_result & r) {
    EXPECT_EQ(r.codes, (std::vector<int32_t>{12345, -42}));
    EXPECT_EQ(r.stop_reason, gpt_sovits::t2s_stop_reason::eos);
}

// ---------------------------------------------------------------------------
// Test-only failing backend wrapper (graph_compute only).
//
// Mirrors the wrapper already used by the SoVITS parity tests: it copies the
// real backend interface and only overrides graph_compute, so no production
// test hook is needed.
// ---------------------------------------------------------------------------

struct FailInjection {
    ggml_backend_t real         = nullptr;  // borrowed
    int            fail_at_call = -1;      // 1-based; -1 disables
    int            calls        = 0;
    int32_t        sampled      = -1;
    int32_t        greedy       = -1;
};

static enum ggml_status fail_injection_graph_compute(ggml_backend_t backend,
                                                     struct ggml_cgraph * cgraph) {
    FailInjection * inj = static_cast<FailInjection *>(backend->context);
    inj->calls += 1;
    if (inj->fail_at_call > 0 && inj->calls >= inj->fail_at_call) {
        return GGML_STATUS_FAILED;
    }
    const auto status = ggml_backend_graph_compute(inj->real, cgraph);
    if (status == GGML_STATUS_SUCCESS) {
        auto * sampled = ggml_graph_get_tensor(cgraph, "sampled");
        auto * greedy = ggml_graph_get_tensor(cgraph, "greedy");
        if (sampled && greedy) {
            ggml_backend_tensor_get(sampled, &inj->sampled, 0, sizeof(int32_t));
            ggml_backend_tensor_get(greedy, &inj->greedy, 0, sizeof(int32_t));
        }
    }
    return status;
}

static const char * fail_injection_get_name(ggml_backend_t) {
    return "test-fail-injection";
}

static void fail_injection_free(ggml_backend_t backend) {
    delete static_cast<FailInjection *>(backend->context);
    delete backend;
}

static ggml_backend_t wrap_failing_backend(ggml_backend_t real,
                                          FailInjection ** out) {
    FailInjection * inj = new FailInjection{real, -1, 0};
    ggml_backend_t wrapped = new ggml_backend{
        /* .guid    = */ real->guid,
        /* .iface   = */ real->iface,
        /* .device  = */ real->device,
        /* .context = */ inj,
    };
    wrapped->iface.graph_compute = fail_injection_graph_compute;
    wrapped->iface.get_name      = fail_injection_get_name;
    wrapped->iface.free          = fail_injection_free;
    *out = inj;
    return wrapped;
}

} // namespace

// ===========================================================================
// EOS handling and termination
// ===========================================================================

TEST(T2SGenerateSynthetic, PrefillEosStopsWithEmptyResult) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(1));

    setup_ln_only(rig.model);
    const int64_t d = rig.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);

    // The final prefill column (last ref-audio token) maps directly to EOS.
    set_tensor_column(rig.model.weights.lm_head_w, spec.vocab - 1, u);
    install_zero_ref(rig, ref_audio_ending_in(d, 2, u));

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    gpt_sovits::t2s_generate_result result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), result, gen_cfg(4)));

    EXPECT_TRUE(result.codes.empty());
    EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::eos);

    // EOS in prefill must not build a decode graph or leave the slot active.
    EXPECT_EQ(rig.session.gf_dec, nullptr);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
    EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(rig.session), 0);
    EXPECT_EQ(rig.session.ref_model, &rig.model);
}

TEST(T2SGenerateSynthetic, GreedyOnlyEosDropsNonEosCandidate) {
    BackendGuard real;
    real.backend = ggml_backend_cpu_init();
    ASSERT_NE(real.backend, nullptr);
    FailInjection * observer = nullptr;
    BackendGuard wrapped;
    wrapped.backend = wrap_failing_backend(real.backend, &observer);
    ModelGuard model;
    const SynthSpec spec;
    create_synthetic_model(spec, wrapped.backend, model.model);
    setup_ln_only(model.model);
    const auto u = zero_mean_dir(spec.d_model, 0);
    const int32_t eos = spec.vocab - 1;
    set_tensor_column(model.model.weights.lm_head_w, eos, scale_vec(u, 0.1f));
    SessionGuard holder;
    auto & session = holder.session;
    init_one_slot_session(session, model.model, 16, make_sampler_config(0));
    install_cached_ref(session, model.model, 2, 2,
                       make_zeros(spec.d_model * 2), ref_audio_ending_in(spec.d_model, 2, u));
    const int32_t tokens[] = {0, 1};
    const auto bert = make_zeros(2048);
    bool observed_non_eos = false;
    for (uint32_t seed = 0; seed < 32 && !observed_non_eos; ++seed) {
        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        ASSERT_TRUE(gpt_sovits::t2s_session_generate(
            session, model.model, tokens, 2, bert.data(), result, gen_cfg(4, seed)));
        ASSERT_EQ(observer->greedy, eos);
        EXPECT_TRUE(result.codes.empty());
        EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::eos);
        EXPECT_EQ(session.gf_dec, nullptr);
        EXPECT_EQ(session.slots[0].n_pos, 0);
        observed_non_eos = observer->sampled != eos;
    }
    EXPECT_TRUE(observed_non_eos) << "must exercise sampled != EOS and greedy == EOS";
}

TEST(T2SGenerateSynthetic, DecodeEosStopsBeforeAppendingCandidate) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(1));

    setup_ln_only(rig.model);
    const int64_t d = rig.model.hparams.hidden_dim;
    const int32_t a = 0;
    const std::vector<float> u_a = zero_mean_dir(d, 0);
    const std::vector<float> u_e = zero_mean_dir(d, 1);

    // Prefill last token -> A.  Decoding A -> EOS.
    set_tensor_column(rig.model.weights.lm_head_w, a, u_a);
    set_tensor_column(rig.model.weights.lm_head_w, spec.vocab - 1, u_e);
    set_tensor_column(rig.model.weights.embed.audio_embedding, a, u_e);
    install_zero_ref(rig, ref_audio_ending_in(d, 2, u_a));

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    gpt_sovits::t2s_generate_result result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), result, gen_cfg(5)));

    ASSERT_EQ(result.codes.size(), 1u);
    EXPECT_EQ(result.codes[0], a);
    EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::eos);

    // A decode graph was needed and is retained; the slot is released.
    EXPECT_NE(rig.session.gf_dec, nullptr);
    EXPECT_EQ(rig.session.decode_model, &rig.model);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
}

TEST(T2SGenerateSynthetic, MaxNewTokensOneCountsPrefillToken) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 6, make_sampler_config(1));

    setup_constant_token(rig.model, 0);
    const int64_t d = rig.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);
    install_zero_ref(rig, ref_audio_ending_in(d, 2, u));

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    gpt_sovits::t2s_generate_result result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), result, gen_cfg(1)));

    EXPECT_EQ(result.codes, (std::vector<int32_t>{0}));
    EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
    EXPECT_EQ(rig.session.gf_dec, nullptr) << "M=1 must not need a decode graph";
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
}

TEST(T2SGenerateSynthetic, ExactMaxAndSlotBoundary) {
    const SynthSpec spec;
    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    const int64_t d = spec.d_model;
    const std::vector<float> u = zero_mean_dir(d, 0);
    const std::vector<float> ref_audio = ref_audio_ending_in(d, 2, u);

    // N = T_ref(2) + T_in(2) + T_prompt(2) = 6.
    // N + M - 1 = 10 == slot_size: exactly fits.
    {
        Rig rig;
        init_rig(rig, spec, 10, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_generate_result result;
        ASSERT_TRUE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(5)));

        EXPECT_EQ(result.codes, (std::vector<int32_t>(5, 0)));
        EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
        EXPECT_NE(rig.session.gf_dec, nullptr);
        EXPECT_EQ(rig.session.decode_model, &rig.model);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
    }

    // One position short: N + M - 1 > slot_size.
    {
        Rig rig;
        init_rig(rig, spec, 9, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(5)));
        expect_result_unchanged(result);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
        EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(rig.session), 0);
    }
}

// ===========================================================================
// Input / config validation
// ===========================================================================

TEST(T2SGenerateSynthetic, InvalidInputsRejectedWithoutSideEffects) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(1));
    setup_constant_token(rig.model, 0);

    const int64_t d = rig.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);
    install_zero_ref(rig, ref_audio_ending_in(d, 2, u));

    std::vector<int32_t> tokens = {0, 1};
    std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    auto expect_fail = [&](const int32_t * tok, int64_t T_in, const float * b) {
        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tok, T_in, b, result, gen_cfg(2)));
        expect_result_unchanged(result);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
        EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(rig.session), 0);
    };

    expect_fail(nullptr, (int64_t) tokens.size(), bert.data());
    expect_fail(tokens.data(), (int64_t) tokens.size(), nullptr);
    expect_fail(tokens.data(), 0, bert.data());
    expect_fail(tokens.data(), -3, bert.data());

    {
        std::vector<int32_t> bad = tokens;
        bad[0] = -1;
        expect_fail(bad.data(), (int64_t) bad.size(), bert.data());
    }
    {
        std::vector<int32_t> bad = tokens;
        bad[1] = (int32_t) spec.phoneme_vocab;
        expect_fail(bad.data(), (int64_t) bad.size(), bert.data());
    }
    {
        std::vector<float> bad = bert;
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        expect_fail(tokens.data(), (int64_t) tokens.size(), bad.data());
    }
    {
        std::vector<float> bad = bert;
        bad.back() = std::numeric_limits<float>::infinity();
        expect_fail(tokens.data(), (int64_t) tokens.size(), bad.data());
    }

    // Tiny host buffers: capacity validation must reject before the token/BERT
    // buffers are scanned.
    int32_t tiny_token = 0;
    float tiny_bert = 0.0f;
    expect_fail(&tiny_token, 16, &tiny_bert);  // N = 6 + 16 > slot_size
    expect_fail(&tiny_token, 2000000000LL, &tiny_bert);
    expect_fail(&tiny_token, std::numeric_limits<int64_t>::max(), &tiny_bert);

    // The rig is still usable.
    gpt_sovits::t2s_generate_result ok;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), ok, gen_cfg(2)));
    EXPECT_EQ(ok.codes, (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(ok.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
}

TEST(T2SGenerateSynthetic, InvalidConfigRejected) {
    const SynthSpec spec;
    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));
    const int64_t d = spec.d_model;
    const std::vector<float> u = zero_mean_dir(d, 0);
    const std::vector<float> ref_audio = ref_audio_ending_in(d, 2, u);

    // Invalid max_new_tokens and capacity.
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        for (int32_t M : {0, -1, 12}) {  // 12 -> N + M - 1 = 17 > 16
            gpt_sovits::t2s_generate_result result = make_sentinel_result();
            EXPECT_FALSE(gpt_sovits::t2s_session_generate(
                rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
                bert.data(), result, gen_cfg(M)))
                << "M=" << M;
            expect_result_unchanged(result);
            EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
        }
    }

    // Invalid sampler configuration (bound at session init).
    struct BadSampler {
        gpt_sovits::t2s_sampler_config cfg;
        const char * what;
    };
    std::vector<BadSampler> bad_samplers;
    {
        auto c = make_sampler_config(1);
        c.top_p = std::numeric_limits<float>::quiet_NaN();
        bad_samplers.push_back({c, "top_p NaN"});
    }
    {
        auto c = make_sampler_config(1);
        c.temperature = std::numeric_limits<float>::infinity();
        bad_samplers.push_back({c, "temperature inf"});
    }
    {
        auto c = make_sampler_config(1);
        c.repetition_penalty = 0.0f;
        bad_samplers.push_back({c, "repetition_penalty 0"});
    }

    for (const auto & bad : bad_samplers) {
        Rig rig;
        init_rig(rig, spec, 16, bad.cfg);
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)))
            << bad.what;
        expect_result_unchanged(result);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0)
            << bad.what;
    }
}

TEST(T2SGenerateSynthetic, RequestContractRejections) {
    const SynthSpec spec;
    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));
    const int64_t d = spec.d_model;
    const std::vector<float> u = zero_mean_dir(d, 0);
    const std::vector<float> ref_audio = ref_audio_ending_in(d, 2, u);

    // --- n_batch must be 1 ---
    {
        BackendGuard backend;
        backend.backend = ggml_backend_cpu_init();
        ASSERT_NE(backend.backend, nullptr);

        ModelGuard model;
        create_synthetic_model(spec, backend.backend, model.model);
        setup_constant_token(model.model, 0);

        SessionGuard session;
        ASSERT_TRUE(gpt_sovits::t2s_session_init(session.session,
                                                 model.model.hparams,
                                                 backend.backend, 2, 16,
                                                 make_sampler_config(1)));
        zero_session_kv_caches(session.session);
        install_cached_ref(session.session, model.model, 2, 2,
                           make_zeros((size_t) (d * 2)), ref_audio);

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            session.session, model.model, tokens.data(),
            (int64_t) tokens.size(), bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
    }

    // --- reference embedding must have been computed ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
        EXPECT_EQ(rig.session.ref_model, nullptr);
    }

    // --- reference belongs to another model ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        ModelGuard other;
        create_synthetic_model(spec, rig.backend, other.model);
        setup_constant_token(other.model, 0);

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, other.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
        EXPECT_EQ(rig.session.ref_model, &rig.model);
    }

    // --- an already-active slot is rejected without modification ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        rig.session.slots[0].n_pos = 1;
        rig.session.slots[0].audio_offset = 0;
        const std::vector<ggml_fp16_t> mask_before = rig.session.mask_host;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
        EXPECT_EQ(rig.session.slots[0].n_pos, 1);
        EXPECT_EQ(rig.session.mask_host, mask_before);
    }

    // --- model and session must share the same backend ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_model bad = rig.model;
        bad.backend = nullptr;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, bad, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
    }

    // --- existing decode graph must belong to the same model ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);
        ASSERT_TRUE(gpt_sovits::t2s_session_build_decode_graph(rig.session,
                                                               rig.model));

        ggml_cgraph * gf_before = rig.session.gf_dec;
        const gpt_sovits::t2s_model * dec_model_before = rig.session.decode_model;

        ModelGuard other;
        create_synthetic_model(spec, rig.backend, other.model);
        setup_constant_token(other.model, 0);

        // Let the reference check pass so the decode-model check is reached.
        rig.session.ref_model = &other.model;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, other.model, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
        EXPECT_EQ(rig.session.gf_dec, gf_before);
        EXPECT_EQ(rig.session.decode_model, dec_model_before);
    }

    // --- incomplete weights ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_model bad = rig.model;
        bad.weights.lm_head_w = nullptr;
        rig.session.ref_model = &bad;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, bad, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
    }

    // --- EOS must be the last vocabulary entry ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        gpt_sovits::t2s_model bad = rig.model;
        bad.hparams.eos = spec.vocab - 2;
        rig.session.ref_model = &bad;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, bad, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);
    }

    // --- codebook must cover every non-EOS token ---
    {
        Rig rig;
        init_rig(rig, spec, 16, make_sampler_config(1));
        setup_constant_token(rig.model, 0);
        install_zero_ref(rig, ref_audio);

        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 2,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * fake_ctx = ggml_init(params);
        ASSERT_NE(fake_ctx, nullptr);

        gpt_sovits::t2s_model bad = rig.model;
        bad.weights.extract_latent.codebook = ggml_new_tensor_2d(
            fake_ctx, GGML_TYPE_F32, kHubertDim,
            (int64_t) spec.vocab - 2);
        rig.session.ref_model = &bad;

        gpt_sovits::t2s_generate_result result = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            rig.session, bad, tokens.data(), (int64_t) tokens.size(),
            bert.data(), result, gen_cfg(2)));
        expect_result_unchanged(result);

        ggml_free(fake_ctx);
    }
}

TEST(T2SGenerateSynthetic, SessionVocabularyMismatchRejectedBeforePrefill) {
    BackendGuard backend;
    backend.backend = ggml_backend_cpu_init();
    ASSERT_NE(backend.backend, nullptr);
    ModelGuard model;
    const SynthSpec spec;
    create_synthetic_model(spec, backend.backend, model.model);
    setup_constant_token(model.model, 0);
    const std::vector<int32_t> ref_tokens = {0, 1};
    const auto ref_bert = make_zeros(2048);
    const auto hubert = make_zeros(768 * 4);
    const std::vector<int32_t> tokens = {1, 2};
    const auto bert = make_zeros(2048);

    for (uint32_t session_vocab : {2u, 6u}) {
        for (float repetition_penalty : {1.0f, 1.35f}) {
            SCOPED_TRACE(::testing::Message() << "session_vocab=" << session_vocab
                                            << ", repetition_penalty=" << repetition_penalty);
            SessionGuard holder;
            auto & session = holder.session;
            auto hp = model.model.hparams;
            hp.vocab_size = session_vocab;
            hp.eos = session_vocab - 1;
            ASSERT_TRUE(gpt_sovits::t2s_session_init(
                session, hp, backend.backend, 1, 16,
                make_sampler_config(1, repetition_penalty)));
            zero_session_kv_caches(session);
            ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
                session, model.model, ref_tokens.data(), ref_tokens.size(),
                ref_bert.data(), hubert.data(), 4));
            ASSERT_EQ(session.ref_T_prompt, 2);
            ASSERT_EQ(session.ref_model, &model.model);
            const auto mask_before = session.mask_host;
            auto * ref_text = session.ref_text_emb;
            auto * ref_audio = session.ref_audio_emb;

            for (int32_t max_new_tokens : {1, 2}) {
                SCOPED_TRACE(max_new_tokens);
                auto result = make_sentinel_result();
                ASSERT_FALSE(gpt_sovits::t2s_session_generate(
                    session, model.model, tokens.data(), tokens.size(),
                    bert.data(), result, gen_cfg(max_new_tokens)));
                expect_result_unchanged(result);
                EXPECT_EQ(session.slots[0].n_pos, 0);
                EXPECT_EQ(session.slots[0].audio_offset, 0);
                EXPECT_EQ(session.mask_host, mask_before);
                EXPECT_EQ(session.ref_text_emb, ref_text);
                EXPECT_EQ(session.ref_audio_emb, ref_audio);
                EXPECT_EQ(session.ref_model, &model.model);
                EXPECT_EQ(session.alloc_flex, nullptr);
                EXPECT_EQ(session.ctx_graph, nullptr);
                EXPECT_EQ(session.gf_dec, nullptr);
                EXPECT_EQ(session.decode_model, nullptr);
            }
        }
    }
}

// ===========================================================================
// Reproducibility and slot / cache reuse
// ===========================================================================

TEST(T2SGenerateSynthetic, SameSeedReproducibleAndDecodeReused) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(0));

    // A/B sampling: both are non-EOS, so the run always reaches max_new_tokens.
    setup_ln_only(rig.model);
    const int64_t d = rig.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);
    const int32_t a = 0;
    const int32_t b = 1;
    const int32_t e = (int32_t) spec.vocab - 1;

    for (int64_t t = 0; t < (int64_t) spec.vocab; t++) {
        set_tensor_column(rig.model.weights.embed.audio_embedding, t, u);
    }
    set_tensor_column(rig.model.weights.lm_head_w, a, u);
    set_tensor_column(rig.model.weights.lm_head_w, b, scale_vec(u, 0.8f));
    set_tensor_column(rig.model.weights.lm_head_w, e, scale_vec(u, -8.0f));
    set_tensor_column(rig.model.weights.lm_head_w, spec.vocab - 2,
                      scale_vec(u, -8.0f));

    const std::vector<float> ref_audio = ref_audio_ending_in(d, 2, u);
    install_zero_ref(rig, ref_audio);

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    gpt_sovits::t2s_generate_result first;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), first, gen_cfg(8, 7)));
    ASSERT_EQ(first.codes.size(), 8u);
    EXPECT_EQ(first.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);

    ggml_cgraph * dec_graph = rig.session.gf_dec;
    ASSERT_NE(dec_graph, nullptr);

    std::vector<uint8_t> ref_text_before((size_t) ggml_nbytes(rig.session.ref_text_emb));
    ggml_backend_tensor_get(rig.session.ref_text_emb, ref_text_before.data(), 0,
                            ref_text_before.size());
    std::vector<uint8_t> ref_audio_before((size_t) ggml_nbytes(rig.session.ref_audio_emb));
    ggml_backend_tensor_get(rig.session.ref_audio_emb, ref_audio_before.data(), 0,
                            ref_audio_before.size());

    gpt_sovits::t2s_generate_result second;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), second, gen_cfg(8, 7)));

    EXPECT_EQ(second.codes, first.codes);
    EXPECT_EQ(second.stop_reason, first.stop_reason);
    EXPECT_EQ(rig.session.gf_dec, dec_graph) << "decode graph must be reused";
    EXPECT_EQ(rig.session.decode_model, &rig.model);
    EXPECT_EQ(rig.session.ref_model, &rig.model);

    std::vector<uint8_t> ref_text_after((size_t) ggml_nbytes(rig.session.ref_text_emb));
    ggml_backend_tensor_get(rig.session.ref_text_emb, ref_text_after.data(), 0,
                            ref_text_after.size());
    std::vector<uint8_t> ref_audio_after((size_t) ggml_nbytes(rig.session.ref_audio_emb));
    ggml_backend_tensor_get(rig.session.ref_audio_emb, ref_audio_after.data(), 0,
                            ref_audio_after.size());
    EXPECT_EQ(ref_text_after, ref_text_before)
        << "reference text cache must be unchanged";
    EXPECT_EQ(ref_audio_after, ref_audio_before)
        << "reference audio cache must be unchanged";

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
    EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(rig.session), 0);
}

TEST(T2SGenerateSynthetic, LongThenShortIsPrefixAndResetsSlot) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(0));

    setup_ln_only(rig.model);
    const int64_t d = rig.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);
    const int32_t a = 0;
    const int32_t b = 1;
    const int32_t e = (int32_t) spec.vocab - 1;

    for (int64_t t = 0; t < (int64_t) spec.vocab; t++) {
        set_tensor_column(rig.model.weights.embed.audio_embedding, t, u);
    }
    set_tensor_column(rig.model.weights.lm_head_w, a, u);
    set_tensor_column(rig.model.weights.lm_head_w, b, scale_vec(u, 0.8f));
    set_tensor_column(rig.model.weights.lm_head_w, e, scale_vec(u, -8.0f));
    set_tensor_column(rig.model.weights.lm_head_w, spec.vocab - 2,
                      scale_vec(u, -8.0f));

    const std::vector<float> ref_audio = ref_audio_ending_in(d, 2, u);
    install_zero_ref(rig, ref_audio);

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    gpt_sovits::t2s_generate_result long_result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), long_result, gen_cfg(8, 3)));
    ASSERT_EQ(long_result.codes.size(), 8u);

    gpt_sovits::t2s_generate_result short_result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        bert.data(), short_result, gen_cfg(3, 3)));

    ASSERT_EQ(short_result.codes.size(), 3u);
    EXPECT_EQ(std::vector<int32_t>(long_result.codes.begin(),
                                   long_result.codes.begin() + 3),
              short_result.codes)
        << "short run must reproduce the long run's prefix";
    EXPECT_EQ(short_result.stop_reason,
              gpt_sovits::t2s_stop_reason::max_new_tokens);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(rig.session, 0), 0);
}

TEST(T2SGenerateSynthetic, ComputeRefEmbThenGenerate) {
    const SynthSpec spec;
    Rig rig;
    init_rig(rig, spec, 16, make_sampler_config(1));

    setup_constant_token(rig.model, 0);

    const int64_t T_ref = 3;
    const int64_t T_hub = 8;
    const int64_t T_prompt = (T_hub - 2) / 2 + 1;  // kernel 2, stride 2

    std::vector<int32_t> ref_tokens((size_t) T_ref, 1);
    std::vector<float> ref_bert((size_t) (1024 * T_ref), 0.25f);
    std::vector<float> hubert((size_t) (768 * T_hub), 0.5f);

    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        rig.session, rig.model, ref_tokens.data(), T_ref,
        ref_bert.data(), hubert.data(), T_hub));

    EXPECT_EQ(rig.session.ref_model, &rig.model);
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_ref(rig.session), T_ref);
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_prompt(rig.session), T_prompt);
    ASSERT_NE(gpt_sovits::t2s_session_get_ref_text_emb(rig.session), nullptr);
    ASSERT_NE(gpt_sovits::t2s_session_get_ref_audio_emb(rig.session), nullptr);

    const std::vector<int32_t> tokens = {1, 2};
    const std::vector<float> target_bert((size_t) (1024 * tokens.size()), 0.1f);

    gpt_sovits::t2s_generate_result result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        rig.session, rig.model, tokens.data(), (int64_t) tokens.size(),
        target_bert.data(), result, gen_cfg(3)));

    EXPECT_EQ(result.codes, (std::vector<int32_t>(3, 0)));
    EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
    EXPECT_NE(rig.session.gf_dec, nullptr);
}

// ===========================================================================
// Compute-failure recovery
// ===========================================================================

TEST(T2SGenerateSynthetic, ComputeFailureReleasesSlotAndRetries) {
    BackendGuard real;
    real.backend = ggml_backend_cpu_init();
    ASSERT_NE(real.backend, nullptr);

    FailInjection * inj = nullptr;
    BackendGuard wrapped;
    wrapped.backend = wrap_failing_backend(real.backend, &inj);
    ASSERT_NE(wrapped.backend, nullptr);
    ASSERT_NE(inj, nullptr);

    const SynthSpec spec;
    ModelGuard model;
    create_synthetic_model(spec, wrapped.backend, model.model);
    setup_constant_token(model.model, 0);

    SessionGuard session;
    init_one_slot_session(session.session, model.model, 16,
                          make_sampler_config(1));

    const int64_t d = model.model.hparams.hidden_dim;
    const std::vector<float> u = zero_mean_dir(d, 0);
    install_cached_ref(session.session, model.model, 2, 2,
                       make_zeros((size_t) (d * 2)),
                       ref_audio_ending_in(d, 2, u));

    const std::vector<int32_t> tokens = {0, 1};
    const std::vector<float> bert = make_zeros((size_t) (1024 * tokens.size()));

    // Calls per run: embedding(1), prefill(2), decode(3..).
    for (int fail_at : {1, 2, 3, 4}) {
        inj->calls        = 0;
        inj->fail_at_call = fail_at;

        gpt_sovits::t2s_generate_result failed = make_sentinel_result();
        EXPECT_FALSE(gpt_sovits::t2s_session_generate(
            session.session, model.model, tokens.data(),
            (int64_t) tokens.size(), bert.data(), failed, gen_cfg(4)))
            << "fail_at=" << fail_at;

        expect_result_unchanged(failed);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session.session, 0), 0)
            << "fail_at=" << fail_at;
        EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(session.session), 0)
            << "fail_at=" << fail_at;

        // Recovery: the same session must be immediately reusable.
        inj->fail_at_call = -1;
        inj->calls        = 0;

        gpt_sovits::t2s_generate_result ok;
        EXPECT_TRUE(gpt_sovits::t2s_session_generate(
            session.session, model.model, tokens.data(),
            (int64_t) tokens.size(), bert.data(), ok, gen_cfg(4)))
            << "retry after fail_at=" << fail_at;
        EXPECT_EQ(ok.codes, (std::vector<int32_t>(4, 0)))
            << "retry after fail_at=" << fail_at;
        EXPECT_EQ(ok.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens)
            << "retry after fail_at=" << fail_at;
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session.session, 0), 0);
    }
}
