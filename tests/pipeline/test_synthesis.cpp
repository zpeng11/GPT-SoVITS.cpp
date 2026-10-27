// tests/pipeline/test_synthesis.cpp
//
// Contract tests for src/pipeline/synthesis_internal.h.  Covers the slot
// budget, config, codebook and models validators without any model weights,
// plus the generate/decode immediate-EOS and pre-compute input rejection
// paths.
//
// A real SoVITS graph is never built.  Max-length success, SoVITS failure and
// the public synthesize_cpu failure-injection ordering are left to the
// integration suite.  The synthetic T2S model is the same minimal LN-only
// construction used by tests/t2s/test_t2s_generate.cpp, reduced to what the
// immediate-EOS path needs.

#include <gtest/gtest.h>

#include "gpt_sovits/hubert.h"
#include "gpt_sovits/pipeline.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"

#include "../../src/pipeline/synthesis_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace gs = gpt_sovits;
namespace {

static gs::t2s_generate_config gen_cfg(int32_t max_new_tokens, uint32_t seed = 42) {
    gs::t2s_generate_config c;
    c.max_new_tokens = max_new_tokens;
    c.seed = seed;
    return c;
}

static gs::synthesis_result sentinel_result() {
    gs::synthesis_result r;
    r.samples = {1.5f, -2.5f};
    r.sample_rate = 123456;
    r.codes = {7, -3};
    r.stop_reason = gs::t2s_stop_reason::eos;
    return r;
}

static void expect_sentinel(const gs::synthesis_result & r) {
    EXPECT_EQ(r.samples, (std::vector<float>{1.5f, -2.5f}));
    EXPECT_EQ(r.sample_rate, 123456);
    EXPECT_EQ(r.codes, (std::vector<int32_t>{7, -3}));
    EXPECT_EQ(r.stop_reason, gs::t2s_stop_reason::eos);
}

// ===========================================================================
// synthesis_slot_size
// ===========================================================================

TEST(SynthesisSlotSize, BoundaryAndFailurePreserveOut) {
    uint32_t out = 0;
    std::string err = "stale";
    EXPECT_TRUE(gs::pipeline_detail::synthesis_slot_size(1, 1, 1, 1, out, &err));
    EXPECT_EQ(out, 3u);
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(gs::pipeline_detail::synthesis_slot_size(1, 1, 1022, 1, out, &err));
    EXPECT_EQ(out, 1024u);  // exactly the limit

    const uint32_t kPreserved = 0xdeadbeefu;
    auto rejects = [&](int64_t r, int64_t t, int64_t p, int64_t m) {
        out = kPreserved;
        err = "stale";
        EXPECT_FALSE(gs::pipeline_detail::synthesis_slot_size(r, t, p, m, out, &err))
            << r << "," << t << "," << p << "," << m;
        EXPECT_EQ(out, kPreserved);
        EXPECT_FALSE(err.empty());
    };
    rejects(1, 1, 1023, 1);  // 1025
    rejects(1, 1, 1, 0);
    rejects(1, 1, 1, -1);
    rejects(0, 1, 1, 1);
    rejects(-1, 1, 1, 1);
    rejects(1, 0, 1, 1);
    rejects(1, 1, 0, 1);
    rejects(std::numeric_limits<int64_t>::max(), 1, 1, 1);
    rejects(1, std::numeric_limits<int64_t>::max(), 1, 1);
    rejects(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());

    out = 0;
    EXPECT_TRUE(gs::pipeline_detail::synthesis_slot_size(2, 3, 4, 5, out, nullptr));
    EXPECT_EQ(out, 13u);
    EXPECT_FALSE(gs::pipeline_detail::synthesis_slot_size(0, 3, 4, 5, out, nullptr));
}

// ===========================================================================
// validate_synthesis_config
// ===========================================================================

TEST(SynthesisConfig, FinitenessAndRanges) {
    gs::synthesis_config cfg;
    std::string err = "stale";
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_config(cfg, &err));
    EXPECT_TRUE(err.empty());

    auto bad = [&](const gs::synthesis_config & c) {
        std::string e = "stale";
        EXPECT_FALSE(gs::pipeline_detail::validate_synthesis_config(c, &e));
        EXPECT_FALSE(e.empty());
    };
    gs::synthesis_config c;
    c = cfg; c.sampler.top_p = std::numeric_limits<float>::quiet_NaN(); bad(c);
    c = cfg; c.sampler.top_p = std::numeric_limits<float>::infinity(); bad(c);
    c = cfg; c.sampler.temperature = std::numeric_limits<float>::quiet_NaN(); bad(c);
    c = cfg; c.sampler.temperature = std::numeric_limits<float>::infinity(); bad(c);
    c = cfg; c.sampler.repetition_penalty = 0.0f; bad(c);
    c = cfg; c.sampler.repetition_penalty = -1.0f; bad(c);
    c = cfg; c.sampler.repetition_penalty = std::numeric_limits<float>::quiet_NaN(); bad(c);
    c = cfg; c.generate.max_new_tokens = 0; bad(c);
    c = cfg; c.generate.max_new_tokens = -1; bad(c);
    c = cfg; c.generate.max_new_tokens = 1025; bad(c);
    c = cfg; c.noise_scale = 0.0f; bad(c);
    c = cfg; c.noise_scale = -0.5f; bad(c);
    c = cfg; c.noise_scale = std::numeric_limits<float>::quiet_NaN(); bad(c);
    c = cfg; c.noise_scale = std::numeric_limits<float>::infinity(); bad(c);

    c = cfg; c.generate.max_new_tokens = 1;
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_config(c, &err));
    c.generate.max_new_tokens = 1024;
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_config(c, &err));
}

// ===========================================================================
// validate_codebooks
// ===========================================================================

constexpr int64_t kCodeDim = 768;
constexpr int64_t kCodeCount = 1024;

struct Book {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t = nullptr;
    ~Book() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

static bool init_book(Book & b, ggml_backend_t backend, ggml_type type,
                      int64_t ne0 = kCodeDim, int64_t ne1 = kCodeCount,
                      bool noncontiguous = false) {
    ggml_init_params p = {ggml_tensor_overhead() * 4, nullptr, true};
    b.ctx = ggml_init(p);
    if (!b.ctx) return false;
    if (noncontiguous) {
        ggml_tensor * base = ggml_new_tensor_2d(b.ctx, type, 2 * ne0, ne1);
        b.t = ggml_view_2d(b.ctx, base, ne0, ne1,
                           (size_t) (2 * ne0) * ggml_type_size(type), 0);
    } else {
        b.t = ggml_new_tensor_2d(b.ctx, type, ne0, ne1);
    }
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    return b.t && b.buf;
}

static void fill_matched(Book & f32, Book & f16) {
    const size_t n = (size_t) (kCodeDim * kCodeCount);
    std::vector<float> values(n);
    std::vector<ggml_fp16_t> half(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = (float) ((int) (i % 2001) - 1000) * 0.001f;
        half[i] = ggml_fp32_to_fp16(values[i]);
    }
    ggml_backend_tensor_set(f32.t, values.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(f16.t, half.data(), 0, n * sizeof(ggml_fp16_t));
}

static void bind_books(gs::t2s_model & t2s, gs::sovits_model & sovits,
                       Book & tb, Book & sb) {
    t2s.weights.extract_latent.codebook = tb.t;
    sovits.quantizer.codebook = sb.t;
}

class CodebookValidation : public ::testing::Test {
protected:
    void SetUp() override {
        backend = ggml_backend_cpu_init();
        ASSERT_NE(backend, nullptr);
    }
    void TearDown() override {
        if (backend) ggml_backend_free(backend);
    }
    ggml_backend_t backend = nullptr;
};

TEST_F(CodebookValidation, MatchedF32ToF16RoundingAccepted) {
    Book tb, sb;
    ASSERT_TRUE(init_book(tb, backend, GGML_TYPE_F32));
    ASSERT_TRUE(init_book(sb, backend, GGML_TYPE_F16));
    fill_matched(tb, sb);
    gs::t2s_model t2s{};
    gs::sovits_model sovits{};
    bind_books(t2s, sovits, tb, sb);
    std::string err = "stale";
    EXPECT_TRUE(gs::pipeline_detail::validate_codebooks(t2s, sovits, &err));
    EXPECT_TRUE(err.empty());
}

TEST_F(CodebookValidation, MismatchAndNonFiniteRejected) {
    Book tb, sb;
    ASSERT_TRUE(init_book(tb, backend, GGML_TYPE_F32));
    ASSERT_TRUE(init_book(sb, backend, GGML_TYPE_F16));
    fill_matched(tb, sb);
    gs::t2s_model t2s{};
    gs::sovits_model sovits{};
    bind_books(t2s, sovits, tb, sb);
    std::string err;

    const ggml_fp16_t mismatched = ggml_fp32_to_fp16(1.0f);  // element 0 is -1
    ggml_backend_tensor_set(sb.t, &mismatched, 0, sizeof(mismatched));
    EXPECT_FALSE(gs::pipeline_detail::validate_codebooks(t2s, sovits, &err));
    EXPECT_FALSE(err.empty());

    fill_matched(tb, sb);
    EXPECT_TRUE(gs::pipeline_detail::validate_codebooks(t2s, sovits, &err));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(tb.t, &nan, 0, sizeof(nan));
    EXPECT_FALSE(gs::pipeline_detail::validate_codebooks(t2s, sovits, &err));

    fill_matched(tb, sb);
    const ggml_fp16_t inf = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    ggml_backend_tensor_set(sb.t, &inf, 0, sizeof(inf));
    EXPECT_FALSE(gs::pipeline_detail::validate_codebooks(t2s, sovits, &err));
}

TEST_F(CodebookValidation, TypeShapeContiguityAndNullRejected) {
    gs::t2s_model t2s{};
    gs::sovits_model sovits{};
    EXPECT_FALSE(gs::pipeline_detail::validate_codebooks(t2s, sovits));  // null

    auto rejects = [&](ggml_type t2s_type, ggml_type sovits_type,
                       int64_t ne0, int64_t ne1, bool noncontiguous) {
        Book tb, sb;
        ASSERT_TRUE(init_book(tb, backend, t2s_type, ne0, ne1, noncontiguous));
        ASSERT_TRUE(init_book(sb, backend, sovits_type, ne0, ne1, noncontiguous));
        bind_books(t2s, sovits, tb, sb);
        EXPECT_FALSE(gs::pipeline_detail::validate_codebooks(t2s, sovits));
    };
    rejects(GGML_TYPE_F16, GGML_TYPE_F32, kCodeDim, kCodeCount, false);  // types
    rejects(GGML_TYPE_F32, GGML_TYPE_F16, kCodeDim - 1, kCodeCount, false);
    rejects(GGML_TYPE_F32, GGML_TYPE_F16, kCodeDim, kCodeCount - 1, false);
    rejects(GGML_TYPE_F32, GGML_TYPE_F16, kCodeDim, kCodeCount, true);  // strides
}

// ===========================================================================
// validate_synthesis_models
// ===========================================================================

// Minimal loaded-model stand-in: a ggml context with one tensor allocated
// onto a backend buffer, so ctx_w/buf_w satisfy the loaded-on checks.
static bool init_weight_slot(ggml_context *& ctx, ggml_backend_buffer_t & buf,
                             ggml_backend_t backend) {
    ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
    ctx = ggml_init(p);
    if (!ctx) return false;
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    return buf != nullptr;
}

static void fill_matched_tensors(ggml_tensor * f32_t, ggml_tensor * f16_t) {
    const size_t n = (size_t) (kCodeDim * kCodeCount);
    std::vector<float> values(n);
    std::vector<ggml_fp16_t> half(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = (float) ((int) (i % 2001) - 1000) * 0.001f;
        half[i] = ggml_fp32_to_fp16(values[i]);
    }
    ggml_backend_tensor_set(f32_t, values.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(f16_t, half.data(), 0, n * sizeof(ggml_fp16_t));
}

class ModelsValidation : public ::testing::Test {
protected:
    void SetUp() override {
        backend = ggml_backend_cpu_init();
        ASSERT_NE(backend, nullptr);
        other_backend = ggml_backend_cpu_init();
        ASSERT_NE(other_backend, nullptr);

        ASSERT_TRUE(init_weight_slot(hubert_state.ctx_w, hubert_state.buf_w, backend));
        hubert_state.backend = backend;
        ASSERT_TRUE(init_weight_slot(roberta_state.ctx_w, roberta_state.buf_w, backend));
        roberta_state.backend = backend;

        // T2S hparams defaults already carry the v2 vocabulary
        // (phoneme 732 / eos 1024 / semantic 1025).
        t2s_state.backend = backend;
        ggml_init_params tp = {ggml_tensor_overhead() * 12, nullptr, true};
        t2s_state.ctx_w = ggml_init(tp);
        ASSERT_NE(t2s_state.ctx_w, nullptr);
        ggml_context * tc = t2s_state.ctx_w;
        auto & el = t2s_state.weights.extract_latent;
        el.ssl_proj_w  = ggml_new_tensor_3d(tc, GGML_TYPE_F32, 2, 768, 768);
        ssl_bad_kernel = ggml_new_tensor_3d(tc, GGML_TYPE_F32, 3, 768, 768);
        ssl_bad_in     = ggml_new_tensor_3d(tc, GGML_TYPE_F32, 2, 767, 768);
        ssl_bad_out    = ggml_new_tensor_3d(tc, GGML_TYPE_F32, 2, 768, 767);
        el.ssl_proj_b  = ggml_new_tensor_1d(tc, GGML_TYPE_F32, 768);
        ssl_bias_bad   = ggml_new_tensor_1d(tc, GGML_TYPE_F32, 767);
        auto & em = t2s_state.weights.embed;
        em.text_embedding = ggml_new_tensor_2d(tc, GGML_TYPE_F32, 4, 732);
        t2s_text_bad      = ggml_new_tensor_2d(tc, GGML_TYPE_F32, 4, 731);
        el.codebook       = ggml_new_tensor_2d(tc, GGML_TYPE_F32, kCodeDim, kCodeCount);
        t2s_state.buf_w = ggml_backend_alloc_ctx_tensors(tc, backend);
        ASSERT_NE(t2s_state.buf_w, nullptr);

        sovits_state.backend = backend;
        sovits_state.hparams.is_v2pro = true;
        sovits_state.hparams.semantic_frame_rate_25hz = true;
        sovits_state.hparams.gin_channels = 1024;
        ggml_init_params sp = {ggml_tensor_overhead() * 8, nullptr, true};
        sovits_state.ctx_w = ggml_init(sp);
        ASSERT_NE(sovits_state.ctx_w, nullptr);
        ggml_context * sc = sovits_state.ctx_w;
        sovits_state.generator.conv_pre.w =
            ggml_new_tensor_3d(sc, GGML_TYPE_F32, 1, 512, 1);
        conv_pre_bad = ggml_new_tensor_3d(sc, GGML_TYPE_F32, 1, 511, 1);
        sovits_state.text_encoder.text.text_embedding =
            ggml_new_tensor_2d(sc, GGML_TYPE_F32, 192, 732);
        sovits_text_bad = ggml_new_tensor_2d(sc, GGML_TYPE_F32, 192, 731);
        sovits_state.quantizer.codebook =
            ggml_new_tensor_2d(sc, GGML_TYPE_F16, kCodeDim, kCodeCount);
        sovits_state.buf_w = ggml_backend_alloc_ctx_tensors(sc, backend);
        ASSERT_NE(sovits_state.buf_w, nullptr);

        sv_state.backend = backend;
        ASSERT_TRUE(init_weight_slot(sv_state.ctx_w, sv_state.buf_w, backend));
        ASSERT_TRUE(init_weight_slot(sv_state.ctx_bn, sv_state.buf_bn, backend));

        fill_matched_tensors(t2s_state.weights.extract_latent.codebook,
                             sovits_state.quantizer.codebook);
    }

    void TearDown() override {
        free_slot(sv_state.ctx_bn, sv_state.buf_bn);
        free_slot(sv_state.ctx_w, sv_state.buf_w);
        free_slot(sovits_state.ctx_w, sovits_state.buf_w);
        free_slot(t2s_state.ctx_w, t2s_state.buf_w);
        free_slot(roberta_state.ctx_w, roberta_state.buf_w);
        free_slot(hubert_state.ctx_w, hubert_state.buf_w);
        if (other_backend) ggml_backend_free(other_backend);
        if (backend) ggml_backend_free(backend);
    }

    gs::synthesis_models models() {
        gs::synthesis_models m;
        m.hubert = &hubert_state;
        m.roberta = &roberta_state;
        m.t2s = &t2s_state;
        m.sovits = &sovits_state;
        m.sv = &sv_state;
        return m;
    }

    // Validates the current fixture state and asserts a diagnostic substring.
    void rejects(const char * expect) {
        std::string err = "stale";
        EXPECT_FALSE(gs::pipeline_detail::validate_synthesis_models(models(), &err));
        EXPECT_NE(err.find(expect), std::string::npos) << err;
    }

    ggml_backend_t backend = nullptr;
    ggml_backend_t other_backend = nullptr;
    gs::hubert_model hubert_state{};
    gs::roberta_model roberta_state{};
    gs::t2s_model t2s_state{};
    gs::sovits_model sovits_state{};
    gs::sv_model sv_state{};

    // Rejected tensor variants kept alive for pointer swaps.
    ggml_tensor * ssl_bad_kernel = nullptr;
    ggml_tensor * ssl_bad_in = nullptr;
    ggml_tensor * ssl_bad_out = nullptr;
    ggml_tensor * ssl_bias_bad = nullptr;
    ggml_tensor * t2s_text_bad = nullptr;
    ggml_tensor * conv_pre_bad = nullptr;
    ggml_tensor * sovits_text_bad = nullptr;

private:
    static void free_slot(ggml_context *& ctx, ggml_backend_buffer_t & buf) {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

TEST_F(ModelsValidation, AcceptsFullyLoadedV2ProSet) {
    std::string err = "stale";
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_models(models(), &err));
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_models(models(), nullptr));

    gs::synthesis_models m = models();
    m.roberta = nullptr;  // pure-English requests need no RoBERTa
    err = "stale";
    EXPECT_TRUE(gs::pipeline_detail::validate_synthesis_models(m, &err));
    EXPECT_TRUE(err.empty());
}

TEST_F(ModelsValidation, MissingModelsRejected) {
    gs::synthesis_models m = models();
    auto rejects_null = [&](const char * expect) {
        std::string err = "stale";
        EXPECT_FALSE(gs::pipeline_detail::validate_synthesis_models(m, &err));
        EXPECT_NE(err.find(expect), std::string::npos) << err;
    };
    const char * const kMessage = "requires HuBERT, T2S, SoVITS and SV";

    m.hubert = nullptr;
    rejects_null(kMessage);
    m = models();
    m.t2s = nullptr;
    rejects_null(kMessage);
    m = models();
    m.sovits = nullptr;
    rejects_null(kMessage);
    m = models();
    m.sv = nullptr;
    rejects_null(kMessage);
}

TEST_F(ModelsValidation, PartialLoadsAndMixedBackendsRejected) {
    ggml_context * ctx = t2s_state.ctx_w;
    t2s_state.ctx_w = nullptr;
    rejects("one CPU backend");
    t2s_state.ctx_w = ctx;

    ggml_backend_buffer_t buf = sovits_state.buf_w;
    sovits_state.buf_w = nullptr;
    rejects("one CPU backend");
    sovits_state.buf_w = buf;

    ggml_backend_t hb = hubert_state.backend;
    hubert_state.backend = nullptr;
    rejects("one CPU backend");
    hubert_state.backend = hb;

    ggml_context * bn = sv_state.ctx_bn;
    sv_state.ctx_bn = nullptr;
    rejects("one CPU backend");
    sv_state.ctx_bn = bn;

    ggml_backend_buffer_t bnb = sv_state.buf_bn;
    sv_state.buf_bn = nullptr;
    rejects("one CPU backend");
    sv_state.buf_bn = bnb;

    // A second CPU backend instance is still a foreign backend.
    sovits_state.backend = other_backend;
    rejects("one CPU backend");
    sovits_state.backend = backend;

    roberta_state.backend = other_backend;
    rejects("one CPU backend");
    roberta_state.backend = backend;
}

TEST_F(ModelsValidation, WrongSovitsProfileRejected) {
    sovits_state.hparams.is_v2pro = false;
    rejects("v2Pro 25 Hz");
    sovits_state.hparams.is_v2pro = true;

    sovits_state.hparams.semantic_frame_rate_25hz = false;
    rejects("v2Pro 25 Hz");
    sovits_state.hparams.semantic_frame_rate_25hz = true;

    sovits_state.hparams.gin_channels = 512;
    rejects("v2Pro 25 Hz");
    sovits_state.hparams.gin_channels = 1024;

    ggml_tensor * conv = sovits_state.generator.conv_pre.w;
    sovits_state.generator.conv_pre.w = nullptr;
    rejects("v2Pro 25 Hz");
    sovits_state.generator.conv_pre.w = conv_pre_bad;  // in_channels 511
    rejects("v2Pro 25 Hz");
    sovits_state.generator.conv_pre.w = conv;
}

TEST_F(ModelsValidation, WrongT2SVocabularyRejected) {
    t2s_state.hparams.phoneme_vocab_size = 731;
    rejects("v2 vocabulary 732");
    t2s_state.hparams.phoneme_vocab_size = 732;

    t2s_state.hparams.eos = 1023;
    rejects("v2 vocabulary 732");
    t2s_state.hparams.eos = 1024;

    t2s_state.hparams.vocab_size = 1024;
    rejects("v2 vocabulary 732");
    t2s_state.hparams.vocab_size = 1025;
}

TEST_F(ModelsValidation, TextEmbeddingMismatchRejected) {
    ggml_tensor * t2s_text = t2s_state.weights.embed.text_embedding;
    ggml_tensor * sovits_text = sovits_state.text_encoder.text.text_embedding;

    t2s_state.weights.embed.text_embedding = nullptr;
    rejects("text embeddings");
    t2s_state.weights.embed.text_embedding = t2s_text_bad;  // 731 entries
    rejects("text embeddings");
    t2s_state.weights.embed.text_embedding = t2s_text;

    sovits_state.text_encoder.text.text_embedding = nullptr;
    rejects("text embeddings");
    sovits_state.text_encoder.text.text_embedding = sovits_text_bad;  // 731
    rejects("text embeddings");
    sovits_state.text_encoder.text.text_embedding = sovits_text;
}

TEST_F(ModelsValidation, BadSslProjRejected) {
    auto & el = t2s_state.weights.extract_latent;
    ggml_tensor * w = el.ssl_proj_w;
    ggml_tensor * b = el.ssl_proj_b;

    el.ssl_proj_w = nullptr;
    rejects("ssl_proj");
    el.ssl_proj_w = ssl_bad_kernel;  // kernel 3
    rejects("ssl_proj");
    el.ssl_proj_w = ssl_bad_in;      // 767 input channels
    rejects("ssl_proj");
    el.ssl_proj_w = ssl_bad_out;     // 767 output channels
    rejects("ssl_proj");
    el.ssl_proj_w = w;

    el.ssl_proj_b = nullptr;
    rejects("ssl_proj");
    el.ssl_proj_b = ssl_bias_bad;    // 767 elements
    rejects("ssl_proj");
    el.ssl_proj_b = b;
}

TEST_F(ModelsValidation, CodebookRejectionPropagates) {
    ggml_tensor * book = t2s_state.weights.extract_latent.codebook;
    t2s_state.weights.extract_latent.codebook = nullptr;
    rejects("T2S and SoVITS codebooks");
    t2s_state.weights.extract_latent.codebook = book;
}

TEST(SynthesizeCpuPublic, InvalidRequestsPreserveResult) {
    gs::synthesis_models models;
    gs::text_frontend_result ref, target;
    ref.phoneme_ids = {1};
    target.phoneme_ids = {2};
    const float pcm = 0.1f;
    gs::synthesis_config config;
    auto rejects = [&](const float * data, int64_t n, int rate, const char * message) {
        auto result = sentinel_result();
        std::string error;
        EXPECT_FALSE(gs::synthesize_cpu(models, data, n, rate, ref, target,
                                        result, config, &error));
        EXPECT_NE(error.find(message), std::string::npos) << error;
        expect_sentinel(result);
    };
    rejects(&pcm, 16000, 16000, "models");
    rejects(nullptr, 16000, 16000, "non-null");
    rejects(&pcm, -1, 16000, "invalid audio");
    rejects(&pcm, INT64_MAX, 44100, "overflow");
    config.generate.max_new_tokens = 1024;
    rejects(&pcm, 16000, 16000, "slot budget");
    config.generate.max_new_tokens = 0;
    rejects(&pcm, 16000, 16000, "max_new_tokens");
}

// ===========================================================================
// generate_and_decode: immediate EOS and pre-compute input rejection
// ===========================================================================

constexpr int64_t kHubertDim = 768;
constexpr int64_t kBertDim = 1024;

struct Spec {
    uint32_t d = 8;
    uint32_t head = 2;
    uint32_t layer = 1;
    uint32_t linear = 16;
    uint32_t vocab = 4;
    uint32_t phoneme = 4;
};

static void put(ggml_tensor * t, const std::vector<float> & v) {
    ASSERT_NE(t, nullptr);
    ASSERT_EQ((size_t) ggml_nelements(t), v.size());
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

static void put_col(ggml_tensor * t, int64_t col, const std::vector<float> & v) {
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(ggml_n_dims(t), 2);
    const int64_t rows = t->ne[0];
    ASSERT_EQ((int64_t) v.size(), rows);
    ASSERT_GE(col, 0);
    ASSERT_LT(col, t->ne[1]);
    std::vector<float> buf((size_t) ggml_nelements(t), 0.0f);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    std::copy(v.begin(), v.end(), buf.begin() + (size_t) (col * rows));
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
}

static std::vector<float> dir(int64_t d, int64_t idx) {
    std::vector<float> v((size_t) d, -1.0f / (float) d);
    if (idx >= 0 && idx < d) v[(size_t) idx] += 1.0f;
    return v;
}

static std::vector<float> ref_tail(int64_t d, int64_t T, int64_t idx) {
    std::vector<float> v((size_t) (d * T), 0.0f);
    const std::vector<float> last = dir(d, idx);
    std::copy(last.begin(), last.end(), v.begin() + (size_t) ((T - 1) * d));
    return v;
}

static void make_model(const Spec & sp, ggml_backend_t backend, gs::t2s_model & m) {
    gs::t2s_hparams hp;
    hp.embedding_dim = sp.d;
    hp.hidden_dim = sp.d;
    hp.n_head = sp.head;
    hp.linear_units = sp.linear;
    hp.n_layer = sp.layer;
    hp.vocab_size = sp.vocab;
    hp.phoneme_vocab_size = sp.phoneme;
    hp.eos = sp.vocab - 1;
    hp.inter_channels = (uint32_t) kHubertDim;
    m.hparams = hp;
    m.backend = backend;

    ggml_init_params p = {ggml_tensor_overhead() * (size_t) (10 + 12 * sp.layer + 4),
                          nullptr, true};
    m.ctx_w = ggml_init(p);
    ASSERT_NE(m.ctx_w, nullptr);
    ggml_context * c = m.ctx_w;
    const int64_t d = sp.d;
    const int64_t df = sp.linear;
    const int64_t v = sp.vocab;
    auto f1 = [&](int64_t x) { return ggml_new_tensor_1d(c, GGML_TYPE_F32, x); };
    auto f2 = [&](int64_t x, int64_t y) { return ggml_new_tensor_2d(c, GGML_TYPE_F32, x, y); };

    auto & el = m.weights.extract_latent;
    el.ssl_proj_w = ggml_new_tensor_3d(c, GGML_TYPE_F16, 2, kHubertDim, kHubertDim);
    el.ssl_proj_b = f1(kHubertDim);
    el.codebook = f2(kHubertDim, v - 1);

    auto & em = m.weights.embed;
    em.text_embedding = f2(d, sp.phoneme);
    em.bert_proj_w = f2(kBertDim, d);
    em.bert_proj_b = f1(d);
    em.text_pos_alpha = f1(1);
    em.audio_embedding = f2(d, v);
    em.audio_pos_alpha = f1(1);

    m.weights.attention.resize(sp.layer);
    for (auto & w : m.weights.attention) {
        w.qkv_w = f2(d, 3 * d); w.qkv_b = f1(3 * d);
        w.out_proj_w = f2(d, d); w.out_proj_b = f1(d);
        w.ln1_w = f1(d); w.ln1_b = f1(d);
        w.ffn_up_w = f2(d, df); w.ffn_up_b = f1(df);
        w.ffn_down_w = f2(df, d); w.ffn_down_b = f1(d);
        w.ln2_w = f1(d); w.ln2_b = f1(d);
    }
    m.weights.lm_head_w = f2(d, v);

    m.buf_w = ggml_backend_alloc_ctx_tensors(c, backend);
    ASSERT_NE(m.buf_w, nullptr);
    ggml_backend_buffer_clear(m.buf_w, 0);
}

static void setup_eos_prefill(gs::t2s_model & m, int32_t eos) {
    const int64_t d = m.hparams.hidden_dim;
    const std::vector<float> ones((size_t) d, 1.0f);
    const std::vector<float> zeros((size_t) d, 0.0f);
    for (auto & l : m.weights.attention) {
        put(l.ln1_w, ones); put(l.ln1_b, zeros);
        put(l.ln2_w, ones); put(l.ln2_b, zeros);
    }
    put(m.weights.embed.text_pos_alpha, {0.0f});
    put(m.weights.embed.audio_pos_alpha, {0.0f});
    put_col(m.weights.lm_head_w, eos, dir(d, 0));
}

static void init_session(gs::t2s_session & s, const gs::t2s_model & m) {
    gs::t2s_sampler_config cfg;
    cfg.top_k = 1;
    cfg.top_p = 1.0f;
    cfg.temperature = 1.0f;
    cfg.repetition_penalty = 1.0f;
    ASSERT_TRUE(gs::t2s_session_init(s, m.hparams, m.backend, 1, 16, cfg));
    for (size_t i = 0; i < s.k_caches.size(); ++i) {
        ggml_backend_tensor_memset(s.k_caches[i], 0, 0, ggml_nbytes(s.k_caches[i]));
        ggml_backend_tensor_memset(s.v_caches[i], 0, 0, ggml_nbytes(s.v_caches[i]));
    }
}

static void install_ref(gs::t2s_session & s, const gs::t2s_model & m,
                        const std::vector<float> & audio) {
    const int64_t d = m.hparams.hidden_dim;
    ASSERT_EQ((int64_t) audio.size(), d * 2);
    ggml_init_params p = {ggml_tensor_overhead() * 3, nullptr, true};
    s.ctx_ref = ggml_init(p);
    ASSERT_NE(s.ctx_ref, nullptr);
    s.ref_text_emb = ggml_new_tensor_2d(s.ctx_ref, GGML_TYPE_F32, d, 2);
    s.ref_audio_emb = ggml_new_tensor_2d(s.ctx_ref, GGML_TYPE_F32, d, 2);
    s.buf_ref = ggml_backend_alloc_ctx_tensors(s.ctx_ref, s.backend);
    ASSERT_NE(s.buf_ref, nullptr);
    put(s.ref_text_emb, std::vector<float>((size_t) (d * 2), 0.0f));
    put(s.ref_audio_emb, audio);
    s.ref_T_ref = 2;
    s.ref_T_prompt = 2;
    s.ref_model = &m;
}

class GenerateAndDecodeEos : public ::testing::Test {
protected:
    void SetUp() override {
        backend = ggml_backend_cpu_init();
        ASSERT_NE(backend, nullptr);
        make_model(Spec{}, backend, model);
        init_session(session, model);
    }
    void TearDown() override {
        gs::t2s_session_free(session);
        gs::t2s_model_free(model);
        if (backend) ggml_backend_free(backend);
    }
    ggml_backend_t backend = nullptr;
    gs::t2s_model model{};
    gs::t2s_session session{};
};

TEST_F(GenerateAndDecodeEos, ImmediateEosPreservesResultAndReleasesSlot) {
    const Spec sp;
    setup_eos_prefill(model, (int32_t) sp.vocab - 1);
    install_ref(session, model, ref_tail(sp.d, 2, 0));

    gs::sovits_session sovits_state{};
    gs::sovits_model sovits_model_state{};
    const std::vector<int32_t> phones = {0, 1};
    const std::vector<float> bert((size_t) (kBertDim * 2), 0.0f);

    gs::synthesis_result result = sentinel_result();
    std::string err = "stale";
    EXPECT_FALSE(gs::pipeline_detail::generate_and_decode(
        session, model, sovits_state, sovits_model_state, phones, bert,
        gen_cfg(4), result, &err));
    expect_sentinel(result);
    // The zeroed SoVITS pair would report "SoVITS waveform decoding failed" if
    // it had been entered; the EOS diagnostic proves the hand-off was skipped.
    EXPECT_NE(err.find("EOS"), std::string::npos);
    EXPECT_EQ(gs::t2s_session_slot_n_pos(session, 0), 0);
    EXPECT_EQ(gs::t2s_session_find_free_slot(session), 0);
}

TEST(GenerateAndDecodeContract, InvalidInputsRejectedBeforeCompute) {
    gs::t2s_session session{};
    gs::t2s_model model{};
    gs::sovits_session sovits_state{};
    gs::sovits_model sovits_model_state{};

    auto rejects = [&](const std::vector<int32_t> & phones,
                       const std::vector<float> & bert) {
        gs::synthesis_result result = sentinel_result();
        std::string err = "stale";
        EXPECT_FALSE(gs::pipeline_detail::generate_and_decode(
            session, model, sovits_state, sovits_model_state, phones, bert,
            gen_cfg(1), result, &err));
        expect_sentinel(result);
    };
    rejects({}, {});
    rejects({0}, std::vector<float>(1023, 0.0f));
    rejects(std::vector<int32_t>(1025, 0), {});
}

}  // namespace
