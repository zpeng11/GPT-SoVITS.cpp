// tests/sovits/test_sovits_text_encoder_mrte.cpp

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "cnpy.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "npy_loader.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelF16 =
    kTestDir + "models/v2-text-encoder-mrte-f16.gguf";
static const std::string kModelF32 =
    kTestDir + "models/v2-text-encoder-mrte-f32.gguf";
static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefYInputNpy = kRefDir + "v2_text_encoder_mrte_y_input.npy";
static const std::string kRefTextInputNpy = kRefDir + "v2_text_encoder_mrte_text_input.npy";
static const std::string kRefGeInputNpy = kRefDir + "v2_text_encoder_mrte_ge_input.npy";
static const std::string kRefOutputNpy = kRefDir + "v2_text_encoder_mrte_output.npy";

static constexpr int64_t kInChannels = 192;
static constexpr int64_t kHidden = 512;
static constexpr int64_t kSslFusedDim = 704;
static constexpr int64_t kKvDim = 1024;
static constexpr int64_t kSslTime = 24;
static constexpr int64_t kTextTime = 17;
static constexpr size_t kMaxNodes = 32768;
static constexpr double kParityMaxAbsTol = 1.05e-2;
static constexpr double kParityRmseTol = 2.0e-3;

struct ErrorStats {
    double max_abs = 0.0;
    double rmse = 0.0;
    double mean_abs = 0.0;
};

struct NpyShapeInfo {
    std::vector<float> data;
    std::vector<size_t> shape;
};

struct GraphContext {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = nullptr;

    explicit GraphContext(size_t max_nodes) {
        const size_t sz = ggml_tensor_overhead() * max_nodes
                        + ggml_graph_overhead_custom(max_nodes, false);
        buf.resize(sz);
        struct ggml_init_params params = {
            /*.mem_size   =*/ sz,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
    }

    ~GraphContext() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;

    operator struct ggml_context *() { return ctx; } // NOLINT
};

static void fill_input(std::vector<float> & data, int64_t channels, int64_t time, float c_scale, float t_scale) {
    for (int64_t t = 0; t < time; ++t) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t idx = static_cast<size_t>(t * channels + c);
            data[idx] = std::sin(static_cast<float>(c) * c_scale)
                      + std::cos(static_cast<float>(t) * t_scale)
                      + 0.001f * static_cast<float>((c + 3 * t) % 13);
        }
    }
}

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static ErrorStats compute_errors(const std::vector<float> & actual,
                                 const std::vector<float> & expected) {
    ErrorStats s{};
    if (actual.size() != expected.size() || actual.empty()) {
        return s;
    }

    double sum_sq = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double err = std::abs(static_cast<double>(actual[i]) -
                                    static_cast<double>(expected[i]));
        s.max_abs = std::max(s.max_abs, err);
        sum_sq += err * err;
        sum_abs += err;
    }

    s.rmse = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    s.mean_abs = sum_abs / static_cast<double>(actual.size());
    return s;
}

static std::vector<float> pack_bct_to_ggml(
    const std::vector<float> & tensor,
    const std::vector<size_t> & shape,
    int64_t channels)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_EQ(shape[1], static_cast<size_t>(channels));
    if (shape.size() != 3 || shape[0] != 1u || shape[1] != static_cast<size_t>(channels)) {
        return {};
    }

    const size_t time = shape[2];
    std::vector<float> packed(static_cast<size_t>(channels) * time);

    for (size_t t = 0; t < time; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(channels); ++c) {
            const size_t src = c * time + t;
            const size_t dst = t * static_cast<size_t>(channels) + c;
            packed[dst] = tensor[src];
        }
    }

    return packed;
}

// Helper: skip test if model file not found.
#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

} // namespace

TEST(SoVITSTextEncoderMrte, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_mrte_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderMrte, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_mrte_model_load(kModelF16, model, backend));

    const auto & w = model.weights;
    ASSERT_NE(w.ssl_fused_w, nullptr);
    ASSERT_NE(w.ssl_fused_b, nullptr);
    ASSERT_NE(w.text_kv_w, nullptr);
    ASSERT_NE(w.text_kv_b, nullptr);
    ASSERT_NE(w.attn_out_w, nullptr);
    ASSERT_NE(w.attn_out_b, nullptr);
    ASSERT_NE(w.ge_out_w, nullptr);
    ASSERT_NE(w.ge_out_b, nullptr);

    EXPECT_EQ(w.ssl_fused_w->ne[0], 1);
    EXPECT_EQ(w.ssl_fused_w->ne[1], kInChannels);
    EXPECT_EQ(w.ssl_fused_w->ne[2], kSslFusedDim);
    EXPECT_EQ(w.ssl_fused_b->ne[0], kSslFusedDim);
    EXPECT_EQ(w.text_kv_w->ne[0], 1);
    EXPECT_EQ(w.text_kv_w->ne[1], kInChannels);
    EXPECT_EQ(w.text_kv_w->ne[2], kKvDim);
    EXPECT_EQ(w.text_kv_b->ne[0], kKvDim);
    EXPECT_EQ(w.attn_out_w->ne[0], 1);
    EXPECT_EQ(w.attn_out_w->ne[1], kHidden);
    EXPECT_EQ(w.attn_out_w->ne[2], kInChannels);
    EXPECT_EQ(w.attn_out_b->ne[0], kInChannels);
    EXPECT_EQ(w.ge_out_w->ne[0], 1);
    EXPECT_EQ(w.ge_out_w->ne[1], kHidden);
    EXPECT_EQ(w.ge_out_w->ne[2], kInChannels);
    EXPECT_EQ(w.ge_out_b->ne[0], kInChannels);

    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderMrte, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_mrte_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, kSslTime);
    ggml_set_name(ssl, "mrte_ssl");
    ggml_set_input(ssl);

    struct ggml_tensor * text = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, kTextTime);
    ggml_set_name(text, "mrte_text");
    ggml_set_input(text);

    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kHidden, 1);
    ggml_set_name(ge, "mrte_ge");
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_text_encoder_mrte_block_forward(gctx, ssl, text, ge, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "mrte_out");
    ggml_set_output(out);

    EXPECT_EQ(out->ne[0], kInChannels);
    EXPECT_EQ(out->ne[1], kSslTime);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    std::vector<float> ssl_input(static_cast<size_t>(kInChannels * kSslTime));
    std::vector<float> text_input(static_cast<size_t>(kInChannels * kTextTime));
    std::vector<float> ge_input(static_cast<size_t>(kHidden));
    fill_input(ssl_input, kInChannels, kSslTime, 0.017f, 0.13f);
    fill_input(text_input, kInChannels, kTextTime, 0.019f, 0.09f);
    fill_input(ge_input, kHidden, 1, 0.007f, 0.0f);

    ggml_backend_tensor_set(ssl, ssl_input.data(), 0, ssl_input.size() * sizeof(float));
    ggml_backend_tensor_set(text, text_input.data(), 0, text_input.size() * sizeof(float));
    ggml_backend_tensor_set(ge, ge_input.data(), 0, ge_input.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kInChannels * kSslTime));
    for (float value : output) {
        EXPECT_TRUE(std::isfinite(value));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderMrte, MissingModelFileFailsCleanly) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_text_encoder_mrte_model_load(
        "/nonexistent/path.gguf", model, backend));

    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderMrte, FreeOnEmptyModelIsSafe) {
    gpt_sovits::sovits_text_encoder_mrte_model model{};
    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
}

TEST(SoVITSTextEncoderMrte, MatchesPythonReference) {
    ASSERT_MODEL_EXISTS(kModelF32);
    ASSERT_MODEL_EXISTS(kRefYInputNpy);
    ASSERT_MODEL_EXISTS(kRefTextInputNpy);
    ASSERT_MODEL_EXISTS(kRefGeInputNpy);
    ASSERT_MODEL_EXISTS(kRefOutputNpy);

    const auto ref_y = load_npy_with_shape(kRefYInputNpy);
    const auto ref_text = load_npy_with_shape(kRefTextInputNpy);
    const auto ref_ge = load_npy_with_shape(kRefGeInputNpy);
    const auto ref_output = load_npy_with_shape(kRefOutputNpy);
    ASSERT_FALSE(ref_y.data.empty());
    ASSERT_FALSE(ref_text.data.empty());
    ASSERT_FALSE(ref_ge.data.empty());
    ASSERT_FALSE(ref_output.data.empty());

    ASSERT_EQ(ref_y.shape.size(), 3u);
    ASSERT_EQ(ref_y.shape[0], 1u);
    ASSERT_EQ(ref_y.shape[1], static_cast<size_t>(kInChannels));

    ASSERT_EQ(ref_text.shape.size(), 3u);
    ASSERT_EQ(ref_text.shape[0], 1u);
    ASSERT_EQ(ref_text.shape[1], static_cast<size_t>(kInChannels));

    ASSERT_EQ(ref_ge.shape.size(), 3u);
    ASSERT_EQ(ref_ge.shape[0], 1u);
    ASSERT_EQ(ref_ge.shape[1], static_cast<size_t>(kHidden));
    ASSERT_EQ(ref_ge.shape[2], 1u);

    ASSERT_EQ(ref_output.shape.size(), 3u);
    ASSERT_EQ(ref_output.shape[0], 1u);
    ASSERT_EQ(ref_output.shape[1], static_cast<size_t>(kInChannels));
    ASSERT_EQ(ref_output.shape[2], ref_y.shape[2]);

    const int64_t y_time = static_cast<int64_t>(ref_y.shape[2]);
    const int64_t text_time = static_cast<int64_t>(ref_text.shape[2]);
    ASSERT_GT(y_time, 0);
    ASSERT_GT(text_time, 0);

    const std::vector<float> packed_y =
        pack_bct_to_ggml(ref_y.data, ref_y.shape, kInChannels);
    const std::vector<float> packed_text =
        pack_bct_to_ggml(ref_text.data, ref_text.shape, kInChannels);
    const std::vector<float> packed_ge =
        pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kHidden);
    const std::vector<float> expected_output =
        pack_bct_to_ggml(ref_output.data, ref_output.shape, kInChannels);
    ASSERT_EQ(packed_y.size(), static_cast<size_t>(kInChannels * y_time));
    ASSERT_EQ(packed_text.size(), static_cast<size_t>(kInChannels * text_time));
    ASSERT_EQ(packed_ge.size(), static_cast<size_t>(kHidden));
    ASSERT_EQ(expected_output.size(), static_cast<size_t>(kInChannels * y_time));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_mrte_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_mrte_model_load(kModelF32, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * y = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, y_time);
    ggml_set_name(y, "mrte_ref_y");
    ggml_set_input(y);

    struct ggml_tensor * text = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, text_time);
    ggml_set_name(text, "mrte_ref_text");
    ggml_set_input(text);

    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kHidden, 1);
    ggml_set_name(ge, "mrte_ref_ge");
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_text_encoder_mrte_block_forward(gctx, y, text, ge, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "mrte_ref_out");
    ggml_set_output(out);

    ASSERT_EQ(out->ne[0], kInChannels);
    ASSERT_EQ(out->ne[1], y_time);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(y, packed_y.data(), 0, packed_y.size() * sizeof(float));
    ggml_backend_tensor_set(text, packed_text.data(), 0, packed_text.size() * sizeof(float));
    ggml_backend_tensor_set(ge, packed_ge.data(), 0, packed_ge.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), expected_output.size());
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const auto err = compute_errors(output, expected_output);
    printf("[text_encoder_mrte parity] Ty=%lld Ttext=%lld max_abs=%.6f rmse=%.6f mean_abs=%.6f\n",
           static_cast<long long>(y_time),
           static_cast<long long>(text_time),
           err.max_abs,
           err.rmse,
           err.mean_abs);
    EXPECT_LT(err.max_abs, kParityMaxAbsTol);
    EXPECT_LT(err.rmse, kParityRmseTol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_mrte_model_free(model);
    ggml_backend_free(backend);
}
