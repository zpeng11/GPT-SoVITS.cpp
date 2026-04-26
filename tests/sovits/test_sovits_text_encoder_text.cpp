// tests/sovits/test_sovits_text_encoder_text.cpp

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
#include <cstdint>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelF16 =
    kTestDir + "models/v2-text-encoder-text-f16.gguf";
static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefInputNpy = kRefDir + "v2_text_encoder_text_input.npy";
static const std::string kRefOutputNpy = kRefDir + "v2_text_encoder_text_output.npy";

static constexpr int64_t kVocab = 732;
static constexpr int64_t kHidden = 192;
static constexpr int64_t kFFN = 768;
static constexpr int64_t kHeads = 2;
static constexpr int64_t kHeadDim = kHidden / kHeads;
static constexpr int64_t kWindow = 4;
static constexpr int64_t kLayers = 6;
static constexpr int64_t kTime = 24;
static constexpr size_t kMaxNodes = 32768;
static constexpr double kParityMaxAbsTol = 1.0e-2;
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

static std::vector<int32_t> make_text_tokens() {
    std::vector<int32_t> tokens(static_cast<size_t>(kTime));
    for (int64_t t = 0; t < kTime; ++t) {
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>((17 * t + 3) % kVocab);
    }
    return tokens;
}

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static std::vector<int32_t> load_npy_as_i32(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    EXPECT_EQ(arr.shape.size(), 2u);
    EXPECT_EQ(arr.shape[0], 1u);
    EXPECT_EQ(arr.word_size, sizeof(int64_t));
    if (arr.shape.size() != 2 || arr.shape[0] != 1u || arr.word_size != sizeof(int64_t)) {
        return {};
    }

    const int64_t * src = arr.data<int64_t>();
    std::vector<int32_t> out(arr.shape[1]);
    for (size_t i = 0; i < arr.shape[1]; ++i) {
        out[i] = static_cast<int32_t>(src[i]);
    }
    return out;
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

TEST(SoVITSTextEncoderText, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_text_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_text_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_text_encoder_text_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderText, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_text_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_text_model_load(kModelF16, model, backend));

    const auto & w = model.weights;
    ASSERT_NE(w.text_embedding, nullptr);
    EXPECT_EQ(w.text_embedding->ne[0], kHidden);
    EXPECT_EQ(w.text_embedding->ne[1], kVocab);

    for (int i = 0; i < kLayers; ++i) {
        const auto & layer = w.layers[i];
        ASSERT_NE(layer.qkv_w, nullptr);
        ASSERT_NE(layer.qkv_b, nullptr);
        ASSERT_NE(layer.out_w, nullptr);
        ASSERT_NE(layer.out_b, nullptr);
        ASSERT_NE(layer.rel_k, nullptr);
        ASSERT_NE(layer.rel_v_t, nullptr);
        ASSERT_NE(layer.ffn_up_w, nullptr);
        ASSERT_NE(layer.ffn_down_w, nullptr);

        EXPECT_EQ(layer.qkv_w->ne[0], 1);
        EXPECT_EQ(layer.qkv_w->ne[1], kHidden);
        EXPECT_EQ(layer.qkv_w->ne[2], 3 * kHidden);
        EXPECT_EQ(layer.qkv_b->ne[0], 3 * kHidden);
        EXPECT_EQ(layer.out_w->ne[0], 1);
        EXPECT_EQ(layer.out_w->ne[1], kHidden);
        EXPECT_EQ(layer.out_w->ne[2], kHidden);
        EXPECT_EQ(layer.out_b->ne[0], kHidden);
        EXPECT_EQ(layer.rel_k->ne[0], kHeadDim);
        EXPECT_EQ(layer.rel_k->ne[1], 2 * kWindow + 1);
        EXPECT_EQ(layer.rel_v_t->ne[0], 2 * kWindow + 1);
        EXPECT_EQ(layer.rel_v_t->ne[1], kHeadDim);
        EXPECT_EQ(layer.ffn_up_w->ne[0], 3);
        EXPECT_EQ(layer.ffn_up_w->ne[1], kHidden);
        EXPECT_EQ(layer.ffn_up_w->ne[2], kFFN);
        EXPECT_EQ(layer.ffn_down_w->ne[0], 3);
        EXPECT_EQ(layer.ffn_down_w->ne[1], kFFN);
        EXPECT_EQ(layer.ffn_down_w->ne[2], kHidden);
    }

    gpt_sovits::sovits_text_encoder_text_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderText, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_text_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_text_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * inp = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kTime);
    ggml_set_name(inp, "text");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::sovits_text_encoder_text_block_forward(gctx, inp, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "encoded_text");
    ggml_set_output(out);

    EXPECT_EQ(out->ne[0], kHidden);
    EXPECT_EQ(out->ne[1], kTime);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    const std::vector<int32_t> tokens = make_text_tokens();
    ggml_backend_tensor_set(inp, tokens.data(), 0, tokens.size() * sizeof(int32_t));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kHidden * kTime));
    for (float value : output) {
        EXPECT_TRUE(std::isfinite(value));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_text_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderText, MissingModelFileFailsCleanly) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_text_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_text_encoder_text_model_load(
        "/nonexistent/path.gguf", model, backend));

    gpt_sovits::sovits_text_encoder_text_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoderText, FreeOnEmptyModelIsSafe) {
    gpt_sovits::sovits_text_encoder_text_model model{};
    gpt_sovits::sovits_text_encoder_text_model_free(model);
}

TEST(SoVITSTextEncoderText, MatchesPythonReference) {
    ASSERT_MODEL_EXISTS(kModelF16);
    ASSERT_MODEL_EXISTS(kRefInputNpy);
    ASSERT_MODEL_EXISTS(kRefOutputNpy);

    const std::vector<int32_t> input = load_npy_as_i32(kRefInputNpy);
    const auto ref_output = load_npy_with_shape(kRefOutputNpy);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(ref_output.data.empty());

    ASSERT_EQ(ref_output.shape.size(), 3u);
    ASSERT_EQ(ref_output.shape[0], 1u);
    ASSERT_EQ(ref_output.shape[1], static_cast<size_t>(kHidden));
    ASSERT_EQ(ref_output.shape[2], input.size());

    const int64_t time = static_cast<int64_t>(input.size());
    ASSERT_GT(time, 0);

    const std::vector<float> expected_output =
        pack_bct_to_ggml(ref_output.data, ref_output.shape, kHidden);
    ASSERT_EQ(expected_output.size(), static_cast<size_t>(kHidden * time));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_text_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_text_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * inp = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, time);
    ggml_set_name(inp, "text");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::sovits_text_encoder_text_block_forward(gctx, inp, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "encoded_text");
    ggml_set_output(out);

    ASSERT_EQ(out->ne[0], kHidden);
    ASSERT_EQ(out->ne[1], time);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(int32_t));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), expected_output.size());
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const auto err = compute_errors(output, expected_output);
    printf("[text_encoder_text parity] T=%lld max_abs=%.6f rmse=%.6f mean_abs=%.6f\n",
           static_cast<long long>(time), err.max_abs, err.rmse, err.mean_abs);
    EXPECT_LT(err.max_abs, kParityMaxAbsTol);
    EXPECT_LT(err.rmse, kParityRmseTol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_text_model_free(model);
    ggml_backend_free(backend);
}
