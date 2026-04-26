// tests/sovits/test_sovits_ref_enc.cpp
//
// Minimal integration test for the SoVITS v2 ref_enc (MelStyleEncoder) block:
//   - loads the dedicated ref_enc GGUF
//   - builds a ggml graph
//   - runs one inference pass on a deterministic input
//   - checks output existence, shape, and finite values

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
    kTestDir + "models/v2-ref-enc-f16.gguf";
static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefInputNpy = kRefDir + "v2_ref_enc_input.npy";
static const std::string kRefOutputNpy = kRefDir + "v2_ref_enc_ge.npy";

static constexpr int64_t kInChannels = 704;
static constexpr int64_t kOutChannels = 512;
static constexpr int64_t kTime = 32;
static constexpr size_t kMaxNodes = 4096;
static constexpr double kParityMaxAbsTol = 5e-2;
static constexpr double kParityRmseTol = 1e-2;

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

static void fill_reference_input(std::vector<float> & data) {
    for (size_t t = 0; t < kTime; ++t) {
        for (size_t c = 0; c < kInChannels; ++c) {
            const size_t idx = t * static_cast<size_t>(kInChannels) + c;
            data[idx] = std::sin(static_cast<float>(c) * 0.01f)
                      + std::cos(static_cast<float>(t) * 0.17f);
        }
    }
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

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static std::vector<float> pack_v2_ref_input_ggml(
    const std::vector<float> & refer,
    const std::vector<size_t> & shape)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_GE(shape[1], static_cast<size_t>(kInChannels));

    if (shape.size() != 3 || shape[0] != 1u || shape[1] < static_cast<size_t>(kInChannels)) {
        return {};
    }

    const size_t full_channels = shape[1];
    const size_t time = shape[2];
    std::vector<float> packed(static_cast<size_t>(kInChannels) * time);

    // Python uses refer[:, :704, :] in BCH layout; ggml input is {channels, time},
    // so re-pack to contiguous [t][c].
    for (size_t t = 0; t < time; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(kInChannels); ++c) {
            const size_t src = c * time + t;
            const size_t dst = t * static_cast<size_t>(kInChannels) + c;
            packed[dst] = refer[src];
        }
    }

    EXPECT_EQ(refer.size(), full_channels * time);
    return packed;
}

// Helper: skip test if model file not found.
#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

} // namespace

TEST(SoVITSRefEnc, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_ref_enc_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_ref_enc_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_ref_enc_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSRefEnc, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_ref_enc_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_ref_enc_model_load(kModelF16, model, backend));

    const auto & w = model.weights;
    ASSERT_NE(w.spectral_1_w, nullptr);
    ASSERT_NE(w.spectral_1_b, nullptr);
    ASSERT_NE(w.spectral_2_w, nullptr);
    ASSERT_NE(w.spectral_2_b, nullptr);
    ASSERT_NE(w.temporal[0].conv_w, nullptr);
    ASSERT_NE(w.temporal[0].conv_b, nullptr);
    ASSERT_NE(w.temporal[1].conv_w, nullptr);
    ASSERT_NE(w.temporal[1].conv_b, nullptr);
    ASSERT_NE(w.attention.q_w, nullptr);
    ASSERT_NE(w.attention.q_b, nullptr);
    ASSERT_NE(w.attention.k_w, nullptr);
    ASSERT_NE(w.attention.k_b, nullptr);
    ASSERT_NE(w.attention.v_w, nullptr);
    ASSERT_NE(w.attention.v_b, nullptr);
    ASSERT_NE(w.attention.out_w, nullptr);
    ASSERT_NE(w.attention.out_b, nullptr);
    ASSERT_NE(w.fc_w, nullptr);
    ASSERT_NE(w.fc_b, nullptr);

    EXPECT_EQ(w.spectral_1_w->ne[0], 704);
    EXPECT_EQ(w.spectral_1_w->ne[1], 128);
    EXPECT_EQ(w.spectral_1_b->ne[0], 128);
    EXPECT_EQ(w.spectral_2_w->ne[0], 128);
    EXPECT_EQ(w.spectral_2_w->ne[1], 128);
    EXPECT_EQ(w.temporal[0].conv_w->ne[0], 5);
    EXPECT_EQ(w.temporal[0].conv_w->ne[1], 128);
    EXPECT_EQ(w.temporal[0].conv_w->ne[2], 256);
    EXPECT_EQ(w.attention.q_w->ne[0], 128);
    EXPECT_EQ(w.attention.q_w->ne[1], 128);
    EXPECT_EQ(w.fc_w->ne[0], 128);
    EXPECT_EQ(w.fc_w->ne[1], 512);
    EXPECT_EQ(w.fc_b->ne[0], 512);

    gpt_sovits::sovits_ref_enc_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSRefEnc, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_ref_enc_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_ref_enc_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, kTime);
    ggml_set_name(inp, "refer");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::sovits_mel_style_encoder_block_forward(gctx, inp, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "ge");
    ggml_set_output(out);

    EXPECT_EQ(out->ne[0], kOutChannels);
    EXPECT_EQ(out->ne[1], 1);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    std::vector<float> input(static_cast<size_t>(kInChannels * kTime));
    fill_reference_input(input);
    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kOutChannels));
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_ref_enc_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSRefEnc, MatchesPythonReference) {
    ASSERT_MODEL_EXISTS(kModelF16);
    ASSERT_MODEL_EXISTS(kRefInputNpy);
    ASSERT_MODEL_EXISTS(kRefOutputNpy);

    const auto ref_input = load_npy_with_shape(kRefInputNpy);
    const auto ref_output = load_npy_with_shape(kRefOutputNpy);
    ASSERT_FALSE(ref_input.data.empty());
    ASSERT_FALSE(ref_output.data.empty());

    ASSERT_EQ(ref_input.shape.size(), 3u);
    ASSERT_EQ(ref_input.shape[0], 1u);
    ASSERT_EQ(ref_input.shape[1], 1025u);

    ASSERT_EQ(ref_output.shape.size(), 3u);
    ASSERT_EQ(ref_output.shape[0], 1u);
    ASSERT_EQ(ref_output.shape[1], static_cast<size_t>(kOutChannels));
    ASSERT_EQ(ref_output.shape[2], 1u);

    const int64_t time = static_cast<int64_t>(ref_input.shape[2]);
    ASSERT_GT(time, 0);

    const std::vector<float> packed_input =
        pack_v2_ref_input_ggml(ref_input.data, ref_input.shape);
    ASSERT_EQ(packed_input.size(), static_cast<size_t>(kInChannels * time));
    ASSERT_EQ(ref_output.data.size(), static_cast<size_t>(kOutChannels));

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_ref_enc_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_ref_enc_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInChannels, time);
    ggml_set_name(inp, "refer");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::sovits_mel_style_encoder_block_forward(gctx, inp, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "ge");
    ggml_set_output(out);

    ASSERT_EQ(out->ne[0], kOutChannels);
    ASSERT_EQ(out->ne[1], 1);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(inp, packed_input.data(), 0,
                            packed_input.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), ref_output.data.size());
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const auto err = compute_errors(output, ref_output.data);
    printf("[ref_enc parity] T=%lld max_abs=%.6f rmse=%.6f mean_abs=%.6f\n",
           static_cast<long long>(time), err.max_abs, err.rmse, err.mean_abs);
    EXPECT_LT(err.max_abs, kParityMaxAbsTol);
    EXPECT_LT(err.rmse, kParityRmseTol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_ref_enc_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSRefEnc, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_ref_enc_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_ref_enc_model_load("/nonexistent/path.gguf", model, backend));

    ggml_backend_free(backend);
}

TEST(SoVITSRefEnc, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_ref_enc_model model{};
    gpt_sovits::sovits_ref_enc_model_free(model);
}
