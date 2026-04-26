// tests/sovits/test_sovits_quantizer.cpp

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "cnpy.h"
#include "npy_loader.h"
#include "test_backend.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelF16 =
    kTestDir + "models/v2-quantizer-f16.gguf";
static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefCodesNpy = kRefDir + "v2_quantizer_codes.npy";
static const std::string kRefDecodedNpy = kRefDir + "v2_quantizer_decoded.npy";

static constexpr int64_t kDim = 768;
static constexpr int64_t kBins = 1024;
static constexpr size_t kMaxNodes = 512;
static constexpr double kParityMaxAbsTol = 1e-3;
static constexpr double kParityRmseTol = 1e-4;

struct ErrorStats {
    double max_abs = 0.0;
    double rmse = 0.0;
    double mean_abs = 0.0;
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

struct NpyShapeInfo {
    std::vector<float> data;
    std::vector<size_t> shape;
};

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static std::vector<int32_t> load_npy_as_i32(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    EXPECT_EQ(arr.word_size, sizeof(int32_t));
    if (arr.word_size != sizeof(int32_t)) {
        return {};
    }

    const int32_t * src = arr.data<int32_t>();
    return std::vector<int32_t>(src, src + arr.num_vals);
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

static std::vector<float> flatten_decoded_bct_to_ggml(
    const std::vector<float> & decoded,
    const std::vector<size_t> & shape)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_EQ(shape[1], static_cast<size_t>(kDim));
    if (shape.size() != 3 || shape[0] != 1u || shape[1] != static_cast<size_t>(kDim)) {
        return {};
    }

    const size_t time = shape[2];
    std::vector<float> packed(static_cast<size_t>(kDim) * time);

    // Reference is saved in NumPy BCT layout; ggml expects contiguous {C, T},
    // so re-pack to [t][c].
    for (size_t t = 0; t < time; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(kDim); ++c) {
            const size_t src = c * time + t;
            const size_t dst = t * static_cast<size_t>(kDim) + c;
            packed[dst] = decoded[src];
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

TEST(SoVITSQuantizer, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_quantizer_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_quantizer_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);
    EXPECT_NE(model.weights.codebook, nullptr);

    gpt_sovits::sovits_quantizer_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSQuantizer, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_quantizer_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_quantizer_model_load(kModelF16, model, backend));

    ASSERT_NE(model.weights.codebook, nullptr);
    EXPECT_EQ(model.weights.codebook->ne[0], kDim);
    EXPECT_EQ(model.weights.codebook->ne[1], kBins);

    gpt_sovits::sovits_quantizer_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSQuantizer, MatchesPythonReference) {
    ASSERT_MODEL_EXISTS(kModelF16);
    ASSERT_MODEL_EXISTS(kRefCodesNpy);
    ASSERT_MODEL_EXISTS(kRefDecodedNpy);

    cnpy::NpyArray codes_arr = cnpy::npy_load(kRefCodesNpy);
    ASSERT_EQ(codes_arr.shape.size(), 3u);
    ASSERT_EQ(codes_arr.shape[0], 1u);
    ASSERT_EQ(codes_arr.shape[1], 1u);
    const int64_t time = static_cast<int64_t>(codes_arr.shape[2]);
    ASSERT_GT(time, 0);

    std::vector<int32_t> codes = load_npy_as_i32(kRefCodesNpy);
    ASSERT_EQ(codes.size(), static_cast<size_t>(time));

    const auto ref_decoded = load_npy_with_shape(kRefDecodedNpy);
    ASSERT_EQ(ref_decoded.shape.size(), 3u);
    ASSERT_EQ(ref_decoded.shape[0], 1u);
    ASSERT_EQ(ref_decoded.shape[1], static_cast<size_t>(kDim));
    ASSERT_EQ(ref_decoded.shape[2], static_cast<size_t>(time));

    const std::vector<float> expected =
        flatten_decoded_bct_to_ggml(ref_decoded.data, ref_decoded.shape);
    ASSERT_EQ(expected.size(), static_cast<size_t>(kDim * time));

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_quantizer_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_quantizer_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * inp = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, time);
    ggml_set_name(inp, "codes");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::sovits_rvq_decode_block_forward(gctx, inp, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "decoded");
    ggml_set_output(out);

    ASSERT_EQ(out->ne[0], kDim);
    ASSERT_EQ(out->ne[1], time);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(inp, codes.data(), 0, codes.size() * sizeof(int32_t));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), expected.size());
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const auto err = compute_errors(output, expected);
    printf("[quantizer parity] T=%lld max_abs=%.6f rmse=%.6f mean_abs=%.6f\n",
           static_cast<long long>(time), err.max_abs, err.rmse, err.mean_abs);
    EXPECT_LT(err.max_abs, kParityMaxAbsTol);
    EXPECT_LT(err.rmse, kParityRmseTol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_quantizer_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSQuantizer, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_quantizer_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_quantizer_model_load("/nonexistent/path.gguf", model, backend));

    ggml_backend_free(backend);
}

TEST(SoVITSQuantizer, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_quantizer_model model{};
    gpt_sovits::sovits_quantizer_model_free(model);
}
