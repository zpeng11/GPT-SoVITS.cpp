// tests/sovits/test_sovits_generator.cpp
//
// Tests for the SoVITS v2 Generator block:
//   - loads the dedicated generator GGUF
//   - builds a ggml graph for one inference pass
//   - checks output existence, shape, and finite values
//   - F32 / Q8 / Q5 / Q4 parity tests against PyTorch reference outputs
//
// y_mask is intentionally absent from the C++ generator path: in
// SynthesizerTrn.forward the mask is always all-ones (models_onnx.py:208),
// so `(z * y_mask)[:, :, :]` in models_onnx.py:905 is a no-op. The
// reference data still records it for completeness.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "cnpy.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "npy_loader.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelF16 =
    kTestDir + "models/v2-generator-f16.gguf";
static const std::string kModelQ8 =
    kTestDir + "models/v2-generator-q8.gguf";
static const std::string kModelQ5 =
    kTestDir + "models/v2-generator-q5.gguf";
static const std::string kModelQ4 =
    kTestDir + "models/v2-generator-q4.gguf";
static const std::string kModelF32 =
    kTestDir + "models/v2-generator-f32.gguf";

static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefZInputNpy  = kRefDir + "v2_dec_input_z.npy";
static const std::string kRefGeInputNpy = kRefDir + "v2_dec_input_ge.npy";
static const std::string kRefOutNpy     = kRefDir + "v2_dec_output_o.npy";

static constexpr int64_t kGeneratorIn = 192;
static constexpr int64_t kGeneratorGin = 512;
static constexpr int64_t kTime = 4;
static constexpr int64_t kUpsample = 640;
static constexpr size_t kMaxNodes = 8192;

// F32 vs PyTorch reference tolerances. The reference output is stored as
// fp16, so even an ideal F32 implementation has an irreducible ~1e-2
// max_abs floor from the fp16 rounding of v2_dec_output_o.npy (the
// generator output is bounded to (-1, 1) by tanh, so absolute error
// equals relative error here). Budgets below give ~50% headroom over
// observed actuals; quantized tiers follow the text-encoder / flow
// progression.
static constexpr double kParityMaxAbsTol = 3.0e-2;
static constexpr double kParityRmseTol   = 6.0e-3;

struct ErrorStats {
    double max_abs  = 0.0;
    double rmse     = 0.0;
    double mean_abs = 0.0;
};

struct NpyShapeInfo {
    std::vector<float>   data;
    std::vector<size_t>  shape;
};

struct GeneratorRefData {
    std::vector<float> packed_z;
    std::vector<float> packed_ge;
    std::vector<float> expected_o;
    int64_t time        = 0;
    int64_t out_samples = 0;
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

static ErrorStats compute_errors(const std::vector<float> & actual,
                                 const std::vector<float> & expected) {
    ErrorStats s{};
    if (actual.size() != expected.size() || actual.empty()) {
        return s;
    }

    double sum_sq  = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double err = std::abs(static_cast<double>(actual[i]) -
                                    static_cast<double>(expected[i]));
        s.max_abs = std::max(s.max_abs, err);
        sum_sq  += err * err;
        sum_abs += err;
    }

    s.rmse     = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    s.mean_abs = sum_abs / static_cast<double>(actual.size());
    return s;
}

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

// PyTorch lay-out is [B, C, T]; ggml 2d tensors are [C, T] row-major
// (t-major). Repack so that index (c, t) goes to dst[t * C + c].
static std::vector<float> pack_bct_to_ggml(
    const std::vector<float> & tensor,
    const std::vector<size_t> & shape,
    int64_t channels)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_EQ(shape[1], static_cast<size_t>(channels));
    if (shape.size() != 3 || shape[0] != 1u ||
        shape[1] != static_cast<size_t>(channels)) {
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

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static GeneratorRefData load_generator_ref_data() {
    EXPECT_TRUE(file_exists(kRefZInputNpy));
    EXPECT_TRUE(file_exists(kRefGeInputNpy));
    EXPECT_TRUE(file_exists(kRefOutNpy));

    const auto ref_z  = load_npy_with_shape(kRefZInputNpy);
    const auto ref_ge = load_npy_with_shape(kRefGeInputNpy);
    const auto ref_o  = load_npy_with_shape(kRefOutNpy);

    EXPECT_FALSE(ref_z.data.empty());
    EXPECT_FALSE(ref_ge.data.empty());
    EXPECT_FALSE(ref_o.data.empty());

    EXPECT_EQ(ref_z.shape.size(), 3u);
    EXPECT_EQ(ref_z.shape[0], 1u);
    EXPECT_EQ(ref_z.shape[1], static_cast<size_t>(kGeneratorIn));

    EXPECT_EQ(ref_ge.shape.size(), 3u);
    EXPECT_EQ(ref_ge.shape[0], 1u);
    EXPECT_EQ(ref_ge.shape[1], static_cast<size_t>(kGeneratorGin));
    EXPECT_EQ(ref_ge.shape[2], 1u);

    EXPECT_EQ(ref_o.shape.size(), 3u);
    EXPECT_EQ(ref_o.shape[0], 1u);
    EXPECT_EQ(ref_o.shape[1], 1u);
    EXPECT_EQ(ref_o.shape[2], ref_z.shape[2] * static_cast<size_t>(kUpsample));

    GeneratorRefData data;
    data.time        = static_cast<int64_t>(ref_z.shape[2]);
    data.out_samples = static_cast<int64_t>(ref_o.shape[2]);
    data.packed_z    = pack_bct_to_ggml(ref_z.data,  ref_z.shape,  kGeneratorIn);
    data.packed_ge   = pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kGeneratorGin);
    // Output is single-channel waveform {1, 1, T*640} — straight flat copy.
    data.expected_o  = ref_o.data;

    EXPECT_GT(data.time, 0);
    EXPECT_EQ(data.packed_z.size(),  static_cast<size_t>(kGeneratorIn * data.time));
    EXPECT_EQ(data.packed_ge.size(), static_cast<size_t>(kGeneratorGin));
    EXPECT_EQ(data.expected_o.size(), static_cast<size_t>(data.out_samples));
    EXPECT_EQ(data.out_samples, data.time * kUpsample);

    return data;
}

static void run_generator_parity(
    const std::string & model_path,
    const char * label,
    double max_abs_tol,
    double rmse_tol)
{
    ASSERT_TRUE(file_exists(model_path));
    GeneratorRefData ref = load_generator_ref_data();
    ASSERT_GT(ref.time, 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_generator_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(model_path, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * z = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorIn, ref.time);
    ggml_set_input(z);
    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorGin, 1);
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_generator_block_forward(gctx, z, ge, model.weights);
    ASSERT_NE(out, nullptr);

    struct ggml_tensor * wav_out = ggml_cont(gctx, out);
    ASSERT_NE(wav_out, nullptr);
    ggml_set_output(wav_out);
    ggml_build_forward_expand(gf, wav_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(z,  ref.packed_z.data(), 0,
                            ref.packed_z.size() * sizeof(float));
    ggml_backend_tensor_set(ge, ref.packed_ge.data(), 0,
                            ref.packed_ge.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_elems  = static_cast<size_t>(ref.out_samples);
    const size_t out_nbytes = out_elems * sizeof(float);
    std::vector<float> actual_o(out_elems);
    ggml_backend_tensor_get(wav_out, actual_o.data(), 0, out_nbytes);

    const auto err = compute_errors(actual_o, ref.expected_o);
    printf("[generator %s parity] T=%lld out=%lld o(max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           label,
           static_cast<long long>(ref.time),
           static_cast<long long>(ref.out_samples),
           err.max_abs, err.rmse, err.mean_abs);

    EXPECT_LT(err.max_abs, max_abs_tol);
    EXPECT_LT(err.rmse,    rmse_tol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_generator_model_free(model);
    ggml_backend_free(backend);
}

static void fill_input(std::vector<float> & data, int64_t channels, int64_t time) {
    for (int64_t t = 0; t < time; ++t) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t idx = static_cast<size_t>(t * channels + c);
            data[idx] = std::sin(static_cast<float>(c) * 0.07f + static_cast<float>(t) * 0.11f);
        }
    }
}

#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

} // namespace

TEST(SoVITSGenerator, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_generator_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_generator_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSGenerator, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_generator_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(kModelF16, model, backend));

    const auto & w = model.weights;
    ASSERT_NE(w.conv_pre.w, nullptr);
    ASSERT_NE(w.conv_pre.b, nullptr);
    ASSERT_NE(w.cond.w, nullptr);
    ASSERT_NE(w.cond.b, nullptr);
    ASSERT_NE(w.conv_post_w, nullptr);

    EXPECT_EQ(w.conv_pre.w->ne[0], 192 * 7);
    EXPECT_EQ(w.conv_pre.w->ne[1], 512);
    EXPECT_EQ(w.conv_pre.w->ne[2], 1);
    EXPECT_EQ(w.cond.w->ne[0], 512);
    EXPECT_EQ(w.cond.w->ne[1], 512);
    EXPECT_EQ(w.cond.w->ne[2], 1);

    const auto & s0 = w.stages[0];
    ASSERT_NE(s0.up.w, nullptr);
    ASSERT_NE(s0.up.b, nullptr);
    EXPECT_EQ(s0.up.w->ne[0], 16);
    EXPECT_EQ(s0.up.w->ne[1], 256);
    EXPECT_EQ(s0.up.w->ne[2], 512);
    EXPECT_EQ(s0.up.b->ne[0], 256);

    const auto & rb0 = s0.resblocks[0];
    ASSERT_NE(rb0.convs1[0].w, nullptr);
    ASSERT_NE(rb0.convs1[0].b, nullptr);
    ASSERT_NE(rb0.convs2[0].w, nullptr);
    ASSERT_NE(rb0.convs2[0].b, nullptr);
    EXPECT_EQ(rb0.convs1[0].w->ne[0], 256 * 3);
    EXPECT_EQ(rb0.convs1[0].w->ne[1], 256);
    EXPECT_EQ(rb0.convs1[0].w->ne[2], 1);

    EXPECT_EQ(w.conv_post_w->ne[0], 16 * 7);
    EXPECT_EQ(w.conv_post_w->ne[1], 1);
    EXPECT_EQ(w.conv_post_w->ne[2], 1);

    gpt_sovits::sovits_generator_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSGenerator, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_generator_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * z = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorIn, kTime);
    ggml_set_name(z, "z");
    ggml_set_input(z);

    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorGin, 1);
    ggml_set_name(ge, "ge");
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_generator_block_forward(gctx, z, ge, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "wav");
    ggml_set_output(out);

    EXPECT_EQ(out->ne[0], 1);
    EXPECT_EQ(out->ne[1], kTime * kUpsample);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    std::vector<float> z_data(static_cast<size_t>(kGeneratorIn * kTime));
    fill_input(z_data, kGeneratorIn, kTime);
    ggml_backend_tensor_set(z, z_data.data(), 0, z_data.size() * sizeof(float));

    std::vector<float> ge_data(static_cast<size_t>(kGeneratorGin));
    fill_input(ge_data, kGeneratorGin, 1);
    ggml_backend_tensor_set(ge, ge_data.data(), 0, ge_data.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kTime * kUpsample));
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_generator_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSGenerator, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_generator_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_generator_model_load("/nonexistent/path.gguf", model, backend));

    ggml_backend_free(backend);
}

TEST(SoVITSGenerator, QuantizedModelsLoadSuccessfully) {
    for (const std::string & path : {kModelQ8, kModelQ5, kModelQ4}) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_generator_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(path, model, backend));

        gpt_sovits::sovits_generator_model_free(model);
        ggml_backend_free(backend);
    }
}

TEST(SoVITSGenerator, QuantizedModelsRunInference) {
    for (const std::string & path : {kModelQ8, kModelQ5, kModelQ4}) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_generator_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_generator_model_load(path, model, backend));

        GraphContext gctx(kMaxNodes);
        ASSERT_NE(gctx.ctx, nullptr);

        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
        ASSERT_NE(gf, nullptr);

        struct ggml_tensor * z = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorIn, kTime);
        ggml_set_input(z);
        struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeneratorGin, 1);
        ggml_set_input(ge);

        struct ggml_tensor * out =
            gpt_sovits::sovits_generator_block_forward(gctx, z, ge, model.weights);
        ASSERT_NE(out, nullptr);
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ASSERT_NE(alloc, nullptr);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        std::vector<float> z_data(static_cast<size_t>(kGeneratorIn * kTime));
        fill_input(z_data, kGeneratorIn, kTime);
        ggml_backend_tensor_set(z, z_data.data(), 0, z_data.size() * sizeof(float));

        std::vector<float> ge_data(static_cast<size_t>(kGeneratorGin));
        fill_input(ge_data, kGeneratorGin, 1);
        ggml_backend_tensor_set(ge, ge_data.data(), 0, ge_data.size() * sizeof(float));

        ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

        std::vector<float> output(ggml_nbytes(out) / sizeof(float));
        ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
        for (float v : output) {
            EXPECT_TRUE(std::isfinite(v));
        }

        ggml_gallocr_free(alloc);
        gpt_sovits::sovits_generator_model_free(model);
        ggml_backend_free(backend);
    }
}

TEST(SoVITSGenerator, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_generator_model model{};
    gpt_sovits::sovits_generator_model_free(model);
}

TEST(SoVITSGenerator, MatchesPythonReference) {
    run_generator_parity(kModelF32, "f32", kParityMaxAbsTol, kParityRmseTol);
}

TEST(SoVITSGenerator, QuantizedQ8MatchesPythonReference) {
    run_generator_parity(kModelQ8, "q8", 8.0e-2, 1.5e-2);
}

TEST(SoVITSGenerator, QuantizedQ5MatchesPythonReference) {
    run_generator_parity(kModelQ5, "q5", 1.5e-1, 3.0e-2);
}

TEST(SoVITSGenerator, QuantizedQ4MatchesPythonReference) {
    run_generator_parity(kModelQ4, "q4", 3.0e-1, 6.0e-2);
}
