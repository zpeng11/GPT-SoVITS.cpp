// tests/sovits/test_sovits_flow.cpp
//
// Tests for the SoVITS v2 flow (ResidualCouplingBlock) block:
//   - loads the dedicated flow GGUF
//   - builds a ggml graph for the inverse pass
//   - runs inference on deterministic inputs
//   - checks output existence, shape, and finite values
//   - F32 / Q8 / Q5 / Q4 parity tests against PyTorch reference outputs
//
// y_mask is intentionally absent from the C++ flow inverse path: in
// SynthesizerTrn.forward the mask returned by enc_p is always all-ones
// (models_onnx.py:208), and every `* x_mask` in ResidualCouplingLayer / WN
// becomes a no-op. Reference data still records it for completeness.

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
    kTestDir + "models/v2-flow-f16.gguf";
static const std::string kModelQ8 =
    kTestDir + "models/v2-flow-q8.gguf";
static const std::string kModelQ5 =
    kTestDir + "models/v2-flow-q5.gguf";
static const std::string kModelQ4 =
    kTestDir + "models/v2-flow-q4.gguf";
static const std::string kModelF32 =
    kTestDir + "models/v2-flow-f32.gguf";

static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefZpInputNpy  = kRefDir + "v2_flow_input_z_p.npy";
static const std::string kRefGeInputNpy  = kRefDir + "v2_flow_input_ge.npy";
static const std::string kRefZOutputNpy  = kRefDir + "v2_flow_output_z.npy";

static constexpr int64_t kFlowChannels = 192;
static constexpr int64_t kFlowGin = 512;
static constexpr int64_t kTime = 32;
static constexpr size_t kMaxNodes = 4096;

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

// F32 vs PyTorch reference tolerances. Flow inverse is pure conv + pointwise
// math (no attention), so it should land well under the text encoder's parity
// budget; we keep the same baseline and tighten empirically if needed.
static constexpr double kParityMaxAbsTol = 1.2e-2;
static constexpr double kParityRmseTol   = 2.5e-3;

struct ErrorStats {
    double max_abs  = 0.0;
    double rmse     = 0.0;
    double mean_abs = 0.0;
};

struct NpyShapeInfo {
    std::vector<float>   data;
    std::vector<size_t>  shape;
};

struct FlowRefData {
    std::vector<float> packed_zp;
    std::vector<float> packed_ge;
    std::vector<float> expected_z;
    int64_t time = 0;
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

static FlowRefData load_flow_ref_data() {
    EXPECT_TRUE(file_exists(kRefZpInputNpy));
    EXPECT_TRUE(file_exists(kRefGeInputNpy));
    EXPECT_TRUE(file_exists(kRefZOutputNpy));

    const auto ref_zp = load_npy_with_shape(kRefZpInputNpy);
    const auto ref_ge = load_npy_with_shape(kRefGeInputNpy);
    const auto ref_z  = load_npy_with_shape(kRefZOutputNpy);

    EXPECT_FALSE(ref_zp.data.empty());
    EXPECT_FALSE(ref_ge.data.empty());
    EXPECT_FALSE(ref_z.data.empty());

    EXPECT_EQ(ref_zp.shape.size(), 3u);
    EXPECT_EQ(ref_zp.shape[0], 1u);
    EXPECT_EQ(ref_zp.shape[1], static_cast<size_t>(kFlowChannels));

    EXPECT_EQ(ref_ge.shape.size(), 3u);
    EXPECT_EQ(ref_ge.shape[0], 1u);
    EXPECT_EQ(ref_ge.shape[1], static_cast<size_t>(kFlowGin));
    EXPECT_EQ(ref_ge.shape[2], 1u);

    EXPECT_EQ(ref_z.shape.size(), 3u);
    EXPECT_EQ(ref_z.shape[0], 1u);
    EXPECT_EQ(ref_z.shape[1], static_cast<size_t>(kFlowChannels));
    EXPECT_EQ(ref_z.shape[2], ref_zp.shape[2]);

    FlowRefData data;
    data.time      = static_cast<int64_t>(ref_zp.shape[2]);
    data.packed_zp = pack_bct_to_ggml(ref_zp.data, ref_zp.shape, kFlowChannels);
    data.packed_ge = pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kFlowGin);
    data.expected_z = pack_bct_to_ggml(ref_z.data,  ref_z.shape,  kFlowChannels);

    EXPECT_GT(data.time, 0);
    EXPECT_EQ(data.packed_zp.size(),  static_cast<size_t>(kFlowChannels * data.time));
    EXPECT_EQ(data.packed_ge.size(),  static_cast<size_t>(kFlowGin));
    EXPECT_EQ(data.expected_z.size(), static_cast<size_t>(kFlowChannels * data.time));

    return data;
}

static void run_flow_parity(
    const std::string & model_path,
    const char * label,
    double max_abs_tol,
    double rmse_tol)
{
    ASSERT_TRUE(file_exists(model_path));
    FlowRefData ref = load_flow_ref_data();
    ASSERT_GT(ref.time, 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_flow_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(model_path, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * z_p = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, ref.time);
    ggml_set_input(z_p);
    struct ggml_tensor * ge  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowGin, 1);
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_flow_block_inverse_forward(gctx, z_p, ge, model.weights);
    ASSERT_NE(out, nullptr);

    struct ggml_tensor * z_out = ggml_cont(gctx, out);
    ASSERT_NE(z_out, nullptr);
    ggml_set_output(z_out);
    ggml_build_forward_expand(gf, z_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(z_p, ref.packed_zp.data(), 0,
                            ref.packed_zp.size() * sizeof(float));
    ggml_backend_tensor_set(ge,  ref.packed_ge.data(), 0,
                            ref.packed_ge.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_elems  = static_cast<size_t>(kFlowChannels * ref.time);
    const size_t out_nbytes = out_elems * sizeof(float);
    std::vector<float> actual_z(out_elems);
    ggml_backend_tensor_get(z_out, actual_z.data(), 0, out_nbytes);

    const auto err = compute_errors(actual_z, ref.expected_z);
    printf("[flow %s parity] T=%lld z(max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           label,
           static_cast<long long>(ref.time),
           err.max_abs, err.rmse, err.mean_abs);

    EXPECT_LT(err.max_abs, max_abs_tol);
    EXPECT_LT(err.rmse,    rmse_tol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_flow_model_free(model);
    ggml_backend_free(backend);
}

static void fill_input(std::vector<float> & data, int64_t channels, int64_t time) {
    for (int64_t t = 0; t < time; ++t) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t idx = static_cast<size_t>(t * channels + c);
            data[idx] = std::sin(static_cast<float>(c) * 0.03f + static_cast<float>(t) * 0.1f);
        }
    }
}

#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

} // namespace

TEST(SoVITSFlow, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_flow_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_flow_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSFlow, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_flow_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(kModelF16, model, backend));

    // Check layer 0
    const auto & l0 = model.weights.layers[0];
    ASSERT_NE(l0.pre_w, nullptr);
    ASSERT_NE(l0.pre_b, nullptr);
    ASSERT_NE(l0.post_w, nullptr);
    ASSERT_NE(l0.post_b, nullptr);

    // pre is exported as a 2D linear weight {in=96, out=192}
    EXPECT_EQ(l0.pre_w->ne[0], 96);
    EXPECT_EQ(l0.pre_w->ne[1], 192);
    EXPECT_EQ(l0.pre_w->ne[2], 1);

    EXPECT_EQ(l0.pre_b->ne[0], 192);

    // post is exported as a 2D linear weight {in=192, out=96}
    EXPECT_EQ(l0.post_w->ne[0], 192);
    EXPECT_EQ(l0.post_w->ne[1], 96);
    EXPECT_EQ(l0.post_w->ne[2], 1);

    EXPECT_EQ(l0.post_b->ne[0], 96);

    // enc cond is exported as a 2D linear weight {in=512, out=1536}
    EXPECT_EQ(l0.enc.cond_w->ne[0], kFlowGin);
    EXPECT_EQ(l0.enc.cond_w->ne[1], 1536);
    EXPECT_EQ(l0.enc.cond_w->ne[2], 1);

    EXPECT_EQ(l0.enc.cond_b->ne[0], 1536);

    // WN layer 0 in is exported as flattened 2D weight {in*k=960, out=384}
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[0], 192 * 5);
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[1], 384);
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[2], 1);
    EXPECT_EQ(l0.enc.layers[0].in_b->ne[0], 384);

    // WN layer 0 rs is exported as a 2D linear weight {in=192, out=384}
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[0], 192);
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[1], 384);
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[2], 1);
    EXPECT_EQ(l0.enc.layers[0].rs_b->ne[0], 384);

    // WN layer 3 rs is exported as a 2D linear weight {in=192, out=192}
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[0], 192);
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[1], 192);
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[2], 1);
    EXPECT_EQ(l0.enc.layers[3].rs_b->ne[0], 192);

    // Spot-check other layers exist
    for (int L = 1; L < 4; ++L) {
        ASSERT_NE(model.weights.layers[L].pre_w, nullptr) << "layer " << L;
        ASSERT_NE(model.weights.layers[L].enc.cond_w, nullptr) << "layer " << L;
        ASSERT_NE(model.weights.layers[L].enc.layers[0].in_w, nullptr) << "layer " << L;
    }

    gpt_sovits::sovits_flow_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSFlow, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_flow_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    // Input tensors
    struct ggml_tensor * z_p = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, kTime);
    ggml_set_name(z_p, "z_p");
    ggml_set_input(z_p);

    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowGin, 1);
    ggml_set_name(ge, "ge");
    ggml_set_input(ge);

    struct ggml_tensor * out =
        gpt_sovits::sovits_flow_block_inverse_forward(gctx, z_p, ge, model.weights);
    ASSERT_NE(out, nullptr);
    ggml_set_name(out, "z");
    ggml_set_output(out);

    EXPECT_EQ(out->ne[0], kFlowChannels);
    EXPECT_EQ(out->ne[1], kTime);

    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    // Fill inputs with deterministic data
    std::vector<float> z_p_data(static_cast<size_t>(kFlowChannels * kTime));
    fill_input(z_p_data, kFlowChannels, kTime);
    ggml_backend_tensor_set(z_p, z_p_data.data(), 0, z_p_data.size() * sizeof(float));

    std::vector<float> ge_data(static_cast<size_t>(kFlowGin));
    fill_input(ge_data, kFlowGin, 1);
    ggml_backend_tensor_set(ge, ge_data.data(), 0, ge_data.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_nbytes = ggml_nbytes(out);
    std::vector<float> output(out_nbytes / sizeof(float));
    ggml_backend_tensor_get(out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kFlowChannels * kTime));
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_flow_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSFlow, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_flow_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_flow_model_load("/nonexistent/path.gguf", model, backend));

    ggml_backend_free(backend);
}

TEST(SoVITSFlow, QuantizedModelsLoadSuccessfully) {
    for (const std::string & path : {kModelQ8, kModelQ5, kModelQ4}) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_flow_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(path, model, backend));

        gpt_sovits::sovits_flow_model_free(model);
        ggml_backend_free(backend);
    }
}

TEST(SoVITSFlow, QuantizedModelsRunInference) {
    for (const std::string & path : {kModelQ8, kModelQ5, kModelQ4}) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_flow_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_flow_model_load(path, model, backend));

        GraphContext gctx(kMaxNodes);
        ASSERT_NE(gctx.ctx, nullptr);

        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
        ASSERT_NE(gf, nullptr);

        struct ggml_tensor * z_p = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, kTime);
        ggml_set_input(z_p);
        struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowGin, 1);
        ggml_set_input(ge);

        struct ggml_tensor * out =
            gpt_sovits::sovits_flow_block_inverse_forward(gctx, z_p, ge, model.weights);
        ASSERT_NE(out, nullptr);
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ASSERT_NE(alloc, nullptr);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        std::vector<float> z_p_data(static_cast<size_t>(kFlowChannels * kTime));
        fill_input(z_p_data, kFlowChannels, kTime);
        ggml_backend_tensor_set(z_p, z_p_data.data(), 0, z_p_data.size() * sizeof(float));

        std::vector<float> ge_data(static_cast<size_t>(kFlowGin));
        fill_input(ge_data, kFlowGin, 1);
        ggml_backend_tensor_set(ge, ge_data.data(), 0, ge_data.size() * sizeof(float));

        ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

        std::vector<float> output(ggml_nbytes(out) / sizeof(float));
        ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
        for (float v : output) {
            EXPECT_TRUE(std::isfinite(v));
        }

        ggml_gallocr_free(alloc);
        gpt_sovits::sovits_flow_model_free(model);
        ggml_backend_free(backend);
    }
}

TEST(SoVITSFlow, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_flow_model model{};
    gpt_sovits::sovits_flow_model_free(model);
}

TEST(SoVITSFlow, MatchesPythonReference) {
    run_flow_parity(kModelF32, "f32", kParityMaxAbsTol, kParityRmseTol);
}

TEST(SoVITSFlow, QuantizedQ8MatchesPythonReference) {
    run_flow_parity(kModelQ8, "q8", 6.0e-2, 8.0e-3);
}

TEST(SoVITSFlow, QuantizedQ5MatchesPythonReference) {
    run_flow_parity(kModelQ5, "q5", 1.2e-1, 1.8e-2);
}

TEST(SoVITSFlow, QuantizedQ4MatchesPythonReference) {
    run_flow_parity(kModelQ4, "q4", 3.0e-1, 4.0e-2);
}
