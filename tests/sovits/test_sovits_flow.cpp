// tests/sovits/test_sovits_flow.cpp
//
// Integration test for the SoVITS v2 flow (ResidualCouplingBlock) block:
//   - loads the dedicated flow GGUF
//   - builds a ggml graph for the inverse pass
//   - runs inference on deterministic inputs
//   - checks output existence, shape, and finite values
//
// No parity test needed — correctness is verified by output shape and finiteness.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
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

    // pre: ggml sees {1, 96, 192} after GGUF reversal from PyTorch (192, 96, 1)
    EXPECT_EQ(l0.pre_w->ne[0], 1);
    EXPECT_EQ(l0.pre_w->ne[1], 96);
    EXPECT_EQ(l0.pre_w->ne[2], 192);

    EXPECT_EQ(l0.pre_b->ne[0], 192);

    // post: ggml sees {1, 192, 96}
    EXPECT_EQ(l0.post_w->ne[0], 1);
    EXPECT_EQ(l0.post_w->ne[1], 192);
    EXPECT_EQ(l0.post_w->ne[2], 96);

    EXPECT_EQ(l0.post_b->ne[0], 96);

    // enc cond: {1, 512, 1536}
    EXPECT_EQ(l0.enc.cond_w->ne[0], 1);
    EXPECT_EQ(l0.enc.cond_w->ne[1], kFlowGin);
    EXPECT_EQ(l0.enc.cond_w->ne[2], 1536);

    EXPECT_EQ(l0.enc.cond_b->ne[0], 1536);

    // WN layer 0 in: {5, 192, 384}
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[0], 5);
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[1], 192);
    EXPECT_EQ(l0.enc.layers[0].in_w->ne[2], 384);
    EXPECT_EQ(l0.enc.layers[0].in_b->ne[0], 384);

    // WN layer 0 rs: {1, 192, 384}
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[0], 1);
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[1], 192);
    EXPECT_EQ(l0.enc.layers[0].rs_w->ne[2], 384);
    EXPECT_EQ(l0.enc.layers[0].rs_b->ne[0], 384);

    // WN layer 3 rs: {1, 192, 192} (last layer, only hidden out)
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[0], 1);
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[1], 192);
    EXPECT_EQ(l0.enc.layers[3].rs_w->ne[2], 192);
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

TEST(SoVITSFlow, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_flow_model model{};
    gpt_sovits::sovits_flow_model_free(model);
}
