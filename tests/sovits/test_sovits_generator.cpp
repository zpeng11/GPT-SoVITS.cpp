// tests/sovits/test_sovits_generator.cpp
//
// Integration test for the SoVITS v2 Generator block:
//   - loads the dedicated generator GGUF
//   - builds a ggml graph for one inference pass
//   - checks output existence, shape, and finite values
//
// No parity test needed.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelF16 =
    kTestDir + "models/v2-generator-f16.gguf";

static constexpr int64_t kGeneratorIn = 192;
static constexpr int64_t kGeneratorGin = 512;
static constexpr int64_t kTime = 4;
static constexpr int64_t kUpsample = 640;
static constexpr size_t kMaxNodes = 8192;

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

    EXPECT_EQ(w.conv_pre.w->ne[0], 7);
    EXPECT_EQ(w.conv_pre.w->ne[1], 192);
    EXPECT_EQ(w.conv_pre.w->ne[2], 512);
    EXPECT_EQ(w.cond.w->ne[0], 1);
    EXPECT_EQ(w.cond.w->ne[1], 512);
    EXPECT_EQ(w.cond.w->ne[2], 512);

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
    EXPECT_EQ(rb0.convs1[0].w->ne[0], 3);
    EXPECT_EQ(rb0.convs1[0].w->ne[1], 256);
    EXPECT_EQ(rb0.convs1[0].w->ne[2], 256);

    EXPECT_EQ(w.conv_post_w->ne[0], 7);
    EXPECT_EQ(w.conv_post_w->ne[1], 16);
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

TEST(SoVITSGenerator, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::sovits_generator_model model{};
    gpt_sovits::sovits_generator_model_free(model);
}
