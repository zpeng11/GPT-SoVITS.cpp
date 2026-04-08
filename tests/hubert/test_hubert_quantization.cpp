// tests/hubert/test_hubert_quantization.cpp
//
// Quantization quality tests: verify that the f16-quantized HuBERT model
// produces outputs within acceptable tolerance of the f32 reference.

#include <gtest/gtest.h>

#include "gpt_sovits/gpt_sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir  = HUBERT_TEST_DIR;
static const std::string kModelF16 = kTestDir + "models/chinese-hubert-base-f16.gguf";

static constexpr size_t kMaxNodes = 8192;

std::vector<float> load_f32_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        ADD_FAILURE() << "Failed to open: " << path;
        return {};
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

struct ErrorStats {
    double max_abs;
    double rmse;
    double mean_abs;
};

ErrorStats compute_errors(const std::vector<float> & actual,
                          const std::vector<float> & expected)
{
    ErrorStats s{0.0, 0.0, 0.0};
    if (actual.size() != expected.size()) return s;
    double sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double e = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        s.max_abs = std::max(s.max_abs, e);
        sum_sq  += e * e;
        sum_abs += e;
    }
    s.rmse     = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    s.mean_abs = sum_abs / static_cast<double>(actual.size());
    return s;
}

struct GraphContext {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = nullptr;

    explicit GraphContext(size_t max_nodes) {
        size_t sz = ggml_tensor_overhead() * max_nodes
                  + ggml_graph_overhead_custom(max_nodes, false);
        buf.resize(sz);
        struct ggml_init_params params = {
            /*.mem_size   =*/ sz,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
    }

    ~GraphContext() { if (ctx) ggml_free(ctx); }

    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;

    operator struct ggml_context *() { return ctx; }   // NOLINT
};

std::vector<float> eval_graph(
    ggml_backend_t              backend,
    struct ggml_cgraph        * gf,
    const char                * input_name,
    const float               * input_data,
    size_t                      input_floats,
    const char                * output_name)
{
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));

    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ADD_FAILURE() << "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        return {};
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, input_name);
    if (!inp) {
        ADD_FAILURE() << "Input tensor '" << input_name << "' not found";
        ggml_gallocr_free(alloc);
        return {};
    }
    ggml_backend_tensor_set(inp, input_data, 0, input_floats * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ADD_FAILURE() << "ggml_backend_graph_compute failed";
        ggml_gallocr_free(alloc);
        return {};
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, output_name);
    if (!out) {
        ADD_FAILURE() << "Output tensor '" << output_name << "' not found";
        ggml_gallocr_free(alloc);
        return {};
    }

    size_t nbytes = ggml_nbytes(out);
    std::vector<float> result(nbytes / sizeof(float));
    ggml_backend_tensor_get(out, result.data(), 0, nbytes);

    ggml_gallocr_free(alloc);
    return result;
}

}  // anonymous namespace

// ===========================================================================
// F16 quantization quality
// ===========================================================================

TEST(HubertQuantization, F16FullModelVsF32Reference) {
    // Load f16 model
    gpt_sovits::hubert_model model{};
    if (!gpt_sovits::hubert_model_load(kModelF16, model)) {
        GTEST_SKIP() << "Could not load " << kModelF16;
    }

    auto input = load_f32_bin(kTestDir + "ref_input.bin");
    auto ref   = load_f32_bin(kTestDir + "ref_model_output.bin");
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(ref.empty());

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);

    struct ggml_tensor * inp = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                                   static_cast<int64_t>(input.size()));
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::hubert_model_block_forward(gctx, inp, model.weights);
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    auto result = eval_graph(model.backend, gf,
                             "input",  input.data(),  input.size(),
                             "output");
    ASSERT_EQ(result.size(), ref.size());

    auto err = compute_errors(result, ref);
    printf("  F16 Full Model  : max_abs=%.4e  rmse=%.4e  mean_abs=%.4e\n",
           err.max_abs, err.rmse, err.mean_abs);

    // F16 weights introduce quantization noise that compounds through 12
    // transformer layers.  Tolerances are significantly relaxed vs f32.
    EXPECT_LT(err.max_abs, 1.0)  << "max abs error too large for f16 model";
    EXPECT_LT(err.rmse,    0.1)  << "RMSE too large for f16 model";

    // Log the cosine similarity as a secondary quality metric.
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < result.size(); i++) {
        dot    += static_cast<double>(result[i]) * static_cast<double>(ref[i]);
        norm_a += static_cast<double>(result[i]) * static_cast<double>(result[i]);
        norm_b += static_cast<double>(ref[i])    * static_cast<double>(ref[i]);
    }
    double cosine_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    printf("  F16 Cosine sim  : %.6f\n", cosine_sim);
    EXPECT_GT(cosine_sim, 0.99) << "Cosine similarity too low for f16 model";

    gpt_sovits::hubert_model_free(model);
}
