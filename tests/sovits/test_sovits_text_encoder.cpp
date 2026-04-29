// tests/sovits/test_sovits_text_encoder.cpp

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
    kTestDir + "models/v2-text-encoder-f16.gguf";
static const std::string kModelF32 =
    kTestDir + "models/v2-text-encoder-f32.gguf";
static const std::string kModelQ8 =
    kTestDir + "models/v2-text-encoder-q8.gguf";
static const std::string kModelQ5 =
    kTestDir + "models/v2-text-encoder-q5.gguf";
static const std::string kModelQ4 =
    kTestDir + "models/v2-text-encoder-q4.gguf";
static const std::string kRefDir = kTestDir + "ref/";
static const std::string kRefQuantizedInputNpy = kRefDir + "v2_enc_p_input_quantized.npy";
static const std::string kRefTextInputNpy = kRefDir + "v2_enc_p_input_text.npy";
static const std::string kRefGeInputNpy = kRefDir + "v2_enc_p_input_ge.npy";
static const std::string kRefXOutputNpy = kRefDir + "v2_enc_p_output_x.npy";
static const std::string kRefMOutputNpy = kRefDir + "v2_enc_p_output_m_p.npy";
static const std::string kRefLogsOutputNpy = kRefDir + "v2_enc_p_output_logs_p.npy";

static constexpr int64_t kSslIn = 768;
static constexpr int64_t kHidden = 192;
static constexpr int64_t kGeDim = 512;
static constexpr int64_t kOutChannels = 192;
static constexpr int64_t kTextVocab = 732;
static constexpr int64_t kSslTime = 24;
static constexpr int64_t kTextTime = 17;
static constexpr size_t kMaxNodes = 65536;
static constexpr double kParityMaxAbsTol = 1.2e-2;
static constexpr double kParityRmseTol = 2.5e-3;

struct ErrorStats {
    double max_abs = 0.0;
    double rmse = 0.0;
    double mean_abs = 0.0;
};

struct NpyShapeInfo {
    std::vector<float> data;
    std::vector<size_t> shape;
};

struct TextEncoderRefData {
    std::vector<float> packed_quantized;
    std::vector<int32_t> text;
    std::vector<float> packed_ge;
    std::vector<float> expected_x;
    std::vector<float> expected_m;
    std::vector<float> expected_logs;
    int64_t ssl_time = 0;
    int64_t text_time = 0;
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

static void fill_ssl(std::vector<float> & data) {
    for (int64_t t = 0; t < kSslTime; ++t) {
        for (int64_t c = 0; c < kSslIn; ++c) {
            const size_t idx = static_cast<size_t>(t * kSslIn + c);
            data[idx] = std::sin(static_cast<float>(c) * 0.013f)
                      + std::cos(static_cast<float>(t) * 0.11f)
                      + 0.001f * static_cast<float>(c % 7);
        }
    }
}

static std::vector<int32_t> make_text_tokens() {
    std::vector<int32_t> tokens(static_cast<size_t>(kTextTime));
    for (int64_t t = 0; t < kTextTime; ++t) {
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>((17 * t + 3) % kTextVocab);
    }
    return tokens;
}

static void fill_ge(std::vector<float> & data) {
    for (int64_t c = 0; c < kGeDim; ++c) {
        data[static_cast<size_t>(c)] = std::sin(static_cast<float>(c) * 0.007f);
    }
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

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static TextEncoderRefData load_text_encoder_ref_data() {
    EXPECT_TRUE(file_exists(kRefQuantizedInputNpy));
    EXPECT_TRUE(file_exists(kRefTextInputNpy));
    EXPECT_TRUE(file_exists(kRefGeInputNpy));
    EXPECT_TRUE(file_exists(kRefXOutputNpy));
    EXPECT_TRUE(file_exists(kRefMOutputNpy));
    EXPECT_TRUE(file_exists(kRefLogsOutputNpy));

    const auto ref_quantized = load_npy_with_shape(kRefQuantizedInputNpy);
    const std::vector<int32_t> ref_text = load_npy_as_i32(kRefTextInputNpy);
    const auto ref_ge = load_npy_with_shape(kRefGeInputNpy);
    const auto ref_x = load_npy_with_shape(kRefXOutputNpy);
    const auto ref_m = load_npy_with_shape(kRefMOutputNpy);
    const auto ref_logs = load_npy_with_shape(kRefLogsOutputNpy);

    EXPECT_FALSE(ref_quantized.data.empty());
    EXPECT_FALSE(ref_text.empty());
    EXPECT_FALSE(ref_ge.data.empty());
    EXPECT_FALSE(ref_x.data.empty());
    EXPECT_FALSE(ref_m.data.empty());
    EXPECT_FALSE(ref_logs.data.empty());

    EXPECT_EQ(ref_quantized.shape.size(), 3u);
    EXPECT_EQ(ref_quantized.shape[0], 1u);
    EXPECT_EQ(ref_quantized.shape[1], static_cast<size_t>(kSslIn));

    EXPECT_EQ(ref_ge.shape.size(), 3u);
    EXPECT_EQ(ref_ge.shape[0], 1u);
    EXPECT_EQ(ref_ge.shape[1], static_cast<size_t>(kGeDim));
    EXPECT_EQ(ref_ge.shape[2], 1u);

    EXPECT_EQ(ref_x.shape.size(), 3u);
    EXPECT_EQ(ref_x.shape[0], 1u);
    EXPECT_EQ(ref_x.shape[1], static_cast<size_t>(kOutChannels));
    EXPECT_EQ(ref_m.shape.size(), 3u);
    EXPECT_EQ(ref_m.shape[0], 1u);
    EXPECT_EQ(ref_m.shape[1], static_cast<size_t>(kOutChannels));
    EXPECT_EQ(ref_logs.shape.size(), 3u);
    EXPECT_EQ(ref_logs.shape[0], 1u);
    EXPECT_EQ(ref_logs.shape[1], static_cast<size_t>(kOutChannels));
    EXPECT_EQ(ref_x.shape[2], ref_quantized.shape[2]);
    EXPECT_EQ(ref_m.shape[2], ref_quantized.shape[2]);
    EXPECT_EQ(ref_logs.shape[2], ref_quantized.shape[2]);

    TextEncoderRefData data;
    data.ssl_time = static_cast<int64_t>(ref_quantized.shape[2]);
    data.text_time = static_cast<int64_t>(ref_text.size());
    data.text = ref_text;
    data.packed_quantized = pack_bct_to_ggml(ref_quantized.data, ref_quantized.shape, kSslIn);
    data.packed_ge = pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kGeDim);
    data.expected_x = pack_bct_to_ggml(ref_x.data, ref_x.shape, kOutChannels);
    data.expected_m = pack_bct_to_ggml(ref_m.data, ref_m.shape, kOutChannels);
    data.expected_logs = pack_bct_to_ggml(ref_logs.data, ref_logs.shape, kOutChannels);

    EXPECT_GT(data.ssl_time, 0);
    EXPECT_GT(data.text_time, 0);
    EXPECT_EQ(data.packed_quantized.size(), static_cast<size_t>(kSslIn * data.ssl_time));
    EXPECT_EQ(data.packed_ge.size(), static_cast<size_t>(kGeDim));
    EXPECT_EQ(data.expected_x.size(), static_cast<size_t>(kOutChannels * data.ssl_time));
    EXPECT_EQ(data.expected_m.size(), static_cast<size_t>(kOutChannels * data.ssl_time));
    EXPECT_EQ(data.expected_logs.size(), static_cast<size_t>(kOutChannels * data.ssl_time));

    return data;
}

static void run_text_encoder_parity(
    const std::string & model_path,
    const char * label,
    double max_abs_tol,
    double rmse_tol)
{
    ASSERT_TRUE(file_exists(model_path));
    TextEncoderRefData ref = load_text_encoder_ref_data();

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(model_path, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kSslIn, ref.ssl_time);
    ggml_set_input(ssl);
    struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, ref.text_time);
    ggml_set_input(text);
    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeDim, 1);
    ggml_set_input(ge);

    const gpt_sovits::sovits_text_encoder_result out =
        gpt_sovits::sovits_text_encoder_block_forward(gctx, ssl, text, ge, model.weights);
    ASSERT_NE(out.x, nullptr);
    ASSERT_NE(out.m, nullptr);
    ASSERT_NE(out.logs, nullptr);

    struct ggml_tensor * x_out = ggml_cont(gctx, out.x);
    struct ggml_tensor * m_out = ggml_cont(gctx, out.m);
    struct ggml_tensor * logs_out = ggml_cont(gctx, out.logs);
    ASSERT_NE(x_out, nullptr);
    ASSERT_NE(m_out, nullptr);
    ASSERT_NE(logs_out, nullptr);
    ggml_set_output(x_out);
    ggml_set_output(m_out);
    ggml_set_output(logs_out);

    ggml_build_forward_expand(gf, x_out);
    ggml_build_forward_expand(gf, m_out);
    ggml_build_forward_expand(gf, logs_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(ssl, ref.packed_quantized.data(), 0, ref.packed_quantized.size() * sizeof(float));
    ggml_backend_tensor_set(text, ref.text.data(), 0, ref.text.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ge, ref.packed_ge.data(), 0, ref.packed_ge.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_elems = static_cast<size_t>(kOutChannels * ref.ssl_time);
    const size_t out_nbytes = out_elems * sizeof(float);
    std::vector<float> actual_x(out_elems);
    std::vector<float> actual_m(out_elems);
    std::vector<float> actual_logs(out_elems);
    ggml_backend_tensor_get(x_out, actual_x.data(), 0, out_nbytes);
    ggml_backend_tensor_get(m_out, actual_m.data(), 0, out_nbytes);
    ggml_backend_tensor_get(logs_out, actual_logs.data(), 0, out_nbytes);

    const auto err_x = compute_errors(actual_x, ref.expected_x);
    const auto err_m = compute_errors(actual_m, ref.expected_m);
    const auto err_logs = compute_errors(actual_logs, ref.expected_logs);
    printf("[text_encoder %s parity] Tssl=%lld Ttext=%lld x(max_abs=%.6f rmse=%.6f mean_abs=%.6f) m(max_abs=%.6f rmse=%.6f mean_abs=%.6f) logs(max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           label,
           static_cast<long long>(ref.ssl_time),
           static_cast<long long>(ref.text_time),
           err_x.max_abs,
           err_x.rmse,
           err_x.mean_abs,
           err_m.max_abs,
           err_m.rmse,
           err_m.mean_abs,
           err_logs.max_abs,
           err_logs.rmse,
           err_logs.mean_abs);

    EXPECT_LT(err_x.max_abs, max_abs_tol);
    EXPECT_LT(err_x.rmse, rmse_tol);
    EXPECT_LT(err_m.max_abs, max_abs_tol);
    EXPECT_LT(err_m.rmse, rmse_tol);
    EXPECT_LT(err_logs.max_abs, max_abs_tol);
    EXPECT_LT(err_logs.rmse, rmse_tol);

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_model_free(model);
    ggml_backend_free(backend);
}

// Helper: skip test if model file not found.
#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

} // namespace

TEST(SoVITSTextEncoder, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(kModelF16, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);

    gpt_sovits::sovits_text_encoder_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoder, WeightPointersAndShapesLookCorrect) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(kModelF16, model, backend));

    const auto & w = model.weights;
    ASSERT_NE(w.ssl.ssl_proj_w, nullptr);
    ASSERT_NE(w.text.text_embedding, nullptr);
    ASSERT_NE(w.mrte.ssl_fused_w, nullptr);
    ASSERT_NE(w.post.proj_w, nullptr);
    ASSERT_NE(w.post.proj_b, nullptr);

    EXPECT_EQ(w.ssl.ssl_proj_w->ne[0], kSslIn);
    EXPECT_EQ(w.ssl.ssl_proj_w->ne[1], kHidden);
    EXPECT_EQ(w.text.text_embedding->ne[0], kHidden);
    EXPECT_EQ(w.text.text_embedding->ne[1], kTextVocab);
    EXPECT_EQ(w.ssl.layers[0].qkv_w->ne[0], kHidden);
    EXPECT_EQ(w.ssl.layers[0].qkv_w->ne[1], 3 * kHidden);
    EXPECT_EQ(w.ssl.layers[0].out_w->ne[0], kHidden);
    EXPECT_EQ(w.ssl.layers[0].out_w->ne[1], kHidden);
    EXPECT_EQ(w.mrte.ssl_fused_w->ne[0], kHidden);
    EXPECT_EQ(w.mrte.ssl_fused_w->ne[1], 704);
    EXPECT_EQ(w.mrte.text_kv_w->ne[0], kHidden);
    EXPECT_EQ(w.mrte.text_kv_w->ne[1], 1024);
    EXPECT_EQ(w.mrte.attn_out_w->ne[0], kGeDim);
    EXPECT_EQ(w.mrte.attn_out_w->ne[1], kHidden);
    EXPECT_EQ(w.post.proj_w->ne[0], kHidden);
    EXPECT_EQ(w.post.proj_w->ne[1], 2 * kOutChannels);
    EXPECT_EQ(w.post.proj_b->ne[0], 2 * kOutChannels);

    gpt_sovits::sovits_text_encoder_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoder, BuildsGraphAndRunsInference) {
    ASSERT_MODEL_EXISTS(kModelF16);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(kModelF16, model, backend));

    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ASSERT_NE(gf, nullptr);

    struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kSslIn, kSslTime);
    ggml_set_name(ssl, "text_encoder_ssl");
    ggml_set_input(ssl);

    struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kTextTime);
    ggml_set_name(text, "text_encoder_text");
    ggml_set_input(text);

    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeDim, 1);
    ggml_set_name(ge, "text_encoder_ge");
    ggml_set_input(ge);

    const gpt_sovits::sovits_text_encoder_result out =
        gpt_sovits::sovits_text_encoder_block_forward(gctx, ssl, text, ge, model.weights);
    ASSERT_NE(out.x, nullptr);
    ASSERT_NE(out.m, nullptr);
    ASSERT_NE(out.logs, nullptr);

    struct ggml_tensor * logs_out = ggml_cont(gctx, out.logs);
    ASSERT_NE(logs_out, nullptr);

    ggml_set_name(logs_out, "text_encoder_logs");
    ggml_set_output(logs_out);
    EXPECT_EQ(logs_out->type, GGML_TYPE_F32);

    EXPECT_EQ(out.x->ne[0], kOutChannels);
    EXPECT_EQ(out.x->ne[1], kSslTime);
    EXPECT_EQ(out.m->ne[0], kOutChannels);
    EXPECT_EQ(out.m->ne[1], kSslTime);
    EXPECT_EQ(out.logs->ne[0], kOutChannels);
    EXPECT_EQ(out.logs->ne[1], kSslTime);
    EXPECT_EQ(logs_out->ne[0], kOutChannels);
    EXPECT_EQ(logs_out->ne[1], kSslTime);

    ggml_build_forward_expand(gf, logs_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    std::vector<float> ssl_input(static_cast<size_t>(kSslIn * kSslTime));
    std::vector<int32_t> text_input = make_text_tokens();
    std::vector<float> ge_input(static_cast<size_t>(kGeDim));
    fill_ssl(ssl_input);
    fill_ge(ge_input);

    ggml_backend_tensor_set(ssl, ssl_input.data(), 0, ssl_input.size() * sizeof(float));
    ggml_backend_tensor_set(text, text_input.data(), 0, text_input.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ge, ge_input.data(), 0, ge_input.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    const size_t out_elems = static_cast<size_t>(logs_out->ne[0] * logs_out->ne[1]);
    const size_t out_nbytes = out_elems * sizeof(float);
    std::vector<float> output(out_elems);
    ggml_backend_tensor_get(logs_out, output.data(), 0, out_nbytes);

    ASSERT_EQ(output.size(), static_cast<size_t>(kOutChannels * kSslTime));
    for (float value : output) {
        EXPECT_TRUE(std::isfinite(value));
    }

    ggml_gallocr_free(alloc);
    gpt_sovits::sovits_text_encoder_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoder, MissingModelFileFailsCleanly) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_text_encoder_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_text_encoder_model_load(
        "/nonexistent/path.gguf", model, backend));

    gpt_sovits::sovits_text_encoder_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSTextEncoder, FreeOnEmptyModelIsSafe) {
    gpt_sovits::sovits_text_encoder_model model{};
    gpt_sovits::sovits_text_encoder_model_free(model);
}

TEST(SoVITSTextEncoder, MatchesPythonReference) {
    run_text_encoder_parity(kModelF32, "f32", kParityMaxAbsTol, kParityRmseTol);
}

TEST(SoVITSTextEncoder, QuantizedQ8MatchesPythonReference) {
    run_text_encoder_parity(kModelQ8, "q8", 6.0e-2, 8.0e-3);
}

TEST(SoVITSTextEncoder, QuantizedQ5MatchesPythonReference) {
    run_text_encoder_parity(kModelQ5, "q5", 1.2e-1, 1.8e-2);
}

TEST(SoVITSTextEncoder, QuantizedQ4MatchesPythonReference) {
    run_text_encoder_parity(kModelQ4, "q4", 2.0e-1, 3.0e-2);
}

TEST(SoVITSTextEncoder, QuantizedQ8RunsInference) {
    auto run_quantized = [&](const std::string & path) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_text_encoder_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(path, model, backend));

        GraphContext gctx(kMaxNodes);
        ASSERT_NE(gctx.ctx, nullptr);

        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
        ASSERT_NE(gf, nullptr);

        struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kSslIn, kSslTime);
        ggml_set_input(ssl);
        struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kTextTime);
        ggml_set_input(text);
        struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeDim, 1);
        ggml_set_input(ge);

        const gpt_sovits::sovits_text_encoder_result out =
            gpt_sovits::sovits_text_encoder_block_forward(gctx, ssl, text, ge, model.weights);
        ASSERT_NE(out.logs, nullptr);

        struct ggml_tensor * logs_out = ggml_cont(gctx, out.logs);
        ASSERT_NE(logs_out, nullptr);
        ggml_set_output(logs_out);
        ggml_build_forward_expand(gf, logs_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ASSERT_NE(alloc, nullptr);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        std::vector<float> ssl_input(static_cast<size_t>(kSslIn * kSslTime));
        std::vector<int32_t> text_input = make_text_tokens();
        std::vector<float> ge_input(static_cast<size_t>(kGeDim));
        fill_ssl(ssl_input);
        fill_ge(ge_input);

        ggml_backend_tensor_set(ssl, ssl_input.data(), 0, ssl_input.size() * sizeof(float));
        ggml_backend_tensor_set(text, text_input.data(), 0, text_input.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ge, ge_input.data(), 0, ge_input.size() * sizeof(float));

        ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

        const size_t out_nbytes = ggml_nbytes(logs_out);
        std::vector<float> output(out_nbytes / sizeof(float));
        ggml_backend_tensor_get(logs_out, output.data(), 0, out_nbytes);
        for (float value : output) {
            EXPECT_TRUE(std::isfinite(value));
        }

        ggml_gallocr_free(alloc);
        gpt_sovits::sovits_text_encoder_model_free(model);
        ggml_backend_free(backend);
    };

    run_quantized(kModelQ8);
}

TEST(SoVITSTextEncoder, QuantizedQ5RunsInference) {
    auto run_quantized = [&](const std::string & path) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_text_encoder_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(path, model, backend));

        GraphContext gctx(kMaxNodes);
        ASSERT_NE(gctx.ctx, nullptr);

        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
        ASSERT_NE(gf, nullptr);

        struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kSslIn, kSslTime);
        ggml_set_input(ssl);
        struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kTextTime);
        ggml_set_input(text);
        struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeDim, 1);
        ggml_set_input(ge);

        const gpt_sovits::sovits_text_encoder_result out =
            gpt_sovits::sovits_text_encoder_block_forward(gctx, ssl, text, ge, model.weights);
        ASSERT_NE(out.logs, nullptr);

        struct ggml_tensor * logs_out = ggml_cont(gctx, out.logs);
        ASSERT_NE(logs_out, nullptr);
        ggml_set_output(logs_out);
        ggml_build_forward_expand(gf, logs_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ASSERT_NE(alloc, nullptr);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        std::vector<float> ssl_input(static_cast<size_t>(kSslIn * kSslTime));
        std::vector<int32_t> text_input = make_text_tokens();
        std::vector<float> ge_input(static_cast<size_t>(kGeDim));
        fill_ssl(ssl_input);
        fill_ge(ge_input);

        ggml_backend_tensor_set(ssl, ssl_input.data(), 0, ssl_input.size() * sizeof(float));
        ggml_backend_tensor_set(text, text_input.data(), 0, text_input.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ge, ge_input.data(), 0, ge_input.size() * sizeof(float));

        ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

        const size_t out_nbytes = ggml_nbytes(logs_out);
        std::vector<float> output(out_nbytes / sizeof(float));
        ggml_backend_tensor_get(logs_out, output.data(), 0, out_nbytes);
        for (float value : output) {
            EXPECT_TRUE(std::isfinite(value));
        }

        ggml_gallocr_free(alloc);
        gpt_sovits::sovits_text_encoder_model_free(model);
        ggml_backend_free(backend);
    };

    run_quantized(kModelQ5);
}

TEST(SoVITSTextEncoder, QuantizedQ4RunsInference) {
    auto run_quantized = [&](const std::string & path) {
        ASSERT_MODEL_EXISTS(path);

        ggml_backend_t backend = create_test_backend();
        ASSERT_NE(backend, nullptr);

        gpt_sovits::sovits_text_encoder_model model{};
        ASSERT_TRUE(gpt_sovits::sovits_text_encoder_model_load(path, model, backend));

        GraphContext gctx(kMaxNodes);
        ASSERT_NE(gctx.ctx, nullptr);

        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
        ASSERT_NE(gf, nullptr);

        struct ggml_tensor * ssl = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kSslIn, kSslTime);
        ggml_set_input(ssl);
        struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kTextTime);
        ggml_set_input(text);
        struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGeDim, 1);
        ggml_set_input(ge);

        const gpt_sovits::sovits_text_encoder_result out =
            gpt_sovits::sovits_text_encoder_block_forward(gctx, ssl, text, ge, model.weights);
        ASSERT_NE(out.logs, nullptr);

        struct ggml_tensor * logs_out = ggml_cont(gctx, out.logs);
        ASSERT_NE(logs_out, nullptr);
        ggml_set_output(logs_out);
        ggml_build_forward_expand(gf, logs_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ASSERT_NE(alloc, nullptr);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        std::vector<float> ssl_input(static_cast<size_t>(kSslIn * kSslTime));
        std::vector<int32_t> text_input = make_text_tokens();
        std::vector<float> ge_input(static_cast<size_t>(kGeDim));
        fill_ssl(ssl_input);
        fill_ge(ge_input);

        ggml_backend_tensor_set(ssl, ssl_input.data(), 0, ssl_input.size() * sizeof(float));
        ggml_backend_tensor_set(text, text_input.data(), 0, text_input.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ge, ge_input.data(), 0, ge_input.size() * sizeof(float));

        ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

        const size_t out_nbytes = ggml_nbytes(logs_out);
        std::vector<float> output(out_nbytes / sizeof(float));
        ggml_backend_tensor_get(logs_out, output.data(), 0, out_nbytes);
        for (float value : output) {
            EXPECT_TRUE(std::isfinite(value));
        }

        ggml_gallocr_free(alloc);
        gpt_sovits::sovits_text_encoder_model_free(model);
        ggml_backend_free(backend);
    };

    run_quantized(kModelQ4);
}
