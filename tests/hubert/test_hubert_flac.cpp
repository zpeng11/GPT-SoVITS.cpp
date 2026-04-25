// tests/hubert/test_hubert_flac.cpp
//
// FLAC-based end-to-end HuBERT inference test.
//
// Decodes a real FLAC audio file using dr_flac, resamples to 16kHz,
// normalizes, runs the full HuBERT model, and compares the output
// against Python-generated reference across all quantization variants
// (f32, f16, q8, q5, q4).

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"

#include <gtest/gtest.h>

#include "gpt_sovits/hubert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Paths & model variants
// ---------------------------------------------------------------------------
static const std::string kTestDir = HUBERT_TEST_DIR;

struct ModelVariant {
    std::string name;
    std::string path;
    double      max_abs_tol;
    double      rmse_tol;
};

static const std::vector<ModelVariant> kModelVariants = {
    {"F32", kTestDir + "models/chinese-hubert-base-f32.gguf", 2e-2, 1e-3},
    {"F16", kTestDir + "models/chinese-hubert-base-f16.gguf", 2e-2, 2e-3},
    {"Q8",  kTestDir + "models/chinese-hubert-base-q8.gguf",  1e-1, 1e-2},
    {"Q5",  kTestDir + "models/chinese-hubert-base-q5.gguf",  2.5e-1, 3e-2},
    {"Q4",  kTestDir + "models/chinese-hubert-base-q4.gguf",  5e-1, 8e-2},
};

static constexpr size_t kMaxNodes = 8192;
static constexpr int kTargetSampleRate = 16000;

// ---------------------------------------------------------------------------
// Helpers (same pattern as test_hubert_parity.cpp)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// FLAC decode + resample + normalize
// ---------------------------------------------------------------------------

// Decode FLAC file to f32 mono. Returns empty on failure.
std::vector<float> decode_flac(const std::string & path,
                                int & out_sample_rate)
{
    drflac * flac = drflac_open_file(path.c_str(), nullptr);
    if (!flac) {
        ADD_FAILURE() << "drflac_open_file failed: " << path;
        return {};
    }

    out_sample_rate = static_cast<int>(flac->sampleRate);
    auto total_frames = static_cast<size_t>(flac->totalPCMFrameCount);
    int channels = flac->channels;

    // Decode all frames (interleaved if multi-channel)
    std::vector<float> pcm(total_frames * channels);
    drflac_uint64 read = drflac_read_pcm_frames_f32(
        flac, total_frames, pcm.data());
    drflac_close(flac);

    if (read != total_frames) {
        ADD_FAILURE() << "FLAC decode mismatch: expected " << total_frames
                      << " frames, got " << read;
        return {};
    }

    // Convert to mono by averaging channels
    if (channels > 1) {
        std::vector<float> mono(total_frames);
        for (size_t i = 0; i < total_frames; i++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                sum += pcm[i * channels + c];
            }
            mono[i] = sum / static_cast<float>(channels);
        }
        return mono;
    }

    return pcm;
}

// Simple resampling by integer-ratio decimation (take every ratio-th sample).
// For 48kHz -> 16kHz the ratio is 3.
std::vector<float> resample_decimate(const std::vector<float> & audio,
                                      int src_rate, int dst_rate)
{
    if (src_rate == dst_rate) return audio;

    int ratio = src_rate / dst_rate;
    if (src_rate != dst_rate * ratio) {
        ADD_FAILURE() << "Cannot decimate " << src_rate << " -> " << dst_rate
                      << " (non-integer ratio)";
        return {};
    }

    std::vector<float> out;
    out.reserve(audio.size() / ratio);
    for (size_t i = 0; i < audio.size(); i += static_cast<size_t>(ratio)) {
        out.push_back(audio[i]);
    }
    return out;
}

// Normalize: zero mean, unit variance (matching Wav2Vec2FeatureExtractor).
void normalize(std::vector<float> & audio) {
    double mean = 0.0;
    for (float s : audio) mean += static_cast<double>(s);
    mean /= static_cast<double>(audio.size());

    double var = 0.0;
    for (float s : audio) {
        double d = static_cast<double>(s) - mean;
        var += d * d;
    }
    var /= static_cast<double>(audio.size());

    float inv_std = static_cast<float>(1.0 / std::sqrt(var + 1e-7));
    float mean_f  = static_cast<float>(mean);
    for (float & s : audio) {
        s = (s - mean_f) * inv_std;
    }
}

}  // anonymous namespace

// ===========================================================================
// Parameterized fixture
// ===========================================================================

class HubertFlacParity : public ::testing::TestWithParam<ModelVariant> {
protected:
    void SetUp() override {
        backend_ = create_test_backend();
        if (!backend_) {
            GTEST_SKIP() << "Could not init CPU backend";
        }
        const auto & v = GetParam();
        if (!gpt_sovits::hubert_model_load(v.path, model_, backend_)) {
            GTEST_SKIP() << "Could not load " << v.path;
        }
    }
    void TearDown() override {
        gpt_sovits::hubert_model_free(model_);
        if (backend_) {
            ggml_backend_free(backend_);
        }
    }

    ggml_backend_t          backend_ = nullptr;
    gpt_sovits::hubert_model model_{};
};

// ---------------------------------------------------------------------------
// Full-model inference from FLAC
// ---------------------------------------------------------------------------

TEST_P(HubertFlacParity, FullModelFromFlac) {
    const auto & variant = GetParam();

    // Load Python reference
    auto ref = load_f32_bin(kTestDir + "ref_flac_model_output.bin");
    ASSERT_FALSE(ref.empty()) << "Missing ref_flac_model_output.bin — "
                                 "run generate_flac_reference.py first";

    // Decode FLAC
    int sample_rate = 0;
    auto pcm = decode_flac(kTestDir + "Narsil.asr_dummy.4.flac", sample_rate);
    ASSERT_FALSE(pcm.empty());
    ASSERT_EQ(sample_rate, 48000) << "Expected 48kHz FLAC";

    // Resample 48kHz -> 16kHz
    auto audio = resample_decimate(pcm, sample_rate, kTargetSampleRate);
    ASSERT_FALSE(audio.empty());

    // Normalize
    normalize(audio);

    // Build and run ggml graph
    GraphContext gctx(kMaxNodes);
    ASSERT_NE(gctx.ctx, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);

    struct ggml_tensor * inp = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                                   static_cast<int64_t>(audio.size()));
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    struct ggml_tensor * out =
        gpt_sovits::hubert_model_block_forward(gctx, inp, model_.weights);
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    auto result = eval_graph(model_.backend, gf,
                             "input", audio.data(), audio.size(),
                             "output");
    ASSERT_EQ(result.size(), ref.size())
        << "output size " << result.size() << " != ref size " << ref.size();

    auto err = compute_errors(result, ref);
    printf("  [%s] FLAC Full Model : max_abs=%.4e  rmse=%.4e  mean_abs=%.4e\n",
           variant.name.c_str(), err.max_abs, err.rmse, err.mean_abs);

    EXPECT_LT(err.max_abs, variant.max_abs_tol)
        << variant.name << " max abs error too large for FLAC full model";
    EXPECT_LT(err.rmse, variant.rmse_tol)
        << variant.name << " RMSE too large for FLAC full model";
}

INSTANTIATE_TEST_SUITE_P(
    HubertFlacParity,
    HubertFlacParity,
    ::testing::ValuesIn(kModelVariants),
    [](const ::testing::TestParamInfo<ModelVariant> & info) {
        return info.param.name;
    });
