// tests/sovits/test_sovits_session.cpp
//
// End-to-end smoke test for the SoVITS v2 session pipeline:
//   - loads the unified sovits GGUF (all 5 blocks) via sovits_model_load
//   - verifies hparams.semantic_frame_rate_25hz is read correctly
//   - runs sovits_session_compute_ge + sovits_session_forward
//   - checks output shape (T_ssl * 640), finiteness, and non-trivial range
//
// This test does not compare against PyTorch reference values — that would
// require matching RNG state across python `torch.randn_like` and our mt19937.
// Helper-level parity (m+randn*exp(logs) and the doubling) is covered
// separately in test_sovits_helpers.cpp. Block-level parity for each sub-model
// is covered by the existing test_sovits_*.cpp suite.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "cnpy.h"
#include "npy_loader.h"
#include "test_backend.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kModelDir = kTestDir + "models/";

// Unified GGUF (all 5 blocks in one file, disjoint tensor-name prefixes).
static const std::string kModelPath = kModelDir + "v2-sovits-f16.gguf";

static const std::string kRefDir       = kTestDir + "ref/";
static const std::string kReferNpy     = kRefDir + "v2_ref_enc_input.npy";      // BCT, may be >704 channels
static const std::string kCodesNpy     = kRefDir + "v2_quantizer_codes.npy";    // {1, 1, T_codes}
static const std::string kTextNpy      = kRefDir + "v2_enc_p_input_text.npy";   // {T_text}

static constexpr int64_t kMelChannels       = 704;
static constexpr int64_t kGeneratorFrameMul = 640;
static constexpr int64_t kExpectedGEChans   = 512;

// SoVITS v2 output sample rate (matches hps.data.sampling_rate in
// GPT_SoVITS/configs/s2.json).
static constexpr uint32_t kSampleRate = 32000;

// Path where the smoke test dumps its wav output for later inspection.
// Resolved under the test source directory so it's stable across runs
// regardless of cwd.
static const std::string kWavOutDir = kTestDir + "out/";
static const std::string kWavOutPath = kWavOutDir + "sovits_v2_forward_smoke.wav";

// Write a mono 32-bit float WAV file. Returns true on success.
// Minimal IEEE-float WAV (WAVE_FORMAT_IEEE_FLOAT = 3, 16-byte fmt chunk).
static bool write_wav_f32(const std::string & path,
                          const float * samples,
                          size_t n_samples,
                          uint32_t sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;

    const uint16_t n_chan        = 1;
    const uint16_t bits_per_samp = 32;
    const uint32_t byte_rate     = sample_rate * n_chan * (bits_per_samp / 8);
    const uint16_t block_align   = n_chan * (bits_per_samp / 8);
    const uint32_t data_size     = static_cast<uint32_t>(n_samples * sizeof(float));

    #pragma pack(push, 1)
    struct WavHeader {
        char     riff[4];        // "RIFF"
        uint32_t riff_size;      // 36 + data_size
        char     wave[4];        // "WAVE"
        char     fmt_[4];        // "fmt "
        uint32_t fmt_size;       // 16
        uint16_t audio_fmt;      // 3 = IEEE float
        uint16_t n_chan;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_samp;
        char     data[4];        // "data"
        uint32_t data_size;
    } header;
    #pragma pack(pop)

    static_assert(sizeof(header) == 44, "WAV header must be 44 bytes");

    memcpy(header.riff,       "RIFF",  4);
    header.riff_size      = 36 + data_size;
    memcpy(header.wave,       "WAVE",  4);
    memcpy(header.fmt_,       "fmt ",  4);
    header.fmt_size       = 16;
    header.audio_fmt      = 3;
    header.n_chan         = n_chan;
    header.sample_rate    = sample_rate;
    header.byte_rate      = byte_rate;
    header.block_align    = block_align;
    header.bits_per_samp  = bits_per_samp;
    memcpy(header.data,      "data",  4);
    header.data_size      = data_size;

    bool ok = (fwrite(&header, sizeof(header), 1, f) == 1)
           && (fwrite(samples, sizeof(float), n_samples, f) == n_samples);
    fclose(f);
    return ok;
}

#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

#define ASSERT_NPY_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Reference npy not found: " << path; \
    fclose(f); \
} while (0)

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
    if (arr.word_size == sizeof(int32_t)) {
        const int32_t * src = arr.data<int32_t>();
        return std::vector<int32_t>(src, src + arr.num_vals);
    }
    if (arr.word_size == sizeof(int64_t)) {
        const int64_t * src = arr.data<int64_t>();
        std::vector<int32_t> out(arr.num_vals);
        for (size_t i = 0; i < arr.num_vals; ++i) {
            out[i] = static_cast<int32_t>(src[i]);
        }
        return out;
    }
    ADD_FAILURE() << "unsupported word_size=" << arr.word_size << " in '" << path << "'";
    return {};
}

// Build a backend for the smoke test.
//
// The combined SoVITS pipeline triggers a Metal shader-compile hang on
// Apple M4 (the conv-transpose + conv1d + flash-attn op mix in a single
// graph deadlocks the Metal pipeline compiler). Tests cover correctness
// via the block-level parity tests; this smoke test only needs to verify
// the session glue, so we run it on the CPU backend to sidestep the
// Metal issue. Production callers using Metal will see the same hang
// until ggml-metal is fixed upstream.
static ggml_backend_t create_smoke_backend() {
    return ggml_backend_cpu_init();
}

// Repack a 3D BCT numpy array into ggml {kMelChannels, T} row-major,
// slicing channel axis to [:704] (the v2 MelStyleEncoder input slice).
// PyTorch saves refer in BCT layout: [b][c][t] = b*(C*T) + c*T + t.
// ggml {C, T} expects [t][c] = t*C + c (column-major equivalent).
static std::vector<float> repack_bct_to_ggml_ct_sliced(
    const std::vector<float> & bct,
    const std::vector<size_t> & shape)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_GE(shape[1], static_cast<size_t>(kMelChannels));
    if (shape.size() != 3 || shape[0] != 1u ||
        shape[1] < static_cast<size_t>(kMelChannels)) {
        return {};
    }

    const size_t full_channels = shape[1];
    const size_t T = shape[2];
    std::vector<float> packed(static_cast<size_t>(kMelChannels) * T);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(kMelChannels); ++c) {
            packed[t * static_cast<size_t>(kMelChannels) + c] =
                bct[c * T + t];
        }
    }
    return packed;
}

} // namespace

// ---------------------------------------------------------------------------
// Unified loader
// ---------------------------------------------------------------------------

TEST(SoVITSModel, LoadsUnifiedModel) {
    ASSERT_MODEL_EXISTS(kModelPath);

    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(kModelPath, model, backend));

    // V2 pretrained checkpoint is 25hz.
    EXPECT_TRUE(model.hparams.semantic_frame_rate_25hz);

    EXPECT_NE(model.ref_enc.spectral_1_w, nullptr);
    EXPECT_NE(model.quantizer.codebook, nullptr);
    EXPECT_NE(model.text_encoder.post.proj_w, nullptr);
    EXPECT_NE(model.flow.layers[0].pre_w, nullptr);
    EXPECT_NE(model.generator.conv_pre.w, nullptr);

    gpt_sovits::sovits_model_free(model);
    ggml_backend_free(backend);
}

TEST(SoVITSModel, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_model model{};
    EXPECT_FALSE(gpt_sovits::sovits_model_load("/nonexistent/sovits.gguf", model, backend));

    gpt_sovits::sovits_model_free(model);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

TEST(SoVITSSession, InitAndFreeAreSafe) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, backend, 0.5f, 12345u));
    EXPECT_EQ(session.noise_scale, 0.5f);
    EXPECT_EQ(session.rng_seed, 12345u);
    EXPECT_EQ(session.ge, nullptr);

    gpt_sovits::sovits_session_free(session);
    EXPECT_EQ(session.ge, nullptr);
    EXPECT_EQ(session.backend, nullptr);

    ggml_backend_free(backend);
}

TEST(SoVITSSession, RejectsInvalidNoiseScale) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_session session{};
    EXPECT_FALSE(gpt_sovits::sovits_session_init(session, backend, 0.0f, 0u));
    EXPECT_FALSE(gpt_sovits::sovits_session_init(session, backend, -1.0f, 0u));

    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Full pipeline end-to-end
// ---------------------------------------------------------------------------

TEST(SoVITSSession, ForwardProducesWaveformOfCorrectShape) {
    ASSERT_MODEL_EXISTS(kModelPath);
    ASSERT_NPY_EXISTS(kReferNpy);
    ASSERT_NPY_EXISTS(kCodesNpy);
    ASSERT_NPY_EXISTS(kTextNpy);

    // CPU backend: see create_smoke_backend() comment.
    ggml_backend_t backend = create_smoke_backend();
    ASSERT_NE(backend, nullptr);

    // Load the unified model (all 5 blocks).
    gpt_sovits::sovits_model model{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(kModelPath, model, backend));
    ASSERT_TRUE(model.hparams.semantic_frame_rate_25hz);

    // Load refer fixture and repack to ggml {704, T_refer} (slice to [:704]).
    const NpyShapeInfo refer_bct = load_npy_with_shape(kReferNpy);
    ASSERT_FALSE(refer_bct.data.empty());
    ASSERT_EQ(refer_bct.shape.size(), 3u);
    ASSERT_EQ(refer_bct.shape[0], 1u);
    ASSERT_GE(refer_bct.shape[1], static_cast<size_t>(kMelChannels));
    const int64_t T_refer = static_cast<int64_t>(refer_bct.shape[2]);
    ASSERT_GT(T_refer, 0);
    const std::vector<float> refer_ggml =
        repack_bct_to_ggml_ct_sliced(refer_bct.data, refer_bct.shape);
    ASSERT_EQ(refer_ggml.size(), static_cast<size_t>(kMelChannels * T_refer));

    // Load codes fixture: {1, 1, T_codes}.
    std::vector<int32_t> codes = load_npy_as_i32(kCodesNpy);
    const int64_t T_codes = static_cast<int64_t>(codes.size());
    ASSERT_GT(T_codes, 0);

    // Load text fixture: {T_text}.
    std::vector<int32_t> text = load_npy_as_i32(kTextNpy);
    const int64_t T_text = static_cast<int64_t>(text.size());
    ASSERT_GT(T_text, 0);

    // Init session.
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, backend, 0.5f, 42u));

    // Compute ge from refer (single-slice list mirrors the previous path).
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        session, model, {{refer_ggml.data(), T_refer}}));
    struct ggml_tensor * ge = gpt_sovits::sovits_session_get_ge(session);
    ASSERT_NE(ge, nullptr);
    EXPECT_EQ(ge->ne[0], kExpectedGEChans);
    EXPECT_EQ(ge->ne[1], 1);

    // Run forward.
    const int64_t T_ssl  = 2 * T_codes;   // 25hz doubling
    const int64_t wav_len = T_ssl * kGeneratorFrameMul;
    std::vector<float> wav(static_cast<size_t>(wav_len), 0.0f);
    ASSERT_TRUE(gpt_sovits::sovits_session_forward(
        session, model,
        codes.data(), T_codes,
        text.data(),  T_text,
        wav.data(),   wav_len));

    // Output checks: shape is implicit in the API contract; verify content.
    size_t n_finite = 0;
    float  vmin = +INFINITY;
    float  vmax = -INFINITY;
    for (float v : wav) {
        if (std::isfinite(v)) {
            ++n_finite;
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
    EXPECT_EQ(n_finite, static_cast<size_t>(wav_len))
        << "wav contains non-finite values";
    // tanh output is in [-1, 1]; a non-degenerate run should span a
    // non-trivial sub-range.
    EXPECT_LT(vmin, 0.0f);
    EXPECT_GT(vmax, 0.0f);
    EXPECT_GT(vmax - vmin, 0.1f)
        << "output range [" << vmin << ", " << vmax << "] suspiciously small";

    // Dump the wav for offline inspection (listen / plot / compare).
    std::filesystem::create_directories(kWavOutDir);
    ASSERT_TRUE(write_wav_f32(kWavOutPath, wav.data(), wav.size(), kSampleRate))
        << "failed to write wav to " << kWavOutPath;
    printf("SoVITSSession smoke wav written: %s\n"
           "  samples=%zu  sample_rate=%u  duration=%.3fs\n"
           "  range=[%g, %g]\n",
           kWavOutPath.c_str(),
           wav.size(), kSampleRate,
           static_cast<double>(wav.size()) / kSampleRate,
           static_cast<double>(vmin), static_cast<double>(vmax));

    gpt_sovits::sovits_session_free(session);
    gpt_sovits::sovits_model_free(model);
    ggml_backend_free(backend);
}
