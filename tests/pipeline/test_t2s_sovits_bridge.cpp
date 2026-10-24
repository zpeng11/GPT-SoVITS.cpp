#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"
#include "gpt_sovits/t2s.h"

#include "cnpy.h"
#include "npy_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

const std::string kT2SModelPath =
    std::string(T2S_TEST_DIR) + "models/s1v3-s2Gv2Pro-f16.gguf";
const std::string kSoVITSModelPath =
    std::string(SOVITS_TEST_DIR) + "models/v2pro-sovits-f16.gguf";
const std::string kReferPath = std::string(SOVITS_TEST_DIR) + "ref/v2pro_refer.npy";
const std::string kSVPath = std::string(SOVITS_TEST_DIR) + "ref/v2pro_sv_emb.npy";

constexpr int64_t kCodeDim = 768;
constexpr int64_t kCodeCount = 1024;
constexpr int64_t kSpecChannels = 704;
constexpr int64_t kSVSize = 20480;
constexpr int64_t kFrameSamples = 640;

class T2SSoVITSBridge : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(std::filesystem::is_regular_file(kReferPath)) << kReferPath;
        ASSERT_TRUE(std::filesystem::is_regular_file(kSVPath)) << kSVPath;
        ASSERT_NO_FATAL_FAILURE(LoadReference());

        std::string missing_models;
        for (const auto & path : {kT2SModelPath, kSoVITSModelPath}) {
            if (!std::filesystem::is_regular_file(path)) {
                missing_models += "\n  " + path;
            }
        }
        if (!missing_models.empty()) {
            GTEST_SKIP() << "Missing local GGUF models:" << missing_models
                         << "\nSee tests/pipeline/README.md for conversion commands.";
        }

        // The pinned Metal CONV_TRANSPOSE_1D kernel can hang on the full decoder.
        backend = ggml_backend_cpu_init();
        ASSERT_NE(backend, nullptr);
        ASSERT_TRUE(gpt_sovits::t2s_model_load(kT2SModelPath, t2s_model, backend));
        ASSERT_TRUE(gpt_sovits::sovits_model_load(kSoVITSModelPath, sovits_model, backend));
    }

    void TearDown() override {
        gpt_sovits::sovits_session_free(sovits_session);
        gpt_sovits::t2s_session_free(t2s_session);
        gpt_sovits::sovits_model_free(sovits_model);
        gpt_sovits::t2s_model_free(t2s_model);
        if (backend) ggml_backend_free(backend);
    }

    void AssertMatchedCodebooks() {
        ASSERT_TRUE(sovits_model.hparams.is_v2pro);
        ASSERT_TRUE(sovits_model.hparams.semantic_frame_rate_25hz);
        ASSERT_EQ(sovits_model.hparams.gin_channels, 1024);
        ASSERT_NE(sovits_model.generator.conv_pre.w, nullptr);
        ASSERT_EQ(sovits_model.generator.conv_pre.w->ne[1], 512);

        ggml_tensor * t2s_book = t2s_model.weights.extract_latent.codebook;
        ggml_tensor * sovits_book = sovits_model.quantizer.codebook;
        ASSERT_NE(t2s_book, nullptr);
        ASSERT_NE(sovits_book, nullptr);
        ASSERT_EQ(t2s_book->type, GGML_TYPE_F32);
        ASSERT_EQ(sovits_book->type, GGML_TYPE_F16);
        for (const auto * book : {t2s_book, sovits_book}) {
            ASSERT_EQ(book->ne[0], kCodeDim);
            ASSERT_EQ(book->ne[1], kCodeCount);
            ASSERT_EQ(ggml_nelements(book), kCodeDim * kCodeCount);
            ASSERT_TRUE(ggml_is_contiguous(book));
        }

        const size_t n = static_cast<size_t>(kCodeDim * kCodeCount);
        std::vector<float> t2s_values(n);
        std::vector<ggml_fp16_t> expected(n), actual(n);
        ggml_backend_tensor_get(t2s_book, t2s_values.data(), 0, n * sizeof(float));
        ggml_backend_tensor_get(sovits_book, actual.data(), 0, n * sizeof(ggml_fp16_t));
        ASSERT_TRUE(std::all_of(t2s_values.begin(), t2s_values.end(),
                               [](float v) { return std::isfinite(v); }));
        ASSERT_TRUE(std::all_of(actual.begin(), actual.end(), [](ggml_fp16_t v) {
            return std::isfinite(ggml_fp16_to_fp32(v));
        }));
        for (size_t i = 0; i < n; ++i) {
            expected[i] = ggml_fp32_to_fp16(t2s_values[i]);
        }
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
        if (mismatch.first != expected.end()) {
            const size_t i = static_cast<size_t>(mismatch.first - expected.begin());
            FAIL() << "Codebooks differ at channel " << i % kCodeDim
                   << ", code " << i / kCodeDim
                   << ": T2S rounded to F16=" << ggml_fp16_to_fp32(expected[i])
                   << ", SoVITS=" << ggml_fp16_to_fp32(actual[i]);
        }
    }

    void LoadReference() {
        const auto refer = cnpy::npy_load(kReferPath);
        ASSERT_EQ(refer.shape, (std::vector<size_t>{1, 1025, 243}));
        ASSERT_EQ(refer.word_size, sizeof(ggml_fp16_t));
        ASSERT_FALSE(refer.fortran_order);
        const auto bct = load_npy_as_f32(kReferPath);
        ASSERT_EQ(bct.size(), refer.num_vals);
        ASSERT_TRUE(std::all_of(bct.begin(), bct.end(),
                               [](float v) { return std::isfinite(v); }));
        refer_frames = static_cast<int64_t>(refer.shape[2]);
        refer_spec.resize(static_cast<size_t>(kSpecChannels * refer_frames));
        for (int64_t t = 0; t < refer_frames; ++t) {
            for (int64_t c = 0; c < kSpecChannels; ++c) {
                refer_spec[static_cast<size_t>(t * kSpecChannels + c)] =
                    bct[static_cast<size_t>(c * refer_frames + t)];
            }
        }

        const auto sv = cnpy::npy_load(kSVPath);
        ASSERT_EQ(sv.shape, (std::vector<size_t>{1, static_cast<size_t>(kSVSize)}));
        ASSERT_EQ(sv.word_size, sizeof(ggml_fp16_t));
        ASSERT_FALSE(sv.fortran_order);
        sv_embedding = load_npy_as_f32(kSVPath);
        ASSERT_EQ(sv_embedding.size(), static_cast<size_t>(kSVSize));
        ASSERT_TRUE(std::all_of(sv_embedding.begin(), sv_embedding.end(),
                               [](float v) { return std::isfinite(v); }));
    }

    ggml_backend_t backend = nullptr;
    gpt_sovits::t2s_model t2s_model{};
    gpt_sovits::sovits_model sovits_model{};
    gpt_sovits::t2s_session t2s_session{};
    gpt_sovits::sovits_session sovits_session{};
    std::vector<float> refer_spec;
    std::vector<float> sv_embedding;
    int64_t refer_frames = 0;
};

TEST_F(T2SSoVITSBridge, CpuF16ProducesWaveform) {
    ASSERT_NO_FATAL_FAILURE(AssertMatchedCodebooks());

    const std::vector<int32_t> ref_tokens = {1, 2, 3};
    const std::vector<float> ref_bert(1024 * ref_tokens.size(), 0.1f);
    const int64_t T_hub = 8;
    const std::vector<float> hubert(static_cast<size_t>(kCodeDim * T_hub), 0.5f);
    const std::vector<int32_t> target_tokens = {7, 8, 9};
    const std::vector<float> target_bert(1024 * target_tokens.size(), 0.1f);

    ASSERT_NE(t2s_model.weights.embed.text_embedding, nullptr);
    ASSERT_NE(sovits_model.text_encoder.text.text_embedding, nullptr);
    for (int32_t id : target_tokens) {
        ASSERT_GE(id, 0);
        ASSERT_LT(id, t2s_model.weights.embed.text_embedding->ne[1]);
        ASSERT_LT(id, sovits_model.text_encoder.text.text_embedding->ne[1]);
    }

    gpt_sovits::t2s_sampler_config sampler;
    sampler.top_k = 1;
    sampler.top_p = 1.0f;
    sampler.temperature = 1.0f;
    sampler.repetition_penalty = 1.0f;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(
        t2s_session, t2s_model.hparams, backend, 1, 64, sampler));
    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        t2s_session, t2s_model, ref_tokens.data(), static_cast<int64_t>(ref_tokens.size()),
        ref_bert.data(), hubert.data(), T_hub));

    gpt_sovits::t2s_generate_config config;
    config.max_new_tokens = 5;
    config.seed = 42;
    gpt_sovits::t2s_generate_result result;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        t2s_session, t2s_model, target_tokens.data(),
        static_cast<int64_t>(target_tokens.size()), target_bert.data(), result, config));

    SCOPED_TRACE(::testing::Message() << "codes=" << result.codes.size()
                 << ", stop_reason=" << static_cast<int>(result.stop_reason));
    ASSERT_FALSE(result.codes.empty()) << "Fixed synthetic target hit immediate EOS";
    ASSERT_LE(result.codes.size(), static_cast<size_t>(config.max_new_tokens));
    for (int32_t code : result.codes) {
        ASSERT_GE(code, 0);
        ASSERT_NE(code, t2s_model.hparams.eos);
        ASSERT_LT(code, sovits_model.quantizer.codebook->ne[1]);
    }
    switch (result.stop_reason) {
        case gpt_sovits::t2s_stop_reason::eos:
            EXPECT_LT(result.codes.size(), static_cast<size_t>(config.max_new_tokens));
            break;
        case gpt_sovits::t2s_stop_reason::max_new_tokens:
            EXPECT_EQ(result.codes.size(), static_cast<size_t>(config.max_new_tokens));
            break;
        default:
            FAIL() << "Unexpected T2S stop reason";
    }

    ASSERT_TRUE(gpt_sovits::sovits_session_init(sovits_session, backend, 0.5f, 7u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        sovits_session, sovits_model,
        {{refer_spec.data(), refer_frames, sv_embedding.data()}}));

    const int64_t T_codes = static_cast<int64_t>(result.codes.size());
    const int64_t T_ssl = sovits_model.hparams.semantic_frame_rate_25hz ? 2 * T_codes : T_codes;
    const size_t wav_len = static_cast<size_t>(T_ssl * kFrameSamples);
    constexpr size_t kGuardSamples = 8;
    constexpr float kSentinel = 12345.0f;
    std::vector<float> wav(wav_len + kGuardSamples, kSentinel);
    std::fill_n(wav.begin(), wav_len, std::numeric_limits<float>::quiet_NaN());
    SCOPED_TRACE(::testing::Message() << "expected waveform samples=" << wav_len);
    ASSERT_TRUE(gpt_sovits::sovits_session_forward(
        sovits_session, sovits_model, result.codes.data(), T_codes,
        target_tokens.data(), static_cast<int64_t>(target_tokens.size()),
        wav.data(), static_cast<int64_t>(wav_len)));

    for (size_t i = wav_len; i < wav.size(); ++i) {
        EXPECT_EQ(wav[i], kSentinel) << "Output guard overwritten at sample " << i;
    }
    double energy = 0.0;
    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < wav_len; ++i) {
        ASSERT_TRUE(std::isfinite(wav[i])) << "Non-finite/unwritten sample " << i;
        energy += static_cast<double>(wav[i]) * wav[i];
        vmin = std::min(vmin, wav[i]);
        vmax = std::max(vmax, wav[i]);
    }
    const double rms = std::sqrt(energy / static_cast<double>(wav_len));
    EXPECT_GT(rms, 1e-8) << "Degenerate waveform, range=[" << vmin << ", " << vmax << "]";
    EXPECT_GT(vmax - vmin, 1e-7f) << "Constant waveform, RMS=" << rms;
    printf("[T2S -> SoVITS CPU] codes=%zu stop=%d samples=%zu range=[%g, %g] rms=%g\n",
           result.codes.size(), static_cast<int>(result.stop_reason), wav_len,
           static_cast<double>(vmin), static_cast<double>(vmax), rms);
}

} // namespace
