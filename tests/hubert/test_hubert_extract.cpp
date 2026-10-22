// tests/hubert/test_hubert_extract.cpp
//
// Tests for the host-side normalized-input HuBERT feature extraction runner
// (hubert_extract_features_from_normalized_pcm) and the output-frame formula.
//
// The parity reference files are produced by tests/hubert/generate_reference.py
// and stored as raw f32 little-endian binaries whose memory layout matches
// ggml convention (ne[0] is the contiguous innermost dimension).

#include <gtest/gtest.h>

#include "gpt_sovits/hubert.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir = HUBERT_TEST_DIR;

// Same model and tolerances as the F32 FullModel case in
// test_hubert_parity.cpp.
static const std::string kModelPath = kTestDir + "models/chinese-hubert-base-f32.gguf";
static constexpr double kMaxAbsTol = 1e-2;
static constexpr double kRmseTol   = 1e-3;

static constexpr int64_t kHiddenSize = 768;

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

}  // anonymous namespace

// ===========================================================================
// Output-frame formula
// ===========================================================================

TEST(HubertOutputFrames, EdgeCases) {
    EXPECT_EQ(gpt_sovits::hubert_output_frames(0),   0);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(10),  0);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(399), 0);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(400), 1);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(401), 1);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(719), 1);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(720), 2);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(16000), 49);
}

// ===========================================================================
// Argument validation (no model weights needed: these all fail before the
// graph is built).
// ===========================================================================

class HubertExtractArgs : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = create_test_backend();
        if (!backend_) {
            GTEST_SKIP() << "Could not init CPU backend";
        }
        // Only the backend is needed for validation; weights stay null.
        model_.backend = backend_;
    }
    void TearDown() override {
        if (backend_) {
            ggml_backend_free(backend_);
        }
    }

    ggml_backend_t          backend_ = nullptr;
    gpt_sovits::hubert_model model_{};
};

TEST_F(HubertExtractArgs, NullPcmReturnsFalse) {
    std::vector<float> out(2 * kHiddenSize, 0.0f);
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, nullptr, 720, out.data(), static_cast<int64_t>(out.size())));
}

TEST_F(HubertExtractArgs, NullOutReturnsFalse) {
    std::vector<float> pcm(720, 0.0f);
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, pcm.data(), static_cast<int64_t>(pcm.size()), nullptr,
        static_cast<int64_t>(2 * kHiddenSize)));
}

TEST_F(HubertExtractArgs, NonPositiveSamplesReturnsFalse) {
    std::vector<float> pcm(720, 0.0f);
    std::vector<float> out(2 * kHiddenSize, 0.0f);
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, pcm.data(), 0, out.data(), static_cast<int64_t>(out.size())));
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, pcm.data(), -1, out.data(), static_cast<int64_t>(out.size())));
}

TEST_F(HubertExtractArgs, OutCapTooSmallReturnsFalse) {
    // 720 samples -> T' = 2 -> 1536 floats required.
    std::vector<float> pcm(720, 0.0f);
    std::vector<float> out(2 * kHiddenSize, 0.0f);
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, pcm.data(), static_cast<int64_t>(pcm.size()), out.data(),
        static_cast<int64_t>(out.size()) - 1));

    // 400 samples -> T' = 1 -> 768 floats required.
    std::vector<float> pcm400(400, 0.0f);
    std::vector<float> out400(kHiddenSize, 0.0f);
    EXPECT_FALSE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, pcm400.data(), static_cast<int64_t>(pcm400.size()),
        out400.data(), static_cast<int64_t>(out400.size()) - 1));
}

// ===========================================================================
// End-to-end parity against the Python reference
// ===========================================================================

class HubertExtract : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = create_test_backend();
        if (!backend_) {
            GTEST_SKIP() << "Could not init CPU backend";
        }
        if (!gpt_sovits::hubert_model_load(kModelPath, model_, backend_)) {
            GTEST_SKIP() << "Could not load " << kModelPath;
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

TEST_F(HubertExtract, MatchesReference) {
    auto input = load_f32_bin(kTestDir + "ref_input.bin");
    auto ref   = load_f32_bin(kTestDir + "ref_model_output.bin");
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(ref.empty());

    const int64_t n_samples = static_cast<int64_t>(input.size());
    const int64_t T         = gpt_sovits::hubert_output_frames(n_samples);
    ASSERT_GT(T, 0);
    ASSERT_EQ(ref.size(), static_cast<size_t>(kHiddenSize * T))
        << "reference shape does not match the frame formula";

    std::vector<float> out(static_cast<size_t>(kHiddenSize * T), 0.0f);
    int64_t frames = -1;

    ASSERT_TRUE(gpt_sovits::hubert_extract_features_from_normalized_pcm(
        model_, input.data(), n_samples, out.data(),
        static_cast<int64_t>(out.size()), &frames));
    EXPECT_EQ(frames, T);

    auto err = compute_errors(out, ref);
    printf("  [Extract] Full Model : max_abs=%.4e  rmse=%.4e  mean_abs=%.4e\n",
           err.max_abs, err.rmse, err.mean_abs);

    EXPECT_LT(err.max_abs, kMaxAbsTol) << "max abs error too large";
    EXPECT_LT(err.rmse,    kRmseTol)   << "RMSE too large";
}
