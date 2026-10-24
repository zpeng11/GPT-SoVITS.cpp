#include <gtest/gtest.h>

#include "gpt_sovits/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using gpt_sovits::audio_reference_data;
using gpt_sovits::audio_prepare_reference;

std::vector<float> make_signal(int n, int rate) {
    std::vector<float> pcm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / rate;
        pcm[static_cast<size_t>(i)] = static_cast<float>(
            0.2 + (0.3 + 0.1 * std::sin(37.0 * t)) * std::sin(2764.6 * t)
                + 0.1 * std::cos(7756.5 * t + 0.3));
    }
    return pcm;
}

void expect_same_data(const audio_reference_data & actual,
                      const audio_reference_data & expected) {
    EXPECT_EQ(actual.pcm16k, expected.pcm16k);
    EXPECT_EQ(actual.normalized_pcm16k, expected.normalized_pcm16k);
    EXPECT_EQ(actual.spec704, expected.spec704);
    EXPECT_EQ(actual.spec_frames, expected.spec_frames);
}

void expect_failure_preserves_output(const float * pcm, int64_t n, int rate) {
    audio_reference_data out{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, 7};
    const auto saved = out;
    const float * raw_ptr = out.pcm16k.data();
    const float * normalized_ptr = out.normalized_pcm16k.data();
    const float * spec_ptr = out.spec704.data();
    const auto raw_cap = out.pcm16k.capacity();
    const auto normalized_cap = out.normalized_pcm16k.capacity();
    const auto spec_cap = out.spec704.capacity();

    EXPECT_FALSE(audio_prepare_reference(pcm, n, rate, out));
    expect_same_data(out, saved);
    EXPECT_EQ(out.pcm16k.data(), raw_ptr);
    EXPECT_EQ(out.normalized_pcm16k.data(), normalized_ptr);
    EXPECT_EQ(out.spec704.data(), spec_ptr);
    EXPECT_EQ(out.pcm16k.capacity(), raw_cap);
    EXPECT_EQ(out.normalized_pcm16k.capacity(), normalized_cap);
    EXPECT_EQ(out.spec704.capacity(), spec_cap);
}

TEST(AudioReference, ResamplesDirectlyFromSourceWithoutMutatingInput) {
    for (int rate : {16000, 32000, 44100, 48000}) {
        SCOPED_TRACE(rate);
        const auto pcm = make_signal(rate / 10 + 17, rate);
        const auto saved = pcm;
        const int64_t n = static_cast<int64_t>(pcm.size());
        std::vector<float> expected16k;
        ASSERT_TRUE(gpt_sovits::audio_resample(pcm.data(), n, rate, 16000, expected16k));

        audio_reference_data out;
        ASSERT_TRUE(audio_prepare_reference(pcm.data(), n, rate, out));
        EXPECT_EQ(pcm, saved);
        EXPECT_EQ(out.pcm16k.size(), static_cast<size_t>((n * 16000 + rate - 1) / rate));
        EXPECT_EQ(out.pcm16k, expected16k);
        EXPECT_EQ(out.normalized_pcm16k.size(), out.pcm16k.size());
        if (rate == 16000) EXPECT_EQ(out.pcm16k, pcm);
        EXPECT_NE(out.pcm16k.data(), pcm.data());
        EXPECT_NE(out.normalized_pcm16k.data(), out.pcm16k.data());
        for (float value : out.normalized_pcm16k) EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(AudioReference, NormalizesKnownPopulationVarianceWithoutClippingSVInput) {
    std::vector<float> pcm(800);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = (i % 2 == 0) ? 2.0f : 4.0f;
    audio_reference_data out;
    ASSERT_TRUE(audio_prepare_reference(pcm.data(), static_cast<int64_t>(pcm.size()), 16000, out));
    EXPECT_EQ(out.pcm16k, pcm);

    const float inv_std = static_cast<float>(1.0 / std::sqrt(1.0 + 1e-7));
    double mean = 0.0, variance = 0.0;
    for (size_t i = 0; i < pcm.size(); ++i) {
        const float value = out.normalized_pcm16k[i];
        EXPECT_FLOAT_EQ(value, (pcm[i] - 3.0f) * inv_std);
        mean += value;
        variance += static_cast<double>(value) * value;
    }
    EXPECT_NEAR(mean / pcm.size(), 0.0, 1e-7);
    EXPECT_NEAR(variance / pcm.size(), 1.0 / (1.0 + 1e-7), 1e-6);
}

TEST(AudioReference, NormalizationPreservesFloatRoundingOrder) {
    const float pattern[] = {0.1f, -0.3f, 0.7f, 1.25f, -0.015625f, 0.33f, 0.019f};
    // Double mean/variance, then float mean and inverse std before subtract/multiply.
    const uint32_t expected[] = {
        0xbed0ebc5u, 0xbf9df234u, 0x3f54afe5u, 0x3ffbb3eeu,
        0xbf2593d5u, 0x3d88e6eeu, 0xbf134681u,
    };
    constexpr size_t period = sizeof(pattern) / sizeof(pattern[0]);
    std::vector<float> pcm;
    for (int repeat = 0; repeat < 64; ++repeat) {
        pcm.insert(pcm.end(), pattern, pattern + period);
    }
    audio_reference_data out;
    ASSERT_TRUE(audio_prepare_reference(pcm.data(), static_cast<int64_t>(pcm.size()), 16000, out));
    ASSERT_EQ(out.normalized_pcm16k.size(), pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        uint32_t actual;
        std::memcpy(&actual, &out.normalized_pcm16k[i], sizeof(actual));
        EXPECT_EQ(actual, expected[i % period]) << "sample " << i;
    }
}

TEST(AudioReference, NearConstantVarianceUsesEpsilonInsideSquareRoot) {
    constexpr float center = 0.125f;
    constexpr float delta = 1.0f / 4096;
    std::vector<float> pcm(800);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = center + (i % 2 == 0 ? -delta : delta);
    }
    audio_reference_data out;
    ASSERT_TRUE(audio_prepare_reference(pcm.data(), static_cast<int64_t>(pcm.size()), 16000, out));

    const double variance = static_cast<double>(delta) * delta;
    const float inv_std = static_cast<float>(1.0 / std::sqrt(variance + 1e-7));
    double actual_variance = 0.0;
    for (size_t i = 0; i < pcm.size(); ++i) {
        const float value = out.normalized_pcm16k[i];
        EXPECT_FLOAT_EQ(value, (pcm[i] - center) * inv_std);
        actual_variance += static_cast<double>(value) * value;
    }
    EXPECT_NEAR(actual_variance / pcm.size(), variance / (variance + 1e-7), 1e-6);
}

TEST(AudioReference, SilenceAndConstantInputRemainFinite) {
    for (float level : {0.0f, 0.25f}) {
        SCOPED_TRACE(level);
        const std::vector<float> pcm(800, level);
        audio_reference_data out;
        ASSERT_TRUE(audio_prepare_reference(pcm.data(), static_cast<int64_t>(pcm.size()), 16000, out));
        EXPECT_EQ(out.pcm16k, pcm);
        for (float value : out.normalized_pcm16k) EXPECT_EQ(value, 0.0f);
        for (float value : out.spec704) {
            EXPECT_TRUE(std::isfinite(value));
            if (level == 0.0f) EXPECT_FLOAT_EQ(value, 0.001f);
        }
    }
}

TEST(AudioReference, SpectrumMatchesPerFramePrefixOfDirect32kSTFT) {
    for (int rate : {16000, 32000, 44100, 48000}) {
        SCOPED_TRACE(rate);
        const auto pcm = make_signal(rate / 10 + 17, rate);
        const auto saved = pcm;
        const int64_t n = static_cast<int64_t>(pcm.size());
        std::vector<float> pcm32k, full;
        ASSERT_TRUE(gpt_sovits::audio_resample(pcm.data(), n, rate, 32000, pcm32k));
        EXPECT_EQ(pcm32k.size(), static_cast<size_t>((n * 32000 + rate - 1) / rate));
        int64_t expected_frames = 0;
        ASSERT_TRUE(gpt_sovits::audio_stft_magnitude(
            pcm32k.data(), static_cast<int64_t>(pcm32k.size()), 2048, 640, 2048,
            full, &expected_frames));

        audio_reference_data out;
        ASSERT_TRUE(audio_prepare_reference(pcm.data(), n, rate, out));
        EXPECT_EQ(pcm, saved);
        ASSERT_GT(expected_frames, 1);
        ASSERT_EQ(out.spec_frames, expected_frames);
        EXPECT_EQ(out.spec_frames, 1 + (static_cast<int64_t>(pcm32k.size()) + 1408 - 2048) / 640);
        ASSERT_EQ(out.spec704.size(), static_cast<size_t>(expected_frames) * 704);
        for (int64_t t = 0; t < expected_frames; ++t) {
            SCOPED_TRACE(t);
            for (int c = 0; c < 704; ++c) {
                EXPECT_EQ(out.spec704[static_cast<size_t>(t) * 704 + c],
                          full[static_cast<size_t>(t) * 1025 + c]) << "bin " << c;
            }
        }
    }
}

TEST(AudioReference, EnforcesFeatureMinimumAndSTFTFrameBoundaries) {
    for (int n : {1, 352, 399}) {
        const auto pcm = make_signal(n, 16000);
        expect_failure_preserves_output(pcm.data(), n, 16000);
    }
    for (int n : {400, 639, 640, 641, 959, 960, 961}) {
        SCOPED_TRACE(n);
        const auto pcm = make_signal(n, 16000);
        audio_reference_data out;
        ASSERT_TRUE(audio_prepare_reference(pcm.data(), n, 16000, out));
        EXPECT_EQ(out.pcm16k.size(), static_cast<size_t>(n));
        EXPECT_EQ(out.spec_frames, n / 320);
        EXPECT_EQ(out.spec704.size(), static_cast<size_t>(n / 320) * 704);
    }
    for (int n : {798, 799, 800, 1279, 1280, 1281}) {
        SCOPED_TRACE(n);
        const auto pcm = make_signal(n, 32000);
        if (n == 798) {
            expect_failure_preserves_output(pcm.data(), n, 32000);
            continue;
        }
        audio_reference_data out;
        ASSERT_TRUE(audio_prepare_reference(pcm.data(), n, 32000, out));
        EXPECT_EQ(out.pcm16k.size(), static_cast<size_t>((n + 1) / 2));
        EXPECT_EQ(out.spec_frames, n / 640);
    }
}

TEST(AudioReference, InvalidArgumentsPreserveExistingOutput) {
    const auto pcm = make_signal(800, 16000);
    expect_failure_preserves_output(nullptr, 800, 16000);
    expect_failure_preserves_output(pcm.data(), 0, 16000);
    expect_failure_preserves_output(pcm.data(), -1, 16000);
    expect_failure_preserves_output(pcm.data(), 800, 0);
    expect_failure_preserves_output(pcm.data(), 800, -16000);
    expect_failure_preserves_output(pcm.data(), 800, std::numeric_limits<int>::max());
}

TEST(AudioReference, UnsupportedResamplingPreservesExistingOutput) {
    const auto pcm = make_signal(30000, 1048577);
    expect_failure_preserves_output(pcm.data(), static_cast<int64_t>(pcm.size()), 1048577);
}

TEST(AudioReference, RejectsNonFiniteInputAndComputationWithoutClipping) {
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
        auto pcm = make_signal(800, 16000);
        pcm.back() = invalid;
        expect_failure_preserves_output(pcm.data(), static_cast<int64_t>(pcm.size()), 16000);
    }
    const std::vector<float> resample_overflow(800, std::numeric_limits<float>::max());
    expect_failure_preserves_output(resample_overflow.data(), 800, 16000);
    const std::vector<float> stft_overflow(800, std::numeric_limits<float>::max() / 1024);
    expect_failure_preserves_output(stft_overflow.data(), 800, 32000);
}

TEST(AudioReference, RejectsLengthOverflowBeforeReadingInput) {
    // The oversized lengths must be rejected before accessing this one-sample buffer.
    const float sample = 0.1f;
    expect_failure_preserves_output(&sample, std::numeric_limits<int64_t>::max(), 16000);
    expect_failure_preserves_output(&sample, std::numeric_limits<int64_t>::max() / 100, 1);
    const uint64_t max_samples = std::vector<float>().max_size();
    const int64_t spectrum_overflow = static_cast<int64_t>((max_samples / 1025 + 1) * 640);
    expect_failure_preserves_output(&sample, spectrum_overflow, 32000);
}

TEST(AudioReference, ReplacesResultsAndRetriesAfterFailure) {
    const auto long_pcm = make_signal(2000, 16000);
    const auto short_pcm = make_signal(400, 16000);
    audio_reference_data out, expected;
    ASSERT_TRUE(audio_prepare_reference(long_pcm.data(), 2000, 16000, out));
    const auto saved = out;
    EXPECT_FALSE(audio_prepare_reference(nullptr, 400, 16000, out));
    expect_same_data(out, saved);
    ASSERT_TRUE(audio_prepare_reference(short_pcm.data(), 400, 16000, expected));
    ASSERT_TRUE(audio_prepare_reference(short_pcm.data(), 400, 16000, out));
    expect_same_data(out, expected);
    ASSERT_EQ(out.pcm16k.size(), 400u);
    EXPECT_EQ(out.spec_frames, 1);
}

TEST(AudioReference, AcceptsInputBorrowedFromExistingOutput) {
    const auto pcm = make_signal(1600, 48000);
    audio_reference_data out, expected;
    ASSERT_TRUE(audio_prepare_reference(pcm.data(), static_cast<int64_t>(pcm.size()), 48000, out));
    const auto copied_input = out.pcm16k;
    ASSERT_TRUE(audio_prepare_reference(copied_input.data(), static_cast<int64_t>(copied_input.size()),
                                       16000, expected));
    ASSERT_TRUE(audio_prepare_reference(out.pcm16k.data(), static_cast<int64_t>(out.pcm16k.size()),
                                       16000, out));
    expect_same_data(out, expected);
}

} // namespace
