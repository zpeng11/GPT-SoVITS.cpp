// tests/audio/test_audio_resample.cpp
//
// Bandlimited resampler tests:
//   * identity when src_rate == dst_rate,
//   * sine-sweep SNR after 44.1 kHz -> 16 kHz and 48 kHz -> 32 kHz,
//   * integer-ratio 48 kHz -> 16 kHz close to naive decimation (envelope),
//   * optional scipy.signal.resample_poly parity when the fixture exists,
//   * invalid-argument rejection.

#include <gtest/gtest.h>

#include "gpt_sovits/audio.h"

#include "cnpy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static std::vector<float> load_f32(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != 4) return {};
    const float * p = arr.data<float>();
    return std::vector<float>(p, p + arr.num_vals);
}

static double phase_sweep(double t, double f0, double f1, double duration) {
    const double pi = 3.14159265358979323846;
    return 2.0 * pi * (f0 * t + 0.5 * (f1 - f0) / duration * t * t);
}

static std::vector<float> make_sweep(int n, int rate, double f0, double f1) {
    std::vector<float> x(static_cast<size_t>(n));
    const double duration = static_cast<double>(n) / rate;
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / rate;
        x[static_cast<size_t>(i)] = static_cast<float>(std::sin(phase_sweep(t, f0, f1, duration)));
    }
    return x;
}

// Returns the expected swept sine sampled at the destination rate, assuming
// the resampler preserves the input time origin.
static std::vector<double> expected_sweep(int n_out, int rate, double f0, double f1, double duration) {
    std::vector<double> y(static_cast<size_t>(n_out));
    for (int m = 0; m < n_out; ++m) {
        const double t = static_cast<double>(m) / rate;
        y[static_cast<size_t>(m)] = std::sin(phase_sweep(t, f0, f1, duration));
    }
    return y;
}

static double snr_db(const std::vector<float> & actual,
                     const std::vector<double> & expected,
                     size_t skip) {
    double sig = 0.0, err = 0.0;
    for (size_t i = skip; i + skip < expected.size() && i < actual.size(); ++i) {
        const double a = actual[i];
        const double e = expected[i];
        sig += e * e;
        err += (a - e) * (a - e);
    }
    if (!(err > 0.0)) return 300.0;   // exact (or numerically silent error)
    return 10.0 * std::log10(sig / err);
}

TEST(AudioResample, Identity) {
    const std::vector<float> x = make_sweep(1000, 16000, 100.0, 6000.0);
    std::vector<float> out;
    ASSERT_TRUE(gpt_sovits::audio_resample(x.data(), static_cast<int64_t>(x.size()),
                                           16000, 16000, out));
    ASSERT_EQ(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) EXPECT_EQ(out[i], x[i]);

    std::vector<float> single(1, 0.25f);
    ASSERT_TRUE(gpt_sovits::audio_resample(single.data(), 1, 44100, 48000, out));
    ASSERT_FALSE(out.empty());
    for (float v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(AudioResample, Sweep44kTo16kSnr) {
    constexpr int src = 44100, dst = 16000;
    constexpr double f0 = 100.0, f1 = 6000.0;   // below dst Nyquist (8 kHz)
    std::vector<float> x = make_sweep(src, src, f0, f1);

    std::vector<float> out;
    ASSERT_TRUE(gpt_sovits::audio_resample(x.data(), static_cast<int64_t>(x.size()), src, dst, out));
    ASSERT_EQ(out.size(), static_cast<size_t>(dst));   // exactly 1 second

    const auto expected = expected_sweep(dst, dst, f0, f1, 1.0);
    const size_t skip = static_cast<size_t>(0.03 * dst);   // 30 ms edge transient
    const double snr = snr_db(out, expected, skip);
    printf("[resample 44100->16000] SNR=%.2f dB\n", snr);
    EXPECT_GT(snr, 40.0);
}

TEST(AudioResample, Sweep48kTo32kSnr) {
    constexpr int src = 48000, dst = 32000;
    constexpr double f0 = 200.0, f1 = 12000.0;  // below dst Nyquist (16 kHz)
    std::vector<float> x = make_sweep(src, src, f0, f1);

    std::vector<float> out;
    ASSERT_TRUE(gpt_sovits::audio_resample(x.data(), static_cast<int64_t>(x.size()), src, dst, out));
    ASSERT_EQ(out.size(), static_cast<size_t>(dst));

    const auto expected = expected_sweep(dst, dst, f0, f1, 1.0);
    const size_t skip = static_cast<size_t>(0.03 * dst);
    const double snr = snr_db(out, expected, skip);
    printf("[resample 48000->32000] SNR=%.2f dB\n", snr);
    EXPECT_GT(snr, 40.0);
}

TEST(AudioResample, IntegerDownsampleMatchesEnvelope) {
    constexpr int src = 48000, dst = 16000;
    constexpr int n = src;   // 1 second -> 16000 output samples
    std::vector<float> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / src;
        x[static_cast<size_t>(i)] =
            static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * t) +
                               0.2 * std::sin(2.0 * 3.14159265358979323846 * 2500.0 * t + 0.4));
    }

    std::vector<float> out;
    ASSERT_TRUE(gpt_sovits::audio_resample(x.data(), n, src, dst, out));
    ASSERT_EQ(out.size(), static_cast<size_t>(n / 3));

    // The filter is not a pure decimator: compare block RMS envelopes, not
    // sample values.
    const size_t block = 64;
    const size_t blocks = out.size() / block;
    ASSERT_GT(blocks, 4u);
    size_t compared = 0;
    double max_rel = 0.0;
    for (size_t b = 2; b + 2 < blocks; ++b) {   // skip edge transients
        double so = 0.0, sd = 0.0;
        for (size_t i = 0; i < block; ++i) {
            const size_t m = b * block + i;
            so += static_cast<double>(out[m]) * out[m];
            sd += static_cast<double>(x[m * 3]) * x[m * 3];
        }
        const double ro = std::sqrt(so / block);
        const double rd = std::sqrt(sd / block);
        const double rel = std::fabs(ro - rd) / std::max(rd, 1e-9);
        max_rel = std::max(max_rel, rel);
        ++compared;
    }
    printf("[resample 48000->16000] max block-RMS relative difference=%.4f\n", max_rel);
    EXPECT_GT(compared, 0u);
    EXPECT_LT(max_rel, 0.05);
}

// Optional parity against scipy.signal.resample_poly, emitted by
// generate_reference.py when scipy is installed.
TEST(AudioResample, ScipyResamplePolyParity) {
    const std::string in_path  = std::string(AUDIO_TEST_DIR) + "ref/resample_input_44k.npy";
    const std::string ref_path = std::string(AUDIO_TEST_DIR) + "ref/resample_poly_16k.npy";
    if (!file_exists(in_path) || !file_exists(ref_path)) {
        GTEST_SKIP() << "scipy resample_poly reference not generated";
    }

    const std::vector<float> x   = load_f32(in_path);
    const std::vector<float> ref = load_f32(ref_path);
    ASSERT_FALSE(x.empty());
    ASSERT_FALSE(ref.empty());

    std::vector<float> out;
    ASSERT_TRUE(gpt_sovits::audio_resample(x.data(), static_cast<int64_t>(x.size()), 44100, 16000, out));
    ASSERT_EQ(out.size(), ref.size());

    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(out[i]) - ref[i]));
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
    }
    printf("[resample scipy parity] max_abs=%.3e max_ref=%.3e\n", max_abs, max_ref);
    EXPECT_LT(max_abs, 1e-3);
}

TEST(AudioResample, RejectsInvalidArguments) {
    std::vector<float> x(100, 0.1f);
    std::vector<float> out;

    EXPECT_FALSE(gpt_sovits::audio_resample(nullptr, 100, 44100, 16000, out));
    EXPECT_FALSE(gpt_sovits::audio_resample(x.data(), 0, 44100, 16000, out));
    EXPECT_FALSE(gpt_sovits::audio_resample(x.data(), -1, 44100, 16000, out));
    EXPECT_FALSE(gpt_sovits::audio_resample(x.data(), 100, 0, 16000, out));
    EXPECT_FALSE(gpt_sovits::audio_resample(x.data(), 100, -44100, 16000, out));
    EXPECT_FALSE(gpt_sovits::audio_resample(x.data(), 100, 44100, 0, out));
}

} // namespace
