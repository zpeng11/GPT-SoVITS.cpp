// tests/audio/test_audio_io.cpp
//
// WAV/FLAC I/O tests:
//   * mono IEEE-float32 WAV round-trip (bit-exact),
//   * PCM16 WAV decode within quantization tolerance,
//   * decoding the repository FLAC fixture (sample rate / length / non-silence),
//   * failure contract for missing/unsupported inputs.

#include <gtest/gtest.h>

#include "gpt_sovits/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

static std::string temp_path(const std::string & name) {
    return testing::TempDir() + name;
}

static void remove_file(const std::string & path) {
    std::remove(path.c_str());
}

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static std::vector<float> make_samples(size_t n, uint32_t seed) {
    std::vector<float> s(n);
    uint32_t state = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;          // LCG, deterministic
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) /
                        static_cast<float>(0x00ffffffu);   // [0, 1]
        s[i] = 2.0f * u - 1.0f;
    }
    return s;
}

// Minimal PCM16 WAV writer used only to produce a decoder input fixture.
static bool write_pcm16_wav(const std::string & path, const std::vector<int16_t> & samples,
                            uint32_t sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;

    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

    #pragma pack(push, 1)
    struct Header {
        char     riff[4];
        uint32_t riff_size;
        char     wave[4];
        char     fmt_[4];
        uint32_t fmt_size;
        uint16_t audio_fmt;
        uint16_t n_chan;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_samp;
        char     data[4];
        uint32_t data_size;
    } h;
    #pragma pack(pop)

    std::memcpy(h.riff, "RIFF", 4);
    h.riff_size     = 36 + data_size;
    std::memcpy(h.wave, "WAVE", 4);
    std::memcpy(h.fmt_, "fmt ", 4);
    h.fmt_size      = 16;
    h.audio_fmt     = 1;                 // PCM
    h.n_chan        = 1;
    h.sample_rate   = sample_rate;
    h.byte_rate     = sample_rate * 2u;
    h.block_align   = 2;
    h.bits_per_samp = 16;
    std::memcpy(h.data, "data", 4);
    h.data_size     = data_size;

    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 &&
              fwrite(samples.data(), sizeof(int16_t), samples.size(), f) == samples.size();
    if (fclose(f) != 0) ok = false;
    return ok;
}

TEST(AudioIO, Float32WavRoundTrip) {
    const std::string path = temp_path("audio_io_f32_roundtrip.wav");
    const std::vector<float> samples = make_samples(4096, 7u);
    constexpr int sample_rate = 32000;

    ASSERT_TRUE(gpt_sovits::audio_write_wav_f32(path, samples.data(),
                                                static_cast<int64_t>(samples.size()), sample_rate));

    std::vector<float> decoded;
    int decoded_rate = 0;
    ASSERT_TRUE(gpt_sovits::audio_read_file(path, decoded, decoded_rate));
    remove_file(path);

    EXPECT_EQ(decoded_rate, sample_rate);
    ASSERT_EQ(decoded.size(), samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_EQ(decoded[i], samples[i]) << "float32 sample " << i << " changed";
    }
}

TEST(AudioIO, Pcm16WavDecode) {
    const std::string path = temp_path("audio_io_pcm16.wav");
    constexpr int sample_rate = 16000;
    constexpr size_t n = 2048;

    const std::vector<float> src = make_samples(n, 11u);
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        pcm[i] = static_cast<int16_t>(std::lround(src[i] * 32767.0f));
    }

    ASSERT_TRUE(write_pcm16_wav(path, pcm, sample_rate));

    std::vector<float> decoded;
    int decoded_rate = 0;
    ASSERT_TRUE(gpt_sovits::audio_read_file(path, decoded, decoded_rate));
    remove_file(path);

    EXPECT_EQ(decoded_rate, sample_rate);
    ASSERT_EQ(decoded.size(), n);
    const float quantum = 1.0f / 32768.0f;
    for (size_t i = 0; i < n; ++i) {
        const float expected = static_cast<float>(pcm[i]) / 32768.0f;
        EXPECT_NEAR(decoded[i], expected, quantum * 1e-3f) << "PCM16 decode mismatch at " << i;
        // And the decoded value stays within the PCM16 quantization bound of
        // the original float sample: rounding (0.5 LSB) plus the asymmetric
        // 32767 vs 32768 scale (another 1 LSB at full scale).
        EXPECT_LE(std::fabs(decoded[i] - src[i]), 1.5f * quantum + 1e-6f);
    }
}

TEST(AudioIO, ReadsFlacFixture) {
    const std::string path = std::string(AUDIO_TEST_DIR) + "../hubert/Narsil.asr_dummy.4.flac";
    if (!file_exists(path)) {
        GTEST_SKIP() << "flac fixture missing: " << path;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    ASSERT_TRUE(gpt_sovits::audio_read_file(path, samples, sample_rate)) << path;

    EXPECT_EQ(sample_rate, 48000);
    // STREAMINFO reports 315648 frames, mono.
    EXPECT_EQ(samples.size(), static_cast<size_t>(315648));

    double ss = 0.0;
    float peak = 0.0f;
    for (float v : samples) {
        ASSERT_TRUE(std::isfinite(v));
        peak = std::max(peak, std::fabs(v));
        ss += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(ss / static_cast<double>(samples.size()));
    printf("[flac] samples=%zu rate=%d peak=%.4f rms=%.4f\n",
           samples.size(), sample_rate, peak, rms);
    EXPECT_GT(peak, 0.01f) << "decoded flac looks silent";
    EXPECT_GT(rms, 0.001) << "decoded flac looks silent";
}

TEST(AudioIO, RejectsMissingAndUnsupportedFiles) {
    std::vector<float> out;
    int rate = 0;

    EXPECT_FALSE(gpt_sovits::audio_read_file("", out, rate));
    EXPECT_FALSE(gpt_sovits::audio_read_file(temp_path("audio_io_missing.wav"), out, rate));
    EXPECT_FALSE(gpt_sovits::audio_read_file(temp_path("audio_io_unsupported.txt"), out, rate));
    EXPECT_FALSE(gpt_sovits::audio_read_file(temp_path("audio_io_unsupported"), out, rate));

    // Writer argument validation.
    std::vector<float> one(1, 0.0f);
    EXPECT_FALSE(gpt_sovits::audio_write_wav_f32("", one.data(), 1, 16000));
    EXPECT_FALSE(gpt_sovits::audio_write_wav_f32(temp_path("x.wav"), nullptr, 1, 16000));
    EXPECT_FALSE(gpt_sovits::audio_write_wav_f32(temp_path("x.wav"), one.data(), 0, 16000));
    EXPECT_FALSE(gpt_sovits::audio_write_wav_f32(temp_path("x.wav"), one.data(), 1, 0));
}

} // namespace
