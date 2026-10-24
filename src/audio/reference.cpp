#include "gpt_sovits/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gpt_sovits {

namespace {

constexpr int kWaveRate = 16000;
constexpr int kSpecRate = 32000;
constexpr int64_t kMinWaveSamples = 400;
constexpr int kFFTSize = 2048;
constexpr int kHopLength = 640;
constexpr int64_t kSpecBins = kFFTSize / 2 + 1;
constexpr int64_t kSpecChannels = 704;
constexpr int64_t kReflectPad = (kFFTSize - kHopLength) / 2;
constexpr double kVarianceEps = 1e-7;

bool fail(const char * message) {
    fprintf(stderr, "audio_prepare_reference: %s\n", message);
    return false;
}

bool resampled_length(int64_t n, int src_rate, int dst_rate, int64_t & length) {
    const int64_t divisor = std::gcd(src_rate, dst_rate);
    const int64_t up = dst_rate / divisor;
    const int64_t down = src_rate / divisor;
    if (n > std::numeric_limits<int64_t>::max() / up) return false;
    const int64_t n_up = n * up;
    length = n_up / down + (n_up % down != 0);
    return true;
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

} // namespace

bool audio_prepare_reference(const float * pcm, int64_t n_samples, int sample_rate,
                             audio_reference_data & out) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0) {
        return fail("pcm must be non-null, length and sample rate must be positive");
    }

    try {
        audio_reference_data pending;
        const uint64_t max_samples = static_cast<uint64_t>(pending.pcm16k.max_size());
        int64_t n16k = 0, n32k = 0;
        if (static_cast<uint64_t>(n_samples) > max_samples ||
            !resampled_length(n_samples, sample_rate, kWaveRate, n16k) ||
            !resampled_length(n_samples, sample_rate, kSpecRate, n32k) ||
            static_cast<uint64_t>(n16k) > max_samples ||
            static_cast<uint64_t>(n32k) > max_samples ||
            n32k > std::numeric_limits<int64_t>::max() - 2 * kReflectPad) {
            return fail("waveform length overflow");
        }
        if (n16k < kMinWaveSamples || n32k <= kReflectPad) {
            return fail("reference is too short for HuBERT / SV / SoVITS inputs");
        }
        const int64_t frames = 1 + (n32k + 2 * kReflectPad - kFFTSize) / kHopLength;
        if (frames <= 0 || static_cast<uint64_t>(frames) > max_samples / kSpecBins ||
            static_cast<uint64_t>(frames) > max_samples / kSpecChannels) {
            return fail("spectrum size overflow");
        }
        if (!std::all_of(pcm, pcm + n_samples,
                         [](float value) { return std::isfinite(value); })) {
            return fail("input contains non-finite samples");
        }

        std::vector<float> pcm32k;
        if (!audio_resample(pcm, n_samples, sample_rate, kWaveRate, pending.pcm16k) ||
            !audio_resample(pcm, n_samples, sample_rate, kSpecRate, pcm32k)) {
            return fail("resampling failed");
        }
        if (pending.pcm16k.size() != static_cast<size_t>(n16k) ||
            pcm32k.size() != static_cast<size_t>(n32k) ||
            !all_finite(pending.pcm16k) || !all_finite(pcm32k)) {
            return fail("invalid resampled waveform");
        }

        double mean = 0.0;
        for (float value : pending.pcm16k) mean += static_cast<double>(value);
        mean /= static_cast<double>(n16k);
        double variance = 0.0;
        for (float value : pending.pcm16k) {
            const double delta = static_cast<double>(value) - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(n16k);
        const float inv_std = static_cast<float>(1.0 / std::sqrt(variance + kVarianceEps));
        const float mean_f = static_cast<float>(mean);
        pending.normalized_pcm16k.resize(static_cast<size_t>(n16k));
        for (size_t i = 0; i < pending.pcm16k.size(); ++i) {
            pending.normalized_pcm16k[i] = (pending.pcm16k[i] - mean_f) * inv_std;
        }
        if (!all_finite(pending.normalized_pcm16k)) {
            return fail("normalization produced non-finite samples");
        }

        std::vector<float> spectrum;
        int64_t actual_frames = 0;
        if (!audio_stft_magnitude(pcm32k.data(), n32k, kFFTSize, kHopLength, kFFTSize,
                                  spectrum, &actual_frames)) {
            return fail("STFT failed");
        }
        if (actual_frames != frames ||
            spectrum.size() != static_cast<size_t>(frames) * kSpecBins ||
            !all_finite(spectrum)) {
            return fail("invalid reference spectrum");
        }
        pending.spec704.resize(static_cast<size_t>(frames) * kSpecChannels);
        for (int64_t t = 0; t < frames; ++t) {
            std::copy_n(spectrum.data() + static_cast<size_t>(t) * kSpecBins,
                        kSpecChannels,
                        pending.spec704.data() + static_cast<size_t>(t) * kSpecChannels);
        }
        pending.spec_frames = frames;
        out = std::move(pending);
        return true;
    } catch (const std::bad_alloc &) {
        return fail("host allocation failed");
    } catch (const std::length_error &) {
        return fail("host allocation size exceeded");
    }
}

} // namespace gpt_sovits
