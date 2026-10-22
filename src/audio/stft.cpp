#include "gpt_sovits/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace gpt_sovits {

namespace {

constexpr float  kPiF      = 3.14159265358979323846f;
constexpr double kPiD      = 3.14159265358979323846;
constexpr float  kStftEps  = 1e-6f;

static bool is_pow2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// Overflow-checked helpers for host-side length arithmetic (operands >= 0).
static bool mul_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) return false;
    out = a * b;
    return true;
}

static bool add_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (b > std::numeric_limits<int64_t>::max() - a) return false;
    out = a + b;
    return true;
}

} // namespace

// Iterative radix-2 Cooley-Tukey FFT (in-place). The arithmetic is kept
// identical to the original sv_fbank implementation so sv_fbank output does
// not change when it switches to this shared entry point.
bool audio_fft_radix2(float * re, float * im, int64_t n) {
    if (!re || !im || !is_pow2(n)) {
        fprintf(stderr, "%s: n must be a positive power of two (got %lld)\n",
                __func__, static_cast<long long>(n));
        return false;
    }
    if (n > std::numeric_limits<int>::max()) {
        fprintf(stderr, "%s: n=%lld exceeds supported FFT size\n",
                __func__, static_cast<long long>(n));
        return false;
    }
    const int m = static_cast<int>(n);
    // bit reversal
    for (int i = 1, j = 0; i < m; ++i) {
        int bit = m >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= m; len <<= 1) {
        const float ang = -2.0f * kPiF / len;
        const float wre = std::cos(ang), wim = std::sin(ang);
        for (int i = 0; i < m; i += len) {
            float cre = 1.0f, cim = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const float ar = re[i + k], ai = im[i + k];
                const float br = re[i + k + len / 2], bi = im[i + k + len / 2];
                const float tr = cre * br - cim * bi;
                const float ti = cre * bi + cim * br;
                re[i + k] = ar + tr; im[i + k] = ai + ti;
                re[i + k + len / 2] = ar - tr; im[i + k + len / 2] = ai - ti;
                const float ncre = cre * wre - cim * wim;
                cim = cre * wim + cim * wre;
                cre = ncre;
            }
        }
    }
    return true;
}

bool audio_stft_magnitude(const float * pcm, int64_t n_samples,
                          int n_fft, int hop_length, int win_length,
                          std::vector<float> & out, int64_t * out_frames) {
    if (out_frames) *out_frames = 0;
    if (!pcm || n_samples <= 0) {
        fprintf(stderr, "%s: pcm must be non-null and n_samples > 0\n", __func__);
        return false;
    }
    if (!is_pow2(n_fft)) {
        fprintf(stderr, "%s: n_fft must be a positive power of two (got %d)\n",
                __func__, n_fft);
        return false;
    }
    if (hop_length <= 0) {
        fprintf(stderr, "%s: hop_length must be > 0 (got %d)\n", __func__, hop_length);
        return false;
    }
    if (win_length <= 0 || win_length > n_fft) {
        fprintf(stderr, "%s: win_length must be in (0, n_fft] (got %d, n_fft=%d)\n",
                __func__, win_length, n_fft);
        return false;
    }
    if (n_fft < hop_length) {
        fprintf(stderr, "%s: n_fft (%d) must be >= hop_length (%d)\n",
                __func__, n_fft, hop_length);
        return false;
    }

    // Upstream: int((n_fft - hop_size) / 2) on both sides.
    const int64_t pad = (static_cast<int64_t>(n_fft) - hop_length) / 2;
    if (pad >= n_samples) {
        // torch/numpy reflect padding requires the pad width to be smaller
        // than the input length.
        fprintf(stderr, "%s: reflect pad (%lld) must be < n_samples (%lld)\n",
                __func__, static_cast<long long>(pad), static_cast<long long>(n_samples));
        return false;
    }

    int64_t n_padded = 0;
    if (!add_ok(n_samples, 2 * pad, n_padded)) {
        fprintf(stderr, "%s: padded length overflow\n", __func__);
        return false;
    }

    int64_t frames = 0;
    if (n_padded >= n_fft) {
        frames = 1 + (n_padded - n_fft) / hop_length;
    }

    const int64_t n_bins = static_cast<int64_t>(n_fft) / 2 + 1;
    int64_t total = 0;
    if (!mul_ok(frames, n_bins, total) ||
        static_cast<uint64_t>(total) > out.max_size()) {
        fprintf(stderr, "%s: output size overflow\n", __func__);
        return false;
    }

    out.assign(static_cast<size_t>(total), 0.0f);
    if (frames == 0) return true;

    // Periodic Hann window (win_length, not win_length-1). torch.stft
    // zero-pads a short window to n_fft with the left side at
    // (n_fft - win_length) / 2 (integer division, "pad center").
    std::vector<float> win(static_cast<size_t>(win_length));
    for (int i = 0; i < win_length; ++i) {
        win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPiD * i / win_length));
    }
    const int64_t win_offset = (static_cast<int64_t>(n_fft) - win_length) / 2;

    std::vector<float> re(static_cast<size_t>(n_fft));
    std::vector<float> im(static_cast<size_t>(n_fft));

    for (int64_t t = 0; t < frames; ++t) {
        const int64_t frame_start = t * hop_length - pad;   // may be negative
        for (int64_t i = 0; i < n_fft; ++i) {
            // Reflect-pad index mapping (numpy/torch mode='reflect'):
            //   -pad .. n-1+pad  ->  0 .. n-1
            int64_t s = frame_start + i;
            if (s < 0) {
                s = -s;
            } else if (s >= n_samples) {
                s = 2 * (n_samples - 1) - s;
            }
            const int64_t wi = i - win_offset;
            re[static_cast<size_t>(i)] =
                (wi >= 0 && wi < win_length) ? pcm[s] * win[static_cast<size_t>(wi)]
                                             : 0.0f;
            im[static_cast<size_t>(i)] = 0.0f;
        }

        if (!audio_fft_radix2(re.data(), im.data(), n_fft)) {
            out.clear();
            return false;
        }

        float * dst = out.data() + static_cast<size_t>(t) * static_cast<size_t>(n_bins);
        for (int64_t k = 0; k < n_bins; ++k) {
            const float r = re[static_cast<size_t>(k)];
            const float ii = im[static_cast<size_t>(k)];
            dst[static_cast<size_t>(k)] = std::sqrt(r * r + ii * ii + kStftEps);
        }
    }

    if (out_frames) *out_frames = frames;
    return true;
}

} // namespace gpt_sovits
