#include "gpt_sovits/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace gpt_sovits {

namespace {

constexpr double kPi = 3.14159265358979323846;
// ~10 zero crossings on each side of the main lobe, matching the usual
// windowed-sinc resampler quality target (scipy resample_poly defaults).
constexpr int64_t kZeroCrossings = 10;
constexpr double  kKaiserBeta    = 5.0;
// Guard against pathological coprime rate pairs whose filter would be huge.
constexpr int64_t kMaxRateFactor = 1 << 20;

static int64_t gcd_i64(int64_t a, int64_t b) {
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

static inline double sinc(double x) {
    if (std::fabs(x) < 1e-12) return 1.0;
    const double px = kPi * x;
    return std::sin(px) / px;
}

// Zeroth-order modified Bessel function of the first kind, I0(x).
static double bessel_i0(double x) {
    double sum  = 1.0;
    double term = 1.0;
    const double y = 0.25 * x * x;
    for (int k = 1; k < 200; ++k) {
        term *= y / (static_cast<double>(k) * k);
        sum  += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

static bool mul_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) return false;
    out = a * b;
    return true;
}

} // namespace

bool audio_resample(const float * in, int64_t n_in, int src_rate, int dst_rate,
                    std::vector<float> & out) {
    out.clear();
    if (!in || n_in <= 0 || src_rate <= 0 || dst_rate <= 0) {
        fprintf(stderr, "%s: invalid arguments (in=%p, n_in=%lld, src=%d, dst=%d)\n",
                __func__, static_cast<const void *>(in),
                static_cast<long long>(n_in), src_rate, dst_rate);
        return false;
    }
    if (src_rate == dst_rate) {
        out.assign(in, in + n_in);
        return true;
    }

    const int64_t g    = gcd_i64(src_rate, dst_rate);
    const int64_t up   = static_cast<int64_t>(dst_rate) / g;   // interpolation factor
    const int64_t down = static_cast<int64_t>(src_rate) / g;   // decimation factor
    if (up <= 0 || down <= 0) {
        fprintf(stderr, "%s: invalid rational ratio\n", __func__);
        return false;
    }
    if (up == down) {
        out.assign(in, in + n_in);
        return true;
    }

    const int64_t max_rate = std::max(up, down);
    if (max_rate > kMaxRateFactor) {
        fprintf(stderr, "%s: rate ratio too large (max factor %lld)\n",
                __func__, static_cast<long long>(max_rate));
        return false;
    }

    const int64_t half = kZeroCrossings * max_rate;   // filter half length
    const int64_t len  = 2 * half + 1;
    if (static_cast<uint64_t>(len) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        fprintf(stderr, "%s: filter size overflow\n", __func__);
        return false;
    }

    // Low-pass cutoff relative to the upsampled Nyquist frequency:
    //   min(src, dst) / (src * up) * 2 = 1 / max(up, down)
    const double fc   = 1.0 / static_cast<double>(max_rate);
    const double i0b  = bessel_i0(kKaiserBeta);

    std::vector<double> h(static_cast<size_t>(len));
    double hsum = 0.0;
    for (int64_t i = 0; i < len; ++i) {
        const double x = static_cast<double>(i - half);
        const double r = x / static_cast<double>(half);
        const double w = bessel_i0(kKaiserBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
        h[static_cast<size_t>(i)] = fc * sinc(fc * x) * w;
        hsum += h[static_cast<size_t>(i)];
    }
    if (!(std::fabs(hsum) > 0.0)) {
        fprintf(stderr, "%s: degenerate filter\n", __func__);
        return false;
    }
    // Unit DC gain before zero-stuffing; compensate the upsampling by `up`.
    const double gain = static_cast<double>(up) / hsum;
    for (double & v : h) v *= gain;

    int64_t n_up = 0;
    if (!mul_ok(n_in, up, n_up)) {
        fprintf(stderr, "%s: output length overflow\n", __func__);
        return false;
    }
    int64_t n_out = n_up / down;
    if (n_up % down != 0) ++n_out;
    if (static_cast<uint64_t>(n_out) > static_cast<uint64_t>(out.max_size())) {
        fprintf(stderr, "%s: output size overflow\n", __func__);
        return false;
    }

    out.resize(static_cast<size_t>(n_out));

    for (int64_t m = 0; m < n_out; ++m) {
        int64_t k = 0;
        if (!mul_ok(m, down, k)) {   // k = m*down
            fprintf(stderr, "%s: index overflow\n", __func__);
            out.clear();
            return false;
        }
        k += half;   // remove the FIR group delay

        // n such that k - n*up is a valid filter tap and n is in [0, n_in).
        int64_t n_lo = k - (len - 1);
        n_lo = (n_lo >= 0) ? (n_lo + up - 1) / up : -((-n_lo) / up);   // ceil(n_lo/up)
        int64_t n_hi = k / up;
        if (n_lo < 0) n_lo = 0;
        if (n_hi > n_in - 1) n_hi = n_in - 1;

        double acc = 0.0;
        for (int64_t n = n_lo; n <= n_hi; ++n) {
            acc += static_cast<double>(in[n]) * h[static_cast<size_t>(k - n * up)];
        }
        out[static_cast<size_t>(m)] = static_cast<float>(acc);
    }
    return true;
}

} // namespace gpt_sovits
