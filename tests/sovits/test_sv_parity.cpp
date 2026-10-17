// tests/sovits/test_sv_parity.cpp
//
// Phase-3 parity for the standalone SV (speaker-verification) pipeline:
//   - fbank:    sv_fbank(audio16k)  vs sv_fbank.npy    (Kaldi.fbank reference)
//   - forward3: sv_forward3(fbank)  vs sv_emb.npy      (ERes2NetV2.forward3)
// The two are checked independently so a regression localizes to the DSP
// frontend or the model. Fixtures are produced by python/dump_sv_ref.py.
//
// This file deliberately depends only on gpt_sovits/sv.h -- not on
// sovits.h/sovits_session. The end-to-end ge wiring check (which needs the
// SoVITS session) lives in test_sovits_v2pro_parity.cpp.
//
// Beyond the value-parity checks it covers the fbank frontend's append/repeat/
// boundary/null-pointer contract and sv_forward3's rejection of invalid
// model/input.

#include <gtest/gtest.h>

#include "gpt_sovits/sv.h"

#include "cnpy.h"
#include "npy_loader.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

const std::string kTestDir = SOVITS_TEST_DIR;
const std::string kModel   = kTestDir + "models/sv-eres2netv2-f32.gguf";
const std::string kRefDir  = kTestDir + "ref/";

static ggml_backend_t sv_test_backend() {
    if (std::getenv("SV_TEST_CPU")) return ggml_backend_cpu_init();
    return create_test_backend();
}

struct ErrorStats {
    double max_abs   = 0.0;
    double rmse      = 0.0;
    double mean_abs  = 0.0;
    // Failure flags: a silent zero return would let EXPECT_LT(0, tol) pass, so
    // a size mismatch or any non-finite sample must poison the stats instead.
    bool   size_mismatch = false;
    bool   nonfinite     = false;
};

// Compare two flat vectors. A size mismatch or any non-finite entry returns
// +inf for every metric (so threshold checks fail loudly) and sets the flag.
static ErrorStats compute_errors(const std::vector<float> & a,
                                 const std::vector<float> & b) {
    const double kInf = std::numeric_limits<double>::infinity();
    ErrorStats s{};
    if (a.size() != b.size() || a.empty()) {
        s.size_mismatch = true;
        s.max_abs = s.rmse = s.mean_abs = kInf;
        return s;
    }
    double ss = 0.0, sa = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            s.nonfinite = true;
            continue;
        }
        const double e = std::fabs(x - y);
        s.max_abs = std::max(s.max_abs, e);
        ss += e * e; sa += e;
    }
    s.rmse = std::sqrt(ss / a.size());
    s.mean_abs = sa / a.size();
    if (s.nonfinite) {   // a NaN/Inf sample must not slip through the tolerance.
        s.max_abs = s.rmse = s.mean_abs = kInf;
    }
    return s;
}

struct NpyShape { std::vector<float> data; std::vector<size_t> shape; };

static NpyShape load_npy(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static bool file_exists(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false; fclose(f); return true;
}

// Deterministic finite test waveform (not the parity fixture): a couple of
// sine partials so the fbank has non-trivial energy to work with.
static std::vector<float> make_audio(size_t n) {
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        a[i] = 0.3f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 440.0 * t)) +
               0.1f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 1337.0 * t));
    }
    return a;
}

// fbank parity: feed the exact 16kHz audio used by dump_sv_ref.py.
TEST(SVParity, Fbank) {
    const NpyShape audio = load_npy(kRefDir + "sv_audio16k.npy");
    const NpyShape ref   = load_npy(kRefDir + "sv_fbank.npy");
    ASSERT_FALSE(audio.data.empty());
    ASSERT_EQ(ref.shape.size(), 2u);
    const int m_ref = static_cast<int>(ref.shape[0]);
    const int mel   = static_cast<int>(ref.shape[1]);

    std::vector<float> feats;
    const int m = gpt_sovits::sv_fbank(audio.data.data(),
                                       static_cast<int64_t>(audio.data.size()), feats);
    ASSERT_EQ(m, m_ref);
    ASSERT_EQ(feats.size(), ref.data.size());

    const auto err = compute_errors(feats, ref.data);
    printf("[sv fbank] m=%d mel=%d (max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           m, mel, err.max_abs, err.rmse, err.mean_abs);
    EXPECT_FALSE(err.size_mismatch) << "frame count / feature size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "fbank produced non-finite values";
    EXPECT_LT(err.max_abs, 1.0e-3);
    EXPECT_LT(err.rmse,    2.0e-4);
}

// forward3 parity: feed the Python fbank directly (isolates the model).
TEST(SVParity, Forward3) {
    if (!file_exists(kModel)) GTEST_SKIP() << "SV model not present: " << kModel;
    const NpyShape fbn = load_npy(kRefDir + "sv_fbank.npy");
    const NpyShape emb = load_npy(kRefDir + "sv_emb.npy");
    ASSERT_EQ(fbn.shape.size(), 2u);
    const int m = static_cast<int>(fbn.shape[0]);
    ASSERT_EQ(emb.data.size(), 20480u);

    // forward3 now runs as a ggml graph on a backend (Metal on macOS). Use the
    // shared helper so the parity check exercises the same path production uses.
    ggml_backend_t backend = sv_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sv_model model{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModel, model, backend));
    EXPECT_EQ(model.hparams.out_dim, 20480);

    std::vector<float> out(20480, 0.0f);
    ASSERT_TRUE(gpt_sovits::sv_forward3(model, fbn.data.data(), m, out.data()));

    const auto err = compute_errors(out, emb.data);
    printf("[sv forward3] m=%d (max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           m, err.max_abs, err.rmse, err.mean_abs);
    EXPECT_FALSE(err.size_mismatch) << "embedding size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "forward3 produced non-finite values";
    EXPECT_LT(err.max_abs, 1.0e-2);
    EXPECT_LT(err.rmse,    3.0e-3);

    gpt_sovits::sv_model_free(model);
    ggml_backend_free(backend);
}

// First-layer stress fixture: a fbank whose forward3 activations exceed the
// Hardtanh(0, 20) clamp bound at the first block. A mis-applied clamp (or any
// value clipped instead of passed through) shows up as a large (~0.16) max error;
// the correct graph matches the recorded embedding to the usual tolerances.
TEST(SVParity, Forward3Stress) {
    const std::string fbank_path = kRefDir + "sv_stress_fbank.npy";
    const std::string emb_path   = kRefDir + "sv_stress_emb.npy";
    ASSERT_TRUE(file_exists(fbank_path));
    ASSERT_TRUE(file_exists(emb_path));

    const NpyShape fbn = load_npy(fbank_path);
    const NpyShape emb = load_npy(emb_path);
    ASSERT_EQ(fbn.shape.size(), 2u);
    ASSERT_EQ(fbn.shape[1], 80u);
    const int m = static_cast<int>(fbn.shape[0]);
    ASSERT_GT(m, 0);
    ASSERT_EQ(emb.data.size(), 20480u);
    if (!file_exists(kModel)) GTEST_SKIP() << "SV model not present: " << kModel;

    ggml_backend_t backend = sv_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sv_model model{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModel, model, backend));

    std::vector<float> out(20480, 0.0f);
    ASSERT_TRUE(gpt_sovits::sv_forward3(model, fbn.data.data(), m, out.data()));

    const auto err = compute_errors(out, emb.data);
    printf("[sv forward3 stress] m=%d (max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           m, err.max_abs, err.rmse, err.mean_abs);
    EXPECT_FALSE(err.size_mismatch) << "embedding size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "forward3 produced non-finite values";
    EXPECT_LT(err.max_abs, 1.0e-2);
    EXPECT_LT(err.rmse,    3.0e-3);

    gpt_sovits::sv_model_free(model);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// fbank frontend contract (append / repeat / boundary / null input)
// ---------------------------------------------------------------------------

// sv_fbank grows out_feats (it does not clear it): a second call appends a
// second m*80 block that is identical to the first, and any pre-existing
// values are preserved.
TEST(SVParity, FbankAppendsAndRepeatsDeterministically) {
    const std::vector<float> audio = make_audio(16000);

    std::vector<float> feats;
    feats.push_back(123.0f);  // sentinel: must survive (append, not clear)
    const int m1 = gpt_sovits::sv_fbank(audio.data(),
                                        static_cast<int64_t>(audio.size()), feats);
    ASSERT_GT(m1, 0);
    ASSERT_EQ(feats.size(), 1u + static_cast<size_t>(m1) * 80u);
    EXPECT_FLOAT_EQ(feats[0], 123.0f) << "existing data must not be cleared";

    const std::vector<float> first(feats.begin() + 1, feats.end());

    const int m2 = gpt_sovits::sv_fbank(audio.data(),
                                        static_cast<int64_t>(audio.size()), feats);
    EXPECT_EQ(m2, m1);
    ASSERT_EQ(feats.size(), 1u + 2u * static_cast<size_t>(m1) * 80u);

    // Appended block must match the first (same input -> same frames).
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(feats[1u + first.size() + i], first[i]) << "frame index " << i;
    }

    // Independent repeat: two fresh calls are bit-identical.
    std::vector<float> a1, a2;
    ASSERT_EQ(gpt_sovits::sv_fbank(audio.data(), (int64_t)audio.size(), a1), m1);
    ASSERT_EQ(gpt_sovits::sv_fbank(audio.data(), (int64_t)audio.size(), a2), m1);
    ASSERT_EQ(a1.size(), a2.size());
    for (size_t i = 0; i < a1.size(); ++i) EXPECT_EQ(a1[i], a2[i]);
}

// Frame-count boundaries: window=400 samples, hop=160 (snip_edges).
TEST(SVParity, FbankBoundaries) {
    const std::vector<float> audio = make_audio(1200);

    std::vector<float> feats;
    // Too short / empty / negative: no frames, output untouched.
    EXPECT_EQ(gpt_sovits::sv_fbank(audio.data(), 0, feats), 0);
    EXPECT_EQ(gpt_sovits::sv_fbank(audio.data(), -7, feats), 0);
    EXPECT_EQ(gpt_sovits::sv_fbank(audio.data(), 399, feats), 0);
    EXPECT_TRUE(feats.empty());

    // Exactly one window -> one frame.
    EXPECT_EQ(gpt_sovits::sv_fbank(audio.data(), 400, feats), 1);
    EXPECT_EQ(feats.size(), 80u);

    // One more hop -> two frames.
    feats.clear();
    EXPECT_EQ(gpt_sovits::sv_fbank(audio.data(), 400 + 160, feats), 2);
    EXPECT_EQ(feats.size(), 160u);

    // General formula: m = 1 + (n - 400) / 160.
    feats.clear();
    const int m = gpt_sovits::sv_fbank(audio.data(), 1000, feats);
    EXPECT_EQ(m, 1 + (1000 - 400) / 160);
    EXPECT_EQ(feats.size(), static_cast<size_t>(m) * 80u);
}

// A null audio pointer must be rejected without dereferencing it, leaving the
// output vector untouched.
TEST(SVParity, FbankRejectsNullAudio) {
    std::vector<float> feats;
    EXPECT_EQ(gpt_sovits::sv_fbank(nullptr, 16000, feats), 0);
    EXPECT_TRUE(feats.empty());
    EXPECT_EQ(gpt_sovits::sv_fbank(nullptr, 0, feats), 0);
    EXPECT_TRUE(feats.empty());
}

// ---------------------------------------------------------------------------
// sv_forward3 invalid model / invalid input
// ---------------------------------------------------------------------------

// Non-positive frame counts are rejected before any graph work.
TEST(SVParity, ForwardRejectsNonPositiveFrames) {
    if (!file_exists(kModel)) GTEST_SKIP() << "SV model not present: " << kModel;
    const NpyShape fbn = load_npy(kRefDir + "sv_fbank.npy");
    ASSERT_EQ(fbn.shape.size(), 2u);

    ggml_backend_t backend = sv_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sv_model model{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModel, model, backend));

    std::vector<float> out(20480, 0.0f);
    EXPECT_FALSE(gpt_sovits::sv_forward3(model, fbn.data.data(),  0, out.data()));
    EXPECT_FALSE(gpt_sovits::sv_forward3(model, fbn.data.data(), -1, out.data()));

    gpt_sovits::sv_model_free(model);
    ggml_backend_free(backend);
}

// An unloaded model (no backend) must be rejected, not asserted on.
TEST(SVParity, ForwardRejectsNullModel) {
    gpt_sovits::sv_model model{};   // backend == nullptr, unloaded
    const std::vector<float> fbank(80, 0.0f);
    std::vector<float> out(20480, 0.0f);
    EXPECT_FALSE(gpt_sovits::sv_forward3(model, fbank.data(), 1, out.data()));
}

// A null feature pointer must be rejected.
TEST(SVParity, ForwardRejectsNullInput) {
    if (!file_exists(kModel)) GTEST_SKIP() << "SV model not present: " << kModel;
    ggml_backend_t backend = sv_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sv_model model{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModel, model, backend));

    std::vector<float> out(20480, 0.0f);
    EXPECT_FALSE(gpt_sovits::sv_forward3(model, nullptr, 1, out.data()));

    gpt_sovits::sv_model_free(model);
    ggml_backend_free(backend);
}

} // namespace
