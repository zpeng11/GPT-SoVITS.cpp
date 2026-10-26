// Synthesis integration tests against one recorded upstream CPU case.
// Test 1 runs no T2S: it loads dumped target phones/codes, ge/ge_enc and the
// actual noise_scaled.npy into sovits_session_forward (noise injected through
// the existing "randn" graph input), then compares against wav_raw.npy.
// Test 2 runs public synthesize_cpu from ref.wav + frontend JSONs on CPU;
// preflight failures happen before the one full run so models must stay usable.
// Fixed-code output is not expected to match Python bitwise.

#include <gtest/gtest.h>

#include "gpt_sovits/audio.h"
#include "gpt_sovits/hubert.h"
#include "gpt_sovits/pipeline.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"
#include "gpt_sovits/text_frontend.h"
#include "cnpy.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

// Test-only: ggml_backend struct/interface is not on the public include path.
#include "../../third_party/ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
constexpr int64_t kInter = 192, kFrameMul = 640, kGe = 1024, kGeEnc = 512, kCodeVocab = 1024;

std::string JoinPath(const std::string & d, const std::string & n) {
    if (d.empty()) return n;
    return d.back() == '/' || d.back() == '\\' ? d + n : d + "/" + n;
}

std::string CaseDirectory() {
    std::string d = PIPELINE_REFERENCE_CASE;
    if (!d.empty() && d.back() != '/') d.push_back('/');
    return d;
}

struct ModelPaths {
    std::string hubert, roberta, t2s, sovits, sv;
};
const ModelPaths kModels = {
    JoinPath(HUBERT_TEST_DIR, "models/chinese-hubert-base-f32.gguf"),
    JoinPath(ROBERTA_MODEL_DIR, "chinese-roberta-wwm-ext-large-f16.gguf"),
    JoinPath(T2S_TEST_DIR, "models/s1v3-s2Gv2Pro-f16.gguf"),
    JoinPath(SOVITS_TEST_DIR, "models/v2pro-sovits-f16.gguf"),
    JoinPath(SOVITS_TEST_DIR, "models/sv-eres2netv2-f32.gguf"),
};

// ---------------------------------------------------------------------------
// Case manifest
// ---------------------------------------------------------------------------

struct CaseInfo {
    std::string dir;
    int64_t T_codes = 0, n_phones = 0;
    double noise_scale = 0.5;
    int source_sample_rate = 0;
};

std::string ReadFile(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

CaseInfo ParseCase(const std::string & dir) {
    const Json m = Json::parse(ReadFile(dir + "manifest.json"));
    auto req = [](bool ok, const char * what) {
        if (!ok) throw std::runtime_error(what);
    };
    req(m.at("status").get<std::string>() == "complete", "case status is not complete");
    req(m.at("schema").get<std::string>() == "gpt-sovits-inference-dump/v1",
        "unexpected dump schema");
    req(m.at("parameters").at("version").get<std::string>() == "v2Pro",
        "case version is not v2Pro");
    req(m.at("sovits_decode").at("semantic_frame_rate").get<std::string>() == "25hz",
        "case is not 25 Hz");
    req(m.at("sovits_decode").at("noise_scale").get<double>() > 0.0,
        "case noise_scale must be positive");

    const Json & sources = m.at("sources");
    req(sources.is_object() && !sources.empty(), "manifest sources are missing");
    for (auto it = sources.begin(); it != sources.end(); ++it) {
        const std::string h = it.value().get<std::string>();
        req(h.size() == 64 && std::all_of(h.begin(), h.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            }), "manifest source digest is not SHA-256");
    }

    const Json & a = m.at("arrays");
    auto shape = [&](const char * n, const char * dtype) {
        const Json & x = a.at(n);
        req(x.at("dtype").get<std::string>() == dtype, "unexpected dump dtype");
        return x.at("shape").get<std::vector<int64_t>>();
    };
    const auto codes = shape("generated_codes.npy", "int64");
    const auto phones = shape("target_phones.npy", "int64");
    req(codes.size() == 3 && codes[0] == 1 && codes[1] == 1 && codes[2] > 0,
        "generated_codes must be {1,1,T}");
    req(phones.size() == 1 && phones[0] > 0, "target_phones must be rank 1");
    const int64_t T = codes[2];
    req(shape("ge.npy", "float32") == std::vector<int64_t>{1, kGe, 1}, "bad ge shape");
    req(shape("ge_enc.npy", "float32") == std::vector<int64_t>{1, kGeEnc, 1},
        "bad ge_enc shape");
    req(shape("noise_scaled.npy", "float32") == std::vector<int64_t>{1, kInter, 2 * T},
        "bad noise_scaled shape");
    req(shape("wav_raw.npy", "float32") == std::vector<int64_t>{1, 1, 2 * T * kFrameMul},
        "bad wav_raw shape");
    req(shape("target_bert.npy", "float32") == std::vector<int64_t>{1024, phones[0]},
        "bad target_bert shape");
    req(m.at("generation").at("step").get<int64_t>() == T,
        "generation step disagrees with generated_codes");

    CaseInfo c;
    c.dir = dir;
    c.T_codes = T;
    c.n_phones = phones[0];
    c.noise_scale = m.at("sovits_decode").at("noise_scale").get<double>();
    c.source_sample_rate = m.at("reference_audio").at("source_sample_rate").get<int>();
    return c;
}

// ---------------------------------------------------------------------------
// .npy and layout helpers
// ---------------------------------------------------------------------------

struct NpyF32 {
    std::vector<float> data;
    std::vector<size_t> shape;
};

NpyF32 LoadF32(const std::string & path) {
    cnpy::NpyArray a = cnpy::npy_load(path);
    if (a.word_size != sizeof(float) || a.fortran_order || a.num_vals == 0) {
        throw std::runtime_error("expected non-empty C-order float32: " + path);
    }
    return {std::vector<float>(a.data<float>(), a.data<float>() + a.num_vals), a.shape};
}

std::vector<int32_t> LoadI32(const std::string & path) {
    cnpy::NpyArray a = cnpy::npy_load(path);
    if (a.fortran_order || a.num_vals == 0) {
        throw std::runtime_error("expected non-empty C-order integer array: " + path);
    }
    if (a.word_size == sizeof(int32_t)) {
        const int32_t * p = a.data<int32_t>();
        return std::vector<int32_t>(p, p + a.num_vals);
    }
    if (a.word_size != sizeof(int64_t)) {
        throw std::runtime_error("unsupported integer array in " + path);
    }
    const int64_t * p = a.data<int64_t>();
    std::vector<int32_t> out(a.num_vals);
    for (size_t i = 0; i < a.num_vals; ++i) {
        if (p[i] < std::numeric_limits<int32_t>::min() ||
            p[i] > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("int64 value out of int32 range: " + path);
        }
        out[i] = static_cast<int32_t>(p[i]);
    }
    return out;
}

// PyTorch [1,C,T] / [C,T] C-order -> channel-fast [T,C].
std::vector<float> PackCT(const std::vector<float> & raw, int64_t c, int64_t t) {
    if (c <= 0 || t <= 0 || raw.size() != static_cast<size_t>(c * t)) {
        throw std::runtime_error("cannot repack to channel-fast layout");
    }
    std::vector<float> out(raw.size());
    for (int64_t i = 0; i < t; ++i)
        for (int64_t j = 0; j < c; ++j)
            out[static_cast<size_t>(i * c + j)] = raw[static_cast<size_t>(j * t + i)];
    return out;
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

struct Metrics {
    size_t n = 0;
    double max_abs = 0.0, rmse = 0.0, correlation = 0.0;
    bool nonfinite = false;
};

Metrics Compare(const std::vector<float> & a, const std::vector<float> & e) {
    if (a.size() != e.size() || a.empty()) throw std::runtime_error("waveform size mismatch");
    Metrics m;
    m.n = a.size();
    double sa = 0, se = 0;
    size_t finite = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(e[i])) m.nonfinite = true;
        else { ++finite; sa += a[i]; se += e[i]; }
    }
    if (!finite) { m.nonfinite = true; return m; }
    const double ma = sa / finite, me = se / finite;
    double va = 0, ve = 0, cov = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(e[i])) continue;
        const double d = std::abs(a[i] - e[i]);
        m.max_abs = std::max(m.max_abs, d);
        m.rmse += d * d;
        va += (a[i] - ma) * (a[i] - ma);
        ve += (e[i] - me) * (e[i] - me);
        cov += (a[i] - ma) * (e[i] - me);
    }
    m.rmse = std::sqrt(m.rmse / finite);
    if (va > 0 && ve > 0) m.correlation = cov / std::sqrt(va * ve);
    if (m.nonfinite) {
        const double inf = std::numeric_limits<double>::infinity();
        m.max_abs = m.rmse = inf;
    }
    return m;
}

struct Range {
    bool finite = true;
    double min = 0, max = 0, rms = 0;
};

Range Analyze(const std::vector<float> & v) {
    if (v.empty()) throw std::runtime_error("empty waveform");
    Range r;
    r.min = std::numeric_limits<double>::infinity();
    r.max = -r.min;
    double sq = 0;
    for (float x : v) {
        if (!std::isfinite(x)) { r.finite = false; continue; }
        r.min = std::min(r.min, static_cast<double>(x));
        r.max = std::max(r.max, static_cast<double>(x));
        sq += static_cast<double>(x) * x;
    }
    r.rms = std::sqrt(sq / static_cast<double>(v.size()));
    return r;
}

// ---------------------------------------------------------------------------
// Test-only body that overrides graph_compute to inject the dumped noise.
// ---------------------------------------------------------------------------

struct NoiseInjector {
    ggml_backend_t real = nullptr;
    const float * data = nullptr;
    size_t count = 0;
    int calls = 0, uploads = 0;
    bool shape_ok = true;
};

enum ggml_status injected_compute(ggml_backend_t b, struct ggml_cgraph * g) {
    NoiseInjector * inj = static_cast<NoiseInjector *>(b->context);
    ++inj->calls;
    struct ggml_tensor * t = ggml_graph_get_tensor(g, "randn");
    if (!t || t->type != GGML_TYPE_F32 || ggml_nbytes(t) != inj->count * sizeof(float)) {
        inj->shape_ok = false;
        return GGML_STATUS_FAILED;
    }
    ggml_backend_tensor_set(t, inj->data, 0, inj->count * sizeof(float));
    ++inj->uploads;
    return ggml_backend_graph_compute(inj->real, g);
}

const char * injected_name(ggml_backend_t) { return "test-noise-injection"; }
void injected_free(ggml_backend_t b) {
    delete static_cast<NoiseInjector *>(b->context);
    delete b;
}

ggml_backend_t wrap_with_noise(ggml_backend_t real, NoiseInjector * inj) {
    ggml_backend_t w = new ggml_backend{};
    w->guid = real->guid;
    w->iface = real->iface;
    w->device = real->device;
    w->context = inj;
    w->iface.graph_compute = injected_compute;
    w->iface.get_name = injected_name;
    w->iface.free = injected_free;
    return w;
}

// Session-owned {N,1} F32 tensor loaded from the dump.
struct OwnedTensor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    struct ggml_tensor * t = nullptr;

    ~OwnedTensor() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
    OwnedTensor() = default;
    OwnedTensor(const OwnedTensor &) = delete;
    OwnedTensor & operator=(const OwnedTensor &) = delete;

    bool init(ggml_backend_t b, const char * name, const std::vector<float> & v, int64_t n) {
        if (v.size() != static_cast<size_t>(n)) return false;
        struct ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
        ctx = ggml_init(p);
        if (!ctx) return false;
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
        if (!t) return false;
        ggml_set_name(t, name);
        buf = ggml_backend_alloc_ctx_tensors(ctx, b);
        if (!buf) return false;
        ggml_backend_tensor_set(t, v.data(), 0, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class SynthesisIntegration : public ::testing::Test {
protected:
    void SetUp() override {
        if (CaseDirectory().empty()) GTEST_SKIP() << "PIPELINE_REFERENCE_CASE is empty";
        ASSERT_TRUE(std::filesystem::is_directory(CaseDirectory()))
            << "missing reference case directory: " << CaseDirectory();
        try {
            case_ = ParseCase(CaseDirectory());
        } catch (const std::exception & e) {
            FAIL() << "invalid reference case: " << e.what();
        }
        for (const std::string & n : {"manifest.json", "ref.wav", "ref_frontend.json",
                                      "target_frontend.json", "target_phones.npy",
                                      "target_bert.npy", "generated_codes.npy", "ge.npy",
                                      "ge_enc.npy", "noise_scaled.npy", "wav_raw.npy"}) {
            ASSERT_TRUE(std::filesystem::is_regular_file(case_.dir + n))
                << "required case file is missing: " << case_.dir + n;
        }
    }

    CaseInfo case_;
};

// ---------------------------------------------------------------------------
// Test 1: fixed dumped codes/phones/ge/ge_enc/noise -> full decode
// ---------------------------------------------------------------------------

struct SovitsRig {
    ggml_backend_t real = nullptr, wrapped = nullptr;
    NoiseInjector * inj = nullptr;
    gpt_sovits::sovits_model model{};
    gpt_sovits::sovits_session session{};

    ~SovitsRig() {
        gpt_sovits::sovits_session_free(session);
        gpt_sovits::sovits_model_free(model);
        if (wrapped) ggml_backend_free(wrapped);
        if (real) ggml_backend_free(real);
    }
};

TEST_F(SynthesisIntegration, FixedCodesDecodeMatchesDump) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModels.sovits))
        << "required local model is missing: " << kModels.sovits;

    std::string error;
    gpt_sovits::text_frontend_result target_text;
    ASSERT_TRUE(gpt_sovits::text_frontend_read_json(
        case_.dir + "target_frontend.json", target_text, &error)) << error;

    std::vector<int32_t> phones, codes;
    NpyF32 ge, ge_enc, noise, wav_dump;
    try {
        phones = LoadI32(case_.dir + "target_phones.npy");
        codes = LoadI32(case_.dir + "generated_codes.npy");
        ge = LoadF32(case_.dir + "ge.npy");
        ge_enc = LoadF32(case_.dir + "ge_enc.npy");
        noise = LoadF32(case_.dir + "noise_scaled.npy");
        wav_dump = LoadF32(case_.dir + "wav_raw.npy");
    } catch (const std::exception & e) { FAIL() << e.what(); }

    ASSERT_EQ(phones, target_text.phoneme_ids);
    ASSERT_EQ(codes.size(), static_cast<size_t>(case_.T_codes));
    ASSERT_EQ(phones.size(), static_cast<size_t>(case_.n_phones));
    ASSERT_EQ(ge.data.size(), static_cast<size_t>(kGe));
    ASSERT_EQ(ge_enc.data.size(), static_cast<size_t>(kGeEnc));
    const int64_t T_ssl = 2 * case_.T_codes, wav_len = T_ssl * kFrameMul;
    ASSERT_EQ(noise.data.size(), static_cast<size_t>(kInter * T_ssl));
    ASSERT_EQ(wav_dump.data.size(), static_cast<size_t>(wav_len));
    for (int32_t c : codes) { ASSERT_GE(c, 0); ASSERT_LT(c, kCodeVocab); }

    std::vector<float> noise_ct;
    try { noise_ct = PackCT(noise.data, kInter, T_ssl); }
    catch (const std::exception & e) { FAIL() << e.what(); }

    SovitsRig rig;
    rig.real = ggml_backend_cpu_init();
    ASSERT_NE(rig.real, nullptr);
    ASSERT_TRUE(gpt_sovits::sovits_model_load(kModels.sovits, rig.model, rig.real));
    ASSERT_TRUE(rig.model.hparams.is_v2pro);
    ASSERT_TRUE(rig.model.hparams.semantic_frame_rate_25hz);
    ASSERT_EQ(rig.model.hparams.gin_channels, kGe);

    rig.inj = new NoiseInjector{rig.real, noise_ct.data(), noise_ct.size()};
    rig.wrapped = wrap_with_noise(rig.real, rig.inj);
    ASSERT_TRUE(gpt_sovits::sovits_session_init(
        rig.session, rig.wrapped, static_cast<float>(case_.noise_scale), 7u));

    OwnedTensor ge_t, enc_t;
    ASSERT_TRUE(ge_t.init(rig.wrapped, "ge", ge.data, kGe));
    ASSERT_TRUE(enc_t.init(rig.wrapped, "ge_enc", ge_enc.data, kGeEnc));
    rig.session.ctx_ge = ge_t.ctx;
    rig.session.buf_ge = ge_t.buf;
    rig.session.ge = ge_t.t;
    ge_t.ctx = nullptr; ge_t.buf = nullptr; ge_t.t = nullptr;
    rig.session.ctx_ge_enc = enc_t.ctx;
    rig.session.buf_ge_enc = enc_t.buf;
    rig.session.ge_enc = enc_t.t;
    enc_t.ctx = nullptr; enc_t.buf = nullptr; enc_t.t = nullptr;

    constexpr size_t kGuard = 8;
    constexpr float kSentinel = 12345.0f;
    std::vector<float> wav(static_cast<size_t>(wav_len) + kGuard,
                           std::numeric_limits<float>::quiet_NaN());
    std::fill(wav.begin() + wav_len, wav.end(), kSentinel);
    ASSERT_TRUE(gpt_sovits::sovits_session_forward(
        rig.session, rig.model, codes.data(), case_.T_codes, phones.data(), case_.n_phones,
        wav.data(), static_cast<int64_t>(wav.size())));
    EXPECT_EQ(rig.inj->calls, 1);
    EXPECT_EQ(rig.inj->uploads, 1);
    EXPECT_TRUE(rig.inj->shape_ok) << "graph input 'randn' was not overridden";
    for (size_t i = wav_len; i < wav.size(); ++i) {
        EXPECT_EQ(wav[i], kSentinel) << "output guard overwritten at " << i;
    }

    const std::vector<float> body(wav.begin(), wav.begin() + wav_len);
    const Metrics m = Compare(body, wav_dump.data);
    const Range r = Analyze(body);
    std::printf("[synthesis integration] fixed decode: n=%zu max_abs=%.6g rmse=%.6g "
                "corr=%.6f range=[%g, %g]\n", m.n, m.max_abs, m.rmse, m.correlation,
                r.min, r.max);
    EXPECT_FALSE(m.nonfinite);
    EXPECT_TRUE(r.finite);
    EXPECT_LT(r.min, 0.0);
    EXPECT_GT(r.max, 0.0);
    EXPECT_GT(r.max - r.min, 0.1) << "output is degenerate";
    // Measured 2026-10-26 CPU Debug/F32: max_abs=1.45e-2, rmse=1.95e-3,
    // corr=0.999923. Limits sit just below/above the single-block dec parity
    // budgets (3e-2 max_abs, 8e-3 RMSE; corr 0.999 ~= 4.5% error RMS), so a
    // wrong block still fails without relaxing to the composed worst case.
    EXPECT_LT(m.max_abs, 3.0e-2);
    EXPECT_LT(m.rmse, 5.0e-3);
    EXPECT_GT(m.correlation, 0.999);
}

// ---------------------------------------------------------------------------
// Test 2: public synthesize_cpu from ref.wav + frontend JSONs
// ---------------------------------------------------------------------------

struct RequestHook {
    ggml_backend_t real = nullptr;
    bool fail_generation = false;
    bool fail_waveform = false;
    bool saw_failure = false;
    int sampler_calls = 0;
};

enum ggml_status request_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * hook = static_cast<RequestHook *>(backend->context);
    const bool sampler = ggml_graph_get_tensor(graph, "sampled") != nullptr;
    const bool waveform = ggml_graph_get_tensor(graph, "wav") != nullptr;
    if (sampler) ++hook->sampler_calls;
    if ((sampler && hook->fail_generation) || (waveform && hook->fail_waveform)) {
        hook->saw_failure = true;
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(hook->real, graph);
}

void request_backend_free(ggml_backend_t backend) {
    delete static_cast<RequestHook *>(backend->context);
    delete backend;
}

struct ModelsRig {
    ggml_backend_t real = nullptr;
    ggml_backend_t backend = nullptr;
    RequestHook * hook = nullptr;
    gpt_sovits::hubert_model hubert{};
    gpt_sovits::roberta_model roberta{};
    gpt_sovits::t2s_model t2s{};
    gpt_sovits::sovits_model sovits{};
    gpt_sovits::sv_model sv{};

    ~ModelsRig() {
        gpt_sovits::sv_model_free(sv);
        gpt_sovits::sovits_model_free(sovits);
        gpt_sovits::t2s_model_free(t2s);
        gpt_sovits::roberta_model_free(roberta);
        gpt_sovits::hubert_model_free(hubert);
        if (backend) ggml_backend_free(backend);
        if (real) ggml_backend_free(real);
    }
};

TEST_F(SynthesisIntegration, PublicSynthesizeCpuEndToEnd) {
    for (const std::string & p : {kModels.hubert, kModels.roberta, kModels.t2s,
                                  kModels.sovits, kModels.sv}) {
        ASSERT_TRUE(std::filesystem::is_regular_file(p))
            << "required local model is missing: " << p;
    }

    ModelsRig rig;
    rig.real = ggml_backend_cpu_init();
    ASSERT_NE(rig.real, nullptr);
    rig.hook = new RequestHook{rig.real};
    rig.backend = new ggml_backend{rig.real->guid, rig.real->iface, rig.real->device, rig.hook};
    rig.backend->iface.graph_compute = request_compute;
    rig.backend->iface.free = request_backend_free;
    ASSERT_TRUE(gpt_sovits::hubert_model_load(kModels.hubert, rig.hubert, rig.backend));
    ASSERT_TRUE(gpt_sovits::roberta_model_load(kModels.roberta, rig.roberta, rig.backend));
    ASSERT_TRUE(gpt_sovits::t2s_model_load(kModels.t2s, rig.t2s, rig.backend));
    ASSERT_TRUE(gpt_sovits::sovits_model_load(kModels.sovits, rig.sovits, rig.backend));
    ASSERT_TRUE(gpt_sovits::sv_model_load(kModels.sv, rig.sv, rig.backend));
    ASSERT_TRUE(rig.sovits.hparams.is_v2pro);
    ASSERT_TRUE(rig.sovits.hparams.semantic_frame_rate_25hz);
    ASSERT_EQ(rig.sovits.hparams.gin_channels, kGe);

    std::vector<float> pcm;
    int sample_rate = 0;
    ASSERT_TRUE(gpt_sovits::audio_read_file(case_.dir + "ref.wav", pcm, sample_rate));
    EXPECT_EQ(sample_rate, case_.source_sample_rate);
    ASSERT_FALSE(pcm.empty());

    std::string error;
    gpt_sovits::text_frontend_result ref_text, target_text;
    ASSERT_TRUE(gpt_sovits::text_frontend_read_json(
        case_.dir + "ref_frontend.json", ref_text, &error)) << error;
    ASSERT_TRUE(gpt_sovits::text_frontend_read_json(
        case_.dir + "target_frontend.json", target_text, &error)) << error;

    gpt_sovits::synthesis_models models{
        &rig.hubert, &rig.roberta, &rig.t2s, &rig.sovits, &rig.sv};
    gpt_sovits::synthesis_config config;
    config.sampler.top_k = 1;
    config.generate.max_new_tokens = 100;
    config.generate.seed = 42;
    config.noise_scale = static_cast<float>(case_.noise_scale);
    config.noise_seed = 7;

    gpt_sovits::synthesis_result result;
    const std::vector<float> sentinel = {0.25f, -0.5f};
    result.samples = sentinel;
    result.sample_rate = 123;
    result.codes = {7};
    result.stop_reason = gpt_sovits::t2s_stop_reason::eos;
    auto expect_sentinel = [&]() {
        EXPECT_EQ(result.samples, sentinel);
        EXPECT_EQ(result.sample_rate, 123);
        EXPECT_EQ(result.codes, std::vector<int32_t>{7});
        EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::eos);
    };

    const float * pcm_data = pcm.data();
    const int64_t n_pcm = static_cast<int64_t>(pcm.size());
    auto call = [&](const gpt_sovits::synthesis_models & m,
                    gpt_sovits::synthesis_result & r,
                    const gpt_sovits::synthesis_config & c) {
        return gpt_sovits::synthesize_cpu(m, pcm_data, n_pcm, sample_rate,
                                          ref_text, target_text, r, c, &error);
    };

    // Preflight failures (missing model, invalid config, over-budget request);
    // all must preserve result and leave the models usable for the run below.
    {
        gpt_sovits::synthesis_models missing = models;
        missing.sovits = nullptr;
        EXPECT_FALSE(call(missing, result, config));
        EXPECT_FALSE(error.empty());
        expect_sentinel();
    }
    {
        gpt_sovits::synthesis_config bad = config;
        bad.generate.max_new_tokens = 0;
        EXPECT_FALSE(call(models, result, bad));
        EXPECT_FALSE(error.empty());
        expect_sentinel();
    }
    {
        gpt_sovits::synthesis_config bad = config;
        bad.generate.max_new_tokens = 1024;  // valid config, N+max-1 > 1024
        EXPECT_FALSE(call(models, result, bad));
        EXPECT_FALSE(error.empty());
        expect_sentinel();
    }
    {
        // One mismatched codebook element must reject pairing before inference;
        // restore it immediately so the following run uses the real models.
        struct ggml_tensor * t2s_book = rig.t2s.weights.extract_latent.codebook;
        struct ggml_tensor * sovits_book = rig.sovits.quantizer.codebook;
        ASSERT_NE(t2s_book, nullptr);
        ASSERT_NE(sovits_book, nullptr);
        float original = 0.0f;
        ggml_fp16_t paired = 0;
        ggml_backend_tensor_get(t2s_book, &original, 0, sizeof(float));
        ggml_backend_tensor_get(sovits_book, &paired, 0, sizeof(ggml_fp16_t));
        float changed = 1.0f;
        if (ggml_fp32_to_fp16(changed) == paired) changed = -1.0f;
        if (ggml_fp32_to_fp16(changed) == paired) changed = 0.5f;
        ASSERT_NE(ggml_fp32_to_fp16(changed), paired);
        ggml_backend_tensor_set(t2s_book, &changed, 0, sizeof(float));
        EXPECT_FALSE(call(models, result, config));
        EXPECT_FALSE(error.empty());
        ggml_backend_tensor_set(t2s_book, &original, 0, sizeof(float));
        expect_sentinel();
    }

    gpt_sovits::synthesis_config short_config = config;
    short_config.generate.max_new_tokens = 1;
    rig.hook->fail_generation = true;
    EXPECT_FALSE(call(models, result, short_config));
    EXPECT_TRUE(rig.hook->saw_failure);
    expect_sentinel();
    rig.hook->fail_generation = false;
    rig.hook->saw_failure = false;
    rig.hook->fail_waveform = true;
    EXPECT_FALSE(call(models, result, short_config));
    EXPECT_TRUE(rig.hook->saw_failure);
    expect_sentinel();
    rig.hook->fail_waveform = false;
    const int calls_before = rig.hook->sampler_calls;
    gpt_sovits::synthesis_result truncated;
    ASSERT_TRUE(call(models, truncated, short_config)) << error;
    ASSERT_EQ(truncated.codes.size(), 1u);
    EXPECT_EQ(truncated.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
    EXPECT_EQ(truncated.samples.size(), 1280u);
    EXPECT_EQ(rig.hook->sampler_calls - calls_before, 1);
    expect_sentinel();
    gpt_sovits::synthesis_result repeated;
    ASSERT_TRUE(call(models, repeated, short_config)) << error;
    EXPECT_EQ(repeated.codes, truncated.codes);
    EXPECT_EQ(repeated.samples, truncated.samples);

    error = "stale";
    ASSERT_TRUE(call(models, result, config)) << error;
    EXPECT_TRUE(error.empty());

    ASSERT_FALSE(result.codes.empty()) << "immediate EOS is not successful synthesis";
    EXPECT_EQ(result.sample_rate, 32000);
    EXPECT_EQ(result.samples.size(), result.codes.size() * 1280u);
    EXPECT_LE(result.codes.size(), static_cast<size_t>(config.generate.max_new_tokens));
    for (int32_t c : result.codes) { EXPECT_GE(c, 0); EXPECT_LT(c, kCodeVocab); }
    if (result.stop_reason == gpt_sovits::t2s_stop_reason::eos) {
        EXPECT_LT(result.codes.size(), static_cast<size_t>(config.generate.max_new_tokens));
    } else {
        EXPECT_EQ(result.stop_reason, gpt_sovits::t2s_stop_reason::max_new_tokens);
        EXPECT_EQ(result.codes.size(), static_cast<size_t>(config.generate.max_new_tokens));
    }

    const Range r = Analyze(result.samples);
    std::printf("[synthesis integration] synthesize_cpu: codes=%zu stop=%d samples=%zu "
                "range=[%g, %g] rms=%g\n", result.codes.size(),
                static_cast<int>(result.stop_reason), result.samples.size(),
                r.min, r.max, r.rms);
    EXPECT_TRUE(r.finite) << "synthesized samples are not finite";
    EXPECT_LT(r.min, 0.0);
    EXPECT_GT(r.max, 0.0);
    EXPECT_GT(r.max - r.min, 1.0e-3) << "waveform is degenerate";
    EXPECT_GT(r.rms, 1.0e-8) << "waveform is silent";

    // Failure after success must preserve the successful result.
    const gpt_sovits::synthesis_result saved = result;
    {
        gpt_sovits::synthesis_config bad = config;
        bad.generate.max_new_tokens = 0;
        EXPECT_FALSE(call(models, result, bad));
        EXPECT_FALSE(error.empty());
        EXPECT_TRUE(result.samples == saved.samples) << "failure modified the waveform";
        EXPECT_EQ(result.codes, saved.codes);
        EXPECT_EQ(result.sample_rate, saved.sample_rate);
        EXPECT_EQ(result.stop_reason, saved.stop_reason);
    }
}

} // namespace
