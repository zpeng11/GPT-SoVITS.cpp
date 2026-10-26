// Reference-pipeline integration tests against one recorded upstream CPU case.
//
// Test 1 feeds the exact dumped component inputs through the individual C++
// entry points. Test 2 runs ref.wav -> prepare_reference -> encode_reference,
// validates the cached shapes, compares against a manual compute_ref_emb +
// compute_ge_audio composition on the same prepared features, and checks that a
// repeated encode is rejected without changing either cache. Resampler
// differences are printed as diagnostics only; no generation or decode runs.

#include <gtest/gtest.h>

#include "gpt_sovits/audio.h"
#include "gpt_sovits/hubert.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"
#include "gpt_sovits/text_frontend.h"
#include "reference_internal.h"

#include "cnpy.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr int64_t kBertDim = 1024;
constexpr int64_t kHubertDim = 768;
constexpr int64_t kSpecTake = 704;

std::string JoinPath(const std::string & directory, const std::string & name) {
    if (directory.empty()) return name;
    std::string out = directory;
    if (out.back() != '/' && out.back() != '\\') out.push_back('/');
    return out + name;
}

std::string CaseDirectory() {
    std::string directory = PIPELINE_REFERENCE_CASE;
    if (!directory.empty() && directory.back() != '/') directory.push_back('/');
    return directory;
}

struct ModelPaths {
    std::string hubert;
    std::string roberta;
    std::string t2s;
    std::string sovits;
    std::string sv;
};

const ModelPaths kModels = {
    JoinPath(HUBERT_TEST_DIR, "models/chinese-hubert-base-f32.gguf"),
    JoinPath(ROBERTA_MODEL_DIR, "chinese-roberta-wwm-ext-large-f16.gguf"),
    JoinPath(T2S_TEST_DIR, "models/s1v3-s2Gv2Pro-f16.gguf"),
    JoinPath(SOVITS_TEST_DIR, "models/v2pro-sovits-f16.gguf"),
    JoinPath(SOVITS_TEST_DIR, "models/sv-eres2netv2-f32.gguf"),
};

// ---------------------------------------------------------------------------
// Manifest-driven case description
// ---------------------------------------------------------------------------

struct CaseInfo {
    std::string dir;
    int64_t T_ref = 0;
    int64_t T_hub = 0;
    int64_t T_prompt = 0;
    int64_t T_spec = 0;
    int64_t n_phones = 0;
    int64_t n_sv_audio = 0;
    int64_t n_hubert_input = 0;
    int64_t n_audio32k = 0;
    int64_t silence_samples = 0;
    int64_t spec_file_channels = 0;
    int64_t sv_dim = 0;
    int64_t ge_channels = 0;
    int64_t ge_enc_channels = 0;
    int source_sample_rate = 0;
};

std::string ReadTextFile(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

CaseInfo ParseCase(const std::string & dir) {
    const Json manifest = Json::parse(ReadTextFile(dir + "manifest.json"));
    if (manifest.at("status") != "complete") throw std::runtime_error("case status is not complete");
    if (manifest.at("schema") != "gpt-sovits-inference-dump/v1") {
        throw std::runtime_error("unexpected dump schema");
    }
    if (manifest.at("parameters").at("version") != "v2Pro") {
        throw std::runtime_error("case version is not v2Pro");
    }
    if (manifest.at("sovits_decode").at("semantic_frame_rate") != "25hz") {
        throw std::runtime_error("case is not a 25 Hz SoVITS model");
    }
    if (manifest.at("hubert_audio").at("sample_rate").get<int>() != 16000) {
        throw std::runtime_error("case HuBERT sample rate is not 16 kHz");
    }

    const Json & arrays = manifest.at("arrays");
    auto shape = [&](const char * name, const char * dtype) {
        const Json & array = arrays.at(name);
        if (array.at("dtype") != dtype) {
            throw std::runtime_error(std::string("unexpected dtype for ") + name);
        }
        return array.at("shape").get<std::vector<int64_t>>();
    };
    auto require = [](bool condition, const char * message) {
        if (!condition) throw std::runtime_error(message);
    };

    const auto phones = shape("ref_phones.npy", "int64");
    const auto bert = shape("ref_bert.npy", "float32");
    const auto hubert = shape("ref_hubert.npy", "float32");
    const auto prompt = shape("ref_prompt_codes.npy", "int64");
    const auto spec = shape("ref_spec.npy", "float32");
    const auto sv_audio = shape("ref_sv_audio16k.npy", "float32");
    const auto hubert_input = shape("ref_hubert_input16k.npy", "float32");
    const auto audio32 = shape("ref_audio32k.npy", "float32");
    const auto sv_emb = shape("ref_sv_emb.npy", "float32");
    const auto ge = shape("ge.npy", "float32");
    const auto ge_enc = shape("ge_enc.npy", "float32");

    require(phones.size() == 1, "ref_phones must be rank 1");
    require(bert.size() == 2 && bert[0] == kBertDim, "ref_bert must be {1024, T_ref}");
    require(hubert.size() == 3 && hubert[0] == 1 && hubert[1] == kHubertDim,
            "ref_hubert must be {1, 768, T_hub}");
    require(prompt.size() == 3 && prompt[0] == 1 && prompt[1] == 1,
            "ref_prompt_codes must be {1, 1, T_prompt}");
    require(spec.size() == 3 && spec[0] == 1 && spec[1] >= kSpecTake,
            "ref_spec must be {1, >=704, T_spec}");
    require(sv_audio.size() == 2 && sv_audio[0] == 1, "ref_sv_audio16k must be {1, N}");
    require(hubert_input.size() == 2 && hubert_input[0] == 1,
            "ref_hubert_input16k must be {1, N}");
    require(audio32.size() == 2 && audio32[0] == 1, "ref_audio32k must be {1, N}");
    require(sv_emb.size() == 2 && sv_emb[0] == 1, "ref_sv_emb must be {1, 20480}");
    require(ge.size() == 3 && ge[0] == 1 && ge[2] == 1, "ge must be {1, C, 1}");
    require(ge_enc.size() == 3 && ge_enc[0] == 1 && ge_enc[2] == 1,
            "ge_enc must be {1, C, 1}");

    CaseInfo c;
    c.dir = dir;
    c.n_phones = phones[0];
    c.T_ref = bert[1];
    c.T_hub = hubert[2];
    c.T_prompt = prompt[2];
    c.T_spec = spec[2];
    c.n_sv_audio = sv_audio[1];
    c.n_hubert_input = hubert_input[1];
    c.n_audio32k = audio32[1];
    c.silence_samples = manifest.at("hubert_audio").at("silence_samples").get<int64_t>();
    c.spec_file_channels = spec[1];
    c.sv_dim = sv_emb[1];
    c.ge_channels = ge[1];
    c.ge_enc_channels = ge_enc[1];
    c.source_sample_rate = manifest.at("reference_audio").at("source_sample_rate").get<int>();

    require(c.T_ref > 0 && c.T_ref == c.n_phones, "T_ref must equal the phone count");
    require(c.T_hub > 0 && c.T_prompt > 0 && c.T_spec > 0, "reference lengths must be positive");
    require(c.T_prompt == c.T_hub / 2, "T_prompt must be T_hub / 2");
    require(c.n_hubert_input == c.n_sv_audio + c.silence_samples,
            "HuBERT input length must equal 16 kHz audio plus silence");
    require(gpt_sovits::hubert_output_frames(c.n_hubert_input) == c.T_hub,
            "HuBERT frame count disagrees with its input length");
    require(c.sv_dim > 0 && c.ge_channels > 0 && c.ge_enc_channels > 0,
            "embedding dimensions must be positive");
    return c;
}

// ---------------------------------------------------------------------------
// .npy loading and layout conversion
// ---------------------------------------------------------------------------

struct NpyF32 {
    std::vector<float> data;
    std::vector<size_t> shape;
};

bool LoadNpyF32(const std::string & path, NpyF32 & out, std::string * error) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        if (arr.word_size != sizeof(float) || arr.fortran_order || arr.num_vals == 0) {
            if (error) *error = "expected non-empty C-order float32: " + path;
            return false;
        }
        out.shape = arr.shape;
        out.data.assign(arr.data<float>(), arr.data<float>() + arr.num_vals);
        return true;
    } catch (const std::exception & e) {
        if (error) *error = std::string("cannot load ") + path + ": " + e.what();
        return false;
    }
}

std::vector<int32_t> LoadNpyI32(const std::string & path, bool * ok, std::string * error) {
    *ok = false;
    try {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        if (arr.fortran_order || arr.num_vals == 0) {
            if (error) *error = "expected non-empty C-order integer array: " + path;
            return {};
        }
        if (arr.word_size == sizeof(int32_t)) {
            const int32_t * src = arr.data<int32_t>();
            *ok = true;
            return std::vector<int32_t>(src, src + arr.num_vals);
        }
        if (arr.word_size == sizeof(int64_t)) {
            const int64_t * src = arr.data<int64_t>();
            std::vector<int32_t> out(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; ++i) {
                if (src[i] < std::numeric_limits<int32_t>::min() ||
                    src[i] > std::numeric_limits<int32_t>::max()) {
                    if (error) *error = "int64 value out of int32 range: " + path;
                    return {};
                }
                out[i] = static_cast<int32_t>(src[i]);
            }
            *ok = true;
            return out;
        }
        if (error) *error = "unsupported integer array in " + path;
        return {};
    } catch (const std::exception & e) {
        if (error) *error = std::string("cannot load ") + path + ": " + e.what();
        return {};
    }
}

// PyTorch [C, T] / [1, C, T] C-order -> channel-fast {take, T}.
std::vector<float> PackChannelTime(const std::vector<float> & raw,
                                   int64_t file_channels, int64_t time, int64_t take) {
    if (file_channels <= 0 || time <= 0 || take <= 0 || take > file_channels ||
        raw.size() != static_cast<size_t>(file_channels * time)) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(take * time));
    for (int64_t t = 0; t < time; ++t) {
        for (int64_t c = 0; c < take; ++c) {
            out[static_cast<size_t>(t * take + c)] =
                raw[static_cast<size_t>(c * time + t)];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

struct ErrorStats {
    double max_abs = 0.0;
    double rmse = 0.0;
    double mean_abs = 0.0;
    bool size_mismatch = false;
    bool nonfinite = false;
};

ErrorStats ComputeErrors(const std::vector<float> & actual,
                         const std::vector<float> & expected) {
    const double inf = std::numeric_limits<double>::infinity();
    ErrorStats s;
    if (actual.size() != expected.size() || actual.empty()) {
        s.size_mismatch = true;
        s.max_abs = s.rmse = s.mean_abs = inf;
        return s;
    }
    double sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            s.nonfinite = true;
            continue;
        }
        const double diff = std::abs(a - e);
        s.max_abs = std::max(s.max_abs, diff);
        sum_sq += diff * diff;
        sum_abs += diff;
    }
    s.rmse = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    s.mean_abs = sum_abs / static_cast<double>(actual.size());
    if (s.nonfinite) s.max_abs = s.rmse = s.mean_abs = inf;
    return s;
}

ErrorStats ReportErrors(const char * label,
                        const std::vector<float> & actual,
                        const std::vector<float> & expected) {
    const ErrorStats s = ComputeErrors(actual, expected);
    std::printf("[reference] %s: n=%zu max_abs=%.6g rmse=%.6g mean_abs=%.6g%s%s\n",
                label, actual.size(), s.max_abs, s.rmse, s.mean_abs,
                s.size_mismatch ? " SIZE_MISMATCH" : "",
                s.nonfinite ? " NONFINITE" : "");
    return s;
}

bool AllFinite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(),
                       [](float v) { return std::isfinite(v); });
}

std::vector<float> ReadTensorF32(struct ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32) return {};
    const size_t n = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> out(n, 0.0f);
    ggml_backend_tensor_get(tensor, out.data(), 0, n * sizeof(float));
    return out;
}

std::vector<float> ReadTensorAnyF32(struct ggml_tensor * tensor) {
    if (!tensor) return {};
    const size_t n = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> out(n, 0.0f);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out.data(), 0, n * sizeof(float));
        return out;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(tensor, raw.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
        return out;
    }
    return {};
}

// ---------------------------------------------------------------------------
// RAII resources: sessions -> models -> backend
// ---------------------------------------------------------------------------

struct ReferenceResources {
    ggml_backend_t backend = nullptr;
    gpt_sovits::hubert_model hubert{};
    gpt_sovits::roberta_model roberta{};
    gpt_sovits::t2s_model t2s{};
    gpt_sovits::sovits_model sovits{};
    gpt_sovits::sv_model sv{};
    gpt_sovits::t2s_session t2s_state{};
    gpt_sovits::sovits_session sovits_state{};
    gpt_sovits::t2s_session t2s_manual{};
    gpt_sovits::sovits_session sovits_manual{};

    ReferenceResources() = default;
    ReferenceResources(const ReferenceResources &) = delete;
    ReferenceResources & operator=(const ReferenceResources &) = delete;

    ~ReferenceResources() {
        gpt_sovits::t2s_session_free(t2s_manual);
        gpt_sovits::sovits_session_free(sovits_manual);
        gpt_sovits::t2s_session_free(t2s_state);
        gpt_sovits::sovits_session_free(sovits_state);
        gpt_sovits::sv_model_free(sv);
        gpt_sovits::sovits_model_free(sovits);
        gpt_sovits::t2s_model_free(t2s);
        gpt_sovits::roberta_model_free(roberta);
        gpt_sovits::hubert_model_free(hubert);
        if (backend) ggml_backend_free(backend);
    }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class ReferenceIntegration : public ::testing::Test {
protected:
    void SetUp() override {
        if (CaseDirectory().empty()) {
            GTEST_SKIP() << "PIPELINE_REFERENCE_CASE is empty";
        }
        ASSERT_TRUE(std::filesystem::is_directory(CaseDirectory()))
            << "missing reference case directory: " << CaseDirectory();
        try {
            case_ = ParseCase(CaseDirectory());
        } catch (const std::exception & e) {
            FAIL() << "invalid reference case: " << e.what();
        }

        for (const std::string & name : {"manifest.json", "ref.wav", "ref_frontend.json",
                                         "ref_phones.npy", "ref_bert.npy", "ref_hubert.npy",
                                         "ref_hubert_input16k.npy", "ref_spec.npy",
                                         "ref_sv_audio16k.npy", "ref_sv_emb.npy",
                                         "ref_audio32k.npy", "ge.npy", "ge_enc.npy"}) {
            ASSERT_TRUE(std::filesystem::is_regular_file(case_.dir + name))
                << "required case file is missing: " << case_.dir + name;
        }
        for (const std::string & path : {kModels.hubert, kModels.roberta, kModels.t2s,
                                         kModels.sovits, kModels.sv}) {
            ASSERT_TRUE(std::filesystem::is_regular_file(path))
                << "required local model is missing: " << path;
        }

        resources_.backend = ggml_backend_cpu_init();
        ASSERT_NE(resources_.backend, nullptr);
        ASSERT_TRUE(gpt_sovits::hubert_model_load(kModels.hubert, resources_.hubert,
                                                 resources_.backend));
        ASSERT_TRUE(gpt_sovits::roberta_model_load(kModels.roberta, resources_.roberta,
                                                  resources_.backend));
        ASSERT_TRUE(gpt_sovits::sovits_model_load(kModels.sovits, resources_.sovits,
                                                 resources_.backend));
        ASSERT_TRUE(gpt_sovits::sv_model_load(kModels.sv, resources_.sv, resources_.backend));

        ASSERT_TRUE(resources_.sovits.hparams.is_v2pro);
        ASSERT_TRUE(resources_.sovits.hparams.semantic_frame_rate_25hz);
        EXPECT_EQ(resources_.sovits.hparams.gin_channels, case_.ge_channels);
        EXPECT_EQ(resources_.sv.hparams.out_dim, case_.sv_dim);
        EXPECT_EQ(case_.ge_enc_channels, int64_t{512});
    }

    ReferenceResources resources_;
    CaseInfo case_;
};

// ---------------------------------------------------------------------------
// Test 1: dumped component inputs
// ---------------------------------------------------------------------------

TEST_F(ReferenceIntegration, DumpComponentsMatchReference) {
    std::string error;
    std::string npy_error;
    bool ok = false;

    // Frontend phones + BERT.
    gpt_sovits::text_frontend_result text;
    ASSERT_TRUE(gpt_sovits::text_frontend_read_json(
        case_.dir + "ref_frontend.json", text, &error)) << error;
    const std::vector<int32_t> ref_phones =
        LoadNpyI32(case_.dir + "ref_phones.npy", &ok, &npy_error);
    ASSERT_TRUE(ok) << npy_error;
    ASSERT_EQ(ref_phones.size(), static_cast<size_t>(case_.n_phones));
    ASSERT_EQ(text.phoneme_ids.size(), ref_phones.size());
    EXPECT_EQ(text.phoneme_ids, ref_phones);

    NpyF32 ref_bert;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_bert.npy", ref_bert, &npy_error)) << npy_error;
    ASSERT_EQ(ref_bert.shape.size(), 2u);
    EXPECT_EQ(ref_bert.shape[0], static_cast<size_t>(kBertDim));
    EXPECT_EQ(ref_bert.shape[1], static_cast<size_t>(case_.T_ref));
    std::vector<float> bert;
    ASSERT_TRUE(gpt_sovits::text_frontend_build_bert_features(
        text, &resources_.roberta, bert, &error)) << error;
    ASSERT_EQ(bert.size(), static_cast<size_t>(kBertDim * case_.T_ref));
    ASSERT_TRUE(AllFinite(bert));
    const std::vector<float> expected_bert =
        PackChannelTime(ref_bert.data, kBertDim, case_.T_ref, kBertDim);
    ASSERT_EQ(expected_bert.size(), bert.size());
    const ErrorStats bert_errors = ReportErrors("ref BERT (f16 RoBERTa)", bert, expected_bert);
    EXPECT_FALSE(bert_errors.size_mismatch);
    EXPECT_FALSE(bert_errors.nonfinite);
    EXPECT_LT(bert_errors.rmse, 1.0e-2);
    for (size_t i = 0; i < bert.size(); ++i) {
        // Existing F16 TTS-extraction tolerance: abs 5e-2 + rel 5e-3.
        const double tolerance = 5.0e-2 + 5.0e-3 * std::abs(static_cast<double>(expected_bert[i]));
        EXPECT_NEAR(bert[i], expected_bert[i], tolerance)
            << "phoneme " << (i / kBertDim) << " channel " << (i % kBertDim);
    }

    // HuBERT on the recorded padded 16 kHz input.
    NpyF32 ref_hubert_input;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_hubert_input16k.npy", ref_hubert_input,
                           &npy_error)) << npy_error;
    ASSERT_EQ(ref_hubert_input.data.size(), static_cast<size_t>(case_.n_hubert_input));
    ASSERT_EQ(gpt_sovits::hubert_output_frames(case_.n_hubert_input), case_.T_hub);
    NpyF32 ref_hubert;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_hubert.npy", ref_hubert, &npy_error)) << npy_error;
    std::vector<float> hubert(static_cast<size_t>(kHubertDim * case_.T_hub), 0.0f);
    int64_t frames = -1;
    ASSERT_TRUE(gpt_sovits::hubert_extract_features_from_pcm(
        resources_.hubert, ref_hubert_input.data.data(), case_.n_hubert_input,
        hubert.data(), static_cast<int64_t>(hubert.size()), &frames));
    ASSERT_EQ(frames, case_.T_hub);
    const std::vector<float> expected_hubert =
        PackChannelTime(ref_hubert.data, kHubertDim, case_.T_hub, kHubertDim);
    ASSERT_EQ(expected_hubert.size(), hubert.size());
    const ErrorStats hubert_errors = ReportErrors("ref HuBERT (F32)", hubert, expected_hubert);
    EXPECT_FALSE(hubert_errors.size_mismatch);
    EXPECT_FALSE(hubert_errors.nonfinite);
    EXPECT_LT(hubert_errors.max_abs, 1.0e-4);
    EXPECT_LT(hubert_errors.rmse, 1.0e-5);

    // SV on the recorded 16 kHz audio.
    NpyF32 ref_sv_audio;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_sv_audio16k.npy", ref_sv_audio,
                           &npy_error)) << npy_error;
    ASSERT_EQ(ref_sv_audio.data.size(), static_cast<size_t>(case_.n_sv_audio));
    NpyF32 ref_sv_emb;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_sv_emb.npy", ref_sv_emb, &npy_error)) << npy_error;
    ASSERT_EQ(ref_sv_emb.data.size(), static_cast<size_t>(case_.sv_dim));
    std::vector<float> fbank;
    const int fbank_frames = gpt_sovits::sv_fbank(
        ref_sv_audio.data.data(), case_.n_sv_audio, fbank);
    ASSERT_GT(fbank_frames, 0);
    std::vector<float> sv_embedding(static_cast<size_t>(case_.sv_dim), 0.0f);
    ASSERT_TRUE(gpt_sovits::sv_forward3(
        resources_.sv, fbank.data(), fbank_frames, sv_embedding.data()));
    ASSERT_TRUE(AllFinite(sv_embedding));
    const ErrorStats sv_errors = ReportErrors("ref SV embedding", sv_embedding, ref_sv_emb.data);
    EXPECT_FALSE(sv_errors.size_mismatch);
    EXPECT_FALSE(sv_errors.nonfinite);
    EXPECT_LT(sv_errors.max_abs, 1.0e-2);
    EXPECT_LT(sv_errors.rmse, 3.0e-3);

    // SoVITS conditioning from the recorded spec + audio.
    NpyF32 ref_spec;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_spec.npy", ref_spec, &npy_error)) << npy_error;
    ASSERT_EQ(ref_spec.data.size(),
              static_cast<size_t>(case_.spec_file_channels * case_.T_spec));
    const std::vector<float> spec704 =
        PackChannelTime(ref_spec.data, case_.spec_file_channels, case_.T_spec, kSpecTake);
    ASSERT_EQ(spec704.size(), static_cast<size_t>(kSpecTake * case_.T_spec));

    NpyF32 ref_ge;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ge.npy", ref_ge, &npy_error)) << npy_error;
    ASSERT_EQ(ref_ge.data.size(), static_cast<size_t>(case_.ge_channels));
    NpyF32 ref_ge_enc;
    ASSERT_TRUE(LoadNpyF32(case_.dir + "ge_enc.npy", ref_ge_enc, &npy_error)) << npy_error;
    ASSERT_EQ(ref_ge_enc.data.size(), static_cast<size_t>(case_.ge_enc_channels));

    ASSERT_TRUE(gpt_sovits::sovits_session_init(
        resources_.sovits_state, resources_.backend, 0.5f, 0u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge_audio(
        resources_.sovits_state, resources_.sovits, resources_.sv,
        {{spec704.data(), case_.T_spec, ref_sv_audio.data.data(), case_.n_sv_audio}}));
    const std::vector<float> ge = ReadTensorF32(
        gpt_sovits::sovits_session_get_ge(resources_.sovits_state));
    const std::vector<float> ge_enc = ReadTensorF32(
        gpt_sovits::sovits_session_get_ge_enc(resources_.sovits_state));
    ASSERT_EQ(ge.size(), static_cast<size_t>(case_.ge_channels));
    ASSERT_EQ(ge_enc.size(), static_cast<size_t>(case_.ge_enc_channels));
    ASSERT_TRUE(AllFinite(ge));
    ASSERT_TRUE(AllFinite(ge_enc));

    const ErrorStats ge_errors = ReportErrors("ref ge (spec + SV audio)", ge, ref_ge.data);
    EXPECT_FALSE(ge_errors.size_mismatch);
    EXPECT_FALSE(ge_errors.nonfinite);
    EXPECT_LT(ge_errors.max_abs, 1.0e-1);
    EXPECT_LT(ge_errors.rmse, 2.0e-2);
    const ErrorStats ge_enc_errors = ReportErrors("ref ge_enc (ge_to512)", ge_enc,
                                                   ref_ge_enc.data);
    EXPECT_FALSE(ge_enc_errors.size_mismatch);
    EXPECT_FALSE(ge_enc_errors.nonfinite);
    EXPECT_LT(ge_enc_errors.max_abs, 8.0e-2);
    EXPECT_LT(ge_enc_errors.rmse, 3.0e-2);

    // Prompt audio embedding: the exact dumped phones/BERT/HuBERT go through
    // t2s_session_compute_ref_emb, and the cached audio embedding is compared
    // against an independent build from ref_prompt_codes.npy. A changed
    // semantic code would move a whole embedding row and fail loudly.
    ASSERT_TRUE(gpt_sovits::t2s_model_load(kModels.t2s, resources_.t2s, resources_.backend));
    const int64_t needed_slots = case_.T_ref + case_.T_prompt;
    ASSERT_LE(needed_slots, std::numeric_limits<uint32_t>::max() - 16);
    const uint32_t slot_size = static_cast<uint32_t>(needed_slots + 16);
    const gpt_sovits::t2s_sampler_config sampler{};
    ASSERT_TRUE(gpt_sovits::t2s_session_init(
        resources_.t2s_state, resources_.t2s.hparams, resources_.backend, 1, slot_size,
        sampler));
    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        resources_.t2s_state, resources_.t2s, text.phoneme_ids.data(), case_.T_ref,
        bert.data(), hubert.data(), case_.T_hub));
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_ref(resources_.t2s_state), case_.T_ref);
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_prompt(resources_.t2s_state),
              case_.T_prompt);

    const std::vector<int32_t> prompt_codes =
        LoadNpyI32(case_.dir + "ref_prompt_codes.npy", &ok, &npy_error);
    ASSERT_TRUE(ok) << npy_error;
    ASSERT_EQ(prompt_codes.size(), static_cast<size_t>(case_.T_prompt));

    struct ggml_tensor * audio_weight = resources_.t2s.weights.embed.audio_embedding;
    struct ggml_tensor * alpha_tensor = resources_.t2s.weights.embed.audio_pos_alpha;
    ASSERT_NE(audio_weight, nullptr);
    ASSERT_NE(alpha_tensor, nullptr);
    const int64_t d_model = audio_weight->ne[0];
    ASSERT_EQ(d_model, static_cast<int64_t>(resources_.t2s.hparams.embedding_dim));
    ASSERT_GE(d_model, 2);
    ASSERT_EQ(d_model % 2, 0);
    const std::vector<float> weight = ReadTensorAnyF32(audio_weight);
    const std::vector<float> alpha = ReadTensorAnyF32(alpha_tensor);
    ASSERT_EQ(weight.size(), static_cast<size_t>(d_model * audio_weight->ne[1]));
    ASSERT_FALSE(alpha.empty());

    // block.cpp add_positional_embedding:
    // channel 2j = sin(pos*freq_j), 2j+1 = cos(pos*freq_j),
    // freq_j = exp(-log(10000)*j/(d_model/2)), scaled by alpha.
    std::vector<float> expected_audio(static_cast<size_t>(d_model * case_.T_prompt));
    const int64_t half = d_model / 2;
    for (int64_t t = 0; t < case_.T_prompt; ++t) {
        const int32_t code = prompt_codes[static_cast<size_t>(t)];
        ASSERT_GE(code, 0);
        ASSERT_LT(code, audio_weight->ne[1]);
        const float * row = weight.data() + static_cast<int64_t>(code) * d_model;
        float * dst = expected_audio.data() + t * d_model;
        for (int64_t j = 0; j < half; ++j) {
            const float freq = std::exp(-std::log(10000.0f) *
                                        static_cast<float>(j) / static_cast<float>(half));
            const float arg = static_cast<float>(t) * freq;
            dst[2 * j]     = row[2 * j]     + alpha[0] * std::sin(arg);
            dst[2 * j + 1] = row[2 * j + 1] + alpha[0] * std::cos(arg);
        }
    }
    const std::vector<float> actual_audio = ReadTensorF32(
        gpt_sovits::t2s_session_get_ref_audio_emb(resources_.t2s_state));
    ASSERT_EQ(actual_audio.size(), expected_audio.size());
    const ErrorStats prompt_errors = ReportErrors(
        "ref prompt audio embedding (dump codes)", actual_audio, expected_audio);
    EXPECT_FALSE(prompt_errors.size_mismatch);
    EXPECT_FALSE(prompt_errors.nonfinite);
    EXPECT_LT(prompt_errors.max_abs, 1.0e-3);
}

// ---------------------------------------------------------------------------
// Test 2: ref.wav -> prepare_reference -> encode_reference
// ---------------------------------------------------------------------------

struct CacheSnapshot {
    struct ggml_tensor * text = nullptr;
    struct ggml_tensor * audio = nullptr;
    struct ggml_tensor * ge = nullptr;
    struct ggml_tensor * ge_enc = nullptr;
    int64_t T_ref = 0;
    int64_t T_prompt = 0;
    std::vector<float> text_values;
    std::vector<float> audio_values;
    std::vector<float> ge_values;
    std::vector<float> ge_enc_values;
};

CacheSnapshot CaptureCache(const gpt_sovits::t2s_session & ts,
                           const gpt_sovits::sovits_session & ss) {
    return {ts.ref_text_emb, ts.ref_audio_emb, ss.ge, ss.ge_enc,
            ts.ref_T_ref, ts.ref_T_prompt,
            ReadTensorF32(ts.ref_text_emb), ReadTensorF32(ts.ref_audio_emb),
            ReadTensorF32(ss.ge), ReadTensorF32(ss.ge_enc)};
}

void ExpectCacheUnchanged(const gpt_sovits::t2s_session & ts,
                          const gpt_sovits::sovits_session & ss,
                          const CacheSnapshot & snap) {
    EXPECT_EQ(ts.ref_text_emb, snap.text);
    EXPECT_EQ(ts.ref_audio_emb, snap.audio);
    EXPECT_EQ(ss.ge, snap.ge);
    EXPECT_EQ(ss.ge_enc, snap.ge_enc);
    EXPECT_EQ(ts.ref_T_ref, snap.T_ref);
    EXPECT_EQ(ts.ref_T_prompt, snap.T_prompt);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(ts.ref_text_emb), snap.text_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(ts.ref_audio_emb), snap.audio_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(ss.ge), snap.ge_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(ss.ge_enc), snap.ge_enc_values).max_abs, 0.0);
}

TEST_F(ReferenceIntegration, PrepareAndEncodeReferenceFromWav) {
    std::string error;
    std::string npy_error;
    bool ok = false;

    std::vector<float> pcm;
    int sample_rate = 0;
    ASSERT_TRUE(gpt_sovits::audio_read_file(case_.dir + "ref.wav", pcm, sample_rate));
    ASSERT_EQ(sample_rate, case_.source_sample_rate);

    gpt_sovits::text_frontend_result text;
    ASSERT_TRUE(gpt_sovits::text_frontend_read_json(
        case_.dir + "ref_frontend.json", text, &error)) << error;

    gpt_sovits::pipeline_detail::reference_features features;
    ASSERT_TRUE(gpt_sovits::pipeline_detail::prepare_reference(
        pcm.data(), static_cast<int64_t>(pcm.size()), sample_rate, text,
        resources_.hubert, &resources_.roberta, features, &error)) << error;
    EXPECT_EQ(features.T_ref, case_.T_ref);
    EXPECT_EQ(features.T_hub, case_.T_hub);
    EXPECT_EQ(features.T_prompt, case_.T_prompt);
    EXPECT_EQ(features.audio.spec_frames, case_.T_spec);
    EXPECT_EQ(features.phones.size(), static_cast<size_t>(case_.n_phones));
    EXPECT_EQ(features.bert.size(), static_cast<size_t>(kBertDim * case_.T_ref));
    EXPECT_EQ(features.hubert.size(), static_cast<size_t>(kHubertDim * case_.T_hub));
    EXPECT_EQ(features.audio.pcm16k.size(), static_cast<size_t>(case_.n_sv_audio));
    EXPECT_EQ(features.audio.spec704.size(), static_cast<size_t>(kSpecTake * case_.T_spec));
    EXPECT_EQ(features.T_prompt, features.T_hub / 2);
    EXPECT_EQ(gpt_sovits::hubert_output_frames(
                  static_cast<int64_t>(features.audio.pcm16k.size()) + case_.silence_samples),
              case_.T_hub);
    ASSERT_TRUE(gpt_sovits::pipeline_detail::validate_reference_features(features, &error))
        << error;
    ASSERT_TRUE(AllFinite(features.bert));
    ASSERT_TRUE(AllFinite(features.hubert));
    std::vector<float> manual_hubert_input = features.audio.pcm16k;
    manual_hubert_input.resize(manual_hubert_input.size() + 9600, 0.0f);
    std::vector<float> manual_hubert(features.hubert.size());
    ASSERT_TRUE(gpt_sovits::hubert_extract_features_from_pcm(
        resources_.hubert, manual_hubert_input.data(), manual_hubert_input.size(),
        manual_hubert.data(), manual_hubert.size()));
    EXPECT_EQ(features.hubert, manual_hubert)
        << "prepare_reference must use unnormalized PCM followed by exactly 9600 zeros";
    ASSERT_TRUE(AllFinite(features.audio.pcm16k));
    ASSERT_TRUE(AllFinite(features.audio.spec704));

    // Resampler diagnostics only: not parity assertions.
    {
        NpyF32 dump16;
        ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_sv_audio16k.npy", dump16, &npy_error))
            << npy_error;
        NpyF32 dump32;
        ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_audio32k.npy", dump32, &npy_error))
            << npy_error;
        std::vector<float> pcm32;
        ASSERT_TRUE(gpt_sovits::audio_resample(
            pcm.data(), static_cast<int64_t>(pcm.size()), sample_rate, 32000, pcm32));
        NpyF32 dump_spec;
        ASSERT_TRUE(LoadNpyF32(case_.dir + "ref_spec.npy", dump_spec, &npy_error))
            << npy_error;
        const std::vector<float> dump_spec704 =
            PackChannelTime(dump_spec.data, case_.spec_file_channels, case_.T_spec, kSpecTake);
        EXPECT_EQ(pcm32.size(), dump32.data.size());
        EXPECT_EQ(features.audio.pcm16k.size(), dump16.data.size());
        EXPECT_EQ(features.audio.spec704.size(), dump_spec704.size());
        (void) ReportErrors("diagnostic 16 kHz PCM (C++ vs dump)", features.audio.pcm16k,
                            dump16.data);
        (void) ReportErrors("diagnostic 32 kHz PCM (C++ vs dump)", pcm32, dump32.data);
        (void) ReportErrors("diagnostic 704-bin spec (C++ vs dump)", features.audio.spec704,
                            dump_spec704);
    }

    ASSERT_TRUE(gpt_sovits::t2s_model_load(kModels.t2s, resources_.t2s, resources_.backend));
    const int64_t needed_slots = case_.T_ref + case_.T_prompt;
    ASSERT_LE(needed_slots, std::numeric_limits<uint32_t>::max() - 16);
    const uint32_t slot_size = static_cast<uint32_t>(needed_slots + 16);
    const gpt_sovits::t2s_sampler_config sampler{};
    ASSERT_TRUE(gpt_sovits::t2s_session_init(
        resources_.t2s_state, resources_.t2s.hparams, resources_.backend, 1, slot_size,
        sampler));
    ASSERT_TRUE(gpt_sovits::sovits_session_init(
        resources_.sovits_state, resources_.backend, 0.5f, 7u));
    ASSERT_TRUE(gpt_sovits::pipeline_detail::encode_reference(
        features, resources_.t2s, resources_.sovits, resources_.sv,
        resources_.t2s_state, resources_.sovits_state, &error)) << error;

    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_ref(resources_.t2s_state), case_.T_ref);
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_prompt(resources_.t2s_state), case_.T_prompt);

    struct ggml_tensor * text_emb =
        gpt_sovits::t2s_session_get_ref_text_emb(resources_.t2s_state);
    struct ggml_tensor * audio_emb =
        gpt_sovits::t2s_session_get_ref_audio_emb(resources_.t2s_state);
    struct ggml_tensor * ge = gpt_sovits::sovits_session_get_ge(resources_.sovits_state);
    struct ggml_tensor * ge_enc =
        gpt_sovits::sovits_session_get_ge_enc(resources_.sovits_state);
    ASSERT_NE(text_emb, nullptr);
    ASSERT_NE(audio_emb, nullptr);
    ASSERT_NE(ge, nullptr);
    ASSERT_NE(ge_enc, nullptr);
    const int64_t d_model = resources_.t2s.hparams.embedding_dim;
    EXPECT_EQ(text_emb->ne[0], d_model);
    EXPECT_EQ(text_emb->ne[1], case_.T_ref);
    EXPECT_EQ(audio_emb->ne[0], d_model);
    EXPECT_EQ(audio_emb->ne[1], case_.T_prompt);
    EXPECT_EQ(ge->ne[0], case_.ge_channels);
    EXPECT_EQ(ge->ne[1], 1);
    EXPECT_EQ(ge_enc->ne[0], case_.ge_enc_channels);
    EXPECT_EQ(ge_enc->ne[1], 1);

    const std::vector<float> text_values = ReadTensorF32(text_emb);
    const std::vector<float> audio_values = ReadTensorF32(audio_emb);
    const std::vector<float> ge_values = ReadTensorF32(ge);
    const std::vector<float> ge_enc_values = ReadTensorF32(ge_enc);
    ASSERT_EQ(text_values.size(), static_cast<size_t>(d_model * case_.T_ref));
    ASSERT_EQ(audio_values.size(), static_cast<size_t>(d_model * case_.T_prompt));
    ASSERT_EQ(ge_values.size(), static_cast<size_t>(case_.ge_channels));
    ASSERT_EQ(ge_enc_values.size(), static_cast<size_t>(case_.ge_enc_channels));
    ASSERT_TRUE(AllFinite(text_values));
    ASSERT_TRUE(AllFinite(audio_values));
    ASSERT_TRUE(AllFinite(ge_values));
    ASSERT_TRUE(AllFinite(ge_enc_values));

    // Independent composition on the same prepared features.
    ASSERT_TRUE(gpt_sovits::t2s_session_init(
        resources_.t2s_manual, resources_.t2s.hparams, resources_.backend, 1, slot_size,
        sampler));
    ASSERT_TRUE(gpt_sovits::sovits_session_init(
        resources_.sovits_manual, resources_.backend, 0.5f, 7u));
    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        resources_.t2s_manual, resources_.t2s, features.phones.data(), features.T_ref,
        features.bert.data(), features.hubert.data(), features.T_hub));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge_audio(
        resources_.sovits_manual, resources_.sovits, resources_.sv,
        {{features.audio.spec704.data(), features.audio.spec_frames,
          features.audio.pcm16k.data(),
          static_cast<int64_t>(features.audio.pcm16k.size())}}));

    EXPECT_EQ(ComputeErrors(ReadTensorF32(gpt_sovits::t2s_session_get_ref_text_emb(
                  resources_.t2s_manual)), text_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(gpt_sovits::t2s_session_get_ref_audio_emb(
                  resources_.t2s_manual)), audio_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(gpt_sovits::sovits_session_get_ge(
                  resources_.sovits_manual)), ge_values).max_abs, 0.0);
    EXPECT_EQ(ComputeErrors(ReadTensorF32(gpt_sovits::sovits_session_get_ge_enc(
                  resources_.sovits_manual)), ge_enc_values).max_abs, 0.0);

    // Repeated encode is rejected and leaves both caches unchanged.
    const CacheSnapshot snapshot = CaptureCache(resources_.t2s_state,
                                                resources_.sovits_state);
    error.clear();
    EXPECT_FALSE(gpt_sovits::pipeline_detail::encode_reference(
        features, resources_.t2s, resources_.sovits, resources_.sv,
        resources_.t2s_state, resources_.sovits_state, &error));
    EXPECT_FALSE(error.empty());
    ExpectCacheUnchanged(resources_.t2s_state, resources_.sovits_state, snapshot);
}

} // namespace
