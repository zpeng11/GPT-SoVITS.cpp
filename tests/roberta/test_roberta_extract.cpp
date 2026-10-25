// tests/roberta/test_roberta_extract.cpp
//
// Tests for the host-side RoBERTa feature-extraction runner and the
// WordPiece-token -> phoneme BERT-feature expansion helper.
//
// Uses the fixtures produced by tests/roberta/generate_reference.py and the
// same F16 model/tolerances as test_roberta_parity.cpp.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "gpt_sovits/roberta.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir   = ROBERTA_MODEL_DIR;
static const std::string kModelPath = kTestDir + "chinese-roberta-wwm-ext-large-f16.gguf";

// F16 full-model parity tolerance, from test_roberta_parity.cpp.
static constexpr double kMaxAbsTol = 1e-1;
static constexpr double kRmseTol   = 1e-2;
static constexpr double kTTSAbsTol = 5e-2;
static constexpr double kTTSRelTol = 5e-3;

static constexpr int64_t kHiddenSize = 1024;

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

template<typename T>
std::vector<T> load_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        ADD_FAILURE() << "Failed to open: " << path;
        return {};
    }
    const std::streamoff size = f.tellg();
    if (size <= 0 || size % sizeof(T) != 0 ||
        size > std::numeric_limits<std::streamsize>::max()) {
        ADD_FAILURE() << "Invalid reference size: " << path;
        return {};
    }
    f.seekg(0);
    std::vector<T> data(static_cast<size_t>(size) / sizeof(T));
    if (!f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size))) {
        ADD_FAILURE() << "Incomplete reference read: " << path;
        return {};
    }
    return data;
}

std::vector<float> load_f32_bin(const std::string & path) { return load_bin<float>(path); }
std::vector<int32_t> load_i32_bin(const std::string & path) { return load_bin<int32_t>(path); }

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

// ---------------------------------------------------------------------------
// Fixture: loads the F16 model once per test
// ---------------------------------------------------------------------------

class RobertaExtract : public ::testing::Test {
protected:
    void SetUp() override {
        FILE * f = fopen(kModelPath.c_str(), "rb");
        if (!f) {
            const int error = errno;
            if (error == ENOENT) GTEST_SKIP() << "Model file not found: " << kModelPath;
            FAIL() << "Cannot open model " << kModelPath << ": " << std::strerror(error);
        }
        fclose(f);

        backend_ = CreateBackend();
        ASSERT_NE(backend_, nullptr);
        ASSERT_TRUE(gpt_sovits::roberta_model_load(kModelPath, model_, backend_));
    }
    void TearDown() override {
        gpt_sovits::roberta_model_free(model_);
        if (backend_) {
            ggml_backend_free(backend_);
        }
    }

    virtual ggml_backend_t CreateBackend() { return create_test_backend(); }

    void AssertTTSMatchesReference() {
        const auto input_ids = load_i32_bin(kTestDir + "ref_input_ids.bin");
        const auto ref = load_f32_bin(kTestDir + "ref_tts_output.bin");
        const auto final_ref = load_f32_bin(kTestDir + "ref_model_output.bin");
        ASSERT_FALSE(input_ids.empty());
        ASSERT_FALSE(ref.empty());
        ASSERT_EQ(ref.size(), input_ids.size() * kHiddenSize);
        ASSERT_EQ(final_ref.size(), ref.size());
        ASSERT_TRUE(std::all_of(ref.begin(), ref.end(), [](float v) { return std::isfinite(v); }));
        ASSERT_TRUE(std::all_of(final_ref.begin(), final_ref.end(), [](float v) { return std::isfinite(v); }));
        ASSERT_GT(compute_errors(ref, final_ref).rmse, kRmseTol * 10)
            << "TTS reference must distinguish hidden_states[-3] from last_hidden_state";

        // Borrow the loaded weights, but leave the unused final two layers absent.
        auto tts_model = model_;
        tts_model.weights.encoder.layers[22] = {};
        tts_model.weights.encoder.layers[23] = {};

        constexpr size_t kGuardSize = 8;
        constexpr float kSentinel = 12345.0f;
        std::vector<float> out(ref.size() + kGuardSize, kSentinel);
        ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(
            tts_model, input_ids.data(), static_cast<int64_t>(input_ids.size()),
            out.data(), static_cast<int64_t>(ref.size())));
        for (size_t i = ref.size(); i < out.size(); ++i) {
            EXPECT_FLOAT_EQ(out[i], kSentinel) << "Output guard overwritten at " << i;
        }
        out.resize(ref.size());
        ASSERT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }));
        const auto err = compute_errors(out, ref);
        printf("  [F16] TTS hidden_states[-3] : max_abs=%.4e  rmse=%.4e  mean_abs=%.4e\n",
               err.max_abs, err.rmse, err.mean_abs);
        for (size_t i = 0; i < ref.size(); ++i) {
            EXPECT_NEAR(out[i], ref[i], kTTSAbsTol + kTTSRelTol * std::abs(ref[i]))
                << "token " << i / kHiddenSize << " channel " << i % kHiddenSize;
        }
        EXPECT_LT(err.rmse, kRmseTol);
    }

    ggml_backend_t           backend_ = nullptr;
    gpt_sovits::roberta_model model_{};
};

class RobertaExtractCPU : public RobertaExtract {
protected:
    ggml_backend_t CreateBackend() override { return ggml_backend_cpu_init(); }
};

// ---------------------------------------------------------------------------
// 1. End-to-end runner parity against the HuggingFace reference
// ---------------------------------------------------------------------------

TEST_F(RobertaExtract, MatchesReferenceOutput) {
    auto input_ids = load_i32_bin(kTestDir + "ref_input_ids.bin");
    auto ref       = load_f32_bin(kTestDir + "ref_model_output.bin");
    ASSERT_FALSE(input_ids.empty());
    ASSERT_FALSE(ref.empty());

    const int64_t T = static_cast<int64_t>(input_ids.size());
    ASSERT_EQ(ref.size(), static_cast<size_t>(T * kHiddenSize));

    std::vector<float> out(ref.size(), 0.0f);

    ASSERT_TRUE(gpt_sovits::roberta_extract_features(
        model_, input_ids.data(), T,
        out.data(), static_cast<int64_t>(out.size())));

    auto err = compute_errors(out, ref);
    printf("  [F16] extract_features : max_abs=%.4e  rmse=%.4e  mean_abs=%.4e\n",
           err.max_abs, err.rmse, err.mean_abs);

    EXPECT_LT(err.max_abs, kMaxAbsTol) << "max abs error too large";
    EXPECT_LT(err.rmse,    kRmseTol)   << "RMSE too large";
}

TEST_F(RobertaExtract, TTSMatchesReferenceOutput) {
    ASSERT_NO_FATAL_FAILURE(AssertTTSMatchesReference());
}

TEST_F(RobertaExtractCPU, TTSMatchesReferenceOutput) {
    ASSERT_NO_FATAL_FAILURE(AssertTTSMatchesReference());
}

TEST_F(RobertaExtractCPU, RepeatedCallsAndExplicitTokenTypes) {
    const int32_t input_ids[] = {101, 1, 102};
    const int32_t zero_types[] = {0, 0, 0};
    const int32_t mixed_types[] = {0, 1, 1};
    constexpr int64_t T = 3;
    constexpr int64_t n = T * kHiddenSize;
    std::vector<float> tts(n), final(n), repeated(n), typed(n);
    ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(model_, input_ids, T, tts.data(), n));
    ASSERT_TRUE(gpt_sovits::roberta_extract_features(model_, input_ids, T, final.data(), n));
    ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(model_, input_ids, T, repeated.data(), n));
    EXPECT_EQ(tts, repeated);
    EXPECT_GT(compute_errors(tts, final).rmse, 0.1);
    ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(model_, input_ids, T, typed.data(), n, zero_types));
    for (size_t i = 0; i < tts.size(); ++i) EXPECT_NEAR(tts[i], typed[i], 1e-5);
    ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(model_, input_ids, T, typed.data(), n, mixed_types));
    ASSERT_TRUE(std::all_of(typed.begin(), typed.end(), [](float v) { return std::isfinite(v); }));
    EXPECT_GT(compute_errors(tts, typed).rmse, 1e-5);
}

TEST_F(RobertaExtractCPU, AcceptsBoundaryLengths) {
    for (int64_t T : {int64_t{1}, int64_t{512}}) {
        SCOPED_TRACE(T);
        const std::vector<int32_t> input_ids(static_cast<size_t>(T), 1);
        std::vector<float> out(static_cast<size_t>(T * kHiddenSize) + 8, 12345.0f);
        ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(
            model_, input_ids.data(), T, out.data(), T * kHiddenSize));
        EXPECT_TRUE(std::all_of(out.begin(), out.begin() + T * kHiddenSize,
                               [](float v) { return std::isfinite(v) && v != 12345.0f; }));
        EXPECT_TRUE(std::all_of(out.begin() + T * kHiddenSize, out.end(),
                               [](float v) { return v == 12345.0f; }));
    }
}

TEST(RobertaReference, RejectsMalformedFiles) {
    const auto path = std::filesystem::temp_directory_path() /
        ("roberta-reference-" + std::to_string(
            reinterpret_cast<uintptr_t>(::testing::UnitTest::GetInstance())) + ".bin");
    ASSERT_FALSE(std::filesystem::exists(path));
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } guard{path};
    const std::string name = path.string();
    for (size_t size : {size_t{0}, size_t{1}, size_t{3}, size_t{5}, size_t{7}}) {
        {
            std::ofstream f(path, std::ios::binary);
            const std::string bytes(size, 'x');
            ASSERT_TRUE(f.write(bytes.data(), static_cast<std::streamsize>(bytes.size())));
        }
        EXPECT_NONFATAL_FAILURE(EXPECT_TRUE(load_f32_bin(name).empty()), "Invalid reference size");
        EXPECT_NONFATAL_FAILURE(EXPECT_TRUE(load_i32_bin(name).empty()), "Invalid reference size");
    }
}

// ---------------------------------------------------------------------------
// 2. expand_bert_features: hand-built mapping
//
//   token 0 = [CLS]  span (-1, -1)
//   token 1          span [0, 1)
//   token 2          span [1, 3)
//   token 3 = [SEP]  span (-1, -1)
//   word2ph = {1, 2, 1}, n_phones = 4
//
//   Expected rows: {tok1, tok2, tok2, tok2}
// ---------------------------------------------------------------------------

TEST(RobertaExpand, HandBuiltMapping) {
    constexpr int64_t n_tok    = 4;
    constexpr int64_t n_chars  = 3;
    constexpr int64_t n_phones = 4;

    std::vector<float> tok_feat(static_cast<size_t>(n_tok * kHiddenSize));
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t c = 0; c < kHiddenSize; ++c) {
            tok_feat[static_cast<size_t>(t * kHiddenSize + c)] =
                static_cast<float>(t) + 0.25f;
        }
    }

    const int32_t char_starts[n_tok] = {-1, 0, 1, -1};
    const int32_t char_ends[n_tok]   = {-1, 1, 3, -1};
    const int32_t word2ph[n_chars]   = {1, 2, 1};

    std::vector<float> out(static_cast<size_t>(n_phones * kHiddenSize), -1.0f);

    ASSERT_TRUE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, char_starts, char_ends,
        word2ph, n_chars, n_phones, out.data()));

    const int64_t expected_rows[n_phones] = {1, 2, 2, 2};
    for (int64_t p = 0; p < n_phones; ++p) {
        for (int64_t c = 0; c < kHiddenSize; ++c) {
            EXPECT_FLOAT_EQ(out[static_cast<size_t>(p * kHiddenSize + c)],
                            tok_feat[static_cast<size_t>(expected_rows[p] * kHiddenSize + c)])
                << "phoneme " << p << " channel " << c;
        }
    }
}

TEST(RobertaExpand, RejectsInvalidInput) {
    constexpr int64_t n_tok    = 4;
    constexpr int64_t n_chars  = 3;
    constexpr int64_t n_phones = 4;

    std::vector<float> tok_feat(static_cast<size_t>(n_tok * kHiddenSize), 1.0f);
    const int32_t char_starts[n_tok] = {-1, 0, 1, -1};
    const int32_t char_ends[n_tok]   = {-1, 1, 3, -1};
    const int32_t word2ph[n_chars]   = {1, 2, 1};
    std::vector<float> out(static_cast<size_t>(n_phones * kHiddenSize), 0.0f);

    // Baseline sanity: the valid input succeeds.
    EXPECT_TRUE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, char_starts, char_ends,
        word2ph, n_chars, n_phones, out.data()));

    // sum(word2ph) != n_phones.
    {
        const int32_t bad_word2ph[n_chars] = {1, 1, 1};  // sum = 3
        EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
            tok_feat.data(), n_tok, char_starts, char_ends,
            bad_word2ph, n_chars, n_phones, out.data()));
    }

    // A character with no covering token: [1, 2) leaves char index 2 uncovered.
    {
        const int32_t short_ends[n_tok] = {-1, 1, 2, -1};
        EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
            tok_feat.data(), n_tok, char_starts, short_ends,
            word2ph, n_chars, n_phones, out.data()));
    }

    // Ambiguous coverage: two tokens cover char 1 -> invalid (Python raises).
    {
        const int32_t wide_starts[n_tok] = {-1, 0, 1, -1};
        const int32_t wide_ends[n_tok]   = {-1, 3, 3, -1};
        EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
            tok_feat.data(), n_tok, wide_starts, wide_ends,
            word2ph, n_chars, n_phones, out.data()));
    }

    // Span out of range.
    {
        const int32_t bad_ends[n_tok] = {-1, 1, 4, -1};  // end 4 > n_chars 3
        EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
            tok_feat.data(), n_tok, char_starts, bad_ends,
            word2ph, n_chars, n_phones, out.data()));
    }

    // Null arguments.
    EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
        nullptr, n_tok, char_starts, char_ends,
        word2ph, n_chars, n_phones, out.data()));
    EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, nullptr, char_ends,
        word2ph, n_chars, n_phones, out.data()));
    EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, char_starts, nullptr,
        word2ph, n_chars, n_phones, out.data()));
    EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, char_starts, char_ends,
        nullptr, n_chars, n_phones, out.data()));
    EXPECT_FALSE(gpt_sovits::roberta_expand_bert_features(
        tok_feat.data(), n_tok, char_starts, char_ends,
        word2ph, n_chars, n_phones, nullptr));
}

// ---------------------------------------------------------------------------
// 3. Runner input-validation edge cases
// ---------------------------------------------------------------------------

TEST_F(RobertaExtract, RejectsInvalidInput) {
    const int32_t token_id = 1;
    const int64_t T = 1;
    const std::vector<float> sentinel(kHiddenSize, 12345.0f);
    std::vector<float> out = sentinel;
    const int64_t out_cap = static_cast<int64_t>(out.size());

    for (auto extract : {gpt_sovits::roberta_extract_features,
                         gpt_sovits::roberta_extract_tts_features}) {
        SCOPED_TRACE(extract == gpt_sovits::roberta_extract_features ? "final" : "TTS");
        EXPECT_FALSE(extract(model_, &token_id, 0, out.data(), out_cap, nullptr));
        EXPECT_FALSE(extract(model_, &token_id, -1, out.data(), out_cap, nullptr));
        EXPECT_FALSE(extract(model_, nullptr, T, out.data(), out_cap, nullptr));
        EXPECT_FALSE(extract(model_, &token_id, T, nullptr, out_cap, nullptr));
        EXPECT_FALSE(extract(model_, &token_id, T, out.data(), out_cap - 1, nullptr));
        EXPECT_FALSE(extract(model_, &token_id, T, out.data(), -1, nullptr));

        const std::vector<int32_t> long_ids(513, token_id);
        EXPECT_FALSE(extract(model_, long_ids.data(), static_cast<int64_t>(long_ids.size()),
                             out.data(), out_cap, nullptr));
        for (int32_t bad_id : {-1, static_cast<int32_t>(model_.weights.embeddings.word_embeddings->ne[1])}) {
            EXPECT_FALSE(extract(model_, &bad_id, T, out.data(), out_cap, nullptr));
        }
        for (int32_t bad_type : {-1, static_cast<int32_t>(model_.weights.embeddings.token_type_embeddings->ne[1])}) {
            EXPECT_FALSE(extract(model_, &token_id, T, out.data(), out_cap, &bad_type));
        }
        EXPECT_EQ(out, sentinel);
    }
}

TEST(RobertaExtractValidation, RejectsUnloadedModel) {
    gpt_sovits::roberta_model model{};
    const int32_t token_id = 1;
    std::vector<float> out(kHiddenSize, 12345.0f);
    for (auto extract : {gpt_sovits::roberta_extract_features,
                         gpt_sovits::roberta_extract_tts_features}) {
        EXPECT_FALSE(extract(model, &token_id, 1, out.data(), kHiddenSize, nullptr));
    }
    model.backend = ggml_backend_cpu_init();
    ASSERT_NE(model.backend, nullptr);
    for (auto extract : {gpt_sovits::roberta_extract_features,
                         gpt_sovits::roberta_extract_tts_features}) {
        EXPECT_FALSE(extract(model, &token_id, 1, out.data(), kHiddenSize, nullptr));
    }
    ggml_backend_free(model.backend);
    EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 12345.0f; }));
}

}  // anonymous namespace
