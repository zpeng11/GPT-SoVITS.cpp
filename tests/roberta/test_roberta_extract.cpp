// tests/roberta/test_roberta_extract.cpp
//
// Tests for the host-side RoBERTa feature-extraction runner and the
// WordPiece-token -> phoneme BERT-feature expansion helper.
//
// Uses the fixtures produced by tests/roberta/generate_reference.py and the
// same F16 model/tolerances as test_roberta_parity.cpp.

#include <gtest/gtest.h>

#include "gpt_sovits/roberta.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

static const std::string kTestDir   = ROBERTA_MODEL_DIR;
static const std::string kModelPath = kTestDir + "chinese-roberta-wwm-ext-large-f16.gguf";

// F16 full-model parity tolerance, from test_roberta_parity.cpp.
static constexpr double kMaxAbsTol = 1e-1;
static constexpr double kRmseTol   = 1e-2;

static constexpr int64_t kHiddenSize = 1024;

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

std::vector<float> load_f32_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        ADD_FAILURE() << "Failed to open: " << path;
        return {};
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

std::vector<int32_t> load_i32_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        ADD_FAILURE() << "Failed to open: " << path;
        return {};
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<int32_t> data(size / sizeof(int32_t));
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

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

#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

// ---------------------------------------------------------------------------
// Fixture: loads the F16 model once per test
// ---------------------------------------------------------------------------

class RobertaExtract : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_MODEL_EXISTS(kModelPath);

        backend_ = create_test_backend();
        ASSERT_NE(backend_, nullptr);
        if (!gpt_sovits::roberta_model_load(kModelPath, model_, backend_)) {
            GTEST_SKIP() << "Could not load " << kModelPath;
        }
    }
    void TearDown() override {
        gpt_sovits::roberta_model_free(model_);
        if (backend_) {
            ggml_backend_free(backend_);
        }
    }

    ggml_backend_t           backend_ = nullptr;
    gpt_sovits::roberta_model model_{};
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
    auto input_ids = load_i32_bin(kTestDir + "ref_input_ids.bin");
    ASSERT_FALSE(input_ids.empty());

    const int64_t T = static_cast<int64_t>(input_ids.size());
    std::vector<float> out(static_cast<size_t>(T * kHiddenSize), 0.0f);
    const int64_t out_cap = static_cast<int64_t>(out.size());

    // T == 0.
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, input_ids.data(), 0, out.data(), out_cap));

    // Null token ids.
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, nullptr, T, out.data(), out_cap));

    // Null destination.
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, input_ids.data(), T, nullptr, out_cap));

    // Destination too small.
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, input_ids.data(), T, out.data(), out_cap - 1));

    // T beyond max_position_embeddings (512).
    std::vector<int32_t> long_ids(513, input_ids.front());
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, long_ids.data(), static_cast<int64_t>(long_ids.size()),
        out.data(), out_cap));

    // Out-of-range token id.
    std::vector<int32_t> bad_ids(input_ids.begin(), input_ids.end());
    bad_ids[0] = -1;
    EXPECT_FALSE(gpt_sovits::roberta_extract_features(
        model_, bad_ids.data(), T, out.data(), out_cap));
}

}  // anonymous namespace
