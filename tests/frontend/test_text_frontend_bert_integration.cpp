// Real-tokenizer, real-F16-CPU-model integration test for the public
// text_frontend_build_bert_features assembly API.  CTest does not run Python.

#include <gtest/gtest.h>

#include "gpt_sovits/roberta.h"
#include "gpt_sovits/text_frontend.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHiddenSize = 1024;
constexpr float   kSentinel   = 12345.0f;

const std::string kTestDir = [] {
    std::string directory = ROBERTA_MODEL_DIR;
    if (directory.empty() || directory.back() != '/') directory.push_back('/');
    return directory;
}();

const std::string kFrontendDir = [] {
    std::string directory = FRONTEND_TEST_DIR;
    if (directory.empty() || directory.back() != '/') directory.push_back('/');
    return directory;
}();

const std::string kModelPath   = kTestDir + "chinese-roberta-wwm-ext-large-f16.gguf";
const std::string kFixturePath = kFrontendDir + "fixtures/bert_mixed.json";

size_t Codepoints(const std::string & text) {
    size_t count = 0;
    for (unsigned char byte : text) {
        if ((byte & 0xc0) != 0x80) ++count;
    }
    return count;
}

class TextFrontendBertIntegration : public ::testing::Test {
protected:
    void SetUp() override {
        std::string error = "stale error";
        ASSERT_TRUE(gpt_sovits::text_frontend_read_json(kFixturePath, text_, &error))
            << "Cannot read fixture " << kFixturePath << ": " << error;
        EXPECT_TRUE(error.empty()) << "Successful read did not clear error: " << error;
        ASSERT_NO_FATAL_FAILURE(ValidateFixture());

        // Only ENOENT is a skip; open/load failures are test failures.
        FILE * file = std::fopen(kModelPath.c_str(), "rb");
        if (!file) {
            const int open_error = errno;
            if (open_error == ENOENT) {
                GTEST_SKIP() << "Model file not found: " << kModelPath;
            }
            FAIL() << "Cannot open model " << kModelPath << ": "
                   << std::strerror(open_error);
        }
        std::fclose(file);

        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr) << "CPU backend initialization failed";
        ASSERT_TRUE(gpt_sovits::roberta_model_load(kModelPath, model_, backend_))
            << "Failed to load model " << kModelPath;
    }

    void TearDown() override {
        gpt_sovits::roberta_model_free(model_);
        if (backend_) ggml_backend_free(backend_);
    }

    void ValidateFixture() {
        ASSERT_EQ(text_.normalized_text, "你好,世界!Hello world!再见.");
        ASSERT_EQ(Codepoints(text_.normalized_text), size_t{21});
        ASSERT_EQ(text_.phoneme_ids.size(), size_t{24});
        ASSERT_EQ(text_.segments.size(), size_t{3});

        const char * languages[] = {"zh", "en", "zh"};
        const int64_t offsets[]  = {0, 10, 19};
        const int64_t counts[]   = {10, 9, 5};
        // Body ids from the real chinese-roberta-wwm-ext-large tokenizer.
        const std::vector<int32_t> ids0 = {101, 872, 1962, 117, 686, 4518, 106, 102};
        const std::vector<int32_t> ids1 = {101, 8701, 8572, 106, 102};
        const std::vector<int32_t> ids2 = {101, 1086, 6224, 119, 102};
        const std::vector<int32_t> * ids[] = {&ids0, &ids1, &ids2};

        for (size_t i = 0; i < text_.segments.size(); ++i) {
            SCOPED_TRACE("segment " + std::to_string(i));
            const auto & segment = text_.segments[i];
            EXPECT_EQ(segment.language, languages[i]);
            EXPECT_EQ(segment.phoneme_offset, offsets[i]);
            EXPECT_EQ(segment.phoneme_count, counts[i]);
            ASSERT_TRUE(segment.tokenizer.has_value())
                << "fixture must carry a real per-segment tokenizer";
            const auto & tokenizer = *segment.tokenizer;
            EXPECT_EQ(tokenizer.input_ids, *ids[i]);
            EXPECT_EQ(tokenizer.body_start, int64_t{1});
            EXPECT_EQ(tokenizer.body_end, static_cast<int64_t>(ids[i]->size()) - 1);
            EXPECT_EQ(tokenizer.input_ids.front(), 101);
            EXPECT_EQ(tokenizer.input_ids.back(), 102);
            EXPECT_EQ(tokenizer.char_starts.front(), -1);
            EXPECT_EQ(tokenizer.char_ends.front(), -1);
            EXPECT_EQ(tokenizer.char_starts.back(), -1);
            EXPECT_EQ(tokenizer.char_ends.back(), -1);
            EXPECT_LE(tokenizer.input_ids.size(), size_t{512});

            if (segment.language == "zh") {
                ASSERT_TRUE(segment.word2ph.has_value());
                EXPECT_EQ(segment.word2ph->size(), Codepoints(segment.normalized_text));
                int64_t sum = 0;
                for (int32_t value : *segment.word2ph) {
                    ASSERT_GE(value, 0);
                    sum += value;
                }
                EXPECT_EQ(sum, segment.phoneme_count);
            } else {
                EXPECT_FALSE(segment.word2ph.has_value());
            }
        }
    }

    void BuildExpected() {
        expected_.assign(static_cast<size_t>(text_.phoneme_ids.size() * kHiddenSize), 0.0f);
        for (size_t i = 0; i < text_.segments.size(); ++i) {
            SCOPED_TRACE("expected segment " + std::to_string(i));
            const auto & segment = text_.segments[i];
            ASSERT_GE(segment.phoneme_offset, 0);
            ASSERT_GE(segment.phoneme_count, 0);
            const int64_t row_begin = segment.phoneme_offset * kHiddenSize;
            const int64_t row_count = segment.phoneme_count * kHiddenSize;
            ASSERT_LE(row_begin, static_cast<int64_t>(expected_.size()));
            ASSERT_LE(row_count, static_cast<int64_t>(expected_.size()) - row_begin);
            if (segment.language == "en") continue;

            ASSERT_EQ(segment.language, "zh");
            ASSERT_TRUE(segment.tokenizer.has_value());
            ASSERT_TRUE(segment.word2ph.has_value());
            const auto & tokenizer = *segment.tokenizer;
            const int64_t n_tok = static_cast<int64_t>(tokenizer.input_ids.size());
            ASSERT_GE(n_tok, 2);
            ASSERT_LE(n_tok, 512);
            ASSERT_EQ(tokenizer.body_start, int64_t{1});
            ASSERT_EQ(tokenizer.body_end, n_tok - 1);
            ASSERT_EQ(static_cast<int64_t>(tokenizer.char_starts.size()), n_tok);
            ASSERT_EQ(static_cast<int64_t>(tokenizer.char_ends.size()), n_tok);

            std::vector<float> token_features(static_cast<size_t>(n_tok * kHiddenSize));
            ASSERT_TRUE(gpt_sovits::roberta_extract_tts_features(
                model_, tokenizer.input_ids.data(), n_tok, token_features.data(),
                static_cast<int64_t>(token_features.size())));

            const int64_t n_chars = static_cast<int64_t>(Codepoints(segment.normalized_text));
            ASSERT_EQ(static_cast<int64_t>(segment.word2ph->size()), n_chars);
            std::vector<float> expanded(static_cast<size_t>(segment.phoneme_count * kHiddenSize));
            ASSERT_TRUE(gpt_sovits::roberta_expand_bert_features(
                token_features.data(), n_tok,
                tokenizer.char_starts.data(), tokenizer.char_ends.data(),
                segment.word2ph->data(), n_chars, segment.phoneme_count,
                expanded.data()));
            std::copy(expanded.begin(), expanded.end(),
                      expected_.begin() + static_cast<std::ptrdiff_t>(row_begin));
        }
    }

    void BuildMixed() {
        const size_t total = static_cast<size_t>(text_.phoneme_ids.size() * kHiddenSize);
        actual_.assign(total, kSentinel);
        std::string error = "stale error";
        ASSERT_TRUE(gpt_sovits::text_frontend_build_bert_features(
            text_, &model_, actual_, &error)) << error;
        ASSERT_EQ(actual_.size(), total);
        EXPECT_TRUE(error.empty()) << "Successful build did not clear error: " << error;
        EXPECT_EQ(static_cast<size_t>(std::count(actual_.begin(), actual_.end(), kSentinel)), size_t{0})
            << "builder left output elements unwritten";
    }

    ggml_backend_t             backend_ = nullptr;
    gpt_sovits::roberta_model  model_{};
    gpt_sovits::text_frontend_result text_;
    std::vector<float>         expected_;
    std::vector<float>         actual_;
};

// Verifies the public mixed assembly against direct per-segment calls, with
// English strictly zero and Chinese finite/nondegenerate.
TEST_F(TextFrontendBertIntegration, MixedPublicAssemblyMatchesLowLevelCalls) {
    ASSERT_NO_FATAL_FAILURE(BuildExpected());
    ASSERT_NO_FATAL_FAILURE(BuildMixed());
    ASSERT_EQ(actual_.size(), expected_.size());

    size_t mismatches = 0;
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < expected_.size(); ++i) {
        if (actual_[i] == expected_[i]) continue;
        ++mismatches;
        if (mismatches <= 8) {
            ADD_FAILURE() << "phone-row " << (i / static_cast<size_t>(kHiddenSize))
                          << " channel " << (i % static_cast<size_t>(kHiddenSize))
                          << ": builder=" << actual_[i]
                          << " low-level=" << expected_[i];
        }
        max_abs_diff = std::max(max_abs_diff,
                                std::abs(static_cast<double>(actual_[i]) -
                                         static_cast<double>(expected_[i])));
    }
    EXPECT_EQ(mismatches, size_t{0})
        << "builder differs from direct per-segment extraction/expansion; "
        << "max_abs_diff=" << max_abs_diff;

    size_t english_phones = 0;
    size_t chinese_segments = 0;
    for (const auto & segment : text_.segments) {
        const size_t begin = static_cast<size_t>(segment.phoneme_offset * kHiddenSize);
        const size_t count = static_cast<size_t>(segment.phoneme_count * kHiddenSize);
        ASSERT_LE(begin, actual_.size());
        ASSERT_LE(count, actual_.size() - begin);
        if (segment.language == "en") {
            english_phones += static_cast<size_t>(segment.phoneme_count);
            EXPECT_TRUE(std::all_of(actual_.begin() + static_cast<std::ptrdiff_t>(begin),
                                    actual_.begin() + static_cast<std::ptrdiff_t>(begin + count),
                                    [](float value) { return value == 0.0f; }))
                << "English phoneme ranges must be strictly zero";
            continue;
        }
        ++chinese_segments;
        double sum_squares = 0.0;
        double max_abs = 0.0;
        for (size_t i = begin; i < begin + count; ++i) {
            ASSERT_TRUE(std::isfinite(actual_[i])) << "non-finite Chinese feature";
            sum_squares += static_cast<double>(actual_[i]) * actual_[i];
            max_abs = std::max(max_abs, std::abs(static_cast<double>(actual_[i])));
        }
        EXPECT_GT(sum_squares, 1e-6) << "Chinese features are all zero";
        EXPECT_GT(max_abs, 1e-3) << "Chinese features are degenerate";
    }
    EXPECT_EQ(english_phones, size_t{9});
    EXPECT_EQ(chinese_segments, size_t{2});
}

// Vocabulary overflow in either Chinese segment must preserve the output.  The
// second segment also covers failure after an earlier segment has succeeded.
TEST_F(TextFrontendBertIntegration, RejectsOutOfVocabularyTokenPreservesOutput) {
    ASSERT_NE(model_.weights.embeddings.word_embeddings, nullptr);
    const int32_t vocab_size =
        static_cast<int32_t>(model_.weights.embeddings.word_embeddings->ne[1]);
    ASSERT_GT(vocab_size, 0);

    const size_t total = static_cast<size_t>(text_.phoneme_ids.size() * kHiddenSize);
    for (size_t segment_index : {size_t{0}, size_t{2}}) {
        SCOPED_TRACE("segment " + std::to_string(segment_index));
        for (int32_t bad_id : {vocab_size, int32_t{-1}}) {
            auto broken = text_;
            ASSERT_TRUE(broken.segments[segment_index].tokenizer.has_value());
            ASSERT_GT(broken.segments[segment_index].tokenizer->input_ids.size(), size_t{1});
            broken.segments[segment_index].tokenizer->input_ids[1] = bad_id;

            std::vector<float> out(total, kSentinel);
            std::string error = "stale error";
            EXPECT_FALSE(gpt_sovits::text_frontend_build_bert_features(
                broken, &model_, out, &error));
            EXPECT_FALSE(error.empty()) << "failure must report a diagnostic";
            EXPECT_EQ(static_cast<size_t>(std::count(out.begin(), out.end(), kSentinel)), total)
                << "failure modified output";
        }
    }
}

}  // namespace
