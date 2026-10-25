#include <gtest/gtest.h>

#include "gpt_sovits/text_frontend.h"
#include "gpt_sovits/roberta.h"
#include "bert_features_internal.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace gpt_sovits;
constexpr size_t kChannels = 1024;

text_frontend_segment chinese() {
    text_frontend_segment segment;
    segment.language = "zh";
    segment.normalized_text = "你𠮷好";
    segment.normalized_end = 3;
    segment.phoneme_count = 4;
    segment.word2ph = std::vector<int32_t>{1, 2, 1};
    segment.tokenizer = text_frontend_tokenizer{
        {101, 11, 12, 102}, 1, 3, {-1, 0, 1, -1}, {-1, 1, 3, -1}};
    return segment;
}

text_frontend_segment english() {
    text_frontend_segment segment;
    segment.language = "en";
    segment.normalized_text = "hi";
    segment.normalized_end = 2;
    segment.phoneme_count = 2;
    return segment;
}

text_frontend_result text_of(std::vector<text_frontend_segment> segments) {
    text_frontend_result text;
    int64_t chars = 0;
    for (auto & segment : segments) {
        const auto n_chars = segment.normalized_end - segment.normalized_start;
        segment.normalized_start = chars;
        segment.normalized_end = chars + n_chars;
        chars += n_chars;
        segment.phoneme_offset = static_cast<int64_t>(text.phoneme_ids.size());
        text.phoneme_ids.insert(text.phoneme_ids.end(), static_cast<size_t>(segment.phoneme_count), 1);
        text.normalized_text += segment.normalized_text;
    }
    text.segments = std::move(segments);
    return text;
}

float feature(int32_t id, size_t channel) {
    return static_cast<float>(id * 2) + static_cast<float>(channel) / 2048.0f;
}

struct ProviderState {
    mutable int calls = 0;
    int fail_at = 0;
    int nonfinite_at = 0;
    bool infinity = false;
    bool throw_allocation = false;
    bool throw_length = false;
};

bool provide(const void * context, const text_frontend_tokenizer & tok,
             float * out, int64_t out_cap, std::string & error) {
    const auto & state = *static_cast<const ProviderState *>(context);
    ++state.calls;
    if (state.throw_allocation) throw std::bad_alloc();
    if (state.throw_length) throw std::length_error("test allocation size");
    if (state.calls == state.fail_at) {
        error = "injected compute failure";
        return false;
    }
    EXPECT_EQ(out_cap, static_cast<int64_t>(tok.input_ids.size() * kChannels));
    for (size_t t = 0; t < tok.input_ids.size(); ++t) {
        for (size_t c = 0; c < kChannels; ++c) out[t * kChannels + c] = feature(tok.input_ids[t], c);
    }
    if (state.calls == state.nonfinite_at) {
        out[kChannels] = state.infinity ? std::numeric_limits<float>::infinity()
                                      : std::numeric_limits<float>::quiet_NaN();
    }
    return true;
}

bool build(const text_frontend_result & text, ProviderState & state,
           std::vector<float> & out, std::string & error) {
    return detail::text_frontend_build_bert_features_with_provider(text, provide, &state, out, &error);
}

void expect_rows(const std::vector<float> & out, const std::vector<int32_t> & ids) {
    ASSERT_EQ(out.size(), ids.size() * kChannels);
    for (size_t p = 0; p < ids.size(); ++p) {
        for (size_t c = 0; c < kChannels; ++c) {
            ASSERT_FLOAT_EQ(out[p * kChannels + c], ids[p] == 0 ? 0.0f : feature(ids[p], c))
                << "phone " << p << " channel " << c;
        }
    }
}

TEST(TextFrontendBert, PureEnglishNeedsNoModelOrTokenizer) {
    auto text = text_of({english()});
    for (bool configured : {false, true}) {
        if (configured) text.segments[0].tokenizer = text_frontend_tokenizer{};
        std::vector<float> out{123.0f};
        std::string error = "old error";
        ASSERT_TRUE(text_frontend_build_bert_features(text, nullptr, out, &error)) << error;
        EXPECT_TRUE(error.empty());
        expect_rows(out, {0, 0});
        roberta_model unloaded;
        ASSERT_TRUE(text_frontend_build_bert_features(text, &unloaded, out, &error)) << error;
        ProviderState state;
        ASSERT_TRUE(build(text, state, out, error)) << error;
        EXPECT_EQ(state.calls, 0);
    }
}

TEST(TextFrontendBert, ChineseUsesBodySpansAndChannelFastLayout) {
    ProviderState state;
    std::vector<float> out;
    std::string error;
    ASSERT_TRUE(build(text_of({chinese()}), state, out, error)) << error;
    EXPECT_EQ(state.calls, 1);
    expect_rows(out, {11, 12, 12, 12});
}

TEST(TextFrontendBert, MixedSegmentsKeepDistinctFeaturesAndOffsets) {
    auto second = chinese();
    second.tokenizer->input_ids = {101, 21, 22, 102};
    ProviderState state;
    std::vector<float> out;
    std::string error;
    ASSERT_TRUE(build(text_of({english(), chinese(), english(), second, english()}), state, out, error)) << error;
    EXPECT_EQ(state.calls, 2);
    expect_rows(out, {0, 0, 11, 12, 12, 12, 0, 0, 21, 22, 22, 22, 0, 0});
}

TEST(TextFrontendBert, ZeroPhonemeChineseNeedsNeitherModelNorTokenizer) {
    auto zero = chinese();
    zero.phoneme_count = 0;
    zero.word2ph = std::vector<int32_t>{0, 0, 0};
    zero.tokenizer.reset();
    auto text = text_of({zero, english(), zero});
    std::vector<float> out;
    std::string error;
    ASSERT_TRUE(text_frontend_build_bert_features(text, nullptr, out, &error)) << error;
    expect_rows(out, {0, 0});
    ProviderState state;
    ASSERT_TRUE(build(text, state, out, error)) << error;
    EXPECT_EQ(state.calls, 0);
}

TEST(TextFrontendBert, ZeroWord2PhCharactersNeedNoUniqueToken) {
    auto segment = chinese();
    segment.phoneme_count = 3;
    segment.word2ph = std::vector<int32_t>{1, 0, 2};
    segment.tokenizer = text_frontend_tokenizer{
        {101, 11, 12, 13, 102}, 1, 4, {-1, 0, 1, 1, -1}, {-1, 1, 2, 3, -1}};
    ProviderState state;
    std::vector<float> out;
    std::string error;
    ASSERT_TRUE(build(text_of({segment}), state, out, error)) << error;
    expect_rows(out, {11, 13, 13});
    segment.tokenizer = text_frontend_tokenizer{
        {101, 11, 13, 102}, 1, 3, {-1, 0, 2, -1}, {-1, 1, 3, -1}};
    ASSERT_TRUE(build(text_of({segment}), state, out, error)) << error;
    expect_rows(out, {11, 13, 13});
}

TEST(TextFrontendBert, ParsesTrackedMixedFixtureForHostAssemblyOnly) {
    text_frontend_result text;
    std::string error;
    ASSERT_TRUE(text_frontend_read_json(std::string(FRONTEND_TEST_DIR) + "fixtures/valid_mixed.json", text, &error)) << error;
    ProviderState state;
    std::vector<float> out;
    ASSERT_TRUE(build(text, state, out, error)) << error;
    std::vector<int32_t> expected;
    int calls = 0;
    for (const auto & segment : text.segments) {
        if (segment.language == "en") {
            expected.insert(expected.end(), static_cast<size_t>(segment.phoneme_count), 0);
            continue;
        }
        ++calls;
        const auto & tok = *segment.tokenizer;
        for (size_t c = 0; c < segment.word2ph->size(); ++c) {
            if ((*segment.word2ph)[c] == 0) continue;
            size_t match = tok.input_ids.size();
            for (size_t t = 1; t + 1 < tok.input_ids.size(); ++t) {
                if (tok.char_starts[t] <= static_cast<int64_t>(c) && static_cast<int64_t>(c) < tok.char_ends[t]) match = t;
            }
            ASSERT_LT(match, tok.input_ids.size());
            expected.insert(expected.end(), static_cast<size_t>((*segment.word2ph)[c]), tok.input_ids[match]);
        }
    }
    EXPECT_EQ(state.calls, calls);
    expect_rows(out, expected);
}

TEST(TextFrontendBert, MissingModelAndTokenizerPreserveOutput) {
    auto text = text_of({chinese()});
    std::vector<float> out{123.0f, 456.0f};
    const auto before = out;
    const float * buffer = out.data();
    std::string error;
    EXPECT_FALSE(text_frontend_build_bert_features(text, nullptr, out, &error));
    EXPECT_NE(error.find("segments[0]"), std::string::npos);
    EXPECT_NE(error.find("model"), std::string::npos);
    EXPECT_EQ(out, before);
    EXPECT_EQ(out.data(), buffer);
    roberta_model unloaded;
    EXPECT_FALSE(text_frontend_build_bert_features(text, &unloaded, out, &error));
    EXPECT_NE(error.find("not loaded"), std::string::npos);
    EXPECT_EQ(out, before);
    text.segments[0].tokenizer.reset();
    ProviderState state;
    EXPECT_FALSE(build(text, state, out, error));
    EXPECT_NE(error.find("segments[0].tokenizer"), std::string::npos);
    EXPECT_EQ(state.calls, 0);
    EXPECT_EQ(out, before);
}

TEST(TextFrontendBert, RejectsMalformedStructureBeforeComputing) {
    using Mutate = std::function<void(text_frontend_result &)>;
    const std::vector<Mutate> mutations = {
        [](auto & t) { t.phoneme_ids.clear(); },
        [](auto & t) { t.segments.clear(); },
        [](auto & t) { t.phoneme_ids[0] = -1; },
        [](auto & t) { t.phoneme_ids[0] = 732; },
        [](auto & t) { t.segments[0].language = "ja"; },
        [](auto & t) { t.segments[0].phoneme_offset = -1; },
        [](auto & t) { t.segments[0].phoneme_offset = 1; },
        [](auto & t) { t.segments[1].phoneme_offset = 0; },
        [](auto & t) { t.segments[0].phoneme_count = -1; },
        [](auto & t) { t.segments[0].phoneme_count = INT64_MAX; },
        [](auto & t) { t.phoneme_ids.push_back(1); },
        [](auto & t) { t.segments[0].normalized_start = -1; },
        [](auto & t) { t.segments[0].normalized_end = 4; },
        [](auto & t) { t.segments[0].normalized_text = "bad"; },
        [](auto & t) { t.normalized_text += "x"; },
        [](auto & t) { t.segments[0].normalized_text = std::string(1, '\xff'); },
        [](auto & t) { t.normalized_text = std::string(1, '\xff'); },
        [](auto & t) { t.segments[0].word2ph.reset(); },
        [](auto & t) { t.segments[0].word2ph = std::vector<int32_t>{1, 3}; },
        [](auto & t) { (*t.segments[0].word2ph)[0] = -1; },
        [](auto & t) { (*t.segments[0].word2ph)[0] = INT32_MAX; },
        [](auto & t) { (*t.segments[0].word2ph)[0] = 0; },
        [](auto & t) { t.segments[1].word2ph = std::vector<int32_t>{}; },
    };
    for (size_t i = 0; i < mutations.size(); ++i) {
        SCOPED_TRACE(i);
        auto text = text_of({chinese(), english()});
        mutations[i](text);
        ProviderState state;
        std::vector<float> out{123.0f};
        const auto * buffer = out.data();
        std::string error;
        EXPECT_FALSE(build(text, state, out, error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(state.calls, 0);
        EXPECT_EQ(out, std::vector<float>{123.0f});
        EXPECT_EQ(out.data(), buffer);
    }
}

TEST(TextFrontendBert, RejectsMalformedTokenizerBeforeComputing) {
    using Mutate = std::function<void(text_frontend_tokenizer &)>;
    const std::vector<Mutate> mutations = {
        [](auto & t) { t.input_ids.clear(); },
        [](auto & t) { t.input_ids.resize(1); },
        [](auto & t) { t.char_starts.pop_back(); },
        [](auto & t) { t.char_ends.push_back(0); },
        [](auto & t) { t.body_start = -1; },
        [](auto & t) { t.body_end = INT64_MAX; },
        [](auto & t) { t.input_ids.back() = t.input_ids.front(); },
        [](auto & t) { t.input_ids[1] = -1; },
        [](auto & t) { t.input_ids[1] = t.input_ids.front(); },
        [](auto & t) { t.input_ids[1] = t.input_ids.back(); },
        [](auto & t) { t.char_starts[0] = 0; },
        [](auto & t) { t.char_ends.back() = 0; },
        [](auto & t) { t.char_starts[1] = -1; },
        [](auto & t) { t.char_ends[1] = 0; },
        [](auto & t) { t.char_ends[2] = 4; },
    };
    for (size_t i = 0; i < mutations.size(); ++i) {
        SCOPED_TRACE(i);
        auto text = text_of({chinese()});
        mutations[i](*text.segments[0].tokenizer);
        ProviderState state;
        std::vector<float> out{123.0f};
        std::string error;
        EXPECT_FALSE(build(text, state, out, error));
        EXPECT_NE(error.find("segments[0].tokenizer"), std::string::npos);
        EXPECT_EQ(state.calls, 0);
        EXPECT_EQ(out, std::vector<float>{123.0f});
    }
}

TEST(TextFrontendBert, InvalidCoveragePreservesOutput) {
    for (bool overlap : {false, true}) {
        auto text = text_of({chinese()});
        if (overlap) text.segments[0].tokenizer->char_ends[1] = 2;
        else text.segments[0].tokenizer->char_starts[2] = 2;
        ProviderState state;
        std::vector<float> out{123.0f};
        std::string error;
        EXPECT_FALSE(build(text, state, out, error));
        EXPECT_NE(error.find("segments[0].bert"), std::string::npos);
        EXPECT_NE(error.find("expansion"), std::string::npos);
        EXPECT_EQ(out, std::vector<float>{123.0f});
    }
}

TEST(TextFrontendBert, TokenLimitCountsSpecialTokens) {
    for (size_t n : {size_t{512}, size_t{513}}) {
        text_frontend_segment segment;
        segment.language = "zh";
        segment.normalized_text = std::string(n - 2, 'a');
        segment.normalized_end = static_cast<int64_t>(n - 2);
        segment.phoneme_count = static_cast<int64_t>(n - 2);
        segment.word2ph = std::vector<int32_t>(n - 2, 1);
        text_frontend_tokenizer tok;
        tok.body_start = 1;
        tok.body_end = static_cast<int64_t>(n - 1);
        tok.input_ids.assign(n, 11);
        tok.input_ids.front() = 101;
        tok.input_ids.back() = 102;
        tok.char_starts.assign(n, -1);
        tok.char_ends.assign(n, -1);
        for (size_t t = 1; t + 1 < n; ++t) {
            tok.char_starts[t] = static_cast<int32_t>(t - 1);
            tok.char_ends[t] = static_cast<int32_t>(t);
        }
        segment.tokenizer = std::move(tok);
        ProviderState state;
        std::vector<float> out{123.0f};
        std::string error;
        EXPECT_EQ(build(text_of({segment}), state, out, error), n == 512) << error;
        if (n == 512) {
            expect_rows(out, std::vector<int32_t>(n - 2, 11));
            EXPECT_EQ(state.calls, 1);
        } else {
            EXPECT_EQ(state.calls, 0);
            EXPECT_EQ(out, std::vector<float>{123.0f});
            EXPECT_NE(error.find("512"), std::string::npos);
        }
    }
}

TEST(TextFrontendBert, LaterComputeFailurePreservesOutputAndRetries) {
    const auto text = text_of({chinese(), english(), chinese()});
    ProviderState state;
    state.fail_at = 2;
    std::vector<float> out{123.0f};
    const auto * buffer = out.data();
    std::string error;
    EXPECT_FALSE(build(text, state, out, error));
    EXPECT_NE(error.find("segments[2].bert"), std::string::npos);
    EXPECT_NE(error.find("injected compute failure"), std::string::npos);
    EXPECT_EQ(out, std::vector<float>{123.0f});
    EXPECT_EQ(out.data(), buffer);
    state = {};
    ASSERT_TRUE(build(text, state, out, error)) << error;
    EXPECT_TRUE(error.empty());
    expect_rows(out, {11, 12, 12, 12, 0, 0, 11, 12, 12, 12});
    ASSERT_TRUE(text_frontend_build_bert_features(text_of({english()}), nullptr, out, &error));
    expect_rows(out, {0, 0});
}

TEST(TextFrontendBert, NonFiniteFeaturesPreserveOutput) {
    for (bool infinity : {false, true}) {
        ProviderState state;
        state.nonfinite_at = 2;
        state.infinity = infinity;
        std::vector<float> out{123.0f};
        std::string error;
        EXPECT_FALSE(build(text_of({chinese(), chinese()}), state, out, error));
        EXPECT_NE(error.find("segments[1].bert"), std::string::npos);
        EXPECT_NE(error.find("non-finite"), std::string::npos);
        EXPECT_EQ(out, std::vector<float>{123.0f});
    }
}

TEST(TextFrontendBert, RecoverableHostExceptionsPreserveOutput) {
    for (bool length : {false, true}) {
        ProviderState state;
        state.throw_allocation = !length;
        state.throw_length = length;
        std::vector<float> out{123.0f};
        std::string error;
        EXPECT_FALSE(build(text_of({chinese()}), state, out, error));
        EXPECT_NE(error.find("host allocation"), std::string::npos);
        EXPECT_EQ(out, std::vector<float>{123.0f});
    }
}

TEST(TextFrontendBert, NullErrorReportsToStderr) {
    std::vector<float> out{123.0f};
    EXPECT_FALSE(text_frontend_build_bert_features(text_of({chinese()}), nullptr, out));
    EXPECT_EQ(out, std::vector<float>{123.0f});
    EXPECT_TRUE(text_frontend_build_bert_features(text_of({english()}), nullptr, out));
    expect_rows(out, {0, 0});
}

} // namespace
