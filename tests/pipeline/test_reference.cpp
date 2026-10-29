#include <gtest/gtest.h>

#include "reference_internal.h"
#include "gpt_sovits/hubert.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace pipeline = gpt_sovits::pipeline_detail;

pipeline::reference_features ValidFeatures() {
    pipeline::reference_features f;
    f.audio.pcm16k.assign(720, 0.2f);
    f.audio.normalized_pcm16k.assign(720, 0.0f);
    f.audio.spec_frames = 2;
    f.audio.spec704.assign(704 * 2, 0.1f);
    f.phones = {1, 2};
    f.T_ref = 2;
    f.bert.assign(1024 * 2, 0.0f);
    f.T_hub = gpt_sovits::hubert_output_frames(720 + 9600);
    f.T_prompt = f.T_hub / 2;
    f.hubert.assign(static_cast<size_t>(768 * f.T_hub), 0.1f);
    return f;
}

TEST(ReferenceInput, UsesUnnormalizedPcmAndExactly9600TailZeros) {
    std::vector<float> pcm(720);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 2.0f + static_cast<float>(i) / 1000;
    std::vector<float> out = {99.0f};
    std::string error = "stale";
    ASSERT_TRUE(pipeline::build_reference_hubert_input(pcm.data(), pcm.size(), out, &error));
    EXPECT_TRUE(error.empty());
    ASSERT_EQ(out.size(), pcm.size() + 9600);
    EXPECT_TRUE(std::equal(pcm.begin(), pcm.end(), out.begin()));
    EXPECT_TRUE(std::all_of(out.begin() + pcm.size(), out.end(), [](float v) { return v == 0.0f; }));
    EXPECT_EQ(pcm.front(), 2.0f);
}

TEST(ReferenceInput, InvalidInputPreservesOutputAndReportsError) {
    std::vector<float> pcm(720, 0.1f), out = {12.0f, 34.0f};
    const auto before = out;
    std::string error;
    EXPECT_FALSE(pipeline::build_reference_hubert_input(nullptr, 720, out, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(out, before);
    EXPECT_FALSE(pipeline::build_reference_hubert_input(pcm.data(), 399, out, &error));
    EXPECT_EQ(out, before);
    EXPECT_FALSE(pipeline::build_reference_hubert_input(pcm.data(), INT64_MAX, out, &error));
    EXPECT_EQ(out, before);
    pcm[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(pipeline::build_reference_hubert_input(pcm.data(), pcm.size(), out, &error));
    EXPECT_EQ(out, before);
}

TEST(ReferenceInput, SupportsAliasedDestinationWithoutChangingSamples) {
    std::vector<float> out(400, 0.3f);
    ASSERT_TRUE(pipeline::build_reference_hubert_input(out.data(), out.size(), out));
    ASSERT_EQ(out.size(), 10000u);
    EXPECT_TRUE(std::all_of(out.begin(), out.begin() + 400, [](float v) { return v == 0.3f; }));
    EXPECT_TRUE(std::all_of(out.begin() + 400, out.end(), [](float v) { return v == 0.0f; }));
}

TEST(ReferenceFeatures, ValidatesShapeAndClearsError) {
    const auto f = ValidFeatures();
    std::string error = "stale";
    EXPECT_TRUE(pipeline::validate_reference_features(f, &error));
    EXPECT_TRUE(error.empty());
}

TEST(ReferenceFeatures, RejectsInvalidTokensDimensionsAndNonfiniteData) {
    const auto original = ValidFeatures();
    auto f = original;
    f.phones[0] = 732;
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.T_ref = 3;
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.bert.pop_back();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.bert[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.T_hub = 1; f.T_prompt = 0; f.hubert.resize(768);
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; ++f.T_prompt;
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.T_hub = INT64_MAX; f.T_prompt = f.T_hub / 2;
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.hubert[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.audio.pcm16k.resize(399);
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.audio.pcm16k[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.audio.spec_frames = 0;
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.audio.spec704.pop_back();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
    f = original; f.audio.spec704[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(pipeline::validate_reference_features(f));
}

TEST(ReferenceFeatures, MissingModelsPreservePreparedOutput) {
    auto out = ValidFeatures();
    const auto before = out;
    const gpt_sovits::hubert_model model{};
    const gpt_sovits::text_frontend_result text{};
    const float pcm = 0.1f;
    std::string error;
    EXPECT_FALSE(pipeline::prepare_reference(&pcm, 1, 16000, text, model, nullptr, out, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(out.phones, before.phones);
    EXPECT_EQ(out.hubert, before.hubert);
    EXPECT_EQ(out.audio.pcm16k, before.audio.pcm16k);
}

TEST(ReferenceEncoding, MissingModelsDoNotMutateSessions) {
    const auto f = ValidFeatures();
    const gpt_sovits::t2s_model t2s{};
    const gpt_sovits::sovits_model sovits{};
    const gpt_sovits::sv_model sv{};
    gpt_sovits::t2s_session ts{};
    gpt_sovits::sovits_session ss{};
    std::string error;
    EXPECT_FALSE(pipeline::encode_reference(f, t2s, sovits, sv, ts, ss, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(ts.ctx_ref, nullptr);
    EXPECT_EQ(ts.ref_model, nullptr);
    EXPECT_EQ(ss.ge, nullptr);
}

TEST(ReferenceEncoding, ExistingCacheIsRejectedBeforeModelsAreAccessed) {
    const auto f = ValidFeatures();
    const gpt_sovits::t2s_model t2s{};
    const gpt_sovits::sovits_model sovits{};
    const gpt_sovits::sv_model sv{};
    gpt_sovits::t2s_session ts{};
    gpt_sovits::sovits_session ss{};
    ts.ref_T_ref = 10;
    std::string error;
    EXPECT_FALSE(pipeline::encode_reference(f, t2s, sovits, sv, ts, ss, &error));
    EXPECT_NE(error.find("fresh sessions"), std::string::npos);
    EXPECT_EQ(ts.ref_T_ref, 10);
    EXPECT_EQ(ts.ctx_ref, nullptr);
    EXPECT_EQ(ss.ge, nullptr);

    ts.ref_T_ref = 0;
    ts.ref_prompt_tokens = {1};
    error.clear();
    EXPECT_FALSE(pipeline::encode_reference(f, t2s, sovits, sv, ts, ss, &error));
    EXPECT_NE(error.find("fresh sessions"), std::string::npos);
    EXPECT_EQ(ts.ref_prompt_tokens, (std::vector<int32_t>{1}));
    EXPECT_EQ(ts.ctx_ref, nullptr);
    EXPECT_EQ(ss.ge, nullptr);
}

} // namespace
