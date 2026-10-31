#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef T2S_SAMPLER_REFERENCE_PATH
#error "T2S_SAMPLER_REFERENCE_PATH must be defined by the test target"
#endif

namespace {

std::vector<float> Logits(const std::vector<float> & probabilities) {
    std::vector<float> logits;
    for (float p : probabilities) logits.push_back(std::log(p));
    return logits;
}

struct SamplerCase {
    const char * name;
    std::vector<float> logits;
    std::vector<float> noise;
    float top_p;
    int32_t sampled;
    int32_t greedy;
    int top_k = 0;
    float temperature = 1.0f;
    float repetition_penalty = 1.0f;
    std::vector<float> seen;
    std::vector<int32_t> suppress_eos;
};

using Json = nlohmann::json;

const char * const kUpstreamNames[] = {
    "default_policy", "top_p_removes_crossing", "top_p_low_keeps_leader",
    "top_p_zero_keeps_leader", "top_p_before_temperature", "top_p_before_top_k",
    "temperature_after_filtering", "signed_penalty_duplicate_history",
    "signed_penalty_unique_history", "penalty_disabled", "top_k_zero",
    "top_k_negative", "top_k_at_vocab", "top_k_above_vocab", "top_k_pivot_ties",
    "top_k_one_leading_ties", "eos_unmasked", "eos_prefix_suppressed",
    "single_token_vocabulary",
};

struct UpstreamCase {
    std::string name;
    SamplerCase sampler{};
    std::vector<int32_t> greedy_candidates;
    float relative_margin;
};

void Require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int32_t ReadInt(const Json & value, int32_t low, int32_t high) {
    Require(value.is_number_integer(), "expected an integer, not a boolean or fraction");
    const double number = value.get<double>();
    Require(number >= low && number <= high, "integer out of range");
    return value.get<int32_t>();
}

float ReadFloat(const Json & value) {
    Require(value.is_number(), "expected a finite F32 number");
    const double number = value.get<double>();
    Require(std::isfinite(number) && std::abs(number) <= std::numeric_limits<float>::max(),
            "expected a finite F32 number");
    const float rounded = static_cast<float>(number);
    Require(static_cast<double>(rounded) == number, "number does not round-trip as F32");
    return rounded;
}

std::vector<float> ReadVector(const Json & value, size_t length = 0) {
    Require(value.is_array() && !value.empty() && value.size() <= 64 &&
            (length == 0 || value.size() == length), "invalid vector length");
    std::vector<float> result;
    for (const auto & item : value) result.push_back(ReadFloat(item));
    return result;
}

Json ReadUpstreamDocument() {
    std::ifstream input(T2S_SAMPLER_REFERENCE_PATH);
    Require(input.is_open(), "cannot open tracked sampler fixture");
    return Json::parse(input);
}

std::vector<UpstreamCase> ReadUpstreamCases(const Json & document) {
    Require(document.at("schema") == "gpt-sovits-t2s-sampler-reference/v1" &&
            document.at("status") == "complete", "unsupported sampler fixture schema/status");
    const Json semantics = {
        {"probabilities", "vocabulary-indexed;upstream-logits_to_probs"},
        {"noise", "vocabulary-indexed;fixed-positive-f32;no-rng"},
        {"sampled", "argmax(probabilities/noise)"},
        {"greedy", "all-maxima-of-adjusted-logits-before-filtering"},
        {"eos", "last-id;optional-prefix-exclusion-before-upstream"},
        {"disabled_top_k", "nonpositive-maps-to-None"},
    };
    Require(document.at("semantics") == semantics, "unsupported sampler fixture semantics");
    const auto & cases = document.at("cases");
    const std::set<std::string> expected_names(std::begin(kUpstreamNames), std::end(kUpstreamNames));
    Require(cases.is_array() && cases.size() == expected_names.size(), "incorrect fixture case count");
    std::set<std::string> names;
    std::vector<UpstreamCase> result;
    for (const auto & item : cases) {
        UpstreamCase ref;
        ref.name = item.at("name").get<std::string>();
        Require(expected_names.count(ref.name) && names.insert(ref.name).second,
                "unexpected or duplicate fixture case: " + ref.name);
        auto & c = ref.sampler;
        c.logits = ReadVector(item.at("logits"));
        c.noise = ReadVector(item.at("noise"), c.logits.size());
        Require(std::all_of(c.noise.begin(), c.noise.end(), [](float v) { return v > 0; }),
                "noise must be positive");
        const auto & params = item.at("parameters");
        c.top_k = ReadInt(params.at("top_k"), INT32_MIN, INT32_MAX);
        c.top_p = ReadFloat(params.at("top_p"));
        c.temperature = ReadFloat(params.at("temperature"));
        c.repetition_penalty = ReadFloat(params.at("repetition_penalty"));
        Require(c.repetition_penalty > 0 && std::isfinite(1.0f / c.repetition_penalty),
                "invalid repetition penalty");
        Require(params.at("suppress_eos").is_boolean(), "suppress_eos must be boolean");
        const bool suppress = params.at("suppress_eos").get<bool>();
        const int32_t active = static_cast<int32_t>(c.logits.size()) - (suppress ? 1 : 0);
        Require(active > 0, "EOS suppression needs a nonempty prefix");
        if (c.logits.size() > 1) c.suppress_eos = {suppress ? 1 : 0};
        c.seen.assign(c.logits.size(), 0.0f);
        const auto & history = item.at("previous_tokens");
        Require(history.is_array(), "previous_tokens must be an array");
        for (const auto & token : history) c.seen[ReadInt(token, 0, active - 1)] = 1.0f;

        const auto & expected = item.at("expected");
        const auto adjusted = ReadVector(expected.at("adjusted_logits"), static_cast<size_t>(active));
        const auto probabilities = ReadVector(expected.at("probabilities"), c.logits.size());
        double sum = 0;
        float best = -1, second = -1;
        int32_t selected = -1;
        for (size_t token = 0; token < probabilities.size(); ++token) {
            const float p = probabilities[token];
            Require(p >= 0 && p <= 1, "invalid reference probability");
            sum += p;
            const float score = p / c.noise[token];
            Require(std::isfinite(score), "non-finite reference replay score");
            if (score > best) { second = best; best = score; selected = static_cast<int32_t>(token); }
            else if (score > second) second = score;
        }
        Require(std::abs(sum - 1) <= 1e-6 && (!suppress || probabilities.back() == 0),
                "invalid probability normalization/EOS exclusion");
        const auto & greedy = expected.at("greedy_candidates");
        Require(greedy.is_array() && !greedy.empty(), "invalid greedy candidates");
        for (const auto & token : greedy) ref.greedy_candidates.push_back(ReadInt(token, 0, active - 1));
        const float maximum = *std::max_element(adjusted.begin(), adjusted.end());
        std::vector<int32_t> maxima;
        for (int32_t token = 0; token < active; ++token) {
            if (adjusted[token] == maximum) {
                Require(probabilities[token] > 0, "greedy candidate must remain eligible");
                maxima.push_back(token);
            }
        }
        Require(ref.greedy_candidates == maxima, "incorrect greedy candidate set");
        c.sampled = ReadInt(expected.at("sampled"), 0, active - 1);
        ref.relative_margin = ReadFloat(expected.at("relative_margin"));
        const float margin = best > 0 ? (best - std::max(second, 0.0f)) / best : 0;
        Require(selected == c.sampled && margin >= 0.01f &&
                std::abs(margin - ref.relative_margin) <= 1e-7f,
                "inconsistent or ambiguous reference sampled token/margin");
        result.push_back(std::move(ref));
    }
    return result;
}

class T2SSampler : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);
        ggml_backend_cpu_set_n_threads(backend_, 1);
        constexpr size_t max_nodes = 1024;
        ctx_ = ggml_init({ggml_tensor_overhead() * max_nodes +
                         ggml_graph_overhead_custom(max_nodes, false), nullptr, true});
        ASSERT_NE(ctx_, nullptr);
        alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        ASSERT_NE(alloc_, nullptr);
    }

    void TearDown() override {
        if (alloc_) ggml_gallocr_free(alloc_);
        if (ctx_) ggml_free(ctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    void Run(const SamplerCase & c, int64_t n_batch = 1) {
        ASSERT_GT(n_batch, 0);
        ASSERT_EQ(c.logits.size() % static_cast<size_t>(n_batch), 0u);
        const int64_t vocab = static_cast<int64_t>(c.logits.size()) / n_batch;
        ASSERT_GT(vocab, 0);
        ASSERT_EQ(c.noise.size(), c.logits.size());
        ASSERT_TRUE(c.seen.empty() || c.seen.size() == c.logits.size());
        ASSERT_TRUE(c.suppress_eos.empty() ||
                    c.suppress_eos.size() == static_cast<size_t>(n_batch));

        ggml_tensor * y = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, vocab, n_batch);
        ggml_tensor * weights = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, vocab, vocab);
        ggml_tensor * noise = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, vocab, n_batch);
        ggml_set_input(y);
        ggml_set_input(weights);
        ggml_set_input(noise);
        ggml_tensor * seen = nullptr;
        if (!c.seen.empty() && c.repetition_penalty != 1.0f) {
            seen = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, vocab, n_batch);
            ggml_set_input(seen);
        }
        ggml_tensor * suppress = nullptr;
        if (!c.suppress_eos.empty()) {
            suppress = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
            ggml_set_input(suppress);
        }
        const auto result = gpt_sovits::t2s_sampler_block_forward(
            ctx_, y, weights, seen, c.top_k, c.top_p, c.temperature,
            c.repetition_penalty, noise, suppress);
        ggml_set_output(result.sampled);
        ggml_set_output(result.greedy);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx_, 1024, false);
        ggml_build_forward_expand(graph, result.sampled);
        ggml_build_forward_expand(graph, result.greedy);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc_, graph));

        std::vector<float> identity(static_cast<size_t>(vocab * vocab), 0.0f);
        for (int64_t i = 0; i < vocab; ++i) identity[static_cast<size_t>(i * vocab + i)] = 1.0f;
        ggml_backend_tensor_set(weights, identity.data(), 0, identity.size() * sizeof(float));
        ggml_backend_tensor_set(y, c.logits.data(), 0, c.logits.size() * sizeof(float));
        // Run consumes sorted-rank noise. Handwritten cases supply it directly;
        // upstream cases use the test-only vocabulary-to-rank adapter below.
        ggml_backend_tensor_set(noise, c.noise.data(), 0, c.noise.size() * sizeof(float));
        if (seen) ggml_backend_tensor_set(seen, c.seen.data(), 0, c.seen.size() * sizeof(float));
        if (suppress) ggml_backend_tensor_set(suppress, c.suppress_eos.data(), 0,
                                              c.suppress_eos.size() * sizeof(int32_t));
        ASSERT_EQ(ggml_backend_graph_compute(backend_, graph), GGML_STATUS_SUCCESS);
        sampled_.resize(static_cast<size_t>(n_batch));
        greedy_.resize(static_cast<size_t>(n_batch));
        ggml_backend_tensor_get(result.sampled, sampled_.data(), 0, sampled_.size() * sizeof(int32_t));
        ggml_backend_tensor_get(result.greedy, greedy_.data(), 0, greedy_.size() * sizeof(int32_t));
    }

    void MapVocabularyNoise(SamplerCase & c, std::vector<int32_t> & permutation) {
        const bool suppress = !c.suppress_eos.empty() && c.suppress_eos[0] == 1;
        const size_t active = c.logits.size() - (suppress ? 1 : 0);
        ASSERT_GT(active, 0u);
        ggml_tensor * logits = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, active);
        ggml_set_input(logits);
        ggml_tensor * adjusted = logits;
        ggml_tensor * seen = nullptr;
        if (!c.seen.empty() && c.repetition_penalty != 1.0f) {
            seen = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, active);
            ggml_set_input(seen);
            // Match only the F32 penalty arithmetic used before argsort. This
            // adapts noise indices; it never computes an expected sampler result.
            const auto lerp = [&](ggml_tensor * base, ggml_tensor * selected, ggml_tensor * mask) {
                return ggml_add(ctx_, base, ggml_mul(ctx_, mask, ggml_sub(ctx_, selected, base)));
            };
            ggml_tensor * positive = ggml_scale(ctx_, logits, 1.0f / c.repetition_penalty);
            ggml_tensor * negative = ggml_scale(ctx_, logits, c.repetition_penalty);
            ggml_tensor * scaled = lerp(positive, negative, ggml_step(ctx_, ggml_neg(ctx_, logits)));
            adjusted = lerp(logits, scaled, ggml_step(ctx_, seen));
        }
        // Same one-row F32 CPU argsort as the sampler: tied IDs may have any
        // order, but this identical input/length reproduces that order locally.
        ggml_tensor * indices = ggml_argsort(ctx_, adjusted, GGML_SORT_ORDER_DESC);
        ggml_set_output(indices);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx_, 1024, false);
        ggml_build_forward_expand(graph, indices);
        ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc_, graph));
        ggml_backend_tensor_set(logits, c.logits.data(), 0, active * sizeof(float));
        if (seen) ggml_backend_tensor_set(seen, c.seen.data(), 0, active * sizeof(float));
        ASSERT_EQ(ggml_backend_graph_compute(backend_, graph), GGML_STATUS_SUCCESS);
        permutation.resize(active);
        ggml_backend_tensor_get(indices, permutation.data(), 0, active * sizeof(int32_t));
        const std::vector<float> vocabulary_noise = c.noise;
        c.noise.assign(c.logits.size(), 1.0f);
        std::vector<bool> visited(active, false);
        for (size_t rank = 0; rank < active; ++rank) {
            const int32_t token = permutation[rank];
            ASSERT_GE(token, 0);
            ASSERT_LT(static_cast<size_t>(token), active);
            ASSERT_FALSE(visited[token]);
            visited[token] = true;
            c.noise[rank] = vocabulary_noise[token];
        }
    }

    std::vector<int32_t> sampled_, greedy_;

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t alloc_ = nullptr;
};

class T2SSamplerUpstream : public T2SSampler, public ::testing::WithParamInterface<const char *> {};

TEST_P(T2SSamplerUpstream, MatchesFixtureDecision) {
    std::vector<UpstreamCase> cases;
    ASSERT_NO_THROW(cases = ReadUpstreamCases(ReadUpstreamDocument()));
    const auto found = std::find_if(cases.begin(), cases.end(),
        [&](const UpstreamCase & ref) { return ref.name == GetParam(); });
    ASSERT_NE(found, cases.end());
    SamplerCase c = found->sampler;
    std::vector<int32_t> permutation;
    ASSERT_NO_FATAL_FAILURE(MapVocabularyNoise(c, permutation));
    SCOPED_TRACE(::testing::Message() << found->name << ": top_k=" << c.top_k
        << ", top_p=" << c.top_p << ", temperature=" << c.temperature
        << ", repetition_penalty=" << c.repetition_penalty
        << ", suppress_eos=" << ::testing::PrintToString(c.suppress_eos)
        << ", sorted token IDs=" << ::testing::PrintToString(permutation)
        << ", rank noise=" << ::testing::PrintToString(c.noise)
        << ", reference margin=" << found->relative_margin);
    ASSERT_NO_FATAL_FAILURE(Run(c));
    EXPECT_EQ(sampled_, std::vector<int32_t>{c.sampled});
    ASSERT_EQ(greedy_.size(), 1u);
    EXPECT_NE(std::find(found->greedy_candidates.begin(), found->greedy_candidates.end(), greedy_[0]),
              found->greedy_candidates.end())
        << "greedy=" << greedy_[0]
        << ", expected one of " << ::testing::PrintToString(found->greedy_candidates);
}

INSTANTIATE_TEST_SUITE_P(, T2SSamplerUpstream, ::testing::ValuesIn(kUpstreamNames),
    [](const ::testing::TestParamInfo<const char *> & info) { return std::string(info.param); });

TEST(T2SSamplerReferenceContract, RejectsInvalidFixtureData) {
    Json valid;
    ASSERT_NO_THROW(valid = ReadUpstreamDocument());
    ASSERT_NO_THROW(ReadUpstreamCases(valid));
    const std::vector<std::pair<const char *, std::function<void(Json &)>>> changes = {
        {"schema", [](Json & j) { j["schema"] = "old"; }},
        {"semantics", [](Json & j) { j["semantics"]["noise"] = "rank-indexed"; }},
        {"missing case", [](Json & j) { j["cases"].erase(j["cases"].begin()); }},
        {"duplicate case", [](Json & j) { j["cases"][1] = j["cases"][0]; }},
        {"unknown case", [](Json & j) { j["cases"][0]["name"] = "unknown"; }},
        {"noise length", [](Json & j) { j["cases"][0]["noise"] = Json::array({1.0}); }},
        {"zero noise", [](Json & j) { j["cases"][0]["noise"][0] = 0.0; }},
        {"non-finite logits", [](Json & j) {
            j["cases"][0]["logits"][0] = std::numeric_limits<double>::infinity(); }},
        {"history ID", [](Json & j) { j["cases"][0]["previous_tokens"] = Json::array({20}); }},
        {"fractional k", [](Json & j) { j["cases"][0]["parameters"]["top_k"] = 1.5; }},
        {"boolean k", [](Json & j) { j["cases"][0]["parameters"]["top_k"] = true; }},
        {"invalid penalty", [](Json & j) { j["cases"][0]["parameters"]["repetition_penalty"] = 0; }},
        {"invalid EOS flag", [](Json & j) { j["cases"][0]["parameters"]["suppress_eos"] = 1; }},
        {"sampled ID", [](Json & j) { j["cases"][0]["expected"]["sampled"] = 20; }},
        {"greedy set", [](Json & j) { j["cases"][0]["expected"]["greedy_candidates"] = Json::array({0}); }},
        {"probability sum", [](Json & j) { j["cases"][0]["expected"]["probabilities"][0] = 1.0; }},
        {"sampled margin", [](Json & j) { j["cases"][0]["expected"]["relative_margin"] = 0.0; }},
        {"suppressed EOS probability", [](Json & j) {
            j["cases"][17]["expected"]["probabilities"] = Json::array({0.0, 0.0, 1.0}); }},
    };
    for (const auto & change : changes) {
        SCOPED_TRACE(change.first);
        Json invalid = valid;
        change.second(invalid);
        EXPECT_THROW(ReadUpstreamCases(invalid), std::exception);
    }
}

class T2SSamplerTopP : public T2SSampler, public ::testing::WithParamInterface<SamplerCase> {};

TEST_P(T2SSamplerTopP, MatchesExpectedToken) {
    const auto & c = GetParam();
    ASSERT_NO_FATAL_FAILURE(Run(c));
    EXPECT_EQ(sampled_, std::vector<int32_t>{c.sampled});
    EXPECT_EQ(greedy_, std::vector<int32_t>{c.greedy});
}

const SamplerCase kTopPCases[] = {
    {"RemovesFirstCrossingToken", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.01f, 1.0f},
     0.7f, 0, 0},
    {"KeepsLeadingTokenAboveThreshold", Logits({0.5f, 0.3f, 0.2f}), {10.0f, 0.01f, 0.01f},
     0.1f, 0, 0},
    {"KeepsLeadingTokenAtZeroThreshold", Logits({0.5f, 0.3f, 0.2f}), {10.0f, 0.01f, 0.01f},
     0.0f, 0, 0},
    {"OneDisablesFiltering", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.01f, 1.0f},
     1.0f, 1, 0},
    {"AboveOneDisablesFiltering", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.01f, 1.0f},
     1.1f, 1, 0},
    {"SingleTokenVocabulary", {0.0f}, {1.0f}, 0.0f, 0, 0},
    {"DefaultTopKAndTopP", {20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                            10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-8f, 1e-12f, 1, 1, 1, 1},
     1.0f, 14, 0, 15, 1.0f, 1.35f},
    {"DisabledTopKAllowsLowerRanks", {20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                    10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-8f, 1e-12f, 1, 1, 1, 1},
     1.0f, 15, 0},
    {"FiltersBeforeTemperature", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.01f, 1.0f},
     0.85f, 1, 0, 0, 0.5f},
    {"AppliesTemperatureAfterFiltering", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.5f, 1.0f},
     0.85f, 0, 0, 0, 0.5f},
    {"FiltersBeforeTopK", Logits({0.4f, 0.3f, 0.2f, 0.1f}), {1.0f, 0.01f, 0.001f, 1.0f},
     0.75f, 1, 0, 2},
    {"TopKStillRemovesLowerRanks", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 0.01f, 0.001f},
     0.85f, 0, 0, 1},
    {"RepetitionPenaltyPrecedesFiltering", Logits({0.5f, 0.3f, 0.2f}), {1.0f, 1.0f, 1.0f},
     0.7f, 1, 1, 0, 1.0f, 2.0f, {1.0f, 0.0f, 0.0f}},
    {"UnsortedLogitsMapBackToTokenIds", Logits({0.2f, 0.5f, 0.3f}), {1.0f, 0.01f, 1.0f},
     0.7f, 1, 1},
    {"UnsuppressedEosUsesFullVocabulary", Logits({0.3f, 0.2f, 0.5f}), {1.0f, 0.01f, 1.0f},
     0.7f, 2, 2, 0, 1.0f, 1.0f, {}, {0}},
    {"SuppressedEosUsesPrefixProbabilities", Logits({0.3f, 0.2f, 0.5f}), {1.0f, 0.01f, 1.0f},
     0.7f, 0, 0, 0, 1.0f, 1.0f, {}, {1}},
    {"SuppressedEosLeavesSingleCandidate", {0.0f, 1.0f}, {1.0f, 0.01f},
     0.0f, 0, 0, 0, 1.0f, 1.0f, {}, {1}},
};

INSTANTIATE_TEST_SUITE_P(, T2SSamplerTopP, ::testing::ValuesIn(kTopPCases),
    [](const ::testing::TestParamInfo<SamplerCase> & info) { return std::string(info.param.name); });

class T2SSamplerTopK : public T2SSampler, public ::testing::WithParamInterface<SamplerCase> {};

TEST_P(T2SSamplerTopK, MatchesExpectedToken) {
    const auto & c = GetParam();
    ASSERT_NO_FATAL_FAILURE(Run(c));
    EXPECT_EQ(sampled_, std::vector<int32_t>{c.sampled});
    EXPECT_EQ(greedy_, std::vector<int32_t>{c.greedy});
}

// Sorted token IDs are {1, 3, 0, 2}; small noise makes the selected rank win
// whenever it remains eligible. Other sampling controls are disabled.
const SamplerCase kTopKCases[] = {
    {"CutoffIncludesKthCandidate", {1, 3, 0, 2}, {1, 1e-4f, 1, 1},
     1.0f, 3, 1, 2},
    {"RemovesCandidateBelowCutoff", {1, 3, 0, 2}, {1, 1, 1e-4f, 1},
     1.0f, 1, 1, 2},
    {"ZeroDisablesFiltering", {1, 3, 0, 2}, {1, 1, 1, 1e-4f},
     1.0f, 2, 1, 0},
    {"NegativeDisablesFiltering", {1, 3, 0, 2}, {1, 1, 1, 1e-4f},
     1.0f, 2, 1, -1},
    {"VocabularySizeDisablesFiltering", {1, 3, 0, 2}, {1, 1, 1, 1e-4f},
     1.0f, 2, 1, 4},
    {"AboveVocabularySizeDisablesFiltering", {1, 3, 0, 2}, {1, 1, 1, 1e-4f},
     1.0f, 2, 1, 5},
};

INSTANTIATE_TEST_SUITE_P(, T2SSamplerTopK, ::testing::ValuesIn(kTopKCases),
    [](const ::testing::TestParamInfo<SamplerCase> & info) { return std::string(info.param.name); });

TEST_F(T2SSampler, TopKKeepsCandidatesTiedWithPivot) {
    // Rank 2 lies beyond the first k=2 entries but ties with the pivot. Force it
    // to win without depending on which tied token ID occupies that rank.
    const SamplerCase c = {"", {2, 3, 0, 2}, {1, 1, 1e-4f, 1},
                           1.0f, 0, 1, 2};
    ASSERT_NO_FATAL_FAILURE(Run(c));
    ASSERT_EQ(sampled_.size(), 1u);
    EXPECT_TRUE(sampled_[0] == 0 || sampled_[0] == 3) << sampled_[0];
    EXPECT_EQ(greedy_, std::vector<int32_t>{1});
}

TEST_F(T2SSampler, TopKOneKeepsTiedLeadingCandidates) {
    // Rank 1 must remain eligible even with k=1. Comparing with greedy avoids
    // assuming a particular ordering of the tied leading token IDs.
    const SamplerCase c = {"", {2, 0, 2, 1}, {1, 1e-4f, 1, 1},
                           1.0f, 0, 0, 1};
    ASSERT_NO_FATAL_FAILURE(Run(c));
    ASSERT_EQ(sampled_.size(), 1u);
    ASSERT_EQ(greedy_.size(), 1u);
    EXPECT_TRUE(sampled_[0] == 0 || sampled_[0] == 2) << sampled_[0];
    EXPECT_TRUE(greedy_[0] == 0 || greedy_[0] == 2) << greedy_[0];
    EXPECT_NE(sampled_[0], greedy_[0]);
}

TEST_F(T2SSampler, ExactCumulativeThresholdRemainsEligible) {
    // Uniform logits give exactly representable probabilities and cumulative
    // sums. Comparing with greedy avoids depending on the sort order of ties.
    const SamplerCase c = {"", {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.01f, 1.0f, 1.0f},
                           0.5f, 0, 0};
    ASSERT_NO_FATAL_FAILURE(Run(c));
    ASSERT_EQ(sampled_.size(), 1u);
    ASSERT_EQ(greedy_.size(), 1u);
    EXPECT_NE(sampled_[0], greedy_[0]);
}

TEST_F(T2SSampler, CumulativeProbabilityAboveThresholdIsRemoved) {
    const SamplerCase c = {"", {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.01f, 1.0f, 1.0f},
                           0.49f, 0, 0};
    ASSERT_NO_FATAL_FAILURE(Run(c));
    EXPECT_EQ(sampled_, greedy_);
}

TEST_F(T2SSampler, BatchColumnsKeepIndependentTopPAndEosMasks) {
    SamplerCase c = {"", {}, {}, 0.7f, 0, 0};
    for (const auto & probabilities : std::vector<std::vector<float>>{
             {0.5f, 0.3f, 0.2f}, {0.3f, 0.2f, 0.5f},
             {0.1f, 0.2f, 0.7f}, {0.2f, 0.5f, 0.3f}}) {
        const auto logits = Logits(probabilities);
        c.logits.insert(c.logits.end(), logits.begin(), logits.end());
        c.noise.insert(c.noise.end(), {1.0f, 0.01f, 1.0f});
    }
    c.suppress_eos = {0, 1, 0, 1};
    ASSERT_NO_FATAL_FAILURE(Run(c, 4));
    EXPECT_EQ(sampled_, (std::vector<int32_t>{0, 0, 2, 1}));
    EXPECT_EQ(greedy_, (std::vector<int32_t>{0, 0, 2, 1}));
}

} // namespace
