#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

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
        if (!c.seen.empty()) {
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
        // Noise uses the existing sorted-rank convention; these tests do not
        // assert vocabulary-indexed noise replay or cross-language RNG parity.
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

    std::vector<int32_t> sampled_, greedy_;

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t alloc_ = nullptr;
};

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
