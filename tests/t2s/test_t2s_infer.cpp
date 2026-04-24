// tests/t2s/test_t2s_infer.cpp
//
// End-to-end T2S autoregressive inference test: prefill via flexible graph,
// decode loop via persistent decode graph.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "npy_loader.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const std::string kTestDir = T2S_TEST_DIR;

static ggml_backend_t create_backend() {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

static gpt_sovits::t2s_sampler_config default_sampler_cfg() {
    return gpt_sovits::t2s_sampler_config{};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct ErrorStats {
    double max_abs;
    double rmse;
    double mean_abs;
};

static ErrorStats compute_errors(const std::vector<float> & actual,
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

// Generate Exp(1) noise for the sampler.  The sampler computes
// score = softmax(sorted_logits) / noise, then picks argmax(score).
// Exp(1) noise implements the Gumbel-max trick for categorical sampling.
static void fill_exp_noise(std::vector<float> & buf, std::mt19937 & rng) {
    std::exponential_distribution<float> dist(1.0f);
    for (auto & v : buf) {
        v = dist(rng);
    }
}

// ---------------------------------------------------------------------------
// Token embedding helper for decode steps
// ---------------------------------------------------------------------------

// Reusable embed helper that persists its allocator across decode steps.
struct embed_helper {
    ggml_backend_t        backend = nullptr;
    ggml_gallocr_t        alloc   = nullptr;
    const int64_t         d_model;

    explicit embed_helper(ggml_backend_t backend, int64_t d_model)
        : backend(backend), d_model(d_model) {
        auto buft = ggml_backend_get_default_buffer_type(backend);
        alloc = ggml_gallocr_new(buft);
    }

    ~embed_helper() {
        if (alloc) { ggml_gallocr_free(alloc); }
    }

    std::vector<float> embed(const gpt_sovits::t2s_model & model,
                             int32_t token_id,
                             int64_t start_position)
    {
        const size_t n_intermediates = 16;
        const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                             ggml_graph_overhead_custom(graph_size, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(token_ids, "token_id");
        ggml_set_input(token_ids);

        struct ggml_tensor * emb = gpt_sovits::t2s_audio_embed_block_forward(
            ctx, token_ids,
            model.weights.embed.audio_embedding,
            model.weights.embed.audio_pos_alpha,
            start_position);
        ggml_set_name(emb, "emb_out");
        ggml_set_output(emb);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(gf, emb);

        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(token_ids, &token_id, 0, sizeof(int32_t));
        ggml_backend_graph_compute(backend, gf);

        std::vector<float> result(d_model);
        ggml_backend_tensor_get(emb, result.data(), 0, d_model * sizeof(float));

        ggml_free(ctx);
        return result;
    }
};

// ---------------------------------------------------------------------------
// Full autoregressive inference test
// ---------------------------------------------------------------------------

TEST(T2SInfer, PrefillAndDecodeLoop) {
    // --- Configuration ---
    const std::string model_path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    const std::string ref_dir    = kTestDir + "ref/";

    // Reference dimensions
    const int64_t T_ref     = 52;
    const int64_t T_in      = 28;
    const int64_t T_prompt  = 113;
    const int64_t T_total   = T_ref + T_in + T_prompt;  // 193

    // Skip if model not present
    {
        FILE * f = fopen(model_path.c_str(), "rb");
        if (!f) GTEST_SKIP() << "Model file not found: " << model_path;
        fclose(f);
    }

    // --- Load reference data ---
    auto ref_xy_pos  = load_npy_as_f32(ref_dir + "xy_pos.npy");
    auto ref_xy_dec  = load_npy_as_f32(ref_dir + "xy_dec.npy");
    auto ref_k_cache = load_npy_as_f32(ref_dir + "k_cache.npy");
    auto ref_v_cache = load_npy_as_f32(ref_dir + "v_cache.npy");
    ASSERT_FALSE(ref_xy_pos.empty());
    ASSERT_FALSE(ref_xy_dec.empty());
    ASSERT_FALSE(ref_k_cache.empty());
    ASSERT_FALSE(ref_v_cache.empty());

    // --- Create backend and load model ---
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(model_path, model, backend));

    const int64_t d_model = model.hparams.hidden_dim;  // 512
    const int     n_layer = (int) model.hparams.n_layer;  // 24
    const int64_t vocab   = model.hparams.vocab_size;  // 1025
    const int32_t eos     = (int32_t) model.hparams.eos;  // 1024

    ASSERT_EQ((int64_t) ref_xy_pos.size(), d_model * T_total);
    ASSERT_EQ((int64_t) ref_xy_dec.size(), d_model * T_total);

    // --- RNG for stochastic sampling ---
    std::mt19937 rng(42);

    // --- Session init ---
    const uint32_t n_batch   = 1;
    const uint32_t slot_size = 512;  // >= 193 + max_decode_steps

    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, backend, n_batch, slot_size));

    // Set reference embedding dimensions for correct mask generation.
    session.ref_T_ref    = T_ref;
    session.ref_T_prompt = T_prompt;

    // =======================================================================
    // Phase 1: Prefill via flexible graph
    // =======================================================================

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {(int) T_total};

    auto sampler_cfg = default_sampler_cfg();
    auto graph = gpt_sovits::t2s_session_build_flex_graph(session, model, plan, sampler_cfg);
    ASSERT_NE(graph.ctx, nullptr);
    ASSERT_EQ(graph.N, (int) T_total);

    // Fill input tensor with reference embedding data.
    ggml_backend_tensor_set(graph.x, ref_xy_pos.data(), 0,
                            (size_t)(d_model * T_total) * sizeof(float));

    // Fill exp_noise with Exp(1) random noise for stochastic sampling.
    ASSERT_NE(graph.exp_noise, nullptr);
    std::vector<float> noise_buf((size_t)(vocab * graph.n_active));
    fill_exp_noise(noise_buf, rng);
    ggml_backend_tensor_set(graph.exp_noise, noise_buf.data(), 0,
                            noise_buf.size() * sizeof(float));

    // Fill seen_mask with zeros (no tokens seen → repetition penalty is a no-op).
    if (graph.seen_mask) {
        std::vector<float> zeros_seen((size_t)(vocab * graph.n_active), 0.0f);
        ggml_backend_tensor_set(graph.seen_mask, zeros_seen.data(), 0,
                                zeros_seen.size() * sizeof(float));
    }

    // Advance session state and execute.
    gpt_sovits::t2s_session_flex_advance(session, plan, graph);
    ASSERT_EQ(ggml_backend_graph_compute(backend, graph.gf), GGML_STATUS_SUCCESS);

    // --- Verify prefill output ---
    {
        std::vector<float> actual_y((size_t)(d_model * T_total));
        ggml_backend_tensor_get(graph.y, actual_y.data(), 0, actual_y.size() * sizeof(float));

        auto err = compute_errors(actual_y, ref_xy_dec);
        printf("  [prefill] Output  max_abs=%.6f  RMSE=%.6f\n", err.max_abs, err.rmse);
        EXPECT_LT(err.max_abs, 2.0) << "Prefill output max_abs error";
        EXPECT_LT(err.rmse, 0.5)    << "Prefill output RMSE";
    }

    // --- Verify KV caches ---
    {
        const int64_t layer_elems = d_model * T_total;
        double worst_k_max = 0, worst_v_max = 0;
        int worst_k_layer = -1, worst_v_layer = -1;

        for (int layer = 0; layer < n_layer; layer++) {
            std::vector<float> actual_k(layer_elems);
            ggml_backend_tensor_get(session.k_caches[layer], actual_k.data(), 0,
                                    layer_elems * sizeof(float));
            std::vector<float> expected_k(ref_k_cache.begin() + layer * layer_elems,
                                          ref_k_cache.begin() + (layer + 1) * layer_elems);

            auto err_k = compute_errors(actual_k, expected_k);
            EXPECT_LT(err_k.max_abs, 2.0) << "K cache layer " << layer;
            if (err_k.max_abs > worst_k_max) {
                worst_k_max = err_k.max_abs;
                worst_k_layer = layer;
            }

            std::vector<float> actual_v(layer_elems);
            ggml_backend_tensor_get(session.v_caches[layer], actual_v.data(), 0,
                                    layer_elems * sizeof(float));
            std::vector<float> expected_v(ref_v_cache.begin() + layer * layer_elems,
                                          ref_v_cache.begin() + (layer + 1) * layer_elems);

            auto err_v = compute_errors(actual_v, expected_v);
            EXPECT_LT(err_v.max_abs, 2.0) << "V cache layer " << layer;
            if (err_v.max_abs > worst_v_max) {
                worst_v_max = err_v.max_abs;
                worst_v_layer = layer;
            }
        }

        printf("  [prefill] K cache worst: layer %d max_abs=%.6f\n", worst_k_layer, worst_k_max);
        printf("  [prefill] V cache worst: layer %d max_abs=%.6f\n", worst_v_layer, worst_v_max);
    }

    // --- Read first sampled token from prefill ---
    int32_t first_sampled = -1, first_greedy = -1;
    ggml_backend_tensor_get(graph.sampled, &first_sampled, 0, sizeof(int32_t));
    ggml_backend_tensor_get(graph.greedy, &first_greedy, 0, sizeof(int32_t));

    printf("  [prefill] first sampled=%d, greedy=%d\n", first_sampled, first_greedy);
    EXPECT_GE(first_sampled, 0);
    EXPECT_LT(first_sampled, (int32_t) vocab);
    EXPECT_GE(first_greedy, 0);
    EXPECT_LT(first_greedy, (int32_t) vocab);

    // Free the flex graph before building the decode graph.
    gpt_sovits::t2s_flex_graph_free(graph);

    // =======================================================================
    // Phase 2: Build persistent decode graph
    // =======================================================================

    ASSERT_TRUE(gpt_sovits::t2s_session_build_decode_graph(session, model, sampler_cfg));

    struct ggml_tensor * dec_exp_noise  = gpt_sovits::t2s_session_get_exp_noise(session);
    struct ggml_tensor * dec_seen_mask  = gpt_sovits::t2s_session_get_seen_mask(session);
    struct ggml_tensor * x_dec          = gpt_sovits::t2s_session_get_x_dec(session);
    ASSERT_NE(dec_exp_noise, nullptr);
    ASSERT_NE(dec_seen_mask, nullptr);
    ASSERT_NE(x_dec, nullptr);

    // Pre-allocate reusable buffers for decode loop inputs.
    std::vector<float> dec_noise_buf((size_t)(vocab * n_batch));
    std::vector<float> seen_buf((size_t)(vocab * n_batch), 0.0f);

    // Mark the first sampled token as seen.
    seen_buf[first_sampled] = 1.0f;

    // =======================================================================
    // Phase 3: Autoregressive decode loop
    // =======================================================================

    const int max_decode_steps = 200;
    int32_t current_token = first_sampled;
    std::vector<int32_t> generated;
    generated.push_back(current_token);

    bool reached_eos = false;
    embed_helper emb(backend, d_model);

    for (int step = 0; step < max_decode_steps; step++) {
        // 1. Embed the current token with PE at the next audio position.
        int64_t position = T_prompt + step;
        auto token_emb = emb.embed(model, current_token, position);
        ASSERT_EQ((int64_t) token_emb.size(), d_model);

        // 2. Feed the embedding into the decode graph.
        ggml_backend_tensor_set(x_dec, token_emb.data(), 0,
                                d_model * sizeof(float));

        // 3. Fill exp_noise with fresh Exp(1) noise.
        fill_exp_noise(dec_noise_buf, rng);
        ggml_backend_tensor_set(dec_exp_noise, dec_noise_buf.data(), 0,
                                dec_noise_buf.size() * sizeof(float));

        // 4. Upload seen_mask with all previously generated tokens marked.
        ggml_backend_tensor_set(dec_seen_mask, seen_buf.data(), 0,
                                seen_buf.size() * sizeof(float));

        // 5. Advance session state (update mask, kv_pos, n_pos).
        gpt_sovits::t2s_session_decode_advance(session);

        // 6. Execute decode graph.
        ASSERT_EQ(ggml_backend_graph_compute(backend,
                    gpt_sovits::t2s_session_get_decode_graph(session)),
                  GGML_STATUS_SUCCESS);

        // 7. Read sampler output.
        int32_t sampled_token = -1, greedy_token = -1;
        ggml_backend_tensor_get(gpt_sovits::t2s_session_get_sampled(session),
                                &sampled_token, 0, sizeof(int32_t));
        ggml_backend_tensor_get(gpt_sovits::t2s_session_get_greedy(session),
                                &greedy_token, 0, sizeof(int32_t));

        // 8. Validate token IDs.
        EXPECT_GE(sampled_token, 0) << "decode step " << step;
        EXPECT_LT(sampled_token, (int32_t) vocab) << "decode step " << step;
        EXPECT_GE(greedy_token, 0) << "decode step " << step;
        EXPECT_LT(greedy_token, (int32_t) vocab) << "decode step " << step;

        generated.push_back(sampled_token);

        // 9. Update seen_mask: mark the newly generated token.
        if (sampled_token >= 0 && sampled_token < (int32_t) vocab) {
            seen_buf[sampled_token] = 1.0f;
        }

        current_token = sampled_token;

        // 10. Check EOS.
        if (current_token == eos) {
            reached_eos = true;
            break;
        }
    }

    // --- Print summary ---
    printf("  [decode] %zu tokens generated, reached_eos=%d\n",
           generated.size(), reached_eos);
    printf("  [decode] first 20 tokens: ");
    for (size_t i = 0; i < std::min(generated.size(), (size_t)20); i++) {
        printf("%d ", generated[i]);
    }
    printf("\n");

    // --- Final assertions ---
    if (reached_eos) {
        EXPECT_GT((int) generated.size(), 1) << "Should have generated at least one token before EOS";
    } else {
        printf("  [decode] EOS not reached within %d steps (this is acceptable for the test)\n",
               max_decode_steps);
    }
    EXPECT_GT((int) generated.size(), 0) << "Should have generated at least the prefill token";

    // =======================================================================
    // Cleanup
    // =======================================================================

    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}
