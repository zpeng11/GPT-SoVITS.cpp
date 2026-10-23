// tests/t2s/test_t2s_generate_integration.cpp
//
// Real-model integration test for t2s_session_generate.
//
// Uses the f16 T2S checkpoint and short synthetic reference/target inputs.
// The new high-level API is checked against an independent low-level runner
// built directly from the block helpers (target embedding + flex prefill +
// persistent decode loop).  The runner deliberately uses top_k == 1 so the
// sampled token is deterministic and no golden fixture is required.
//
// A CPU baseline always runs when the model is present.  A separate Metal
// case runs only when Metal was compiled in and can initialize at runtime.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const std::string kTestDir = T2S_TEST_DIR;

namespace {

// ---------------------------------------------------------------------------
// Small RAII helpers
// ---------------------------------------------------------------------------

struct backend_guard {
    ggml_backend_t b = nullptr;
    explicit backend_guard(ggml_backend_t backend) : b(backend) {}
    ~backend_guard() { if (b) ggml_backend_free(b); }
    backend_guard(const backend_guard &) = delete;
    backend_guard & operator=(const backend_guard &) = delete;
};

struct model_guard {
    gpt_sovits::t2s_model m;
    model_guard() = default;
    ~model_guard() { gpt_sovits::t2s_model_free(m); }
    model_guard(const model_guard &) = delete;
    model_guard & operator=(const model_guard &) = delete;
};

struct session_guard {
    gpt_sovits::t2s_session s;
    session_guard() = default;
    ~session_guard() { gpt_sovits::t2s_session_free(s); }
    session_guard(const session_guard &) = delete;
    session_guard & operator=(const session_guard &) = delete;
};

struct ctx_guard {
    ggml_context * p = nullptr;
    explicit ctx_guard(ggml_context * ctx) : p(ctx) {}
    ~ctx_guard() { if (p) ggml_free(p); }
    ctx_guard(const ctx_guard &) = delete;
    ctx_guard & operator=(const ctx_guard &) = delete;
};

struct gallocr_guard {
    ggml_gallocr_t p = nullptr;
    explicit gallocr_guard(ggml_gallocr_t a) : p(a) {}
    ~gallocr_guard() { if (p) ggml_gallocr_free(p); }
    gallocr_guard(const gallocr_guard &) = delete;
    gallocr_guard & operator=(const gallocr_guard &) = delete;
};

struct flex_guard {
    gpt_sovits::t2s_flex_graph & g;
    explicit flex_guard(gpt_sovits::t2s_flex_graph & graph) : g(graph) {}
    ~flex_guard() { gpt_sovits::t2s_flex_graph_free(g); }
    flex_guard(const flex_guard &) = delete;
    flex_guard & operator=(const flex_guard &) = delete;
};

struct slot_guard {
    gpt_sovits::t2s_session & s;
    int id;
    slot_guard(gpt_sovits::t2s_session & session, int slot_id)
        : s(session), id(slot_id) {}
    ~slot_guard() { gpt_sovits::t2s_session_slot_release(s, id); }
    slot_guard(const slot_guard &) = delete;
    slot_guard & operator=(const slot_guard &) = delete;
};

// ---------------------------------------------------------------------------
// Helpers shared with the low-level runner
// ---------------------------------------------------------------------------

static std::vector<uint8_t> read_tensor_bytes(ggml_tensor * t) {
    std::vector<uint8_t> bytes((size_t) ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return bytes;
}

// Same Exp(1) stream as t2s_session_generate (positive resample on zero).
static void fill_exp_noise(std::vector<float> & buf,
                           std::mt19937 & rng,
                           std::exponential_distribution<float> & dist) {
    for (float & v : buf) {
        float x = 0.0f;
        do {
            x = dist(rng);
        } while (!(x > 0.0f));
        v = x;
    }
}

static void read_sampled_greedy(const gpt_sovits::t2s_flex_graph & g,
                                int32_t & sampled, int32_t & greedy) {
    sampled = -1;
    greedy  = -1;
    ggml_backend_tensor_get(g.sampled, &sampled, 0, sizeof(int32_t));
    ggml_backend_tensor_get(g.greedy,  &greedy,  0, sizeof(int32_t));
}

// Compute the target text embedding exactly as generate does, but as a
// standalone step so the manual runner does not call the high-level API.
static bool compute_target_embedding(gpt_sovits::t2s_session & session,
                                     const gpt_sovits::t2s_model & model,
                                     const int32_t * tokens,
                                     int64_t T_in,
                                     const float * bert,
                                     int64_t T_ref,
                                     std::vector<float> & out) {
    const int64_t d_model = model.hparams.hidden_dim;
    out.resize((size_t) (d_model * T_in));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 +
                         ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx_guard ctx(ggml_init(params));
    if (ctx.p == nullptr) return false;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx.p, GGML_DEFAULT_GRAPH_SIZE, false);
    if (gf == nullptr) return false;

    struct ggml_tensor * token_t = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T_in);
    struct ggml_tensor * bert_t  = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, 1024, T_in);
    ggml_set_input(token_t);
    ggml_set_input(bert_t);

    struct ggml_tensor * emb = gpt_sovits::t2s_embed_input_forward(
        ctx.p, token_t, bert_t, T_ref, model.weights.embed);
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(session.backend);
    if (buft == nullptr) return false;

    gallocr_guard alloc(ggml_gallocr_new(buft));
    if (alloc.p == nullptr) return false;
    if (!ggml_gallocr_alloc_graph(alloc.p, gf)) return false;

    ggml_backend_tensor_set(token_t, tokens, 0,
                            (size_t) (T_in * (int64_t) sizeof(int32_t)));
    ggml_backend_tensor_set(bert_t, bert, 0,
                            (size_t) (1024 * T_in * (int64_t) sizeof(float)));

    if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
        return false;
    }

    ggml_backend_tensor_get(emb, out.data(), 0,
                            (size_t) (d_model * T_in * (int64_t) sizeof(float)));
    return true;
}

// Independent low-level generation.  Reuses only the block-level embedding
// helper; prefill/decode and sampling are driven directly here.
static bool manual_generate(gpt_sovits::t2s_session & session,
                            const gpt_sovits::t2s_model & model,
                            const int32_t * input_tokens,
                            int64_t T_in,
                            const float * input_bert,
                            int32_t max_new_tokens,
                            uint32_t seed,
                            std::vector<int32_t> & codes) {
    codes.clear();

    if (session.n_batch != 1 || session.slots.size() != 1) return false;

    const int64_t d_model  = model.hparams.hidden_dim;
    const int64_t vocab    = model.hparams.vocab_size;
    const int32_t eos      = (int32_t) model.hparams.eos;
    const int64_t T_ref    = session.ref_T_ref;
    const int64_t T_prompt = session.ref_T_prompt;
    if (T_ref <= 0 || T_prompt <= 0 || max_new_tokens <= 0) return false;

    const int64_t N = T_ref + T_in + T_prompt;
    if (N + max_new_tokens - 1 > (int64_t) session.slot_size) return false;

    // --- 1. Target embedding at position T_ref. ---
    std::vector<float> x_host((size_t) (d_model * N));
    {
        std::vector<float> target;
        if (!compute_target_embedding(session, model, input_tokens, T_in, input_bert,
                                      T_ref, target)) {
            return false;
        }
        std::copy(target.begin(), target.end(),
                  x_host.begin() + (size_t) (d_model * T_ref));
    }

    // --- 2. Backend three-part concat: [ref text | target text | ref audio]. ---
    ggml_backend_tensor_get(session.ref_text_emb, x_host.data(), 0,
                            (size_t) (d_model * T_ref * (int64_t) sizeof(float)));
    ggml_backend_tensor_get(session.ref_audio_emb,
                            x_host.data() + d_model * (T_ref + T_in), 0,
                            (size_t) (d_model * T_prompt * (int64_t) sizeof(float)));

    const int slot = gpt_sovits::t2s_session_find_free_slot(session);
    if (slot < 0) return false;
    slot_guard sg(session, slot);

    std::mt19937 rng(seed);
    std::exponential_distribution<float> dist(1.0f);
    std::vector<float> noise((size_t) vocab);
    std::vector<float> seen((size_t) vocab, 0.0f);

    int32_t current = -1;

    // --- 3. Flex prefill. ---
    {
        gpt_sovits::t2s_batch_plan plan;
        plan.n_query = { (int) N };

        gpt_sovits::t2s_flex_graph graph =
            gpt_sovits::t2s_session_build_flex_graph(session, model, plan);
        if (graph.ctx == nullptr) return false;
        flex_guard fg(graph);

        if (graph.x_prefill == nullptr || graph.n_active != 1) return false;

        ggml_backend_tensor_set(graph.x_prefill, x_host.data(), 0,
                                (size_t) (d_model * N * (int64_t) sizeof(float)));
        if (graph.seen_mask != nullptr) {
            ggml_backend_tensor_set(graph.seen_mask, seen.data(), 0,
                                    (size_t) (vocab * (int64_t) sizeof(float)));
        }
        fill_exp_noise(noise, rng, dist);
        ggml_backend_tensor_set(graph.exp_noise, noise.data(), 0,
                                (size_t) (vocab * (int64_t) sizeof(float)));

        gpt_sovits::t2s_session_flex_advance(session, plan, graph);
        if (ggml_backend_graph_compute(session.backend, graph.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }

        int32_t sampled = -1, greedy = -1;
        read_sampled_greedy(graph, sampled, greedy);

        // EOS from either sampler ends the request before accepting a token.
        if (sampled == eos || greedy == eos) {
            return true;
        }

        codes.push_back(sampled);
        seen[(size_t) sampled] = 1.0f;
        current = sampled;
    }

    if ((int32_t) codes.size() >= max_new_tokens) return true;

    // --- 4. Persistent decode loop. ---
    if (session.gf_dec == nullptr) {
        if (!gpt_sovits::t2s_session_build_decode_graph(session, model)) {
            return false;
        }
    }
    if (session.gf_dec == nullptr) return false;

    while ((int32_t) codes.size() < max_new_tokens) {
        if (session.slots[slot].n_pos >= (int) session.slot_size) return false;

        ggml_backend_tensor_set(session.token_id, &current, 0, sizeof(int32_t));

        fill_exp_noise(noise, rng, dist);
        ggml_backend_tensor_set(session.exp_noise, noise.data(), 0,
                                (size_t) (vocab * (int64_t) sizeof(float)));
        ggml_backend_tensor_set(session.seen_mask, seen.data(), 0,
                                (size_t) (vocab * (int64_t) sizeof(float)));

        gpt_sovits::t2s_session_decode_advance(session);
        if (ggml_backend_graph_compute(session.backend, session.gf_dec) !=
            GGML_STATUS_SUCCESS) {
            return false;
        }

        int32_t sampled = -1, greedy = -1;
        ggml_backend_tensor_get(session.sampled, &sampled, 0, sizeof(int32_t));
        ggml_backend_tensor_get(session.greedy,  &greedy,  0, sizeof(int32_t));

        if (sampled == eos || greedy == eos) {
            return true;  // do not append the EOS candidate
        }

        codes.push_back(sampled);
        seen[(size_t) sampled] = 1.0f;
        current = sampled;
    }

    return true;
}

// ---------------------------------------------------------------------------
// End-to-end scenario shared by the CPU and Metal cases.
// ---------------------------------------------------------------------------

static void run_generate_integration(ggml_backend_t backend, bool stochastic = false) {
    backend_guard backend_holder(backend);
    ASSERT_NE(backend, nullptr);

    const std::string model_path = kTestDir + "models/s1v3-s2Gv2Pro-f16.gguf";

    {
        FILE * f = fopen(model_path.c_str(), "rb");
        if (!f) GTEST_SKIP() << "Model file not found: " << model_path;
        fclose(f);
    }

    model_guard model_holder;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(model_path, model_holder.m, backend));
    gpt_sovits::t2s_model & model = model_holder.m;

    if (model.weights.extract_latent.ssl_proj_w == nullptr ||
        model.weights.extract_latent.ssl_proj_w->ne[0] != 2) {
        GTEST_SKIP() << "Model extract_latent weights have incompatible shape";
    }

    // Short synthetic reference: T_ref=3, HuBERT frames=8 -> T_prompt=4.
    const int64_t T_ref = 3;
    const int64_t T_hub = 8;
    const std::vector<int32_t> ref_tokens = { 1, 2, 3 };
    std::vector<float> ref_bert((size_t) (1024 * T_ref), 0.1f);
    std::vector<float> hubert((size_t) (768 * T_hub), 0.5f);

    // top_k=1 makes sampled == greedy, so both paths are deterministic.
    gpt_sovits::t2s_sampler_config sampler_cfg;
    sampler_cfg.top_k              = stochastic ? 15 : 1;
    sampler_cfg.top_p              = stochastic ? 0.95f : 1.0f;
    sampler_cfg.temperature        = 1.0f;
    sampler_cfg.repetition_penalty = stochastic ? 1.35f : 1.0f;

    session_guard session_holder;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session_holder.s, model.hparams,
                                             backend, 1, 64, sampler_cfg));
    gpt_sovits::t2s_session & session = session_holder.s;

    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        session, model, ref_tokens.data(), T_ref, ref_bert.data(),
        hubert.data(), T_hub));

    ASSERT_NE(session.ref_text_emb, nullptr);
    ASSERT_NE(session.ref_audio_emb, nullptr);
    EXPECT_EQ(session.ref_T_ref, T_ref);
    EXPECT_EQ(session.ref_model, &model);

    const std::vector<uint8_t> ref_text_before = read_tensor_bytes(session.ref_text_emb);
    const std::vector<uint8_t> ref_audio_before = read_tensor_bytes(session.ref_audio_emb);
    ggml_tensor * ref_text_ptr = session.ref_text_emb;
    ggml_tensor * ref_audio_ptr = session.ref_audio_emb;

    // Short synthetic target.
    const int64_t T_in = 3;
    const std::vector<int32_t> target_tokens = { 7, 8, 9 };
    std::vector<float> target_bert((size_t) (1024 * T_in), 0.1f);

    gpt_sovits::t2s_generate_config gen_cfg;
    gen_cfg.max_new_tokens = 5;
    gen_cfg.seed           = 42;

    // --- First high-level call. ---
    gpt_sovits::t2s_generate_result first;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        session, model, target_tokens.data(), T_in, target_bert.data(), first, gen_cfg));

    ASSERT_FALSE(first.codes.empty()) << "synthetic target hit EOS during prefill";
    EXPECT_LE(first.codes.size(), (size_t) gen_cfg.max_new_tokens);

    // Slot released, reference retained, decode graph bound to this model.
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 0);
    EXPECT_EQ(session.slots[0].audio_offset, 0);
    EXPECT_EQ(session.ref_model, &model);
    EXPECT_NE(session.ref_text_emb, nullptr);
    EXPECT_NE(session.ref_audio_emb, nullptr);
    EXPECT_EQ(session.ref_text_emb, ref_text_ptr);
    EXPECT_EQ(session.ref_audio_emb, ref_audio_ptr);
    EXPECT_EQ(read_tensor_bytes(session.ref_text_emb), ref_text_before);
    EXPECT_EQ(read_tensor_bytes(session.ref_audio_emb), ref_audio_before);

    EXPECT_NE(session.gf_dec, nullptr);
    EXPECT_EQ(session.decode_model, &model);
    ggml_cgraph * first_graph = session.gf_dec;

    // --- Second call with the same seed reuses the decode graph. ---
    gpt_sovits::t2s_generate_result second;
    ASSERT_TRUE(gpt_sovits::t2s_session_generate(
        session, model, target_tokens.data(), T_in, target_bert.data(), second, gen_cfg));

    EXPECT_EQ(second.codes, first.codes);
    EXPECT_EQ(second.stop_reason, first.stop_reason);
    EXPECT_EQ(session.gf_dec, first_graph) << "decode graph should be reused";
    EXPECT_EQ(session.decode_model, &model);

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 0);
    EXPECT_EQ(session.slots[0].audio_offset, 0);
    EXPECT_EQ(session.ref_model, &model);
    EXPECT_EQ(session.ref_text_emb, ref_text_ptr);
    EXPECT_EQ(session.ref_audio_emb, ref_audio_ptr);
    EXPECT_EQ(read_tensor_bytes(session.ref_text_emb), ref_text_before);
    EXPECT_EQ(read_tensor_bytes(session.ref_audio_emb), ref_audio_before);

    session_guard manual_holder;
    auto & manual_session = manual_holder.s;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(manual_session, model.hparams,
                                           backend, 1, 64, sampler_cfg));
    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        manual_session, model, ref_tokens.data(), T_ref, ref_bert.data(),
        hubert.data(), T_hub));
    std::vector<int32_t> manual_codes;
    ASSERT_TRUE(manual_generate(manual_session, model, target_tokens.data(), T_in,
                                target_bert.data(), gen_cfg.max_new_tokens,
                                gen_cfg.seed, manual_codes));

    EXPECT_EQ(manual_codes, first.codes)
        << "manual block-level runner and t2s_session_generate diverged";
    for (int target_length : {5, 1, 3}) {
        SCOPED_TRACE(target_length);
        std::vector<int32_t> varied_tokens((size_t) target_length);
        std::vector<float> varied_bert((size_t) target_length * 1024);
        for (int t = 0; t < target_length; ++t) {
            varied_tokens[t] = 10 + t;
            for (int c = 0; c < 1024; ++c) {
                varied_bert[(size_t) t * 1024 + c] = 0.01f * ((c + 3 * t) % 13 - 6);
            }
        }
        gpt_sovits::t2s_generate_result actual;
        ASSERT_TRUE(gpt_sovits::t2s_session_generate(
            session, model, varied_tokens.data(), target_length,
            varied_bert.data(), actual, gen_cfg));
        ASSERT_TRUE(manual_generate(manual_session, model, varied_tokens.data(),
                                    target_length, varied_bert.data(),
                                    gen_cfg.max_new_tokens, gen_cfg.seed, manual_codes));
        EXPECT_EQ(actual.codes, manual_codes);
        EXPECT_EQ(session.gf_dec, first_graph);
    }

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 0);
    EXPECT_EQ(session.slots[0].audio_offset, 0);
    EXPECT_EQ(session.gf_dec, first_graph);
    EXPECT_EQ(session.ref_model, &model);
    EXPECT_EQ(session.ref_text_emb, ref_text_ptr);
    EXPECT_EQ(session.ref_audio_emb, ref_audio_ptr);
    EXPECT_EQ(read_tensor_bytes(session.ref_text_emb), ref_text_before);
    EXPECT_EQ(read_tensor_bytes(session.ref_audio_emb), ref_audio_before);
}

} // namespace

TEST(T2SGenerateIntegration, CpuBaselineF16) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    run_generate_integration(backend);
}

TEST(T2SGenerateIntegration, CpuSeededSamplingF16) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    run_generate_integration(backend, true);
}

TEST(T2SGenerateIntegration, MetalF16) {
#if defined(GGML_USE_METAL)
    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend == nullptr) {
        GTEST_SKIP() << "Metal backend unavailable at runtime";
    }
    run_generate_integration(backend);
#else
    GTEST_SKIP() << "Metal backend was not compiled in";
#endif
}
