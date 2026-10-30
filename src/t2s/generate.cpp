// src/t2s/generate.cpp
//
// Single-request synchronous T2S generation.
//
// One prefill pass over [ref text | target text | ref audio] produces the first
// candidate token.  Each subsequent pass decodes one token with the persistent
// decode graph.  The caller owns the reference cache and may reuse a decode
// graph across calls.

#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <random>
#include <utility>
#include <vector>

namespace gpt_sovits {

namespace {

static constexpr int64_t kBertDim = 1024;
static constexpr size_t kEmbedNodes = 64;

struct ctx_guard {
    ggml_context * p = nullptr;
    explicit ctx_guard(ggml_context * q) : p(q) {}
    ~ctx_guard() { if (p) ggml_free(p); }
    ctx_guard(const ctx_guard &) = delete;
    ctx_guard & operator=(const ctx_guard &) = delete;
};

struct gallocr_guard {
    ggml_gallocr_t p = nullptr;
    explicit gallocr_guard(ggml_gallocr_t q) : p(q) {}
    ~gallocr_guard() { if (p) ggml_gallocr_free(p); }
    gallocr_guard(const gallocr_guard &) = delete;
    gallocr_guard & operator=(const gallocr_guard &) = delete;
};

struct flex_graph_guard {
    t2s_flex_graph g;
    flex_graph_guard() = default;
    ~flex_graph_guard() { t2s_flex_graph_free(g); }
    flex_graph_guard(const flex_graph_guard &) = delete;
    flex_graph_guard & operator=(const flex_graph_guard &) = delete;
};

struct slot_release_guard {
    t2s_session & session;
    int slot_id;
    slot_release_guard(t2s_session & s, int id) : session(s), slot_id(id) {}
    ~slot_release_guard() { t2s_session_slot_release(session, slot_id); }
    slot_release_guard(const slot_release_guard &) = delete;
    slot_release_guard & operator=(const slot_release_guard &) = delete;
};

// Overflow-checked arithmetic for host-side length calculations (operands >= 0).
static bool mul_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) return false;
    out = a * b;
    return true;
}

static bool add_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a > std::numeric_limits<int64_t>::max() - b) return false;
    out = a + b;
    return true;
}

// Validate all non-mutating preconditions and compute the prefill length.
// On success N_out = T_ref + T_in + T_prompt.
static bool validate_request(const t2s_session       & session,
                             const t2s_model         & model,
                             const int32_t           * input_tokens,
                             int64_t                   T_in,
                             const float             * input_bert,
                             const t2s_generate_config & config,
                             int64_t                 & N_out)
{
    if (session.backend == nullptr || session.ctx_kv == nullptr ||
        session.buf_kv == nullptr || session.n_batch != 1 ||
        session.slot_size == 0 || session.slots.size() != 1) {
        fprintf(stderr, "%s: session must be initialized with n_batch=1\n", __func__);
        return false;
    }
    if (session.slots[0].n_pos != 0) {
        fprintf(stderr, "%s: the single slot is already active\n", __func__);
        return false;
    }
    if (model.backend == nullptr || model.ctx_w == nullptr ||
        model.buf_w == nullptr || model.backend != session.backend) {
        fprintf(stderr, "%s: model and session must share a live backend\n", __func__);
        return false;
    }

    const t2s_hparams & hp = model.hparams;
    const int64_t d_model  = (int64_t) hp.hidden_dim;
    const int64_t vocab    = (int64_t) hp.vocab_size;

    if (d_model <= 0 || d_model % 2 != 0 ||
        hp.n_head == 0 || d_model % hp.n_head != 0 ||
        hp.n_layer == 0 || hp.linear_units == 0 ||
        vocab <= 0 || hp.phoneme_vocab_size == 0) {
        fprintf(stderr, "%s: invalid model hyperparameters\n", __func__);
        return false;
    }
    if (vocab - 1 > (int64_t) std::numeric_limits<int32_t>::max()) {
        fprintf(stderr, "%s: vocabulary does not fit int32 token ids\n", __func__);
        return false;
    }
    // The sampler treats the last vocabulary entry as EOS.  Its non-EOS output
    // must be inside the SoVITS codebook.
    if ((int64_t) hp.eos <= 0 || (int64_t) hp.eos != vocab - 1) {
        fprintf(stderr, "%s: EOS (%u) must be the last vocabulary entry (%lld)\n",
                __func__, hp.eos, (long long) (vocab - 1));
        return false;
    }

    const t2s_model_weights & w = model.weights;
    if (w.lm_head_w == nullptr ||
        w.embed.text_embedding == nullptr ||
        w.embed.bert_proj_w == nullptr ||
        w.embed.bert_proj_b == nullptr ||
        w.embed.text_pos_alpha == nullptr ||
        w.embed.audio_embedding == nullptr ||
        w.embed.audio_pos_alpha == nullptr ||
        w.extract_latent.codebook == nullptr ||
        w.attention.size() != (size_t) hp.n_layer) {
        fprintf(stderr, "%s: model weights are incomplete\n", __func__);
        return false;
    }

    if (w.embed.text_embedding->ne[0] != d_model ||
        w.embed.text_embedding->ne[1] != (int64_t) hp.phoneme_vocab_size ||
        w.embed.audio_embedding->ne[0] != d_model ||
        w.embed.audio_embedding->ne[1] != vocab ||
        w.lm_head_w->ne[0] != d_model ||
        w.lm_head_w->ne[1] != vocab ||
        w.embed.bert_proj_w->ne[0] != kBertDim ||
        w.embed.bert_proj_w->ne[1] != d_model ||
        w.embed.bert_proj_b->ne[0] != d_model) {
        fprintf(stderr, "%s: model embedding/projection shapes are inconsistent\n", __func__);
        return false;
    }

    if ((int64_t) w.extract_latent.codebook->ne[1] < (int64_t) hp.eos) {
        fprintf(stderr, "%s: codebook (%lld codes) does not cover all non-EOS tokens (%u)\n",
                __func__, (long long) w.extract_latent.codebook->ne[1], hp.eos);
        return false;
    }

    for (uint32_t i = 0; i < hp.n_layer; i++) {
        const t2s_attention_block_weights & layer = w.attention[i];
        if (layer.qkv_w == nullptr || layer.qkv_b == nullptr ||
            layer.out_proj_w == nullptr || layer.out_proj_b == nullptr ||
            layer.ln1_w == nullptr || layer.ln1_b == nullptr ||
            layer.ffn_up_w == nullptr || layer.ffn_up_b == nullptr ||
            layer.ffn_down_w == nullptr || layer.ffn_down_b == nullptr ||
            layer.ln2_w == nullptr || layer.ln2_b == nullptr) {
            fprintf(stderr, "%s: attention layer %u weights are incomplete\n", __func__, i);
            return false;
        }
        if (layer.qkv_w->ne[0] != d_model || layer.qkv_w->ne[1] != 3 * d_model ||
            layer.out_proj_w->ne[0] != d_model ||
            layer.ffn_up_w->ne[0] != d_model ||
            layer.ffn_up_w->ne[1] != (int64_t) hp.linear_units ||
            layer.ffn_down_w->ne[0] != (int64_t) hp.linear_units ||
            layer.ffn_down_w->ne[1] != d_model) {
            fprintf(stderr, "%s: attention layer %u has inconsistent shapes\n", __func__, i);
            return false;
        }
    }

    if (session.k_caches.size() != (size_t) hp.n_layer ||
        session.v_caches.size() != (size_t) hp.n_layer ||
        session.k_caches[0]->ne[0] != d_model ||
        session.v_caches[0]->ne[0] != d_model) {
        fprintf(stderr, "%s: session KV caches do not match the model\n", __func__);
        return false;
    }

    const ggml_tensor * sampler_inputs[] = {session.seen_mask, session.exp_noise};
    for (const auto * tensor : sampler_inputs) {
        if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
            tensor->ne[0] != vocab || tensor->ne[1] != session.n_batch ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            !ggml_is_contiguous(tensor)) {
            fprintf(stderr, "%s: session sampler inputs do not match the model vocabulary\n", __func__);
            return false;
        }
    }

    if (session.suppress_eos == nullptr || session.suppress_eos->type != GGML_TYPE_I32 ||
        session.suppress_eos->ne[0] != session.n_batch ||
        !ggml_is_vector(session.suppress_eos) || !ggml_is_contiguous(session.suppress_eos)) {
        fprintf(stderr, "%s: session EOS suppression input is invalid\n", __func__);
        return false;
    }

    const int64_t T_ref    = session.ref_T_ref;
    const int64_t T_prompt = session.ref_T_prompt;
    if (session.ref_model != &model ||
        session.ref_text_emb == nullptr || session.ref_audio_emb == nullptr ||
        T_ref <= 0 || T_prompt <= 0 ||
        session.ref_text_emb->ne[0] != d_model ||
        session.ref_text_emb->ne[1] != T_ref ||
        session.ref_audio_emb->ne[0] != d_model ||
        session.ref_audio_emb->ne[1] != T_prompt) {
        fprintf(stderr, "%s: cached reference embeddings are missing or incompatible\n", __func__);
        return false;
    }
    if (session.ref_prompt_tokens.size() != (uint64_t) T_prompt) {
        fprintf(stderr, "%s: cached reference prompt token count does not match embeddings\n", __func__);
        return false;
    }
    for (int32_t token : session.ref_prompt_tokens) {
        if (token < 0 || (uint32_t) token >= hp.eos) {
            fprintf(stderr, "%s: cached reference prompt token %d outside [0, %u)\n",
                    __func__, token, hp.eos);
            return false;
        }
    }

    if (session.gf_dec != nullptr && session.decode_model != &model) {
        fprintf(stderr, "%s: existing decode graph was built for another model\n", __func__);
        return false;
    }

    const t2s_sampler_config & sc = session.sampler_cfg;
    if (!std::isfinite(sc.top_p) ||
        !std::isfinite(sc.temperature) ||
        !std::isfinite(sc.repetition_penalty) ||
        !(sc.repetition_penalty > 0.0f)) {
        fprintf(stderr, "%s: sampler configuration must be finite with repetition_penalty > 0\n",
                __func__);
        return false;
    }

    if (input_tokens == nullptr || input_bert == nullptr) {
        fprintf(stderr, "%s: input_tokens and input_bert must be non-null\n", __func__);
        return false;
    }
    if (T_in <= 0) {
        fprintf(stderr, "%s: T_in=%lld must be > 0\n", __func__, (long long) T_in);
        return false;
    }

    int64_t token_bytes = 0;
    int64_t bert_elems  = 0;
    int64_t bert_bytes  = 0;
    if (!mul_ok(T_in, (int64_t) sizeof(int32_t), token_bytes) ||
        !mul_ok(T_in, kBertDim, bert_elems) ||
        !mul_ok(bert_elems, (int64_t) sizeof(float), bert_bytes)) {
        fprintf(stderr, "%s: input lengths overflow byte counts\n", __func__);
        return false;
    }

    // Validate lengths before touching the caller's input buffers.  A bogus
    // huge T_in must be rejected here rather than scanned out of bounds.
    int64_t N = 0;
    if (!add_ok(T_ref, T_in, N) || !add_ok(N, T_prompt, N)) {
        fprintf(stderr, "%s: sequence length overflows\n", __func__);
        return false;
    }

    if (config.min_new_tokens < 0) {
        fprintf(stderr, "%s: min_new_tokens must be nonnegative\n", __func__);
        return false;
    }
    const int64_t M = (int64_t) config.max_new_tokens;
    if (M <= 0) {
        fprintf(stderr, "%s: max_new_tokens must be > 0\n", __func__);
        return false;
    }
    constexpr int64_t kMaxInt = (int64_t) std::numeric_limits<int>::max();
    if (N > kMaxInt || M > kMaxInt || (int64_t) session.slot_size > kMaxInt) {
        fprintf(stderr, "%s: lengths do not fit in int indices\n", __func__);
        return false;
    }
    // Positions used: prefill writes N tokens; the remaining M-1 predictions
    // need one decode position each.
    const int64_t slot_size = (int64_t) session.slot_size;
    if (N > slot_size || M - 1 > slot_size - N) {
        fprintf(stderr, "%s: N=%lld + max_new_tokens=%lld - 1 exceeds slot_size=%lld\n",
                __func__, (long long) N, (long long) M, (long long) slot_size);
        return false;
    }

    int64_t x_elems = 0;
    int64_t x_bytes = 0;
    if (!mul_ok(d_model, N, x_elems) ||
        !mul_ok(x_elems, (int64_t) sizeof(float), x_bytes)) {
        fprintf(stderr, "%s: prefill buffer size overflows\n", __func__);
        return false;
    }

    for (int64_t i = 0; i < T_in; i++) {
        if ((int64_t) input_tokens[i] < 0 ||
            (int64_t) input_tokens[i] >= (int64_t) hp.phoneme_vocab_size) {
            fprintf(stderr, "%s: input token %lld (%d) is outside the phoneme vocabulary\n",
                    __func__, (long long) i, input_tokens[i]);
            return false;
        }
    }
    for (int64_t i = 0; i < bert_elems; i++) {
        if (!std::isfinite(input_bert[i])) {
            fprintf(stderr, "%s: BERT input contains a non-finite value at %lld\n",
                    __func__, (long long) i);
            return false;
        }
    }

    N_out = N;
    return true;
}

static void fill_exp_noise(std::vector<float> & buf,
                           std::mt19937 & rng,
                           std::exponential_distribution<float> & dist) {
    for (float & v : buf) {
        // Exp(1) samples are strictly positive; resample on the zero boundary
        // so the sampler's probs / noise cannot divide by zero.
        float x = 0.0f;
        do {
            x = dist(rng);
        } while (!(x > 0.0f));
        v = x;
    }
}

static bool token_is_legal(int32_t token, int64_t vocab) {
    return token >= 0 && (int64_t) token < vocab;
}

} // namespace

bool t2s_session_generate(t2s_session               & session,
                          const t2s_model             & model,
                          const int32_t               * input_tokens,
                          int64_t                       T_in,
                          const float                 * input_bert,
                          t2s_generate_result         & result,
                          const t2s_generate_config   & config)
{
    int64_t N = 0;
    if (!validate_request(session, model, input_tokens, T_in, input_bert, config, N)) {
        return false;
    }

    const int64_t d_model  = (int64_t) model.hparams.hidden_dim;
    const int64_t vocab    = (int64_t) model.hparams.vocab_size;
    const int64_t T_ref    = session.ref_T_ref;
    const int64_t T_prompt = session.ref_T_prompt;
    const int64_t M        = (int64_t) config.max_new_tokens;
    const int32_t eos      = (int32_t) model.hparams.eos;

    try {
        const int slot_id = t2s_session_find_free_slot(session);
        if (slot_id < 0) {
            fprintf(stderr, "%s: no free slot\n", __func__);
            return false;
        }
        slot_release_guard slot_guard(session, slot_id);

        // --- 1. Temporary graph: target text embedding at position T_ref. ---
        const int64_t target_offset = d_model * T_ref;
        const int64_t target_bytes  = d_model * T_in * (int64_t) sizeof(float);
        std::vector<float> x_host((size_t) (d_model * N));

        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * kEmbedNodes +
                                 ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ctx_guard ctx(ggml_init(params));
            if (ctx.p == nullptr) {
                fprintf(stderr, "%s: ggml_init() for embedding graph failed\n", __func__);
                return false;
            }

            struct ggml_cgraph * gf =
                ggml_new_graph_custom(ctx.p, GGML_DEFAULT_GRAPH_SIZE, false);
            if (gf == nullptr) {
                fprintf(stderr, "%s: ggml_new_graph_custom() failed\n", __func__);
                return false;
            }

            struct ggml_tensor * input_token_t =
                ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T_in);
            ggml_set_name(input_token_t, "input_token");
            ggml_set_input(input_token_t);

            struct ggml_tensor * input_bert_t =
                ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, kBertDim, T_in);
            ggml_set_name(input_bert_t, "input_bert");
            ggml_set_input(input_bert_t);

            struct ggml_tensor * target_emb = t2s_embed_input_forward(
                ctx.p, input_token_t, input_bert_t, T_ref, model.weights.embed);
            ggml_set_name(target_emb, "target_emb");
            ggml_set_output(target_emb);
            ggml_build_forward_expand(gf, target_emb);

            if (target_emb->ne[0] != d_model || target_emb->ne[1] != T_in) {
                fprintf(stderr, "%s: target embedding shape {%lld, %lld} != {%lld, %lld}\n",
                        __func__, (long long) target_emb->ne[0],
                        (long long) target_emb->ne[1],
                        (long long) d_model, (long long) T_in);
                return false;
            }

            ggml_backend_buffer_type_t buft =
                ggml_backend_get_default_buffer_type(session.backend);
            if (buft == nullptr) {
                fprintf(stderr, "%s: backend has no default buffer type\n", __func__);
                return false;
            }

            gallocr_guard alloc(ggml_gallocr_new(buft));
            if (alloc.p == nullptr) {
                fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
                return false;
            }
            if (!ggml_gallocr_alloc_graph(alloc.p, gf)) {
                fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
                return false;
            }

            ggml_backend_tensor_set(input_token_t, input_tokens, 0,
                                    (size_t) (T_in * (int64_t) sizeof(int32_t)));
            ggml_backend_tensor_set(input_bert_t, input_bert, 0,
                                    (size_t) (kBertDim * T_in * (int64_t) sizeof(float)));

            if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: embedding graph compute failed\n", __func__);
                return false;
            }

            ggml_backend_tensor_get(target_emb, x_host.data() + target_offset, 0,
                                    (size_t) target_bytes);
        }

        // --- 2. Host concat: [ref text | target text | ref audio]. ---
        ggml_backend_tensor_get(session.ref_text_emb, x_host.data(), 0,
                                (size_t) (d_model * T_ref * (int64_t) sizeof(float)));
        ggml_backend_tensor_get(session.ref_audio_emb,
                                x_host.data() + d_model * (T_ref + T_in), 0,
                                (size_t) (d_model * T_prompt * (int64_t) sizeof(float)));

        // --- 3. Prefill through the flexible graph. ---
        std::mt19937 rng(config.seed);
        std::exponential_distribution<float> dist(1.0f);
        std::vector<float> noise((size_t) vocab);
        std::vector<float> seen((size_t) vocab, 0.0f);
        for (int32_t token : session.ref_prompt_tokens) {
            seen[(size_t) token] = 1.0f;
        }
        std::vector<int32_t> codes;
        codes.reserve((size_t) M);

        t2s_stop_reason reason = t2s_stop_reason::max_new_tokens;
        int32_t current_token  = -1;
        bool need_decode       = false;

        {
            t2s_batch_plan plan;
            plan.n_query = { (int) N };

            flex_graph_guard flex;
            flex.g = t2s_session_build_flex_graph(session, model, plan);
            if (flex.g.ctx == nullptr) {
                fprintf(stderr, "%s: prefill graph build failed\n", __func__);
                return false;
            }
            if (flex.g.x_prefill == nullptr || flex.g.suppress_eos == nullptr ||
                flex.g.n_active != 1) {
                fprintf(stderr, "%s: prefill graph has an unexpected layout\n", __func__);
                return false;
            }

            ggml_backend_tensor_set(flex.g.x_prefill, x_host.data(), 0,
                                    (size_t) (d_model * N * (int64_t) sizeof(float)));

            if (flex.g.seen_mask != nullptr) {
                ggml_backend_tensor_set(flex.g.seen_mask, seen.data(), 0,
                                        (size_t) (vocab * (int64_t) sizeof(float)));
            }

            const int32_t suppress_eos = config.min_new_tokens > 0 ? 1 : 0;
            ggml_backend_tensor_set(flex.g.suppress_eos, &suppress_eos, 0, sizeof(suppress_eos));
            fill_exp_noise(noise, rng, dist);
            ggml_backend_tensor_set(flex.g.exp_noise, noise.data(), 0,
                                    (size_t) (vocab * (int64_t) sizeof(float)));

            t2s_session_flex_advance(session, plan, flex.g);
            if (ggml_backend_graph_compute(session.backend, flex.g.gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: prefill graph compute failed\n", __func__);
                return false;
            }

            int32_t sampled = -1;
            int32_t greedy  = -1;
            ggml_backend_tensor_get(flex.g.sampled, &sampled, 0, sizeof(int32_t));
            ggml_backend_tensor_get(flex.g.greedy,  &greedy,  0, sizeof(int32_t));

            if (!token_is_legal(sampled, vocab) || !token_is_legal(greedy, vocab)) {
                fprintf(stderr, "%s: sampler produced an out-of-range token (%d, %d)\n",
                        __func__, sampled, greedy);
                return false;
            }

            if (sampled == eos || greedy == eos) {
                reason = t2s_stop_reason::eos;
            } else {
                codes.push_back(sampled);
                seen[(size_t) sampled] = 1.0f;
                current_token = sampled;
                need_decode   = codes.size() < (size_t) M;
            }
        } // flex graph metadata is released here, before the decode graph is built.

        // --- 4. Decode loop with the persistent decode graph. ---
        if (need_decode) {
            if (session.gf_dec == nullptr) {
                if (!t2s_session_build_decode_graph(session, model)) {
                    fprintf(stderr, "%s: decode graph build failed\n", __func__);
                    return false;
                }
            }
            if (session.gf_dec == nullptr || session.decode_model != &model ||
                session.token_id == nullptr || session.position == nullptr ||
                session.seen_mask == nullptr || session.exp_noise == nullptr ||
                session.suppress_eos == nullptr || session.sampled == nullptr ||
                session.greedy == nullptr) {
                fprintf(stderr, "%s: decode graph is incomplete or belongs to another model\n",
                        __func__);
                return false;
            }

            while (codes.size() < (size_t) M) {
                // Guard the position written by this decode step.
                if (session.slots[slot_id].n_pos >= (int) session.slot_size) {
                    fprintf(stderr, "%s: decode position exceeds slot_size\n", __func__);
                    return false;
                }

                ggml_backend_tensor_set(session.token_id, &current_token, 0,
                                        sizeof(int32_t));

                const int32_t suppress_eos = codes.size() < (size_t) config.min_new_tokens ? 1 : 0;
                ggml_backend_tensor_set(session.suppress_eos, &suppress_eos, 0, sizeof(suppress_eos));
                fill_exp_noise(noise, rng, dist);
                ggml_backend_tensor_set(session.exp_noise, noise.data(), 0,
                                        (size_t) (vocab * (int64_t) sizeof(float)));
                ggml_backend_tensor_set(session.seen_mask, seen.data(), 0,
                                        (size_t) (vocab * (int64_t) sizeof(float)));

                t2s_session_decode_advance(session);
                if (ggml_backend_graph_compute(session.backend, session.gf_dec) !=
                    GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: decode graph compute failed\n", __func__);
                    return false;
                }

                int32_t sampled = -1;
                int32_t greedy  = -1;
                ggml_backend_tensor_get(session.sampled, &sampled, 0, sizeof(int32_t));
                ggml_backend_tensor_get(session.greedy,  &greedy,  0, sizeof(int32_t));

                if (!token_is_legal(sampled, vocab) || !token_is_legal(greedy, vocab)) {
                    fprintf(stderr, "%s: sampler produced an out-of-range token (%d, %d)\n",
                            __func__, sampled, greedy);
                    return false;
                }

                // Either sampler hitting EOS terminates before accepting the
                // candidate from this step.
                if (sampled == eos || greedy == eos) {
                    reason = t2s_stop_reason::eos;
                    break;
                }

                codes.push_back(sampled);
                seen[(size_t) sampled] = 1.0f;
                current_token = sampled;
            }
        }

        // Commit only on success; failures leave the caller's result untouched.
        result.codes       = std::move(codes);
        result.stop_reason = reason;
        return true;
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        return false;
    }
}

} // namespace gpt_sovits
