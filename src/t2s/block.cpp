#include "gpt_sovits/t2s.h"
#include "ggml.h"

#include <cmath>

namespace gpt_sovits {

// -1e10 causes softmax to underflow to zero without needing a dedicated
// masked_fill op (matches Python AR.models.utils behavior).
static constexpr float kSamplerMaskLogit = -1.0e10f;

struct t2s_sampler_probs_result {
    ::ggml_tensor * probs_sorted;
    ::ggml_tensor * sorted_indices;
};

static ::ggml_tensor * flatten_vector_like(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);
    GGML_ASSERT(x->ne[1] == 1 || x->ne[0] == 1);

    if (x->ne[1] == 1) {
        return ggml_reshape_1d(ctx, x, x->ne[0]);
    }

    ::ggml_tensor * flat = ggml_reshape_1d(ctx, x, x->ne[1]);
    return ggml_cont(ctx, flat);
}

static ::ggml_tensor * ensure_scalar_rows(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    ::ggml_tensor * flat = flatten_vector_like(ctx, x);
    return ggml_reshape_2d(ctx, flat, 1, flat->ne[0]);
}

static ::ggml_tensor * masked_lerp(
    ::ggml_context * ctx,
    ::ggml_tensor  * base,
    ::ggml_tensor  * selected,
    ::ggml_tensor  * mask)
{
    // base + mask * (selected - base), with mask in {0, 1}
    return ggml_add(ctx, base, ggml_mul(ctx, mask, ggml_sub(ctx, selected, base)));
}

// Top-p filtering in sorted-logit space (matches AR.models.utils).
static ::ggml_tensor * apply_top_p_sorted(
    ::ggml_context * ctx,
    ::ggml_tensor  * sorted_logits,
    float            top_p)
{
    if (!(top_p < 1.0f)) {
        return sorted_logits;
    }

    const int64_t vocab = sorted_logits->ne[0];
    if (vocab <= 1) {
        return sorted_logits;
    }

    ::ggml_tensor * sorted_probs = ggml_soft_max(ctx, sorted_logits);
    ::ggml_tensor * cum_probs = ggml_cumsum(ctx, sorted_probs);

    ::ggml_tensor * threshold = ggml_fill(ctx, sorted_logits, top_p);
    ::ggml_tensor * remove_mask = ggml_step(ctx, ggml_sub(ctx, cum_probs, threshold));

    // Shift right by one so the first token above the threshold is kept.
    ::ggml_tensor * keep_head = ggml_view_1d(ctx, remove_mask, 1, 0);
    keep_head = ggml_cont(ctx, keep_head);
    keep_head = ggml_fill(ctx, keep_head, 0.0f);
    ::ggml_tensor * remove_tail = ggml_view_1d(
        ctx,
        remove_mask,
        vocab - 1,
        /* offset = */ ggml_element_size(remove_mask));
    ::ggml_tensor * shifted_remove = ggml_concat(ctx, keep_head, remove_tail, 0);
    shifted_remove = ggml_cont(ctx, shifted_remove);

    ::ggml_tensor * masked_value = ggml_fill(ctx, sorted_logits, kSamplerMaskLogit);
    return masked_lerp(ctx, sorted_logits, masked_value, shifted_remove);
}

static ::ggml_tensor * apply_top_k_sorted(
    ::ggml_context * ctx,
    ::ggml_tensor  * sorted_logits,
    int              top_k)
{
    if (top_k <= 0) {
        return sorted_logits;
    }

    const int64_t vocab = sorted_logits->ne[0];
    if (top_k >= vocab) {
        return sorted_logits;
    }

    const size_t element_size = ggml_element_size(sorted_logits);
    ::ggml_tensor * pivot = ggml_view_1d(
        ctx,
        sorted_logits,
        1,
        /* offset = */ (size_t) (top_k - 1) * element_size);
    pivot = ggml_repeat(ctx, pivot, sorted_logits);

    ::ggml_tensor * remove_mask = ggml_step(ctx, ggml_sub(ctx, pivot, sorted_logits));
    ::ggml_tensor * masked_value = ggml_fill(ctx, sorted_logits, kSamplerMaskLogit);

    return masked_lerp(ctx, sorted_logits, masked_value, remove_mask);
}

static ::ggml_tensor * apply_repetition_penalty(
    ::ggml_context * ctx,
    ::ggml_tensor  * logits,
    ::ggml_tensor  * seen_mask,
    float            repetition_penalty)
{
    if (seen_mask == nullptr || repetition_penalty == 1.0f) {
        return logits;
    }

    GGML_ASSERT(seen_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->ne[0] == seen_mask->ne[0]);

    ::ggml_tensor * seen = ggml_step(ctx, seen_mask);
    ::ggml_tensor * neg_mask = ggml_step(ctx, ggml_neg(ctx, logits));

    ::ggml_tensor * pos_scaled = ggml_scale(ctx, logits, 1.0f / repetition_penalty);
    ::ggml_tensor * neg_scaled = ggml_scale(ctx, logits, repetition_penalty);
    ::ggml_tensor * selected = masked_lerp(ctx, pos_scaled, neg_scaled, neg_mask);

    return masked_lerp(ctx, logits, selected, seen);
}

static ::ggml_tensor * sovits_extract_latent_block_forward(
    ::ggml_context                       * ctx,
    ::ggml_tensor                        * hubert_feature,
    const sovits_extract_latent_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(hubert_feature != nullptr);
    GGML_ASSERT(weights.ssl_proj_w != nullptr);
    GGML_ASSERT(weights.ssl_proj_b != nullptr);
    GGML_ASSERT(weights.codebook != nullptr);

    GGML_ASSERT(hubert_feature->ne[0] == weights.ssl_proj_w->ne[1]);
    GGML_ASSERT(weights.ssl_proj_w->ne[0] == 2);
    GGML_ASSERT(weights.ssl_proj_w->ne[1] == weights.codebook->ne[0]);
    GGML_ASSERT(weights.ssl_proj_w->ne[2] == weights.ssl_proj_b->ne[0]);

    // ggml_conv_1d expects data laid out as {time, channels}. Convert from the
    // project's usual {channels, time} convention, then transpose back.
    ::ggml_tensor * ssl_input = ggml_permute(ctx, hubert_feature, 1, 0, 2, 3);
    ssl_input = ggml_cont(ctx, ssl_input);

    ::ggml_tensor * ssl = ggml_conv_1d(
        ctx,
        weights.ssl_proj_w,
        ssl_input,
        /* s0 = */ 2,
        /* p0 = */ 0,
        /* d0 = */ 1);
    ssl = ggml_permute(ctx, ssl, 1, 0, 2, 3);
    ssl = ggml_cont(ctx, ssl);
    ssl = ggml_add(ctx, ssl, weights.ssl_proj_b);

    // Nearest-code lookup for the single RVQ layer.
    //
    // Python computes:
    //   dist = -(||x||^2 - 2 xE^T + ||E||^2)
    //   codes = argmax(dist)
    // The per-token ||x||^2 term does not change argmax, so we use the simpler
    // equivalent score = xE^T - 0.5 ||E||^2.
    ::ggml_tensor * score = ggml_mul_mat(ctx, weights.codebook, ssl);

    ::ggml_tensor * codebook_sq = ggml_sqr(ctx, weights.codebook);
    ::ggml_tensor * codebook_norm = ggml_sum_rows(ctx, codebook_sq);
    codebook_norm = ggml_reshape_1d(ctx, codebook_norm, weights.codebook->ne[1]);
    codebook_norm = ggml_scale(ctx, codebook_norm, 0.5f);

    score = ggml_sub(ctx, score, codebook_norm);

    return ggml_argmax(ctx, score);
}

static t2s_sampler_probs_result build_sampler_probs(
    ::ggml_context * ctx,
    ::ggml_tensor  * logits,
    ::ggml_tensor  * seen_mask,
    int              top_k,
    float            top_p,
    float            temperature,
    float            repetition_penalty)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(logits != nullptr);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(repetition_penalty > 0.0f);

    ::ggml_tensor * logits_vec = flatten_vector_like(ctx, logits);
    const int64_t vocab = logits_vec->ne[0];
    ::ggml_tensor * seen_mask_vec = nullptr;

    if (seen_mask != nullptr) {
        GGML_ASSERT(seen_mask->type == GGML_TYPE_F32);
        seen_mask_vec = flatten_vector_like(ctx, seen_mask);
        GGML_ASSERT(seen_mask_vec->ne[0] == vocab);
    }

    ::ggml_tensor * adjusted = apply_repetition_penalty(
        ctx,
        logits_vec,
        seen_mask_vec,
        repetition_penalty);

    ::ggml_tensor * sorted_indices = ggml_argsort(ctx, adjusted, GGML_SORT_ORDER_DESC);

    ::ggml_tensor * logits_for_gather = ensure_scalar_rows(ctx, adjusted);
    ::ggml_tensor * sorted_logits = ggml_get_rows(ctx, logits_for_gather, sorted_indices);
    sorted_logits = ggml_reshape_1d(ctx, sorted_logits, vocab);

    sorted_logits = apply_top_p_sorted(ctx, sorted_logits, top_p);

    const float inv_temperature = 1.0f / fmaxf(temperature, 1.0e-5f);
    sorted_logits = ggml_scale(ctx, sorted_logits, inv_temperature);

    sorted_logits = apply_top_k_sorted(ctx, sorted_logits, top_k);

    t2s_sampler_probs_result result;
    result.probs_sorted = ggml_soft_max(ctx, sorted_logits);
    result.sorted_indices = sorted_indices;
    return result;
}

static ::ggml_tensor * pick_sampler_token(
    ::ggml_context * ctx,
    ::ggml_tensor  * probs_sorted,
    ::ggml_tensor  * sorted_indices,
    ::ggml_tensor  * exp_noise)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(probs_sorted != nullptr);
    GGML_ASSERT(sorted_indices != nullptr);
    GGML_ASSERT(probs_sorted->type == GGML_TYPE_F32);
    GGML_ASSERT(sorted_indices->type == GGML_TYPE_I32);

    ::ggml_tensor * probs_vec = flatten_vector_like(ctx, probs_sorted);
    ::ggml_tensor * indices_vec = flatten_vector_like(ctx, sorted_indices);
    const int64_t vocab = probs_vec->ne[0];

    GGML_ASSERT(indices_vec->ne[0] == vocab);

    ::ggml_tensor * score = probs_vec;
    if (exp_noise != nullptr) {
        GGML_ASSERT(exp_noise->type == GGML_TYPE_F32);
        ::ggml_tensor * noise_vec = flatten_vector_like(ctx, exp_noise);
        GGML_ASSERT(noise_vec->ne[0] == vocab);
        score = ggml_div(ctx, probs_vec, noise_vec);
    }

    ::ggml_tensor * score_matrix = ggml_reshape_2d(ctx, score, vocab, 1);
    ::ggml_tensor * picked_rank = ggml_argmax(ctx, score_matrix);

    ::ggml_tensor * picked_token = ggml_get_rows(ctx, ensure_scalar_rows(ctx, indices_vec), picked_rank);
    return ggml_reshape_1d(ctx, picked_token, 1);
}

t2s_sampler_result t2s_sampler_block_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * y,
    ::ggml_tensor  * lm_head_w,
    ::ggml_tensor  * seen_mask,
    int              top_k,
    float            top_p,
    float            temperature,
    float            repetition_penalty,
    ::ggml_tensor  * exp_noise)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(y != nullptr);
    GGML_ASSERT(lm_head_w != nullptr);

    // Project hidden states to logits: {d_model, n_batch} → {vocab, n_batch}
    ::ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_w, y);

    const int64_t vocab   = logits->ne[0];
    const int64_t n_batch = logits->ne[1];
    GGML_ASSERT(n_batch >= 1);

    // Process each batch element independently via 1D column views.
    // Columns of a contiguous {vocab, n_batch} tensor are contiguous
    // (ne[0] is the innermost dimension), so ggml_view_1d works directly.
    std::vector<::ggml_tensor *> sampled_list;
    std::vector<::ggml_tensor *> greedy_list;

    for (int64_t b = 0; b < n_batch; b++) {
        // Extract 1D column view for batch element b.
        ::ggml_tensor * logits_b = ggml_view_1d(ctx, logits, vocab, b * logits->nb[1]);

        ::ggml_tensor * seen_mask_b = nullptr;
        if (seen_mask != nullptr) {
            seen_mask_b = ggml_view_1d(ctx, seen_mask, vocab, b * seen_mask->nb[1]);
        }

        ::ggml_tensor * exp_noise_b = nullptr;
        if (exp_noise != nullptr) {
            exp_noise_b = ggml_view_1d(ctx, exp_noise, vocab, b * exp_noise->nb[1]);
        }

        // Run 1D sampler logic on this batch element.
        t2s_sampler_probs_result probs = build_sampler_probs(
            ctx, logits_b, seen_mask_b, top_k, top_p, temperature, repetition_penalty);

        // Greedy token: sorted_indices[0] is the argmax (highest logit after
        // repetition penalty, before temperature/noise).
        ::ggml_tensor * indices_vec = flatten_vector_like(ctx, probs.sorted_indices);
        ::ggml_tensor * greedy = ggml_view_1d(ctx, indices_vec, 1, 0);
        greedy = ggml_reshape_1d(ctx, ggml_cont(ctx, greedy), 1);

        // Sampled token.
        ::ggml_tensor * sampled = pick_sampler_token(
            ctx, probs.probs_sorted, probs.sorted_indices, exp_noise_b);

        sampled_list.push_back(sampled);
        greedy_list.push_back(greedy);
    }

    // Combine per-batch results via concat along dim 0.
    t2s_sampler_result result;
    if (n_batch == 1) {
        result.sampled = sampled_list[0];
        result.greedy  = greedy_list[0];
    } else {
        result.sampled = sampled_list[0];
        result.greedy  = greedy_list[0];
        for (int64_t b = 1; b < n_batch; b++) {
            result.sampled = ggml_concat(ctx, result.sampled, sampled_list[b], 0);
            result.greedy  = ggml_concat(ctx, result.greedy,  greedy_list[b],  0);
        }
    }

    return result;
}

static ::ggml_tensor * build_sine_positional_embedding(
    ::ggml_context * ctx,
    int64_t          d_model,
    int64_t          n_tokens,
    int64_t          start_position = 0)
{
    GGML_ASSERT(d_model > 0);
    GGML_ASSERT(n_tokens >= 0);
    GGML_ASSERT(start_position >= 0);
    GGML_ASSERT(d_model % 2 == 0);

    if (n_tokens == 0) {
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 0);
    }

    ::ggml_tensor * positions = ggml_arange(
        ctx, (float) start_position, (float) (start_position + n_tokens), 1.0f);

    // ggml_timestep_embedding uses the same frequency schedule as the Python
    // implementation, but outputs [cos half | sin half]. Reorder it to the
    // GPT-SoVITS layout [sin0, cos0, sin1, cos1, ...] explicitly instead of
    // relying on reshape/permute stride tricks.
    ::ggml_tensor * pe_half = ggml_timestep_embedding(ctx, positions, (int) d_model, 10000);
    const int64_t half = d_model / 2;
    const size_t element_size = ggml_element_size(pe_half);

    // Build {1, half, T} views so concat(dim=0) materializes [sin_j, cos_j]
    // pairs for each channel group and token.
    ::ggml_tensor * sin_half = ggml_view_3d(
        ctx,
        pe_half,
        1,
        half,
        n_tokens,
        /* nb1 = */ element_size,
        /* nb2 = */ pe_half->nb[1],
        /* offset = */ half * element_size);
    ::ggml_tensor * cos_half = ggml_view_3d(
        ctx,
        pe_half,
        1,
        half,
        n_tokens,
        /* nb1 = */ element_size,
        /* nb2 = */ pe_half->nb[1],
        /* offset = */ 0);

    ::ggml_tensor * pe = ggml_concat(ctx, sin_half, cos_half, 0);
    pe = ggml_cont(ctx, pe);

    return ggml_reshape_2d(ctx, pe, d_model, n_tokens);
}

static ::ggml_tensor * add_positional_embedding(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * alpha,
    int64_t          start_position = 0)
{
    const int64_t d_model = x->ne[0];
    const int64_t n_tokens = x->ne[1];

    GGML_ASSERT(alpha != nullptr);

    if (n_tokens == 0) {
        return x;
    }

    ::ggml_tensor * pe = build_sine_positional_embedding(ctx, d_model, n_tokens, start_position);
    ::ggml_tensor * scaled_pe = ggml_mul(ctx, pe, alpha);

    return ggml_add(ctx, x, scaled_pe);
}

static ::ggml_tensor * embed_text_forward(
    ::ggml_context                    * ctx,
    ::ggml_tensor                     * x_tokens,
    ::ggml_tensor                     * bert_feature,
    const t2s_embed_block_weights  & weights,
    int64_t                             start_position = 0)
{
    GGML_ASSERT(x_tokens != nullptr);
    GGML_ASSERT(bert_feature != nullptr);
    GGML_ASSERT(weights.text_embedding != nullptr);
    GGML_ASSERT(weights.bert_proj_w != nullptr);
    GGML_ASSERT(weights.bert_proj_b != nullptr);
    GGML_ASSERT(weights.text_pos_alpha != nullptr);
    GGML_ASSERT(x_tokens->type == GGML_TYPE_I32);
    GGML_ASSERT(bert_feature->ne[0] == weights.bert_proj_w->ne[0]);
    GGML_ASSERT(bert_feature->ne[1] == x_tokens->ne[0]);

    // Text token embedding: {d_model, T_x}
    ::ggml_tensor * x = ggml_get_rows(ctx, weights.text_embedding, x_tokens);

    // BERT projection: {1024, T_x} -> {d_model, T_x}
    ::ggml_tensor * bert_proj = ggml_mul_mat(ctx, weights.bert_proj_w, bert_feature);
    bert_proj = ggml_add(ctx, bert_proj, weights.bert_proj_b);

    x = ggml_add(ctx, x, bert_proj);
    x = add_positional_embedding(ctx, x, weights.text_pos_alpha, start_position);
    return x;
}

static ::ggml_tensor * t2s_audio_embed_block_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * token_ids,
    ::ggml_tensor  * audio_embedding,
    ::ggml_tensor  * audio_pos_alpha,
    int64_t          start_position)
{
    GGML_ASSERT(token_ids != nullptr);
    GGML_ASSERT(token_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(audio_embedding != nullptr);
    GGML_ASSERT(audio_pos_alpha != nullptr);
    GGML_ASSERT(token_ids->ne[0] > 0);

    ::ggml_tensor * y = ggml_get_rows(ctx, audio_embedding, token_ids);
    return add_positional_embedding(ctx, y, audio_pos_alpha, start_position);
}

::ggml_tensor * t2s_embed_ref_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * ref_token,
    ::ggml_tensor  * ref_bert_feature,
    ::ggml_tensor  * hubert_feature,
    const sovits_extract_latent_block_weights & extract_latent_weights,
    const t2s_embed_block_weights  & embed_weights)
{
    // 1. Ref text embedding: token embed + BERT proj + PE from position 0.
    ::ggml_tensor * ref_text_emb = embed_text_forward(
        ctx, ref_token, ref_bert_feature, embed_weights,
        /*start_position=*/0);

    // 2. Extract prompt semantic tokens from HuBERT.
    ::ggml_tensor * prompt_tokens = sovits_extract_latent_block_forward(
        ctx, hubert_feature, extract_latent_weights);

    // 3. Audio embed prompt tokens + PE from position 0.
    GGML_ASSERT(embed_weights.audio_embedding != nullptr);
    GGML_ASSERT(embed_weights.audio_pos_alpha != nullptr);
    GGML_ASSERT(prompt_tokens->type == GGML_TYPE_I32);

    ::ggml_tensor * prompt_audio_emb = t2s_audio_embed_block_forward(
        ctx, prompt_tokens, embed_weights.audio_embedding,
        embed_weights.audio_pos_alpha,
        /*start_position=*/0);

    // 4. Concatenate → [ref_text_emb | prompt_audio_emb].
    return ggml_concat(ctx, ref_text_emb, prompt_audio_emb, 1);
}

::ggml_tensor * t2s_embed_input_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * input_token,
    ::ggml_tensor  * input_bert_feature,
    int64_t           T_ref,
    const t2s_embed_block_weights & embed_weights)
{
    return embed_text_forward(
        ctx, input_token, input_bert_feature, embed_weights,
        /*start_position=*/T_ref);
}

struct ggml_tensor * t2s_attention_block_forward(
    struct ggml_context       * ctx,
    struct ggml_cgraph        * gf,
    struct ggml_tensor        * x,
    struct ggml_tensor        * mask,
    struct ggml_tensor        * k_cache,
    struct ggml_tensor        * v_cache,
    struct ggml_tensor        * kv_pos,
    const t2s_attention_block_weights   & weights,
    int                         n_kv,
    int                         n_head,
    float                       eps)
{
    const int64_t d_model  = x->ne[0];
    const int64_t N        = x->ne[1];
    const int64_t head_dim = d_model / n_head;
    const size_t  esz      = ggml_element_size(x);

    // Cache strides — must respect quantized block layout.
    // For F32: cache_ts=4, cache_bs=1 → same as esz.
    // For Q8_0: cache_ts=34, cache_bs=32 → different byte counts.
    const size_t cache_ts  = ggml_type_size(k_cache->type);
    const size_t cache_bs  = ggml_blck_size(k_cache->type);
    const size_t cache_nb1 = cache_ts * (head_dim / cache_bs);  // bytes per head
    const size_t cache_nb2 = cache_ts * (d_model / cache_bs);   // bytes per row

    // ── 1. QKV projection ───────────────────────────────────────
    //   qkv_w {d_model, 3·d_model}  ×  x {d_model, N}  →  {3·d_model, N}
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, weights.qkv_w, x);
    qkv = ggml_add(ctx, qkv, weights.qkv_b);

    // ── 2. Split Q / K / V and scatter-write into KV cache ──────
    //
    // qkv layout along ne[0]: [Q (d_model) | K (d_model) | V (d_model)]
    //
    // Q — view directly into the Q portion, reshape to multi-head
    //   view_3d → {head_dim, n_head, N}   then permute → {head_dim, N, n_head}
    struct ggml_tensor * q = ggml_view_3d(ctx, qkv,
        head_dim, n_head, N,
        /*nb1=*/ esz * head_dim,
        /*nb2=*/ qkv->nb[1],
        /*off=*/ 0);
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // {head_dim, N, n_head}

    // K — extract {d_model, N} (strided), make contiguous, scatter-write
    struct ggml_tensor * k_new = ggml_view_2d(ctx, qkv,
        d_model, N,
        /*nb1=*/ qkv->nb[1],
        /*off=*/ (size_t)(d_model * esz));
    k_new = ggml_cont(ctx, k_new);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_new, kv_pos));

    // V — extract {d_model, N} (strided), make contiguous, scatter-write
    struct ggml_tensor * v_new = ggml_view_2d(ctx, qkv,
        d_model, N,
        /*nb1=*/ qkv->nb[1],
        /*off=*/ (size_t)(2 * d_model * esz));
    v_new = ggml_cont(ctx, v_new);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_new, kv_pos));

    // Read full K from cache → multi-head view for flash attention
    //   view_3d → {head_dim, n_head, n_kv}  then permute → {head_dim, n_kv, n_head}
    struct ggml_tensor * k = ggml_view_3d(ctx, k_cache,
        head_dim, n_head, n_kv,
        /*nb1=*/ cache_nb1,
        /*nb2=*/ cache_nb2,
        /*off=*/ 0);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // {head_dim, n_kv, n_head}

    // Read full V from cache → multi-head view for flash attention
    struct ggml_tensor * v = ggml_view_3d(ctx, v_cache,
        head_dim, n_head, n_kv,
        /*nb1=*/ cache_nb1,
        /*nb2=*/ cache_nb2,
        /*off=*/ 0);
    v = ggml_permute(ctx, v, 0, 2, 1, 3); // {head_dim, n_kv, n_head}

    // ── 3. Flash attention ──────────────────────────────────────
    //   q {head_dim, N, n_head}      (n_embd_k, n_batch, n_head)
    //   k {head_dim, n_kv, n_head}   (n_embd_k, n_kv,    n_head_kv)
    //   v {head_dim, n_kv, n_head}   (n_embd_v, n_kv,    n_head_kv)
    //   mask {n_kv, N}               (f16, contiguous)
    //   → {head_dim, n_head, N}      (n_embd_v, n_head, n_batch)
    const float scale = 1.0f / sqrtf((float) head_dim);

    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx,
        q, k, v, mask,
        scale,
        /*max_bias=*/      0.0f,
        /*logit_softcap=*/ 0.0f);

    // ── 4. Reshape back to {d_model, N} ─────────────────────────
    //   flash_attn_ext output is contiguous {head_dim, n_head, N}
    attn = ggml_reshape_2d(ctx, attn, d_model, N);

    // ── 5. Output projection ────────────────────────────────────
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, weights.out_proj_w, attn);
    attn_out = ggml_add(ctx, attn_out, weights.out_proj_b);

    // ── 6. Residual + LayerNorm 1 (post-norm) ───────────────────
    struct ggml_tensor * res1 = ggml_add(ctx, x, attn_out);

    struct ggml_tensor * ln1 = ggml_norm(ctx, res1, eps);
    ln1 = ggml_mul(ctx, ln1, weights.ln1_w);
    ln1 = ggml_add(ctx, ln1, weights.ln1_b);

    // ── 7. FFN: up-project → ReLU → down-project ───────────────
    struct ggml_tensor * ffn = ggml_mul_mat(ctx, weights.ffn_up_w, ln1);
    ffn = ggml_add(ctx, ffn, weights.ffn_up_b);
    ffn = ggml_relu(ctx, ffn);

    ffn = ggml_mul_mat(ctx, weights.ffn_down_w, ffn);
    ffn = ggml_add(ctx, ffn, weights.ffn_down_b);

    // ── 8. Residual + LayerNorm 2 (post-norm) ───────────────────
    struct ggml_tensor * res2 = ggml_add(ctx, ln1, ffn);

    struct ggml_tensor * ln2 = ggml_norm(ctx, res2, eps);
    ln2 = ggml_mul(ctx, ln2, weights.ln2_w);
    ln2 = ggml_add(ctx, ln2, weights.ln2_b);

    return ln2;
}

} // namespace gpt_sovits
