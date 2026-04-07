#include "gpt_sovits/gpt_sovits.h"
#include "ggml.h"

#include <cmath>

namespace gpt_sovits {

::ggml_tensor * sovits_extract_latent_block_forward(
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

static ::ggml_tensor * build_sine_positional_embedding(
    ::ggml_context * ctx,
    int64_t          d_model,
    int64_t          n_tokens)
{
    GGML_ASSERT(d_model > 0);
    GGML_ASSERT(n_tokens >= 0);
    GGML_ASSERT(d_model % 2 == 0);

    if (n_tokens == 0) {
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 0);
    }

    ::ggml_tensor * positions = ggml_arange(ctx, 0.0f, (float) n_tokens, 1.0f);

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
    ::ggml_tensor  * alpha)
{
    const int64_t d_model = x->ne[0];
    const int64_t n_tokens = x->ne[1];

    GGML_ASSERT(alpha != nullptr);

    if (n_tokens == 0) {
        return x;
    }

    ::ggml_tensor * pe = build_sine_positional_embedding(ctx, d_model, n_tokens);
    ::ggml_tensor * scaled_pe = ggml_mul(ctx, pe, alpha);

    return ggml_add(ctx, x, scaled_pe);
}

::ggml_tensor * t2s_encoder_block_forward(
    ::ggml_context                    * ctx,
    ::ggml_tensor                     * x_tokens,
    ::ggml_tensor                     * bert_feature,
    ::ggml_tensor                     * prompt_tokens,
    const t2s_encoder_block_weights  & weights)
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
    x = add_positional_embedding(ctx, x, weights.text_pos_alpha);

    if (prompt_tokens == nullptr) {
        return x;
    }

    GGML_ASSERT(weights.audio_embedding != nullptr);
    GGML_ASSERT(weights.audio_pos_alpha != nullptr);
    GGML_ASSERT(prompt_tokens->type == GGML_TYPE_I32);

    if (prompt_tokens->ne[0] == 0) {
        return x;
    }

    // Prompt token embedding: {d_model, T_y}
    ::ggml_tensor * y = ggml_get_rows(ctx, weights.audio_embedding, prompt_tokens);
    y = add_positional_embedding(ctx, y, weights.audio_pos_alpha);

    // Concatenate text and prompt sequence along the token dimension.
    return ggml_concat(ctx, x, y, 1);
}

struct ggml_tensor * t2s_attention_block_forward(
    struct ggml_context       * ctx,
    struct ggml_cgraph        * gf,
    struct ggml_tensor        * x,
    struct ggml_tensor        * mask,
    struct ggml_tensor        * k_cache,
    struct ggml_tensor        * v_cache,
    const t2s_attention_block_weights   & weights,
    int                         n_past,
    int                         n_head,
    float                       eps)
{
    const int64_t d_model  = x->ne[0];
    const int64_t N        = x->ne[1];
    const int64_t n_kv     = n_past + N;
    const int64_t head_dim = d_model / n_head;
    const size_t  esz      = ggml_element_size(x);

    // ── 1. QKV projection ───────────────────────────────────────
    //   qkv_w {d_model, 3·d_model}  ×  x {d_model, N}  →  {3·d_model, N}
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, weights.qkv_w, x);
    qkv = ggml_add(ctx, qkv, weights.qkv_b);

    // ── 2. Split Q / K / V and populate KV cache ────────────────
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

    // K — extract {d_model, N} (strided), copy into cache slice
    struct ggml_tensor * k_new = ggml_view_2d(ctx, qkv,
        d_model, N,
        /*nb1=*/ qkv->nb[1],
        /*off=*/ (size_t)(d_model * esz));

    struct ggml_tensor * k_dst = ggml_view_2d(ctx, k_cache,
        d_model, N,
        /*nb1=*/ d_model * esz,
        /*off=*/ (size_t)(n_past * d_model * esz));

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k_new, k_dst));

    // V — extract {d_model, N} (strided), copy into cache slice
    struct ggml_tensor * v_new = ggml_view_2d(ctx, qkv,
        d_model, N,
        /*nb1=*/ qkv->nb[1],
        /*off=*/ (size_t)(2 * d_model * esz));

    struct ggml_tensor * v_dst = ggml_view_2d(ctx, v_cache,
        d_model, N,
        /*nb1=*/ d_model * esz,
        /*off=*/ (size_t)(n_past * d_model * esz));

    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_new, v_dst));

    // Read full K from cache → multi-head view for flash attention
    //   view_3d → {head_dim, n_head, n_kv}  then permute → {head_dim, n_kv, n_head}
    struct ggml_tensor * k = ggml_view_3d(ctx, k_cache,
        head_dim, n_head, n_kv,
        /*nb1=*/ esz * head_dim,
        /*nb2=*/ d_model * esz,
        /*off=*/ 0);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // {head_dim, n_kv, n_head}

    // Read full V from cache → multi-head view for flash attention
    struct ggml_tensor * v = ggml_view_3d(ctx, v_cache,
        head_dim, n_head, n_kv,
        /*nb1=*/ esz * head_dim,
        /*nb2=*/ d_model * esz,
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
