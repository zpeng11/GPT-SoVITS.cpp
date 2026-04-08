#include "gpt_sovits/roberta.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace gpt_sovits {

namespace {

static constexpr int64_t kHiddenSize       = 1024;
static constexpr int64_t kNumHeads         = 16;
static constexpr int64_t kHeadDim          = kHiddenSize / kNumHeads;  // 64
static constexpr int64_t kIntermediateSize = 4096;
static constexpr int64_t kNumLayers        = 24;
static constexpr int64_t kMaxPosEmbed      = 512;
static constexpr float   kLayerNormEps     = 1.0e-12f;

// ---------------------------------------------------------------------------
// Shared helpers (same pattern as hubert/block.cpp)
// ---------------------------------------------------------------------------

static ::ggml_tensor * apply_affine_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(weight->ne[0] == x->ne[0]);
    GGML_ASSERT(bias->ne[0] == x->ne[0]);

    ::ggml_tensor * weight_2d = ggml_reshape_2d(ctx, weight, weight->ne[0], 1);
    ::ggml_tensor * bias_2d   = ggml_reshape_2d(ctx, bias,   bias->ne[0],   1);

    x = ggml_mul(ctx, x, weight_2d);
    x = ggml_add(ctx, x, bias_2d);
    return x;
}

static ::ggml_tensor * layer_norm_last_dim_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias,
    float            eps)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == weight->ne[0]);
    GGML_ASSERT(weight->ne[0] == bias->ne[0]);

    ::ggml_tensor * normed = ggml_norm(ctx, x, eps);
    return apply_affine_2d(ctx, normed, weight, bias);
}

static ::ggml_tensor * linear_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(weight->ne[0] == x->ne[0]);
    GGML_ASSERT(weight->ne[1] == bias->ne[0]);

    ::ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    return ggml_add(ctx, y, bias);
}

} // namespace

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

::ggml_tensor * roberta_embeddings_block_forward(
    ::ggml_context                       * ctx,
    ::ggml_tensor                        * input_ids,
    ::ggml_tensor                        * token_type_ids,
    const roberta_embeddings_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input_ids != nullptr);
    GGML_ASSERT(input_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(weights.word_embeddings != nullptr);
    GGML_ASSERT(weights.position_embeddings != nullptr);
    GGML_ASSERT(weights.layer_norm_w != nullptr);
    GGML_ASSERT(weights.layer_norm_b != nullptr);

    const int64_t T = input_ids->ne[0];

    // Word embeddings: gather rows from the embedding table.
    ::ggml_tensor * x = ggml_get_rows(ctx, weights.word_embeddings, input_ids);
    // x: {1024, T}

    // Learned position embeddings: slice the first T rows.
    GGML_ASSERT(T <= kMaxPosEmbed);
    GGML_ASSERT(weights.position_embeddings->ne[0] == kHiddenSize);
    GGML_ASSERT(weights.position_embeddings->ne[1] == kMaxPosEmbed);

    const size_t esz = ggml_element_size(weights.position_embeddings);
    ::ggml_tensor * pos_embed = ggml_view_2d(
        ctx,
        weights.position_embeddings,
        kHiddenSize, T,
        /* nb1 = */ kHiddenSize * esz,
        /* offset = */ 0);
    // pos_embed: {1024, T} — cast to f32 to match ggml_get_rows output type.
    pos_embed = ggml_cast(ctx, pos_embed, GGML_TYPE_F32);

    x = ggml_add(ctx, x, pos_embed);

    // Token-type embeddings.
    // When token_type_ids is nullptr, default to all zeros (matches HuggingFace
    // BertModel behavior where None → torch.zeros).
    GGML_ASSERT(weights.token_type_embeddings != nullptr);
    if (token_type_ids == nullptr) {
        // Slice row 0 of token_type_embeddings {1024, 2} → {1024, 1}, then
        // broadcast: new_tensor_1d(1024) + x {1024, T} works via ggml_add broadcast.
        const size_t tesz = ggml_element_size(weights.token_type_embeddings);
        ::ggml_tensor * type_zero = ggml_view_2d(
            ctx,
            weights.token_type_embeddings,
            kHiddenSize, 1,
            /* nb1 = */ kHiddenSize * tesz,
            /* offset = */ 0);
        // type_zero: {1024, 1} — cast to f32, then ggml_add broadcasts along dim 1.
        type_zero = ggml_cast(ctx, type_zero, GGML_TYPE_F32);
        x = ggml_add(ctx, x, type_zero);
    } else {
        GGML_ASSERT(token_type_ids->type == GGML_TYPE_I32);
        GGML_ASSERT(token_type_ids->ne[0] == T);

        ::ggml_tensor * type_embed = ggml_get_rows(ctx, weights.token_type_embeddings, token_type_ids);
        x = ggml_add(ctx, x, type_embed);
    }

    return layer_norm_last_dim_2d(ctx, x, weights.layer_norm_w, weights.layer_norm_b, kLayerNormEps);
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

::ggml_tensor * roberta_self_attention_block_forward(
    ::ggml_context                          * ctx,
    ::ggml_tensor                           * x,
    const roberta_self_attention_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kHiddenSize);
    GGML_ASSERT(weights.q_w != nullptr);
    GGML_ASSERT(weights.q_b != nullptr);
    GGML_ASSERT(weights.k_w != nullptr);
    GGML_ASSERT(weights.k_b != nullptr);
    GGML_ASSERT(weights.v_w != nullptr);
    GGML_ASSERT(weights.v_b != nullptr);
    GGML_ASSERT(weights.out_w != nullptr);
    GGML_ASSERT(weights.out_b != nullptr);

    const int64_t T   = x->ne[1];
    const size_t  esz = ggml_element_size(x);

    // ── Q/K/V projections ────────────────────────────────────────
    ::ggml_tensor * Q = linear_2d(ctx, x, weights.q_w, weights.q_b);  // {1024, T}
    ::ggml_tensor * K = linear_2d(ctx, x, weights.k_w, weights.k_b);
    ::ggml_tensor * V = linear_2d(ctx, x, weights.v_w, weights.v_b);

    // ── Reshape to multi-head layout for flash_attn_ext ──────────
    //   view_3d → {head_dim, n_head, T}   then permute → {head_dim, T, n_head}
    ::ggml_tensor * q_3d = ggml_view_3d(ctx, Q,
        kHeadDim, kNumHeads, T,
        /* nb1 = */ esz * kHeadDim,
        /* nb2 = */ Q->nb[1],
        /* off = */ 0);
    q_3d = ggml_permute(ctx, q_3d, 0, 2, 1, 3);  // {64, T, 16}

    ::ggml_tensor * k_3d = ggml_view_3d(ctx, K,
        kHeadDim, kNumHeads, T,
        /* nb1 = */ esz * kHeadDim,
        /* nb2 = */ K->nb[1],
        /* off = */ 0);
    k_3d = ggml_permute(ctx, k_3d, 0, 2, 1, 3);  // {64, T, 16}

    ::ggml_tensor * v_3d = ggml_view_3d(ctx, V,
        kHeadDim, kNumHeads, T,
        /* nb1 = */ esz * kHeadDim,
        /* nb2 = */ V->nb[1],
        /* off = */ 0);
    v_3d = ggml_permute(ctx, v_3d, 0, 2, 1, 3);  // {64, T, 16}

    // ── Flash attention ──────────────────────────────────────────
    //   Q {64, T, 16}    (n_embd_k, n_batch,  n_head)
    //   K {64, T, 16}    (n_embd_k, n_kv,     n_head_kv)
    //   V {64, T, 16}    (n_embd_v, n_kv,     n_head_kv)
    //   → {64, 16, T}    (n_embd_v, n_head,   n_batch)
    const float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));

    ::ggml_tensor * attn = ggml_flash_attn_ext(
        ctx,
        q_3d, k_3d, v_3d,
        /* mask = */ nullptr,
        scale,
        /* max_bias      = */ 0.0f,
        /* logit_softcap = */ 0.0f);

    // ── Reshape back to {1024, T} ────────────────────────────────
    attn = ggml_reshape_2d(ctx, attn, kHiddenSize, T);

    // ── Output projection ────────────────────────────────────────
    return linear_2d(ctx, attn, weights.out_w, weights.out_b);
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

::ggml_tensor * roberta_encoder_layer_block_forward(
    ::ggml_context                             * ctx,
    ::ggml_tensor                              * x,
    const roberta_encoder_layer_block_weights  & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kHiddenSize);

    // Self-attention → residual + LayerNorm
    ::ggml_tensor * attn_out = roberta_self_attention_block_forward(ctx, x, weights.attention);
    ::ggml_tensor * res1     = ggml_add(ctx, x, attn_out);
    ::ggml_tensor * ln1      = layer_norm_last_dim_2d(ctx, res1, weights.attn_ln_w, weights.attn_ln_b, kLayerNormEps);

    // FFN: up → GELU → down
    ::ggml_tensor * ffn = linear_2d(ctx, ln1, weights.ffn_up_w, weights.ffn_up_b);
    ffn = ggml_gelu(ctx, ffn);
    ffn = linear_2d(ctx, ffn, weights.ffn_down_w, weights.ffn_down_b);

    // Residual + LayerNorm
    ::ggml_tensor * res2 = ggml_add(ctx, ln1, ffn);
    return layer_norm_last_dim_2d(ctx, res2, weights.ffn_ln_w, weights.ffn_ln_b, kLayerNormEps);
}

// ---------------------------------------------------------------------------
// Encoder (24 layers)
// ---------------------------------------------------------------------------

::ggml_tensor * roberta_encoder_block_forward(
    ::ggml_context                     * ctx,
    ::ggml_tensor                      * x,
    const roberta_encoder_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kHiddenSize);

    for (const roberta_encoder_layer_block_weights & layer : weights.layers) {
        x = roberta_encoder_layer_block_forward(ctx, x, layer);
    }

    return x;
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

::ggml_tensor * roberta_model_block_forward(
    ::ggml_context                   * ctx,
    ::ggml_tensor                    * input_ids,
    ::ggml_tensor                    * token_type_ids,
    const roberta_model_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input_ids != nullptr);

    ::ggml_tensor * x = roberta_embeddings_block_forward(ctx, input_ids, token_type_ids, weights.embeddings);
    return roberta_encoder_block_forward(ctx, x, weights.encoder);
}

} // namespace gpt_sovits
