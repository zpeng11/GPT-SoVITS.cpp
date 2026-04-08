#include "gpt_sovits/hubert.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace gpt_sovits {

namespace {

static constexpr int64_t kHubertHiddenSize = 768;
static constexpr int64_t kHubertConvDim = 512;
static constexpr int64_t kHubertIntermediateSize = 3072;
static constexpr int64_t kHubertNumHeads = 12;
static constexpr int64_t kHubertHeadDim = kHubertHiddenSize / kHubertNumHeads;
static constexpr int64_t kHubertPosConvGroups = 16;
static constexpr int64_t kHubertPosConvChannelsPerGroup = kHubertHiddenSize / kHubertPosConvGroups;
static constexpr int64_t kHubertPosConvKernel = 128;
static constexpr float kLayerNormEps = 1.0e-5f;

static ::ggml_tensor * ensure_waveform_1d(
    ::ggml_context * ctx,
    ::ggml_tensor  * input_values)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input_values != nullptr);
    GGML_ASSERT(input_values->type == GGML_TYPE_F32);
    GGML_ASSERT(input_values->ne[2] == 1);
    GGML_ASSERT(input_values->ne[3] == 1);

    if (input_values->ne[1] == 1) {
        return ggml_reshape_1d(ctx, input_values, input_values->ne[0]);
    }

    GGML_ASSERT(input_values->ne[0] == 1);
    ::ggml_tensor * flat = ggml_reshape_1d(ctx, input_values, input_values->ne[1]);
    return ggml_cont(ctx, flat);
}

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
    ::ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);

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

static ::ggml_tensor * to_conv_input_layout(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    ::ggml_tensor * conv = ggml_permute(ctx, x, 1, 0, 2, 3);
    return ggml_cont(ctx, conv);
}

static ::ggml_tensor * from_conv_output_layout(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    ::ggml_tensor * seq = ggml_permute(ctx, x, 1, 0, 2, 3);
    return ggml_cont(ctx, seq);
}

static ::ggml_tensor * conv1d_forward_channels_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    int              stride,
    int              padding)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(weight->ne[1] == x->ne[0]);

    ::ggml_tensor * conv_in = to_conv_input_layout(ctx, x);

    // ggml_conv_1d hardcodes F16 im2col. Build the op manually so F32 and
    // future quantized weights are not forced through an F16 kernel path.
    const ggml_type im2col_type = weight->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ::ggml_tensor * im2col = ggml_im2col(
        ctx,
        weight,
        conv_in,
        stride,
        /*s1=*/0,
        padding,
        /*p1=*/0,
        /*d0=*/1,
        /*d1=*/0,
        /*is_2D=*/false,
        im2col_type);

    ::ggml_tensor * patches = ggml_reshape_2d(
        ctx,
        im2col,
        im2col->ne[0],
        im2col->ne[1] * im2col->ne[2]);
    ::ggml_tensor * kernel = ggml_reshape_2d(ctx, weight, weight->ne[0] * weight->ne[1], weight->ne[2]);
    return ggml_mul_mat(ctx, kernel, patches);
}

static ::ggml_tensor * apply_group_norm_channels_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias,
    int              n_groups,
    float            eps)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(x->ne[0] == weight->ne[0]);
    GGML_ASSERT(weight->ne[0] == bias->ne[0]);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    const int64_t channels = x->ne[0];
    const int64_t time = x->ne[1];

    // ggml_group_norm treats ne[2] as channels, so transpose to {time, channels}
    // first, then reinterpret as {time, 1, channels}.
    ::ggml_tensor * x_tc = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_tc = ggml_cont(ctx, x_tc);

    ::ggml_tensor * x_3d = ggml_reshape_3d(ctx, x_tc, time, 1, channels);
    ::ggml_tensor * normed = ggml_group_norm(ctx, x_3d, n_groups, eps);

    ::ggml_tensor * weight_3d = ggml_reshape_3d(ctx, weight, 1, 1, channels);
    ::ggml_tensor * bias_3d = ggml_reshape_3d(ctx, bias, 1, 1, channels);
    normed = ggml_mul(ctx, normed, weight_3d);
    normed = ggml_add(ctx, normed, bias_3d);

    normed = ggml_reshape_2d(ctx, normed, time, channels);
    normed = ggml_permute(ctx, normed, 1, 0, 2, 3);
    return ggml_cont(ctx, normed);
}

static ::ggml_tensor * materialize_pos_conv_weight(
    ::ggml_context * ctx,
    const hubert_positional_conv_block_weights & weights)
{
    GGML_ASSERT(weights.weight_v != nullptr);
    GGML_ASSERT(weights.weight_g != nullptr);
    GGML_ASSERT(weights.weight_v->ne[0] == kHubertPosConvKernel);
    GGML_ASSERT(weights.weight_v->ne[1] == kHubertPosConvChannelsPerGroup);
    GGML_ASSERT(weights.weight_v->ne[2] == kHubertHiddenSize);

    ::ggml_tensor * v = ggml_cast(ctx, weights.weight_v, GGML_TYPE_F32);
    ::ggml_tensor * v_sq = ggml_sqr(ctx, v);

    // weight_norm(..., dim=2) keeps the kernel dimension and reduces over the
    // output-channel and in-group-channel dimensions. Reorder to {ICg, OC, K}
    // so each kernel position becomes one row after flattening.
    ::ggml_tensor * v_sq_perm = ggml_permute(ctx, v_sq, 2, 0, 1, 3); // {ICg, OC, K}
    v_sq_perm = ggml_cont(ctx, v_sq_perm);
    ::ggml_tensor * v_sq_2d = ggml_reshape_2d(
        ctx,
        v_sq_perm,
        kHubertPosConvChannelsPerGroup * kHubertHiddenSize,
        kHubertPosConvKernel);                                         // {ICg * OC, K}
    ::ggml_tensor * norm_sq = ggml_sum_rows(ctx, v_sq_2d);             // {1, K}
    ::ggml_tensor * norm = ggml_sqrt(ctx, norm_sq);
    norm = ggml_reshape_1d(ctx, norm, kHubertPosConvKernel);

    ::ggml_tensor * g = ggml_cast(ctx, weights.weight_g, GGML_TYPE_F32);
    const int64_t g_ne = g->ne[0] * g->ne[1] * g->ne[2] * g->ne[3];
    GGML_ASSERT(g_ne == kHubertPosConvKernel);
    g = ggml_reshape_1d(ctx, g, kHubertPosConvKernel);

    ::ggml_tensor * scale = ggml_div(ctx, g, norm);
    ::ggml_tensor * scale_3d = ggml_reshape_3d(ctx, scale, kHubertPosConvKernel, 1, 1);

    ::ggml_tensor * weight = ggml_mul(ctx, v, scale_3d);
    return ggml_cont(ctx, weight);
}

static ::ggml_tensor * split_channels(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          start_channel,
    int64_t          n_channels)
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(start_channel >= 0);
    GGML_ASSERT(start_channel + n_channels <= x->ne[0]);

    const size_t offset = (size_t) start_channel * ggml_element_size(x);
    ::ggml_tensor * view = ggml_view_2d(ctx, x, n_channels, x->ne[1], x->nb[1], offset);
    return ggml_cont(ctx, view);
}

static ::ggml_tensor * split_weight_out_channels(
    ::ggml_context * ctx,
    ::ggml_tensor  * weight,
    int64_t          start_channel,
    int64_t          n_channels)
{
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(weight->ne[0] == kHubertPosConvKernel);
    GGML_ASSERT(start_channel >= 0);
    GGML_ASSERT(start_channel + n_channels <= weight->ne[2]);

    const size_t offset = (size_t) start_channel * weight->nb[2];
    ::ggml_tensor * view = ggml_view_3d(
        ctx,
        weight,
        weight->ne[0],
        weight->ne[1],
        n_channels,
        weight->nb[1],
        weight->nb[2],
        offset);
    return ggml_cont(ctx, view);
}

static ::ggml_tensor * split_bias(
    ::ggml_context * ctx,
    ::ggml_tensor  * bias,
    int64_t          start_channel,
    int64_t          n_channels)
{
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(start_channel >= 0);
    GGML_ASSERT(start_channel + n_channels <= bias->ne[0]);

    const size_t offset = (size_t) start_channel * ggml_element_size(bias);
    ::ggml_tensor * view = ggml_view_1d(ctx, bias, n_channels, offset);
    return ggml_cont(ctx, view);
}

static ::ggml_tensor * concat_channel_chunks(
    ::ggml_context * ctx,
    ::ggml_tensor  * a,
    ::ggml_tensor  * b)
{
    GGML_ASSERT(a != nullptr);
    GGML_ASSERT(b != nullptr);
    return ggml_concat(ctx, a, b, /*dim=*/0);
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
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, bias->ne[0], 1));
}

static ::ggml_tensor * reshape_heads_for_attention(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    ::ggml_tensor * q = ggml_reshape_3d(ctx, x, kHubertHeadDim, kHubertNumHeads, x->ne[1]);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    return ggml_cont(ctx, q);
}

static ::ggml_tensor * transpose_sequence_and_head_dim(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    ::ggml_tensor * y = ggml_permute(ctx, x, 1, 0, 2, 3);
    return ggml_cont(ctx, y);
}

} // namespace

::ggml_tensor * hubert_feature_encoder_block_forward(
    ::ggml_context                        * ctx,
    ::ggml_tensor                         * input_values,
    const hubert_feature_encoder_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input_values != nullptr);
    GGML_ASSERT(weights.conv0_norm_w != nullptr);
    GGML_ASSERT(weights.conv0_norm_b != nullptr);

    static constexpr int kStrides[7] = {5, 2, 2, 2, 2, 2, 2};
    static constexpr int kPaddings[7] = {0, 0, 0, 0, 0, 0, 0};

    ::ggml_tensor * wave = ensure_waveform_1d(ctx, input_values);
    ::ggml_tensor * x = ggml_reshape_2d(ctx, wave, 1, wave->ne[0]);

    for (int i = 0; i < 7; ++i) {
        GGML_ASSERT(weights.conv_w[i] != nullptr);
        x = conv1d_forward_channels_first(ctx, x, weights.conv_w[i], kStrides[i], kPaddings[i]);

        if (i == 0) {
            x = apply_group_norm_channels_first(
                ctx,
                x,
                weights.conv0_norm_w,
                weights.conv0_norm_b,
                /*n_groups=*/512,
                kLayerNormEps);
        }

        x = ggml_gelu(ctx, x);
    }

    return x;
}

::ggml_tensor * hubert_feature_projection_block_forward(
    ::ggml_context                           * ctx,
    ::ggml_tensor                            * features,
    const hubert_feature_projection_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(features != nullptr);
    GGML_ASSERT(features->ne[0] == kHubertConvDim);
    GGML_ASSERT(weights.layer_norm_w != nullptr);
    GGML_ASSERT(weights.layer_norm_b != nullptr);
    GGML_ASSERT(weights.projection_w != nullptr);
    GGML_ASSERT(weights.projection_b != nullptr);
    GGML_ASSERT(weights.layer_norm_w->ne[0] == kHubertConvDim);
    GGML_ASSERT(weights.layer_norm_b->ne[0] == kHubertConvDim);
    GGML_ASSERT(weights.projection_w->ne[0] == kHubertConvDim);
    GGML_ASSERT(weights.projection_w->ne[1] == kHubertHiddenSize);
    GGML_ASSERT(weights.projection_b->ne[0] == kHubertHiddenSize);

    ::ggml_tensor * x = layer_norm_last_dim_2d(
        ctx,
        features,
        weights.layer_norm_w,
        weights.layer_norm_b,
        kLayerNormEps);
    return linear_2d(ctx, x, weights.projection_w, weights.projection_b);
}

::ggml_tensor * hubert_positional_conv_block_forward(
    ::ggml_context                       * ctx,
    ::ggml_tensor                        * hidden_states,
    const hubert_positional_conv_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(hidden_states != nullptr);
    GGML_ASSERT(hidden_states->ne[0] == kHubertHiddenSize);
    GGML_ASSERT(weights.bias != nullptr);
    GGML_ASSERT(weights.bias->ne[0] == kHubertHiddenSize);

    ::ggml_tensor * conv_weight = materialize_pos_conv_weight(ctx, weights);
    GGML_ASSERT(conv_weight->ne[0] == kHubertPosConvKernel);
    GGML_ASSERT(conv_weight->ne[1] == kHubertPosConvChannelsPerGroup);
    GGML_ASSERT(conv_weight->ne[2] == kHubertHiddenSize);

    ::ggml_tensor * out = nullptr;
    for (int64_t group = 0; group < kHubertPosConvGroups; ++group) {
        const int64_t start = group * kHubertPosConvChannelsPerGroup;

        ::ggml_tensor * x_group = split_channels(
            ctx,
            hidden_states,
            start,
            kHubertPosConvChannelsPerGroup);
        ::ggml_tensor * w_group = split_weight_out_channels(
            ctx,
            conv_weight,
            start,
            kHubertPosConvChannelsPerGroup);
        ::ggml_tensor * y_group = conv1d_forward_channels_first(
            ctx,
            x_group,
            w_group,
            /*stride=*/1,
            /*padding=*/64);
        ::ggml_tensor * b_group = split_bias(
            ctx,
            weights.bias,
            start,
            kHubertPosConvChannelsPerGroup);
        y_group = ggml_add(ctx, y_group, ggml_reshape_2d(ctx, b_group, b_group->ne[0], 1));

        out = out == nullptr ? y_group : concat_channel_chunks(ctx, out, y_group);
    }

    GGML_ASSERT(out != nullptr);
    GGML_ASSERT(out->ne[1] == hidden_states->ne[1] + 1);

    ::ggml_tensor * trimmed = ggml_view_2d(ctx, out, out->ne[0], hidden_states->ne[1], out->nb[1], 0);
    trimmed = ggml_cont(ctx, trimmed);
    return ggml_gelu(ctx, trimmed);
}

::ggml_tensor * hubert_attention_block_forward(
    ::ggml_context                    * ctx,
    ::ggml_tensor                     * x,
    const hubert_attention_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kHubertHiddenSize);
    GGML_ASSERT(weights.q_proj_w != nullptr);
    GGML_ASSERT(weights.q_proj_b != nullptr);
    GGML_ASSERT(weights.k_proj_w != nullptr);
    GGML_ASSERT(weights.k_proj_b != nullptr);
    GGML_ASSERT(weights.v_proj_w != nullptr);
    GGML_ASSERT(weights.v_proj_b != nullptr);
    GGML_ASSERT(weights.out_proj_w != nullptr);
    GGML_ASSERT(weights.out_proj_b != nullptr);

    ::ggml_tensor * q = linear_2d(ctx, x, weights.q_proj_w, weights.q_proj_b);
    ::ggml_tensor * k = linear_2d(ctx, x, weights.k_proj_w, weights.k_proj_b);
    ::ggml_tensor * v = linear_2d(ctx, x, weights.v_proj_w, weights.v_proj_b);

    q = reshape_heads_for_attention(ctx, q); // {head_dim, T, n_head}
    k = reshape_heads_for_attention(ctx, k); // {head_dim, T, n_head}
    v = reshape_heads_for_attention(ctx, v); // {head_dim, T, n_head}

    ::ggml_tensor * attn_scores = ggml_mul_mat(ctx, k, q); // {T, T, n_head}
    attn_scores = ggml_soft_max_ext(
        ctx,
        attn_scores,
        /*mask=*/nullptr,
        1.0f / std::sqrt(static_cast<float>(kHubertHeadDim)),
        /*max_bias=*/0.0f);

    ::ggml_tensor * v_rhs = transpose_sequence_and_head_dim(ctx, v);  // {T, head_dim, n_head}
    ::ggml_tensor * attn = ggml_mul_mat(ctx, v_rhs, attn_scores);     // {head_dim, T, n_head}
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);                       // {head_dim, n_head, T}
    attn = ggml_cont_2d(ctx, attn, kHubertHiddenSize, x->ne[1]);

    return linear_2d(ctx, attn, weights.out_proj_w, weights.out_proj_b);
}

::ggml_tensor * hubert_encoder_layer_block_forward(
    ::ggml_context                         * ctx,
    ::ggml_tensor                          * x,
    const hubert_encoder_layer_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weights.ln1_w != nullptr);
    GGML_ASSERT(weights.ln1_b != nullptr);
    GGML_ASSERT(weights.ffn_up_w != nullptr);
    GGML_ASSERT(weights.ffn_up_b != nullptr);
    GGML_ASSERT(weights.ffn_down_w != nullptr);
    GGML_ASSERT(weights.ffn_down_b != nullptr);
    GGML_ASSERT(weights.ln2_w != nullptr);
    GGML_ASSERT(weights.ln2_b != nullptr);
    GGML_ASSERT(weights.ffn_up_w->ne[0] == kHubertHiddenSize);
    GGML_ASSERT(weights.ffn_up_w->ne[1] == kHubertIntermediateSize);
    GGML_ASSERT(weights.ffn_up_b->ne[0] == kHubertIntermediateSize);
    GGML_ASSERT(weights.ffn_down_w->ne[0] == kHubertIntermediateSize);
    GGML_ASSERT(weights.ffn_down_w->ne[1] == kHubertHiddenSize);
    GGML_ASSERT(weights.ffn_down_b->ne[0] == kHubertHiddenSize);

    ::ggml_tensor * attn_out = hubert_attention_block_forward(ctx, x, weights.attention);
    ::ggml_tensor * res1 = ggml_add(ctx, x, attn_out);
    ::ggml_tensor * ln1 = layer_norm_last_dim_2d(ctx, res1, weights.ln1_w, weights.ln1_b, kLayerNormEps);

    ::ggml_tensor * ffn = linear_2d(ctx, ln1, weights.ffn_up_w, weights.ffn_up_b);
    ffn = ggml_gelu(ctx, ffn);
    ffn = linear_2d(ctx, ffn, weights.ffn_down_w, weights.ffn_down_b);

    ::ggml_tensor * res2 = ggml_add(ctx, ln1, ffn);
    return layer_norm_last_dim_2d(ctx, res2, weights.ln2_w, weights.ln2_b, kLayerNormEps);
}

::ggml_tensor * hubert_encoder_block_forward(
    ::ggml_context                   * ctx,
    ::ggml_tensor                    * x,
    const hubert_encoder_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kHubertHiddenSize);
    GGML_ASSERT(weights.layer_norm_w != nullptr);
    GGML_ASSERT(weights.layer_norm_b != nullptr);
    GGML_ASSERT(weights.layer_norm_w->ne[0] == kHubertHiddenSize);
    GGML_ASSERT(weights.layer_norm_b->ne[0] == kHubertHiddenSize);

    ::ggml_tensor * pos = hubert_positional_conv_block_forward(ctx, x, weights.pos_conv);
    ::ggml_tensor * hidden = ggml_add(ctx, x, pos);
    hidden = layer_norm_last_dim_2d(ctx, hidden, weights.layer_norm_w, weights.layer_norm_b, kLayerNormEps);

    for (const hubert_encoder_layer_block_weights & layer : weights.layers) {
        hidden = hubert_encoder_layer_block_forward(ctx, hidden, layer);
    }

    return hidden;
}

::ggml_tensor * hubert_model_block_forward(
    ::ggml_context                  * ctx,
    ::ggml_tensor                   * input_values,
    const hubert_model_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input_values != nullptr);

    ::ggml_tensor * features = hubert_feature_encoder_block_forward(ctx, input_values, weights.feature_encoder);
    ::ggml_tensor * projected = hubert_feature_projection_block_forward(ctx, features, weights.feature_projection);
    return hubert_encoder_block_forward(ctx, projected, weights.encoder);
}

} // namespace gpt_sovits
