#include "gpt_sovits/sovits.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace gpt_sovits {

namespace {

static constexpr int64_t kMelChannels = 704;
static constexpr int64_t kRVQDim = 768;
static constexpr int64_t kRVQBins = 1024;
static constexpr int64_t kStyleHidden = 128;
static constexpr int64_t kStyleOut = 512;
static constexpr int64_t kStyleKernel = 5;
static constexpr int64_t kStyleHeads = 2;
static constexpr int64_t kStyleHeadDim = kStyleHidden / kStyleHeads;
static constexpr int64_t kTextEncoderSslIn = 768;
static constexpr int64_t kTextEncoderSslHidden = 192;
static constexpr int64_t kTextEncoderSslFFN = 768;
static constexpr int64_t kTextEncoderSslHeads = 2;
static constexpr int64_t kTextEncoderSslHeadDim =
    kTextEncoderSslHidden / kTextEncoderSslHeads;
static constexpr int64_t kTextEncoderSslKernel = 3;
static constexpr int64_t kTextEncoderSslWindow = 4;
static constexpr int64_t kTextEncoderSslRelSize = 2 * kTextEncoderSslWindow + 1;
static constexpr int64_t kTextEncoderMrteHidden = 512;
static constexpr int64_t kTextEncoderMrteHeads = 4;
static constexpr int64_t kTextEncoderMrteHeadDim =
    kTextEncoderMrteHidden / kTextEncoderMrteHeads;
static constexpr int64_t kTextEncoderMrteQDim = 512;
static constexpr int64_t kTextEncoderMrteSkipDim = 192;
static constexpr int64_t kTextEncoderMrteKvDim = 2 * kTextEncoderMrteHidden;
static constexpr int64_t kTextEncoderMrteSslFusedDim =
    kTextEncoderMrteQDim + kTextEncoderMrteSkipDim;
static constexpr int64_t kTextEncoderTextVocabV2 = 732;
static constexpr float kLayerNormEps = 1.0e-5f;

static ::ggml_tensor * flatten_vector_1d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(ctx != nullptr);
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

static ::ggml_tensor * reshape_bias_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(bias != nullptr);
    return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
}

static ::ggml_tensor * layer_norm_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(weight->ne[0] == x->ne[0]);
    GGML_ASSERT(bias->ne[0] == x->ne[0]);

    ::ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y = ggml_mul(ctx, y, reshape_bias_2d(ctx, weight));
    return ggml_add(ctx, y, reshape_bias_2d(ctx, bias));
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
    return ggml_add(ctx, y, reshape_bias_2d(ctx, bias));
}

static ::ggml_tensor * mish_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
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
    return ggml_view_2d(ctx, x, n_channels, x->ne[1], x->nb[1], offset);
}

static ::ggml_tensor * split_qkv_projection(
    ::ggml_context * ctx,
    ::ggml_tensor  * qkv,
    int64_t          index)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(qkv != nullptr);
    GGML_ASSERT(index >= 0 && index < 3);
    GGML_ASSERT(qkv->ne[0] == 3 * kTextEncoderSslHidden);

    return split_channels(ctx, qkv, index * kTextEncoderSslHidden, kTextEncoderSslHidden);
}

static ::ggml_tensor * ensure_f32(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    if (x->type == GGML_TYPE_F32) {
        return x;
    }
    return ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
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

    ::ggml_tensor * conv_in = ggml_permute(ctx, x, 1, 0, 2, 3);
    conv_in = ggml_cont(ctx, conv_in);

    const ggml_type im2col_type =
        (weight->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F16)
            ? GGML_TYPE_F16
            : GGML_TYPE_F32;
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
    if (im2col_type == GGML_TYPE_F32) {
        patches = ensure_f32(ctx, patches);
    }

    ::ggml_tensor * kernel_src = weight;
    if (im2col_type == GGML_TYPE_F32 && weight->type != GGML_TYPE_F32) {
        kernel_src = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    ::ggml_tensor * kernel = ggml_reshape_2d(ctx, kernel_src, kernel_src->ne[0] * kernel_src->ne[1], kernel_src->ne[2]);

    return ggml_mul_mat(ctx, kernel, patches);
}

static ::ggml_tensor * conv1d_with_bias_channels_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias,
    int              stride,
    int              padding)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);

    // Conv1d with kernel=1/stride=1/pad=0 is exactly a per-time-step linear
    // projection on the channel axis, so avoid the im2col path.
    if (weight->ne[0] == 1 && stride == 1 && padding == 0) {
        ::ggml_tensor * linear_w = ggml_reshape_2d(ctx, weight, weight->ne[1], weight->ne[2]);
        return linear_2d(ctx, x, linear_w, bias);
    }

    ::ggml_tensor * y = conv1d_forward_channels_first(ctx, x, weight, stride, padding);
    y = ensure_f32(ctx, y);
    return ggml_add(ctx, y, reshape_bias_2d(ctx, bias));
}

static ::ggml_tensor * slice_time_range(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          start,
    int64_t          length)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(start >= 0);
    GGML_ASSERT(length >= 0);
    GGML_ASSERT(start + length <= x->ne[1]);

    const size_t offset = (size_t) start * x->nb[1];
    ::ggml_tensor * view = ggml_view_2d(ctx, x, x->ne[0], length, x->nb[1], offset);
    return ggml_cont(ctx, view);
}

static ::ggml_tensor * zeros_2d(
    ::ggml_context * ctx,
    int64_t          ne0,
    int64_t          ne1)
{
    ::ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    return ggml_fill(ctx, tmpl, 0.0f);
}

static ::ggml_tensor * add_relative_logits_for_head(
    ::ggml_context * ctx,
    ::ggml_tensor  * scores,
    ::ggml_tensor  * q_head,
    ::ggml_tensor  * rel_k)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(scores != nullptr);
    GGML_ASSERT(q_head != nullptr);
    GGML_ASSERT(rel_k != nullptr);
    GGML_ASSERT(scores->type == GGML_TYPE_F32);
    GGML_ASSERT(scores->ne[0] == q_head->ne[1]);
    GGML_ASSERT(scores->ne[1] == q_head->ne[1]);
    GGML_ASSERT(q_head->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[1] == kTextEncoderSslRelSize);

    const int64_t time = q_head->ne[1];
    ::ggml_tensor * rel_dot = ggml_mul_mat(ctx, rel_k, q_head); // {rel, T}
    rel_dot = ensure_f32(ctx, rel_dot);

    for (int64_t rel_idx = 0; rel_idx < kTextEncoderSslRelSize; ++rel_idx) {
        const int64_t shift = rel_idx - kTextEncoderSslWindow;
        int64_t length = time - (shift >= 0 ? shift : -shift);
        if (length <= 0) {
            continue;
        }

        const int64_t q_start = shift >= 0 ? 0 : -shift;
        const int64_t key_start = shift >= 0 ? shift : 0;
        ::ggml_tensor * dot = ggml_view_2d(
            ctx,
            rel_dot,
            1,
            length,
            rel_dot->nb[1],
            (size_t) rel_idx * rel_dot->nb[0] + (size_t) q_start * rel_dot->nb[1]);
        dot = ensure_f32(ctx, dot);

        scores = ggml_acc_inplace(
            ctx,
            scores,
            dot,
            scores->nb[0] + scores->nb[1],
            scores->nb[2],
            scores->nb[3],
            (size_t) key_start * scores->nb[0] + (size_t) q_start * scores->nb[1]);
    }

    return scores;
}

static ::ggml_tensor * build_relative_value_for_head(
    ::ggml_context * ctx,
    ::ggml_tensor  * attn_head,
    ::ggml_tensor  * rel_v_t)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn_head != nullptr);
    GGML_ASSERT(rel_v_t != nullptr);
    GGML_ASSERT(rel_v_t->ne[0] == kTextEncoderSslRelSize);
    GGML_ASSERT(rel_v_t->ne[1] == kTextEncoderSslHeadDim);

    const int64_t time = attn_head->ne[0];
    ::ggml_tensor * rel_weights = zeros_2d(ctx, kTextEncoderSslRelSize, time);

    for (int64_t rel_idx = 0; rel_idx < kTextEncoderSslRelSize; ++rel_idx) {
        const int64_t shift = rel_idx - kTextEncoderSslWindow;
        int64_t length = time - (shift >= 0 ? shift : -shift);
        if (length <= 0) {
            continue;
        }

        const int64_t key_start = shift >= 0 ? shift : 0;
        const int64_t query_start = shift >= 0 ? 0 : -shift;
        const size_t offset = (size_t) key_start * attn_head->nb[0] + (size_t) query_start * attn_head->nb[1];
        ::ggml_tensor * diag = ggml_view_2d(ctx, attn_head, 1, length, attn_head->nb[0] + attn_head->nb[1], offset);
        diag = ensure_f32(ctx, diag);

        rel_weights = ggml_set_2d_inplace(
            ctx,
            rel_weights,
            diag,
            rel_weights->nb[1],
            (size_t) rel_idx * rel_weights->nb[0] + (size_t) query_start * rel_weights->nb[1]);
    }

    ::ggml_tensor * output = ggml_mul_mat(ctx, rel_v_t, rel_weights);
    return ensure_f32(ctx, output);
}

static ::ggml_tensor * self_attention_with_relative_position(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_relpos_encoder_layer_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.qkv_w != nullptr);
    GGML_ASSERT(weights.qkv_b != nullptr);
    GGML_ASSERT(weights.out_w != nullptr);
    GGML_ASSERT(weights.out_b != nullptr);
    GGML_ASSERT(weights.rel_k != nullptr);
    GGML_ASSERT(weights.rel_v_t != nullptr);

    ::ggml_tensor * qkv = conv1d_with_bias_channels_first(ctx, x, weights.qkv_w, weights.qkv_b, 1, 0);
    ::ggml_tensor * q = split_qkv_projection(ctx, qkv, 0);
    ::ggml_tensor * k = split_qkv_projection(ctx, qkv, 1);
    ::ggml_tensor * v = split_qkv_projection(ctx, qkv, 2);

    const int64_t time = x->ne[1];
    ::ggml_tensor * merged = zeros_2d(ctx, kTextEncoderSslHidden, time);
    ::ggml_tensor * rel_k = weights.rel_k;
    ::ggml_tensor * rel_v_t = weights.rel_v_t;

    for (int64_t h = 0; h < kTextEncoderSslHeads; ++h) {
        const float attn_scale = 1.0f / sqrtf((float) kTextEncoderSslHeadDim);
        const int64_t ch0 = h * kTextEncoderSslHeadDim;
        ::ggml_tensor * q_head = split_channels(ctx, q, ch0, kTextEncoderSslHeadDim);
        ::ggml_tensor * k_head = split_channels(ctx, k, ch0, kTextEncoderSslHeadDim);
        ::ggml_tensor * v_head = split_channels(ctx, v, ch0, kTextEncoderSslHeadDim);

        ::ggml_tensor * content_scores = ggml_mul_mat(ctx, k_head, q_head);
        content_scores = ensure_f32(ctx, content_scores);

        ::ggml_tensor * scores = add_relative_logits_for_head(ctx, content_scores, q_head, rel_k);
        ::ggml_tensor * attn = ggml_soft_max_ext(ctx, scores, nullptr, attn_scale, 0.0f);

        ::ggml_tensor * v_head_t = ggml_transpose(ctx, v_head);
        v_head_t = ggml_cont(ctx, v_head_t);
        ::ggml_tensor * content_out = ggml_mul_mat(ctx, attn, v_head_t);
        content_out = ggml_transpose(ctx, content_out);
        content_out = ggml_cont(ctx, content_out);

        ::ggml_tensor * rel_out = build_relative_value_for_head(ctx, attn, rel_v_t);
        ::ggml_tensor * head_out = ggml_add(ctx, content_out, rel_out);
        GGML_ASSERT(head_out->ne[0] == kTextEncoderSslHeadDim);
        GGML_ASSERT(head_out->ne[1] == time);

        merged = ggml_set_2d_inplace(
            ctx,
            merged,
            head_out,
            merged->nb[1],
            (size_t) ch0 * ggml_element_size(merged));
    }

    return conv1d_with_bias_channels_first(ctx, merged, weights.out_w, weights.out_b, 1, 0);
}

static ::ggml_tensor * text_encoder_ssl_layer_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_relpos_encoder_layer_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);

    ::ggml_tensor * attn_out = self_attention_with_relative_position(ctx, x, weights);
    ::ggml_tensor * x1 = layer_norm_2d(ctx, ggml_add(ctx, x, attn_out), weights.ln1_w, weights.ln1_b);

    ::ggml_tensor * ffn = conv1d_with_bias_channels_first(
        ctx,
        x1,
        weights.ffn_up_w,
        weights.ffn_up_b,
        /*stride=*/1,
        /*padding=*/1);
    ffn = ggml_relu(ctx, ffn);
    ffn = conv1d_with_bias_channels_first(
        ctx,
        ffn,
        weights.ffn_down_w,
        weights.ffn_down_b,
        /*stride=*/1,
        /*padding=*/1);

    return layer_norm_2d(ctx, ggml_add(ctx, x1, ffn), weights.ln2_w, weights.ln2_b);
}

template <size_t N>
static ::ggml_tensor * relpos_encoder_stack_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const std::array<sovits_relpos_encoder_layer_weights, N> & layers)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);

    for (const sovits_relpos_encoder_layer_weights & layer : layers) {
        x = text_encoder_ssl_layer_forward(ctx, x, layer);
    }

    return x;
}

static ::ggml_tensor * masked_temporal_avg_pool(
    ::ggml_context * ctx,
    ::ggml_tensor  * x)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);

    ::ggml_tensor * x_tc = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_tc = ggml_cont(ctx, x_tc);
    ::ggml_tensor * summed = ggml_sum_rows(ctx, x_tc); // {1, C}
    summed = ggml_reshape_2d(ctx, summed, x->ne[0], 1);
    return ggml_scale(ctx, summed, 1.0f / static_cast<float>(x->ne[1]));
}

static ::ggml_tensor * conv_glu_block_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_mel_style_encoder_conv_glu_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weights.conv_w != nullptr);
    GGML_ASSERT(weights.conv_b != nullptr);
    GGML_ASSERT(x->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.conv_w->ne[0] == kStyleKernel);
    GGML_ASSERT(weights.conv_w->ne[1] == kStyleHidden);
    GGML_ASSERT(weights.conv_w->ne[2] == 2 * kStyleHidden);
    GGML_ASSERT(weights.conv_b->ne[0] == 2 * kStyleHidden);

    ::ggml_tensor * conv = conv1d_forward_channels_first(ctx, x, weights.conv_w, /*stride=*/1, /*padding=*/2);
    conv = ggml_add(ctx, conv, reshape_bias_2d(ctx, weights.conv_b));

    ::ggml_tensor * a = split_channels(ctx, conv, 0, kStyleHidden);
    ::ggml_tensor * b = split_channels(ctx, conv, kStyleHidden, kStyleHidden);
    ::ggml_tensor * gated = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
    return ggml_add(ctx, x, gated);
}

static ::ggml_tensor * attention_block_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_mel_style_encoder_attention_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.q_w != nullptr);
    GGML_ASSERT(weights.q_b != nullptr);
    GGML_ASSERT(weights.k_w != nullptr);
    GGML_ASSERT(weights.k_b != nullptr);
    GGML_ASSERT(weights.v_w != nullptr);
    GGML_ASSERT(weights.v_b != nullptr);
    GGML_ASSERT(weights.out_w != nullptr);
    GGML_ASSERT(weights.out_b != nullptr);

    const int64_t time = x->ne[1];
    const size_t esz = ggml_element_size(x);

    ::ggml_tensor * q = linear_2d(ctx, x, weights.q_w, weights.q_b);
    ::ggml_tensor * k = linear_2d(ctx, x, weights.k_w, weights.k_b);
    ::ggml_tensor * v = linear_2d(ctx, x, weights.v_w, weights.v_b);

    ::ggml_tensor * q_3d = ggml_view_3d(ctx, q,
        kStyleHeadDim, kStyleHeads, time,
        /*nb1=*/ esz * kStyleHeadDim,
        /*nb2=*/ q->nb[1],
        /*off=*/ 0);
    q_3d = ggml_permute(ctx, q_3d, 0, 2, 1, 3);

    ::ggml_tensor * k_3d = ggml_view_3d(ctx, k,
        kStyleHeadDim, kStyleHeads, time,
        /*nb1=*/ esz * kStyleHeadDim,
        /*nb2=*/ k->nb[1],
        /*off=*/ 0);
    k_3d = ggml_permute(ctx, k_3d, 0, 2, 1, 3);

    ::ggml_tensor * v_3d = ggml_view_3d(ctx, v,
        kStyleHeadDim, kStyleHeads, time,
        /*nb1=*/ esz * kStyleHeadDim,
        /*nb2=*/ v->nb[1],
        /*off=*/ 0);
    v_3d = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

    const float scale = 1.0f / sqrtf(static_cast<float>(kStyleHidden));
    ::ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_3d, k_3d, v_3d, nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, kStyleHidden, time);

    ::ggml_tensor * projected = linear_2d(ctx, attn, weights.out_w, weights.out_b);
    return ggml_add(ctx, projected, x);
}

static ::ggml_tensor * mrte_cross_attention_from_qkv(
    ::ggml_context * ctx,
    ::ggml_tensor  * q,
    ::ggml_tensor  * k,
    ::ggml_tensor  * v)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(q != nullptr);
    GGML_ASSERT(k != nullptr);
    GGML_ASSERT(v != nullptr);
    GGML_ASSERT(q->ne[0] == kTextEncoderMrteHidden);
    GGML_ASSERT(k->ne[0] == kTextEncoderMrteHidden);
    GGML_ASSERT(v->ne[0] == kTextEncoderMrteHidden);

    const int64_t q_time = q->ne[1];
    const int64_t kv_time = k->ne[1];
    const size_t esz = ggml_element_size(q);

    ::ggml_tensor * q_3d = ggml_view_3d(ctx, q,
        kTextEncoderMrteHeadDim, kTextEncoderMrteHeads, q_time,
        /*nb1=*/ esz * kTextEncoderMrteHeadDim,
        /*nb2=*/ q->nb[1],
        /*off=*/ 0);
    q_3d = ggml_permute(ctx, q_3d, 0, 2, 1, 3);

    ::ggml_tensor * k_3d = ggml_view_3d(ctx, k,
        kTextEncoderMrteHeadDim, kTextEncoderMrteHeads, kv_time,
        /*nb1=*/ esz * kTextEncoderMrteHeadDim,
        /*nb2=*/ k->nb[1],
        /*off=*/ 0);
    k_3d = ggml_permute(ctx, k_3d, 0, 2, 1, 3);

    ::ggml_tensor * v_3d = ggml_view_3d(ctx, v,
        kTextEncoderMrteHeadDim, kTextEncoderMrteHeads, kv_time,
        /*nb1=*/ esz * kTextEncoderMrteHeadDim,
        /*nb2=*/ v->nb[1],
        /*off=*/ 0);
    v_3d = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

    const float scale = 1.0f / sqrtf(static_cast<float>(kTextEncoderMrteHeadDim));
    ::ggml_tensor * attn = ggml_flash_attn_ext(
        ctx,
        q_3d,
        k_3d,
        v_3d,
        /*mask=*/ nullptr,
        scale,
        /*max_bias=*/ 0.0f,
        /*logit_softcap=*/ 0.0f);
    return ggml_reshape_2d(ctx, attn, kTextEncoderMrteHidden, q_time);
}

} // namespace

::ggml_tensor * sovits_mel_style_encoder_block_forward(
    ::ggml_context                           * ctx,
    ::ggml_tensor                            * refer,
    const sovits_mel_style_encoder_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(refer != nullptr);
    GGML_ASSERT(refer->ne[0] == kMelChannels);
    GGML_ASSERT(weights.spectral_1_w != nullptr);
    GGML_ASSERT(weights.spectral_1_b != nullptr);
    GGML_ASSERT(weights.spectral_2_w != nullptr);
    GGML_ASSERT(weights.spectral_2_b != nullptr);
    GGML_ASSERT(weights.fc_w != nullptr);
    GGML_ASSERT(weights.fc_b != nullptr);
    GGML_ASSERT(weights.spectral_1_w->ne[0] == kMelChannels);
    GGML_ASSERT(weights.spectral_1_w->ne[1] == kStyleHidden);
    GGML_ASSERT(weights.spectral_1_b->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.spectral_2_w->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.spectral_2_w->ne[1] == kStyleHidden);
    GGML_ASSERT(weights.spectral_2_b->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.fc_w->ne[0] == kStyleHidden);
    GGML_ASSERT(weights.fc_w->ne[1] == kStyleOut);
    GGML_ASSERT(weights.fc_b->ne[0] == kStyleOut);

    ::ggml_tensor * x = linear_2d(ctx, refer, weights.spectral_1_w, weights.spectral_1_b);
    x = mish_2d(ctx, x);
    x = linear_2d(ctx, x, weights.spectral_2_w, weights.spectral_2_b);
    x = mish_2d(ctx, x);

    for (const sovits_mel_style_encoder_conv_glu_block_weights & block : weights.temporal) {
        x = conv_glu_block_forward(ctx, x, block);
    }

    x = attention_block_forward(ctx, x, weights.attention);
    x = linear_2d(ctx, x, weights.fc_w, weights.fc_b);
    return masked_temporal_avg_pool(ctx, x);
}

::ggml_tensor * sovits_rvq_decode_block_forward(
    ::ggml_context                       * ctx,
    ::ggml_tensor                        * codes,
    const sovits_rvq_decode_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(codes != nullptr);
    GGML_ASSERT(weights.codebook != nullptr);
    GGML_ASSERT(codes->type == GGML_TYPE_I32);
    GGML_ASSERT(weights.codebook->ne[0] == kRVQDim);
    GGML_ASSERT(weights.codebook->ne[1] == kRVQBins);

    ::ggml_tensor * codes_vec = flatten_vector_1d(ctx, codes);
    GGML_ASSERT(codes_vec->ne[0] >= 0);

    return ggml_get_rows(ctx, weights.codebook, codes_vec);
}

::ggml_tensor * sovits_text_encoder_ssl_block_forward(
    ::ggml_context                         * ctx,
    ::ggml_tensor                          * ssl,
    const sovits_text_encoder_ssl_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ssl != nullptr);
    GGML_ASSERT(weights.ssl_proj_w != nullptr);
    GGML_ASSERT(weights.ssl_proj_b != nullptr);
    GGML_ASSERT(ssl->ne[0] == kTextEncoderSslIn);
    GGML_ASSERT(weights.ssl_proj_w->ne[0] == 1);
    GGML_ASSERT(weights.ssl_proj_w->ne[1] == kTextEncoderSslIn);
    GGML_ASSERT(weights.ssl_proj_w->ne[2] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.ssl_proj_b->ne[0] == kTextEncoderSslHidden);

    ::ggml_tensor * x = conv1d_with_bias_channels_first(
        ctx,
        ssl,
        weights.ssl_proj_w,
        weights.ssl_proj_b,
        /*stride=*/1,
        /*padding=*/0);

    return relpos_encoder_stack_forward(ctx, x, weights.layers);
}

::ggml_tensor * sovits_text_encoder_text_block_forward(
    ::ggml_context                          * ctx,
    ::ggml_tensor                           * text,
    const sovits_text_encoder_text_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(text != nullptr);
    GGML_ASSERT(text->type == GGML_TYPE_I32);
    GGML_ASSERT(weights.text_embedding != nullptr);
    GGML_ASSERT(weights.text_embedding->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.text_embedding->ne[1] == kTextEncoderTextVocabV2);
    GGML_ASSERT(text->ne[0] > 0);

    ::ggml_tensor * x = ggml_get_rows(ctx, weights.text_embedding, text);
    return relpos_encoder_stack_forward(ctx, x, weights.layers);
}

::ggml_tensor * sovits_text_encoder_mrte_block_forward(
    ::ggml_context                                * ctx,
    ::ggml_tensor                                 * ssl,
    ::ggml_tensor                                 * text,
    ::ggml_tensor                                 * ge,
    const sovits_text_encoder_mrte_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ssl != nullptr);
    GGML_ASSERT(text != nullptr);
    GGML_ASSERT(ge != nullptr);
    GGML_ASSERT(weights.ssl_fused_w != nullptr);
    GGML_ASSERT(weights.ssl_fused_b != nullptr);
    GGML_ASSERT(weights.text_kv_w != nullptr);
    GGML_ASSERT(weights.text_kv_b != nullptr);
    GGML_ASSERT(weights.attn_out_w != nullptr);
    GGML_ASSERT(weights.attn_out_b != nullptr);
    GGML_ASSERT(weights.ge_out_w != nullptr);
    GGML_ASSERT(weights.ge_out_b != nullptr);
    GGML_ASSERT(ssl->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(text->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(ge->ne[0] == kTextEncoderMrteHidden);
    GGML_ASSERT(ge->ne[1] == 1);
    GGML_ASSERT(weights.ssl_fused_w->ne[0] == 1);
    GGML_ASSERT(weights.ssl_fused_w->ne[1] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.ssl_fused_w->ne[2] == kTextEncoderMrteSslFusedDim);
    GGML_ASSERT(weights.ssl_fused_b->ne[0] == kTextEncoderMrteSslFusedDim);
    GGML_ASSERT(weights.text_kv_w->ne[0] == 1);
    GGML_ASSERT(weights.text_kv_w->ne[1] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.text_kv_w->ne[2] == kTextEncoderMrteKvDim);
    GGML_ASSERT(weights.text_kv_b->ne[0] == kTextEncoderMrteKvDim);
    GGML_ASSERT(weights.attn_out_w->ne[0] == 1);
    GGML_ASSERT(weights.attn_out_w->ne[1] == kTextEncoderMrteHidden);
    GGML_ASSERT(weights.attn_out_w->ne[2] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.attn_out_b->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.ge_out_w->ne[0] == 1);
    GGML_ASSERT(weights.ge_out_w->ne[1] == kTextEncoderMrteHidden);
    GGML_ASSERT(weights.ge_out_w->ne[2] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.ge_out_b->ne[0] == kTextEncoderSslHidden);

    ::ggml_tensor * ssl_fused = conv1d_with_bias_channels_first(
        ctx,
        ssl,
        weights.ssl_fused_w,
        weights.ssl_fused_b,
        /*stride=*/1,
        /*padding=*/0);
    ::ggml_tensor * q = split_channels(ctx, ssl_fused, 0, kTextEncoderMrteQDim);
    ::ggml_tensor * skip = split_channels(ctx, ssl_fused, kTextEncoderMrteQDim, kTextEncoderMrteSkipDim);

    ::ggml_tensor * text_kv = conv1d_with_bias_channels_first(
        ctx,
        text,
        weights.text_kv_w,
        weights.text_kv_b,
        /*stride=*/1,
        /*padding=*/0);
    ::ggml_tensor * k = split_channels(ctx, text_kv, 0, kTextEncoderMrteHidden);
    ::ggml_tensor * v = split_channels(ctx, text_kv, kTextEncoderMrteHidden, kTextEncoderMrteHidden);

    ::ggml_tensor * attn = mrte_cross_attention_from_qkv(ctx, q, k, v);
    ::ggml_tensor * attn_out = conv1d_with_bias_channels_first(
        ctx,
        attn,
        weights.attn_out_w,
        weights.attn_out_b,
        /*stride=*/1,
        /*padding=*/0);
    ::ggml_tensor * ge_out = conv1d_with_bias_channels_first(
        ctx,
        ge,
        weights.ge_out_w,
        weights.ge_out_b,
        /*stride=*/1,
        /*padding=*/0);
    ge_out = ggml_repeat(ctx, ge_out, skip);

    return ggml_add(ctx, ggml_add(ctx, attn_out, skip), ge_out);
}

::ggml_tensor * sovits_text_encoder_post_block_forward(
    ::ggml_context                          * ctx,
    ::ggml_tensor                           * x,
    const sovits_text_encoder_post_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);

    return relpos_encoder_stack_forward(ctx, x, weights.layers);
}

} // namespace gpt_sovits
