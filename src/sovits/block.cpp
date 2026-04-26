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
    ::ggml_tensor * view = ggml_view_2d(ctx, x, n_channels, x->ne[1], x->nb[1], offset);
    return ggml_cont(ctx, view);
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

static ::ggml_tensor * build_relative_logits_for_head(
    ::ggml_context * ctx,
    ::ggml_tensor  * q_head,
    ::ggml_tensor  * rel_k)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(q_head != nullptr);
    GGML_ASSERT(rel_k != nullptr);
    GGML_ASSERT(q_head->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[1] == kTextEncoderSslRelSize);

    const int64_t time = q_head->ne[1];
    ::ggml_tensor * scores = zeros_2d(ctx, time, time);

    for (int64_t rel_idx = 0; rel_idx < kTextEncoderSslRelSize; ++rel_idx) {
        const int64_t shift = rel_idx - kTextEncoderSslWindow;
        int64_t length = time - (shift >= 0 ? shift : -shift);
        if (length <= 0) {
            continue;
        }

        const int64_t q_start = shift >= 0 ? 0 : -shift;
        const int64_t key_start = shift >= 0 ? shift : 0;
        ::ggml_tensor * q_slice = slice_time_range(ctx, q_head, q_start, length);

        ::ggml_tensor * rel_vec = ggml_view_2d(
            ctx,
            rel_k,
            kTextEncoderSslHeadDim,
            1,
            rel_k->nb[1],
            (size_t) rel_idx * rel_k->nb[1]);
        rel_vec = ggml_cont(ctx, rel_vec);

        ::ggml_tensor * dot = ggml_mul_mat(ctx, rel_vec, q_slice);
        dot = ensure_f32(ctx, ggml_cont(ctx, dot));

        ::ggml_tensor * dot_vec = flatten_vector_1d(ctx, dot);
        ::ggml_tensor * diag = ggml_diag(ctx, dot_vec);
        diag = ensure_f32(ctx, diag);
        diag = ggml_pad_ext(
            ctx,
            diag,
            key_start,
            time - length - key_start,
            q_start,
            time - length - q_start,
            0,
            0,
            0,
            0);
        scores = ggml_add(ctx, scores, diag);
    }

    return scores;
}

static ::ggml_tensor * build_relative_value_for_head(
    ::ggml_context * ctx,
    ::ggml_tensor  * attn_head,
    ::ggml_tensor  * rel_v)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn_head != nullptr);
    GGML_ASSERT(rel_v != nullptr);
    GGML_ASSERT(rel_v->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_v->ne[1] == kTextEncoderSslRelSize);

    const int64_t time = attn_head->ne[0];
    ::ggml_tensor * output = zeros_2d(ctx, kTextEncoderSslHeadDim, time);

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
        diag = ensure_f32(ctx, ggml_cont(ctx, diag));

        ::ggml_tensor * rel_vec = ggml_view_2d(
            ctx,
            rel_v,
            kTextEncoderSslHeadDim,
            1,
            rel_v->nb[1],
            (size_t) rel_idx * rel_v->nb[1]);
        rel_vec = ggml_cont(ctx, rel_vec);

        ::ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kTextEncoderSslHeadDim, length);
        ::ggml_tensor * rel_slice = ggml_repeat(ctx, rel_vec, target);
        rel_slice = ensure_f32(ctx, rel_slice);
        ::ggml_tensor * diag_repeat = ggml_repeat(ctx, diag, rel_slice);
        diag_repeat = ensure_f32(ctx, diag_repeat);
        ::ggml_tensor * contrib = ggml_mul(ctx, rel_slice, diag_repeat);
        contrib = ensure_f32(ctx, contrib);
        contrib = ggml_pad_ext(
            ctx,
            contrib,
            0,
            0,
            query_start,
            time - length - query_start,
            0,
            0,
            0,
            0);
        output = ggml_add(ctx, output, contrib);
    }

    return output;
}

static ::ggml_tensor * self_attention_with_relative_position(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_text_encoder_ssl_layer_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.q_w != nullptr);
    GGML_ASSERT(weights.q_b != nullptr);
    GGML_ASSERT(weights.k_w != nullptr);
    GGML_ASSERT(weights.k_b != nullptr);
    GGML_ASSERT(weights.v_w != nullptr);
    GGML_ASSERT(weights.v_b != nullptr);
    GGML_ASSERT(weights.out_w != nullptr);
    GGML_ASSERT(weights.out_b != nullptr);
    GGML_ASSERT(weights.rel_k != nullptr);
    GGML_ASSERT(weights.rel_v != nullptr);

    ::ggml_tensor * q = conv1d_with_bias_channels_first(ctx, x, weights.q_w, weights.q_b, 1, 0);
    ::ggml_tensor * k = conv1d_with_bias_channels_first(ctx, x, weights.k_w, weights.k_b, 1, 0);
    ::ggml_tensor * v = conv1d_with_bias_channels_first(ctx, x, weights.v_w, weights.v_b, 1, 0);

    const int64_t time = x->ne[1];
    ::ggml_tensor * heads[2] = { nullptr, nullptr };

    for (int64_t h = 0; h < kTextEncoderSslHeads; ++h) {
        const int64_t ch0 = h * kTextEncoderSslHeadDim;
        ::ggml_tensor * q_head = split_channels(ctx, q, ch0, kTextEncoderSslHeadDim);
        ::ggml_tensor * k_head = split_channels(ctx, k, ch0, kTextEncoderSslHeadDim);
        ::ggml_tensor * v_head = split_channels(ctx, v, ch0, kTextEncoderSslHeadDim);

        ::ggml_tensor * content_scores = ggml_mul_mat(ctx, k_head, q_head);
        content_scores = ensure_f32(ctx, content_scores);
        content_scores = ggml_scale(ctx, content_scores, 1.0f / sqrtf((float) kTextEncoderSslHeadDim));

        ::ggml_tensor * rel_k = ggml_view_2d(
            ctx,
            weights.rel_k,
            kTextEncoderSslHeadDim,
            kTextEncoderSslRelSize,
            weights.rel_k->nb[1],
            0);
        rel_k = ggml_cont(ctx, rel_k);

        ::ggml_tensor * rel_scores = build_relative_logits_for_head(ctx, q_head, rel_k);
        rel_scores = ggml_scale(ctx, rel_scores, 1.0f / sqrtf((float) kTextEncoderSslHeadDim));
        ::ggml_tensor * scores = ggml_add(ctx, content_scores, rel_scores);
        ::ggml_tensor * attn = ggml_soft_max(ctx, scores);

        ::ggml_tensor * v_head_t = ggml_transpose(ctx, v_head);
        v_head_t = ggml_cont(ctx, v_head_t);
        ::ggml_tensor * content_out = ggml_mul_mat(ctx, attn, v_head_t);
        content_out = ggml_transpose(ctx, content_out);
        content_out = ggml_cont(ctx, content_out);

        ::ggml_tensor * rel_v = ggml_view_2d(
            ctx,
            weights.rel_v,
            kTextEncoderSslHeadDim,
            kTextEncoderSslRelSize,
            weights.rel_v->nb[1],
            0);
        rel_v = ggml_cont(ctx, rel_v);

        ::ggml_tensor * rel_out = build_relative_value_for_head(ctx, attn, rel_v);
        heads[h] = ggml_add(ctx, content_out, rel_out);
        GGML_ASSERT(heads[h]->ne[0] == kTextEncoderSslHeadDim);
        GGML_ASSERT(heads[h]->ne[1] == time);
    }

    ::ggml_tensor * merged = ggml_concat(ctx, heads[0], heads[1], 0);
    merged = ggml_cont(ctx, merged);
    return conv1d_with_bias_channels_first(ctx, merged, weights.out_w, weights.out_b, 1, 0);
}

static ::ggml_tensor * text_encoder_ssl_layer_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_text_encoder_ssl_layer_weights & weights)
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

    for (const sovits_text_encoder_ssl_layer_weights & layer : weights.layers) {
        x = text_encoder_ssl_layer_forward(ctx, x, layer);
    }

    return x;
}

} // namespace gpt_sovits
