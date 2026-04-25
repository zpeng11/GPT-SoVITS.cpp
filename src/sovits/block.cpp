#include "gpt_sovits/sovits.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace gpt_sovits {

namespace {

static constexpr int64_t kMelChannels = 704;
static constexpr int64_t kStyleHidden = 128;
static constexpr int64_t kStyleOut = 512;
static constexpr int64_t kStyleKernel = 5;
static constexpr int64_t kStyleHeads = 2;
static constexpr int64_t kStyleHeadDim = kStyleHidden / kStyleHeads;

static ::ggml_tensor * reshape_bias_2d(
    ::ggml_context * ctx,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(bias != nullptr);
    return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
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

} // namespace gpt_sovits
