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
static constexpr float kGeneratorLreluSlope = 0.1f;
static constexpr std::array<int64_t, kSovitsGeneratorStages> kGeneratorStageOutChannels = {
    256, 128, 64, 32, 16,
};
static constexpr std::array<int, kSovitsGeneratorStages> kGeneratorUpsampleStrides = {
    10, 8, 2, 2, 2,
};
static constexpr std::array<int, kSovitsGeneratorStages> kGeneratorUpsamplePaddings = {
    3, 4, 3, 0, 0,
};
static constexpr std::array<int, kSovitsGeneratorBranches> kGeneratorResblockKernels = {
    3, 7, 11,
};
static constexpr std::array<int, kSovitsGeneratorResLayers> kGeneratorResblockDilations = {
    1, 3, 5,
};

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

static ::ggml_tensor * slice_time_range(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          start,
    int64_t          length);

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

static ::ggml_tensor * split_heads_3d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          head_dim,
    int64_t          n_heads)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == head_dim * n_heads);

    const size_t esz = ggml_element_size(x);
    return ggml_view_3d(
        ctx,
        x,
        head_dim,
        x->ne[1],
        n_heads,
        /*nb1=*/ x->nb[1],
        /*nb2=*/ esz * head_dim,
        /*offset=*/ 0);
}

static ::ggml_tensor * merge_heads_3d(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          hidden)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] * x->ne[2] == hidden);

    ::ggml_tensor * x_dht = ggml_permute(ctx, x, 0, 2, 1, 3);
    x_dht = ggml_cont(ctx, x_dht);
    return ggml_reshape_2d(ctx, x_dht, hidden, x->ne[1]);
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
    int              padding,
    int              dilation = 1)
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
        /*d0=*/dilation,
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
    int              padding,
    int              dilation = 1)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);

    // Conv1d with kernel=1/stride=1/pad=0 is exactly a per-time-step linear
    // projection on the channel axis, so avoid the im2col path.
    if (weight->ne[0] == 1 && stride == 1 && padding == 0 && dilation == 1) {
        ::ggml_tensor * linear_w = ggml_reshape_2d(ctx, weight, weight->ne[1], weight->ne[2]);
        return linear_2d(ctx, x, linear_w, bias);
    }

    ::ggml_tensor * y = conv1d_forward_channels_first(ctx, x, weight, stride, padding, dilation);
    y = ensure_f32(ctx, y);
    return ggml_add(ctx, y, reshape_bias_2d(ctx, bias));
}

static ::ggml_tensor * reshape_bias_2d_time_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * bias)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(bias != nullptr);
    return ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
}

static ::ggml_tensor * slice_time_range_time_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    int64_t          start,
    int64_t          length)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(start >= 0);
    GGML_ASSERT(length >= 0);
    GGML_ASSERT(start + length <= x->ne[0]);

    const size_t offset = (size_t) start * x->nb[0];
    ::ggml_tensor * view = ggml_view_2d(ctx, x, length, x->ne[1], x->nb[1], offset);
    return ggml_cont(ctx, view);
}

static ::ggml_tensor * conv1d_forward_time_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    int              stride,
    int              padding,
    int              dilation = 1)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(weight->ne[1] == x->ne[1]);

    const ggml_type im2col_type = x->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ::ggml_tensor * im2col = ggml_im2col(
        ctx,
        weight,
        x,
        stride,
        /*s1=*/0,
        padding,
        /*p1=*/0,
        /*d0=*/dilation,
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

    // Keep weights as src0 so ggml backends can pick native mixed-dtype or
    // quantized matmul kernels instead of forcing an explicit cast to F32.
    ::ggml_tensor * kernel = ggml_reshape_2d(
        ctx,
        weight,
        weight->ne[0] * weight->ne[1],
        weight->ne[2]);

    ::ggml_tensor * y = ggml_mul_mat(ctx, kernel, patches); // {C_out, T_out}
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));  // {T_out, C_out}
    return y;
}

static ::ggml_tensor * conv1d_with_bias_time_first(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * weight,
    ::ggml_tensor  * bias,
    int              stride,
    int              padding,
    int              dilation = 1)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(bias != nullptr);

    ::ggml_tensor * y = conv1d_forward_time_first(ctx, x, weight, stride, padding, dilation);
    y = ensure_f32(ctx, y);
    return ggml_add(ctx, y, reshape_bias_2d_time_first(ctx, bias));
}

static ::ggml_tensor * conv_transpose1d_with_bias_time_first(
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
    GGML_ASSERT(weight->ne[2] == x->ne[1]);

    ::ggml_tensor * y = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);

    if (padding != 0) {
        GGML_ASSERT(y->ne[0] >= 2 * padding);
        y = slice_time_range_time_first(ctx, y, padding, y->ne[0] - 2 * padding);
    }

    y = ensure_f32(ctx, y);
    return ggml_add(ctx, y, reshape_bias_2d_time_first(ctx, bias));
}

static ::ggml_tensor * generator_resblock1_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_generator_resblock1_weights & weights,
    int              kernel_size)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);

    for (int i = 0; i < kSovitsGeneratorResLayers; ++i) {
        const auto & c1 = weights.convs1[i];
        const auto & c2 = weights.convs2[i];
        GGML_ASSERT(c1.w != nullptr);
        GGML_ASSERT(c1.b != nullptr);
        GGML_ASSERT(c2.w != nullptr);
        GGML_ASSERT(c2.b != nullptr);

        ::ggml_tensor * xt = ggml_leaky_relu(ctx, x, kGeneratorLreluSlope, false);
        xt = conv1d_with_bias_time_first(
            ctx,
            xt,
            c1.w,
            c1.b,
            /*stride=*/1,
            /*padding=*/((kernel_size * kGeneratorResblockDilations[i]) - kGeneratorResblockDilations[i]) / 2,
            /*dilation=*/kGeneratorResblockDilations[i]);
        xt = ggml_leaky_relu(ctx, xt, kGeneratorLreluSlope, false);
        xt = conv1d_with_bias_time_first(
            ctx,
            xt,
            c2.w,
            c2.b,
            /*stride=*/1,
            /*padding=*/(kernel_size - 1) / 2,
            /*dilation=*/1);
        x = ggml_add(ctx, x, xt);
    }

    return x;
}

static ::ggml_tensor * generator_stage_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    const sovits_generator_stage_weights & weights,
    int              stage_idx)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(stage_idx >= 0 && stage_idx < kSovitsGeneratorStages);
    GGML_ASSERT(weights.up.w != nullptr);
    GGML_ASSERT(weights.up.b != nullptr);

    x = ggml_leaky_relu(ctx, x, kGeneratorLreluSlope, false);
    x = conv_transpose1d_with_bias_time_first(
        ctx,
        x,
        weights.up.w,
        weights.up.b,
        kGeneratorUpsampleStrides[stage_idx],
        kGeneratorUpsamplePaddings[stage_idx]);

    ::ggml_tensor * sum = nullptr;
    for (int branch = 0; branch < kSovitsGeneratorBranches; ++branch) {
        ::ggml_tensor * y = generator_resblock1_forward(
            ctx,
            x,
            weights.resblocks[branch],
            kGeneratorResblockKernels[branch]);
        sum = sum == nullptr ? y : ggml_add(ctx, sum, y);
    }

    return ggml_scale(ctx, sum, 1.0f / static_cast<float>(kSovitsGeneratorBranches));
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

static ::ggml_tensor * add_relative_logits_batched(
    ::ggml_context * ctx,
    ::ggml_tensor  * scores,
    ::ggml_tensor  * q_heads,
    ::ggml_tensor  * rel_k)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(scores != nullptr);
    GGML_ASSERT(q_heads != nullptr);
    GGML_ASSERT(rel_k != nullptr);
    GGML_ASSERT(scores->type == GGML_TYPE_F32);
    GGML_ASSERT(scores->ne[0] == q_heads->ne[1]);
    GGML_ASSERT(scores->ne[1] == q_heads->ne[1]);
    GGML_ASSERT(scores->ne[2] == q_heads->ne[2]);
    GGML_ASSERT(q_heads->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[0] == kTextEncoderSslHeadDim);
    GGML_ASSERT(rel_k->ne[1] == kTextEncoderSslRelSize);

    const int64_t time = q_heads->ne[1];
    const int64_t n_heads = q_heads->ne[2];
    ::ggml_tensor * rel_dot = ggml_mul_mat(ctx, rel_k, q_heads); // {rel, T, H}
    rel_dot = ensure_f32(ctx, rel_dot);

    for (int64_t rel_idx = 0; rel_idx < kTextEncoderSslRelSize; ++rel_idx) {
        const int64_t shift = rel_idx - kTextEncoderSslWindow;
        int64_t length = time - (shift >= 0 ? shift : -shift);
        if (length <= 0) {
            continue;
        }

        const int64_t q_start = shift >= 0 ? 0 : -shift;
        const int64_t key_start = shift >= 0 ? shift : 0;
        ::ggml_tensor * dot = ggml_view_3d(
            ctx,
            rel_dot,
            1,
            length,
            n_heads,
            rel_dot->nb[1],
            rel_dot->nb[2],
            (size_t) rel_idx * rel_dot->nb[0] + (size_t) q_start * rel_dot->nb[1]);

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

static ::ggml_tensor * build_relative_value_batched(
    ::ggml_context * ctx,
    ::ggml_tensor  * attn,
    ::ggml_tensor  * rel_v_t)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(attn != nullptr);
    GGML_ASSERT(rel_v_t != nullptr);
    GGML_ASSERT(rel_v_t->ne[0] == kTextEncoderSslRelSize);
    GGML_ASSERT(rel_v_t->ne[1] == kTextEncoderSslHeadDim);

    const int64_t time = attn->ne[0];
    const int64_t n_heads = attn->ne[2];
    ::ggml_tensor * rel_weights = ggml_new_tensor_3d(
        ctx,
        GGML_TYPE_F32,
        kTextEncoderSslRelSize,
        time,
        n_heads);
    rel_weights = ggml_fill(ctx, rel_weights, 0.0f);

    for (int64_t rel_idx = 0; rel_idx < kTextEncoderSslRelSize; ++rel_idx) {
        const int64_t shift = rel_idx - kTextEncoderSslWindow;
        int64_t length = time - (shift >= 0 ? shift : -shift);
        if (length <= 0) {
            continue;
        }

        const int64_t key_start = shift >= 0 ? shift : 0;
        const int64_t query_start = shift >= 0 ? 0 : -shift;
        const size_t offset = (size_t) key_start * attn->nb[0] + (size_t) query_start * attn->nb[1];
        ::ggml_tensor * diag = ggml_view_3d(
            ctx,
            attn,
            1,
            length,
            n_heads,
            attn->nb[0] + attn->nb[1],
            attn->nb[2],
            offset);

        rel_weights = ggml_set_inplace(
            ctx,
            rel_weights,
            diag,
            rel_weights->nb[1],
            rel_weights->nb[2],
            rel_weights->nb[3],
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

    const float attn_scale = 1.0f / sqrtf((float) kTextEncoderSslHeadDim);
    ::ggml_tensor * q_heads = split_heads_3d(ctx, q, kTextEncoderSslHeadDim, kTextEncoderSslHeads);
    ::ggml_tensor * k_heads = split_heads_3d(ctx, k, kTextEncoderSslHeadDim, kTextEncoderSslHeads);
    ::ggml_tensor * v_heads = split_heads_3d(ctx, v, kTextEncoderSslHeadDim, kTextEncoderSslHeads);

    ::ggml_tensor * scores = ggml_mul_mat(ctx, k_heads, q_heads); // {T, T, H}
    scores = ensure_f32(ctx, scores);
    scores = add_relative_logits_batched(ctx, scores, q_heads, weights.rel_k);

    ::ggml_tensor * attn = ggml_soft_max_ext(ctx, scores, nullptr, attn_scale, 0.0f);

    ::ggml_tensor * v_t = ggml_permute(ctx, v_heads, 1, 0, 2, 3);
    v_t = ggml_cont(ctx, v_t);
    ::ggml_tensor * content_out = ggml_mul_mat(ctx, attn, v_t); // {T, D, H}
    content_out = ensure_f32(ctx, content_out);
    content_out = ggml_permute(ctx, content_out, 1, 0, 2, 3);
    content_out = ggml_cont(ctx, content_out);

    ::ggml_tensor * rel_out = build_relative_value_batched(ctx, attn, weights.rel_v_t);
    ::ggml_tensor * merged = ggml_add(ctx, content_out, rel_out); // {D, T, H}
    merged = merge_heads_3d(ctx, merged, kTextEncoderSslHidden);

    return conv1d_with_bias_channels_first(ctx, merged, weights.out_w, weights.out_b, 1, 0);
}

static ::ggml_tensor * relpos_encoder_layer_forward(
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

    // Shared by the ssl, text, and post encoder branches.
    for (const sovits_relpos_encoder_layer_weights & layer : layers) {
        x = relpos_encoder_layer_forward(ctx, x, layer);
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

static ::ggml_tensor * sovits_text_encoder_ssl_block_forward(
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

static ::ggml_tensor * sovits_text_encoder_text_block_forward(
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

static ::ggml_tensor * sovits_text_encoder_mrte_block_forward(
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

static ::ggml_tensor * sovits_text_encoder_post_block_forward(
    ::ggml_context                          * ctx,
    ::ggml_tensor                           * x,
    const sovits_text_encoder_post_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->ne[0] == kTextEncoderSslHidden);

    return relpos_encoder_stack_forward(ctx, x, weights.layers);
}

sovits_text_encoder_result sovits_text_encoder_block_forward(
    ::ggml_context                    * ctx,
    ::ggml_tensor                     * ssl,
    ::ggml_tensor                     * text,
    ::ggml_tensor                     * ge,
    const sovits_text_encoder_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(ssl != nullptr);
    GGML_ASSERT(text != nullptr);
    GGML_ASSERT(ge != nullptr);
    GGML_ASSERT(weights.post.proj_w != nullptr);
    GGML_ASSERT(weights.post.proj_b != nullptr);
    GGML_ASSERT(weights.post.proj_w->ne[0] == 1);
    GGML_ASSERT(weights.post.proj_w->ne[1] == kTextEncoderSslHidden);
    GGML_ASSERT(weights.post.proj_w->ne[2] == 2 * kTextEncoderSslHidden);
    GGML_ASSERT(weights.post.proj_b->ne[0] == 2 * kTextEncoderSslHidden);

    ::ggml_tensor * ssl_x = sovits_text_encoder_ssl_block_forward(ctx, ssl, weights.ssl);
    ::ggml_tensor * text_x = sovits_text_encoder_text_block_forward(ctx, text, weights.text);
    ::ggml_tensor * fused = sovits_text_encoder_mrte_block_forward(ctx, ssl_x, text_x, ge, weights.mrte);
    ::ggml_tensor * x = sovits_text_encoder_post_block_forward(ctx, fused, weights.post);

    ::ggml_tensor * stats = conv1d_with_bias_channels_first(
        ctx,
        x,
        weights.post.proj_w,
        weights.post.proj_b,
        /*stride=*/1,
        /*padding=*/0);

    sovits_text_encoder_result result;
    result.x = x;
    result.m = split_channels(ctx, stats, 0, kTextEncoderSslHidden);
    result.logs = split_channels(ctx, stats, kTextEncoderSslHidden, kTextEncoderSslHidden);
    return result;
}

// ---------------------------------------------------------------------------
// Flow block helpers
// ---------------------------------------------------------------------------

static ::ggml_tensor * fused_add_tanh_sigmoid_multiply(
    ::ggml_context * ctx,
    ::ggml_tensor  * input_a,
    ::ggml_tensor  * input_b,
    int64_t          n_channels)
{
    GGML_ASSERT(input_a != nullptr);
    GGML_ASSERT(input_b != nullptr);
    GGML_ASSERT(input_a->ne[0] == 2 * n_channels);

    ::ggml_tensor * in_act = ggml_add(ctx, input_a, input_b);
    ::ggml_tensor * t_act = split_channels(ctx, in_act, 0, n_channels);
    ::ggml_tensor * s_act = split_channels(ctx, in_act, n_channels, n_channels);
    t_act = ggml_tanh(ctx, t_act);
    s_act = ggml_sigmoid(ctx, s_act);
    return ggml_mul(ctx, t_act, s_act);
}

static ::ggml_tensor * sovits_wn_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * g,
    const sovits_wn_weights & w)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(g != nullptr);
    GGML_ASSERT(x->ne[0] == kSovitsFlowHidden);
    GGML_ASSERT(g->ne[0] == kSovitsFlowGin);
    GGML_ASSERT(w.cond_w != nullptr);
    GGML_ASSERT(w.cond_b != nullptr);

    const int64_t H = kSovitsFlowHidden;
    const int64_t T = x->ne[1];

    // cond_layer: Conv1d(gin, 2*H*N, k=1)
    ::ggml_tensor * cond_w_2d = ggml_reshape_2d(ctx, w.cond_w, w.cond_w->ne[1], w.cond_w->ne[2]);
    ::ggml_tensor * cond = linear_2d(ctx, g, cond_w_2d, w.cond_b);  // {2*H*N, 1}

    // Broadcast condition to full time dimension once (shared across all WN layers)
    cond = ggml_repeat_4d(ctx, cond, cond->ne[0], T, 1, 1);  // {2*H*N, T}

    ::ggml_tensor * output = zeros_2d(ctx, H, T);

    for (int i = 0; i < kSovitsFlowWNLayers; ++i) {
        GGML_ASSERT(w.layers[i].in_w != nullptr);
        GGML_ASSERT(w.layers[i].in_b != nullptr);
        GGML_ASSERT(w.layers[i].rs_w != nullptr);
        GGML_ASSERT(w.layers[i].rs_b != nullptr);

        // in_layer: dilated Conv1d(H, 2*H, K=5, dil=1, pad=2)
        ::ggml_tensor * x_in = conv1d_with_bias_channels_first(
            ctx, x, w.layers[i].in_w, w.layers[i].in_b, 1, 2);  // {2H, T}

        // Condition slice — already broadcasted to full time
        ::ggml_tensor * g_l = split_channels(ctx, cond, i * 2 * H, 2 * H);  // {2H, T}

        // Gated activation: tanh(a) * sigmoid(b)
        ::ggml_tensor * acts = fused_add_tanh_sigmoid_multiply(ctx, x_in, g_l, H);  // {H, T}

        // res_skip_layer: Conv1d(H, out_ch, K=1)
        ::ggml_tensor * res_skip = conv1d_with_bias_channels_first(
            ctx, acts, w.layers[i].rs_w, w.layers[i].rs_b, 1, 0);  // {2H, T} or {H, T}

        const int64_t out_ch = w.layers[i].rs_w->ne[2];  // 2*H for layers 0..2, H for layer 3

        if (i < kSovitsFlowWNLayers - 1) {
            GGML_ASSERT(out_ch == 2 * H);
            ::ggml_tensor * res  = split_channels(ctx, res_skip, 0, H);
            ::ggml_tensor * skip = split_channels(ctx, res_skip, H, H);
            x = ggml_add(ctx, x, res);
            output = ggml_add(ctx, output, skip);
        } else {
            output = ggml_add(ctx, output, res_skip);
        }
    }

    return output;
}

static ::ggml_tensor * sovits_flow_layer_inverse_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * g,
    const sovits_flow_layer_weights & w,
    bool             flip_input)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(g != nullptr);
    GGML_ASSERT(x->ne[0] == kSovitsFlowChannels);
    GGML_ASSERT(w.pre_w != nullptr);
    GGML_ASSERT(w.pre_b != nullptr);
    GGML_ASSERT(w.post_w != nullptr);
    GGML_ASSERT(w.post_b != nullptr);

    const int64_t half = kSovitsFlowHalfChannels;

    // Split: when flip_input=true swap which half is used as conditioning
    ::ggml_tensor * x0;  // conditioning half (fed into pre → WN → post)
    ::ggml_tensor * x1;  // transform half (will be x1 = x1 - m)
    if (flip_input) {
        x0 = split_channels(ctx, x, half, half);   // second half → conditioning
        x1 = split_channels(ctx, x, 0,    half);    // first half  → transform
    } else {
        x0 = split_channels(ctx, x, 0,    half);    // first half  → conditioning
        x1 = split_channels(ctx, x, half, half);    // second half → transform
    }

    // pre: Conv1d(half, H, k=1)
    ::ggml_tensor * pre_w_2d = ggml_reshape_2d(ctx, w.pre_w, w.pre_w->ne[1], w.pre_w->ne[2]);
    ::ggml_tensor * h = linear_2d(ctx, x0, pre_w_2d, w.pre_b);  // {H, T}

    // WaveNet
    h = sovits_wn_forward(ctx, h, g, w.enc);  // {H, T}

    // post: Conv1d(H, half, k=1) → mean_only → only m, no logs
    // Inverse step: x1 = (x1 - m) * exp(-logs), with logs=0 → x1 = x1 - m
    ::ggml_tensor * post_w_2d = ggml_reshape_2d(ctx, w.post_w, w.post_w->ne[1], w.post_w->ne[2]);
    ::ggml_tensor * m = linear_2d(ctx, h, post_w_2d, w.post_b);  // {half, T}
    x1 = ggml_sub(ctx, x1, m);  // output is contiguous (no ggml_cont needed)

    // Merge: cat in original channel order
    // x0 is a view — must make contiguous before concat
    if (flip_input) {
        // x1 (modified first half) goes first, x0 (original second half) goes second
        x0 = ggml_cont(ctx, x0);
        return ggml_concat(ctx, x1, x0, 0);
    } else {
        x0 = ggml_cont(ctx, x0);
        return ggml_concat(ctx, x0, x1, 0);
    }
}

// ---------------------------------------------------------------------------
// Flow block public interface (inference-only, inverse pass)
// ---------------------------------------------------------------------------

::ggml_tensor * sovits_flow_block_inverse_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * x,
    ::ggml_tensor  * g,
    const sovits_flow_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(g != nullptr);
    GGML_ASSERT(x->ne[0] == kSovitsFlowChannels);
    GGML_ASSERT(g->ne[0] == kSovitsFlowGin);

    // Inverse pass in reverse order, flip_input toggles per layer to
    // alternate which half is conditioned on — replaces explicit channel flips.
    bool flip_input = true;
    for (int i = kSovitsFlowNFlows - 1; i >= 0; --i) {
        x = sovits_flow_layer_inverse_forward(ctx, x, g, weights.layers[i], flip_input);
        flip_input = !flip_input;
    }

    return x;
}

::ggml_tensor * sovits_generator_block_forward(
    ::ggml_context * ctx,
    ::ggml_tensor  * z,
    ::ggml_tensor  * g,
    const sovits_generator_block_weights & weights)
{
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(z != nullptr);
    GGML_ASSERT(g != nullptr);
    GGML_ASSERT(z->ne[0] == kSovitsGeneratorIn);
    GGML_ASSERT(g->ne[0] == kSovitsGeneratorGin);
    GGML_ASSERT(g->ne[1] == 1);
    GGML_ASSERT(weights.conv_pre.w != nullptr);
    GGML_ASSERT(weights.conv_pre.b != nullptr);
    GGML_ASSERT(weights.cond.w != nullptr);
    GGML_ASSERT(weights.cond.b != nullptr);
    GGML_ASSERT(weights.conv_post_w != nullptr);

    ::ggml_tensor * z_tf = ggml_cont(ctx, ggml_permute(ctx, z, 1, 0, 2, 3));
    ::ggml_tensor * g_tf = ggml_cont(ctx, ggml_permute(ctx, g, 1, 0, 2, 3));

    ::ggml_tensor * x = conv1d_with_bias_time_first(
        ctx,
        z_tf,
        weights.conv_pre.w,
        weights.conv_pre.b,
        /*stride=*/1,
        /*padding=*/3);

    ::ggml_tensor * cond = conv1d_with_bias_time_first(
        ctx,
        g_tf,
        weights.cond.w,
        weights.cond.b,
        /*stride=*/1,
        /*padding=*/0);
    x = ggml_add(ctx, x, cond);

    for (int stage = 0; stage < kSovitsGeneratorStages; ++stage) {
        GGML_ASSERT(x->ne[1] == (stage == 0 ? 512 : kGeneratorStageOutChannels[stage - 1]));
        x = generator_stage_forward(ctx, x, weights.stages[stage], stage);
        GGML_ASSERT(x->ne[1] == kGeneratorStageOutChannels[stage]);
    }

    x = ggml_leaky_relu(ctx, x, kGeneratorLreluSlope, false);
    x = conv1d_forward_time_first(ctx, x, weights.conv_post_w, /*stride=*/1, /*padding=*/3);
    x = ggml_tanh(ctx, ensure_f32(ctx, x));
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
}

} // namespace gpt_sovits
