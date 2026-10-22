#include "gpt_sovits/sv.h"
#include "gpt_sovits/audio.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gpt_sovits {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsBN = 1e-5f;          // PyTorch BatchNorm default eps
constexpr float kFbankEps = 1.1920928955078125e-07f; // float32 epsilon

// ---------------------------------------------------------------------------
// Weight / derived-constant lookup helpers.
//
// Conv weights live in model.ctx_w (backend-resident). BatchNorm layers are
// pre-folded at load time into per-channel scale/shift [C] tensors stored in
// model.ctx_bn under "<bn-prefix>.scale" / "<bn-prefix>.shift".
//
// ggml graph ops take non-const tensor pointers; weights are graph inputs that
// are never mutated, so the const_cast is safe.
// ---------------------------------------------------------------------------
static ggml_tensor * gw(const sv_model & m, const std::string & name) {
    return const_cast<ggml_tensor *>(sv_model_tensor(m, name));
}

static bool has_tensor(const sv_model & m, const std::string & name) {
    return sv_model_tensor(m, name) != nullptr;
}

// 2D convolution. Weight w is [kw,kh,Cin,Cout] (PyTorch {Cout,Cin,kh,kw}
// reversed), data x is [W,H,C,N]; stride/pad apply to both H and W. No bias.
static ggml_tensor * sv_conv2d(ggml_context * ctx, ggml_tensor * x,
                               ggml_tensor * w, int stride, int pad) {
    return ggml_conv_2d(ctx, w, x, stride, stride, pad, pad, 1, 1);
}

// 2D convolution + per-output-channel bias.
static ggml_tensor * sv_conv2d_bias(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * w, ggml_tensor * bias,
                                    int stride, int pad) {
    ggml_tensor * y = sv_conv2d(ctx, x, w, stride, pad);   // [Wout,Hout,Cout,1]
    ggml_tensor * b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(ctx, y, b);
}

// BatchNorm2d (eval) using pre-folded scale/shift: out = scale * x + shift,
// broadcast over W,H. x is [W,H,C,1]; scale/shift are [C] -> [1,1,C,1].
static ggml_tensor * sv_bn2d(ggml_context * ctx, ggml_tensor * x,
                             const sv_model & m, const std::string & prefix) {
    ggml_tensor * scale = gw(m, prefix + ".scale");
    ggml_tensor * shift = gw(m, prefix + ".shift");
    ggml_tensor * s = ggml_reshape_4d(ctx, scale, 1, 1, scale->ne[0], 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, shift, 1, 1, shift->ne[0], 1);
    return ggml_add(ctx, ggml_mul(ctx, x, s), b);
}

// ERes2NetV2.ReLU = Hardtanh(0, 20).
static ggml_tensor * sv_relu20(ggml_context * ctx, ggml_tensor * x) {
    return ggml_clamp(ctx, x, 0.0f, 20.0f);
}

// AFF attentional feature fusion: out = a*att + b*(2-att), att = 1 + tanh(h).
// Equivalent (no constants needed): out = (a+b) + tanh(h)*(a-b).
// h = local_att(cat(a,b)): Conv2d(2C->C/r)+bias -> BN -> SiLU
//                         -> Conv2d(C/r->C)+bias -> BN
static ggml_tensor * sv_aff(ggml_context * ctx, const sv_model & m,
                            ggml_tensor * a, ggml_tensor * b,
                            const std::string & prefix) {
    ggml_tensor * cat = ggml_concat(ctx, a, b, 2);   // channel dim -> [W,H,2C,1]
    ggml_tensor * h = sv_conv2d_bias(ctx, cat,
            gw(m, prefix + ".local_att.0.weight"),
            gw(m, prefix + ".local_att.0.bias"), 1, 0);
    h = sv_bn2d(ctx, h, m, prefix + ".local_att.1");
    h = ggml_silu(ctx, h);
    h = sv_conv2d_bias(ctx, h,
            gw(m, prefix + ".local_att.3.weight"),
            gw(m, prefix + ".local_att.3.bias"), 1, 0);
    h = sv_bn2d(ctx, h, m, prefix + ".local_att.4");

    ggml_tensor * t   = ggml_tanh(ctx, h);
    ggml_tensor * apb = ggml_add(ctx, a, b);
    ggml_tensor * amb = ggml_sub(ctx, a, b);
    return ggml_add(ctx, apb, ggml_mul(ctx, t, amb));
}

// One BasicBlock (plain and AFF-fuse variant). prefix e.g. "sv.layer3.1".
static ggml_tensor * sv_basic_block(ggml_context * ctx, const sv_model & m,
                                    ggml_tensor * x, int stride,
                                    const std::string & prefix) {
    const int scale = m.hparams.scale;

    // conv1 (1x1, stride) + bn1 + relu
    ggml_tensor * out = sv_relu20(ctx, sv_bn2d(ctx,
            sv_conv2d(ctx, x, gw(m, prefix + ".conv1.weight"), stride, 0),
            m, prefix + ".bn1"));

    const int width = static_cast<int>(out->ne[2]) / scale;   // per-slice channels
    // Slice `scale` channel groups from out [W,H,Cout,1]; each [W,H,width,1].
    auto channel_slice = [&](int i) -> ggml_tensor * {
        const size_t offset = static_cast<size_t>(i) * width * out->nb[2];
        return ggml_cont(ctx, ggml_view_4d(ctx, out,
                out->ne[0], out->ne[1], width, 1,
                out->nb[1], out->nb[2], out->nb[3], offset));
    };

    const bool has_fuse =
        has_tensor(m, prefix + ".fuse_models.0.local_att.0.weight");

    // Res2Net split residual: sp carries the previous slice's post-conv output.
    ggml_tensor * sp = channel_slice(0);
    std::vector<ggml_tensor *> outs;
    for (int i = 0; i < scale; ++i) {
        if (i > 0) {
            ggml_tensor * spx = channel_slice(i);
            if (has_fuse) {
                sp = sv_aff(ctx, m, sp, spx,
                            prefix + ".fuse_models." + std::to_string(i - 1));
            } else {
                sp = ggml_add(ctx, sp, spx);
            }
        }
        ggml_tensor * ci = sv_relu20(ctx, sv_bn2d(ctx,
                sv_conv2d(ctx, sp,
                          gw(m, prefix + ".convs." + std::to_string(i) + ".weight"), 1, 1),
                m, prefix + ".bns." + std::to_string(i)));
        outs.push_back(ci);
        sp = ci;   // next residual base is this slice's conv output
    }

    // Reassemble slices along the channel dim -> [W,H,Cout,1].
    ggml_tensor * assembled = outs[0];
    for (int i = 1; i < scale; ++i) {
        assembled = ggml_concat(ctx, assembled, outs[i], 2);
    }

    // conv3 (1x1) + bn3 (no activation yet).
    ggml_tensor * o = sv_bn2d(ctx,
            sv_conv2d(ctx, assembled, gw(m, prefix + ".conv3.weight"), 1, 0),
            m, prefix + ".bn3");

    // Residual: shortcut(x) if present else x; add then relu.
    ggml_tensor * residual = x;
    if (has_tensor(m, prefix + ".shortcut.0.weight")) {
        residual = sv_bn2d(ctx,
                sv_conv2d(ctx, x, gw(m, prefix + ".shortcut.0.weight"), stride, 0),
                m, prefix + ".shortcut.1");
    }
    return sv_relu20(ctx, ggml_add(ctx, o, residual));
}

static ggml_tensor * sv_run_layer(ggml_context * ctx, const sv_model & m,
                                  ggml_tensor * x, int blocks, int stride,
                                  const std::string & layer_prefix) {
    for (int b = 0; b < blocks; ++b) {
        x = sv_basic_block(ctx, m, x, b == 0 ? stride : 1,
                           layer_prefix + "." + std::to_string(b));
    }
    return x;
}

// ---------------------------------------------------------------------------
// Pre-fold every BatchNorm group in the host-loaded weights into scale/shift.
// A group is a 1D "<prefix>.weight" that has .bias/.running_mean/.running_var
// siblings (conv weights lack running_*). Produces, per group, two [C] tensors
// named "<prefix>.scale" and "<prefix>.shift".
// ---------------------------------------------------------------------------
static bool fold_batchnorms(gguf_context * gctx, ggml_context * hctx,
                            std::vector<std::pair<std::string, std::vector<float>>> & out) {
    const int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; ++i) {
        const char * cname = gguf_get_tensor_name(gctx, i);
        std::string name = cname ? cname : "";
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) continue;
        const std::string prefix = name.substr(0, name.size() - 7);

        const ggml_tensor * w  = ggml_get_tensor(hctx, (prefix + ".weight").c_str());
        const ggml_tensor * b  = ggml_get_tensor(hctx, (prefix + ".bias").c_str());
        const ggml_tensor * rm = ggml_get_tensor(hctx, (prefix + ".running_mean").c_str());
        const ggml_tensor * rv = ggml_get_tensor(hctx, (prefix + ".running_var").c_str());
        if (!rm && !rv) continue;
        for (const ggml_tensor * t : {w, b, rm, rv}) {
            if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != w->ne[0] ||
                t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) {
                fprintf(stderr, "%s: invalid BatchNorm group '%s'\n", __func__, prefix.c_str());
                return false;
            }
        }

        const int C = static_cast<int>(w->ne[0]);
        const float * wp = static_cast<const float *>(w->data);
        const float * bp = static_cast<const float *>(b->data);
        const float * mp = static_cast<const float *>(rm->data);
        const float * vp = static_cast<const float *>(rv->data);

        std::vector<float> scale(C), shift(C);
        for (int c = 0; c < C; ++c) {
            if (!std::isfinite(wp[c]) || !std::isfinite(bp[c]) ||
                !std::isfinite(mp[c]) || !std::isfinite(vp[c]) || vp[c] < 0.0f) {
                fprintf(stderr, "%s: invalid BatchNorm values in '%s'\n", __func__, prefix.c_str());
                return false;
            }
            const float s = wp[c] / std::sqrt(vp[c] + kEpsBN);
            scale[c] = s;
            shift[c] = bp[c] - mp[c] * s;
            if (!std::isfinite(scale[c]) || !std::isfinite(shift[c])) return false;
        }
        out.emplace_back(prefix + ".scale", std::move(scale));
        out.emplace_back(prefix + ".shift", std::move(shift));
    }
    return true;
}

static bool validate_model(gguf_context * gctx, ggml_context * ctx) {
    auto string_value = [&](const char * key, const char * expected) {
        const int64_t id = gguf_find_key(gctx, key);
        return id >= 0 && gguf_get_kv_type(gctx, id) == GGUF_TYPE_STRING &&
               std::strcmp(gguf_get_val_str(gctx, id), expected) == 0;
    };
    auto integer_value = [&](const char * key, uint32_t expected) {
        const int64_t id = gguf_find_key(gctx, key);
        return id >= 0 && gguf_get_kv_type(gctx, id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(gctx, id) == expected;
    };
    if (!string_value("general.architecture", "sovits-sv") ||
        !string_value("sv.arch", "ERes2NetV2") ||
        !integer_value("sv.base_width", 24) || !integer_value("sv.scale", 4) ||
        !integer_value("sv.expansion", 4) || !integer_value("sv.feat_dim", 80) ||
        !integer_value("sv.out_dim", 20480)) {
        fprintf(stderr, "%s: unsupported or missing SV architecture metadata\n", __func__);
        return false;
    }

    int64_t validated_tensors = 0;
    auto tensor = [&](const std::string & name, std::initializer_list<int64_t> shape,
                      bool conv = false) {
        const ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        bool valid = t && (t->type == GGML_TYPE_F32 || (conv && t->type == GGML_TYPE_F16));
        int dim = 0;
        for (int64_t size : shape) {
            valid = valid && t->ne[dim] == size;
            ++dim;
        }
        for (; dim < GGML_MAX_DIMS; ++dim) valid = valid && t->ne[dim] == 1;
        if (!valid) fprintf(stderr, "%s: missing or invalid tensor '%s'\n", __func__, name.c_str());
        ++validated_tensors;
        return valid;
    };
    auto bn = [&](const std::string & prefix, int channels) {
        for (const char * suffix : {".weight", ".bias", ".running_mean", ".running_var"}) {
            if (!tensor(prefix + suffix, {channels})) return false;
        }
        return true;
    };
    auto aff = [&](const std::string & prefix, int channels) {
        const int hidden = channels / 4;
        return tensor(prefix + ".local_att.0.weight", {1, 1, 2 * channels, hidden}, true) &&
               tensor(prefix + ".local_att.0.bias", {hidden}) &&
               bn(prefix + ".local_att.1", hidden) &&
               tensor(prefix + ".local_att.3.weight", {1, 1, hidden, channels}, true) &&
               tensor(prefix + ".local_att.3.bias", {channels}) &&
               bn(prefix + ".local_att.4", channels);
    };
    if (!tensor("sv.conv1.weight", {3, 3, 1, 64}, true) || !bn("sv.bn1", 64)) return false;
    const int blocks[] = {3, 4, 6, 3};
    int input_channels = 64;
    for (int layer = 0; layer < 4; ++layer) {
        const int width = 24 << layer;
        const int channels = 256 << layer;
        for (int block = 0; block < blocks[layer]; ++block) {
            const std::string prefix = "sv.layer" + std::to_string(layer + 1) + "." + std::to_string(block);
            if (!tensor(prefix + ".conv1.weight", {1, 1, input_channels, width * 4}, true) ||
                !bn(prefix + ".bn1", width * 4) ||
                !tensor(prefix + ".conv3.weight", {1, 1, width * 4, channels}, true) ||
                !bn(prefix + ".bn3", channels)) return false;
            for (int split = 0; split < 4; ++split) {
                const std::string index = std::to_string(split);
                if (!tensor(prefix + ".convs." + index + ".weight", {3, 3, width, width}, true) ||
                    !bn(prefix + ".bns." + index, width)) return false;
                if (layer >= 2 && split < 3 && !aff(prefix + ".fuse_models." + index, width)) return false;
            }
            if (block == 0) {
                if (!tensor(prefix + ".shortcut.0.weight", {1, 1, input_channels, channels}, true) ||
                    !bn(prefix + ".shortcut.1", channels)) return false;
            } else if (ggml_get_tensor(ctx, (prefix + ".shortcut.0.weight").c_str())) {
                fprintf(stderr, "%s: unexpected shortcut in '%s'\n", __func__, prefix.c_str());
                return false;
            }
            if (layer < 2 && ggml_get_tensor(ctx, (prefix + ".fuse_models.0.local_att.0.weight").c_str())) {
                fprintf(stderr, "%s: unexpected AFF in '%s'\n", __func__, prefix.c_str());
                return false;
            }
            input_channels = channels;
        }
    }
    if (!tensor("sv.layer3_ds.weight", {3, 3, 1024, 2048}, true) || !aff("sv.fuse34", 2048)) return false;
    // Extra inputs must not shadow derived BatchNorm constants or select a
    // different graph branch through tensor-presence checks.
    if (gguf_get_n_tensors(gctx) != validated_tensors) {
        fprintf(stderr, "%s: unexpected tensors in fixed SV architecture\n", __func__);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kaldi fbank frontend
// ---------------------------------------------------------------------------

static float mel_scale_scalar(float freq) { return 1127.0f * std::log(1.0f + freq / 700.0f); }

// Triangular mel filterbank: [num_bins, num_fft_bins/2] (last col dropped at use).
static std::vector<float> get_mel_banks(int num_bins, int padded_window, float fs,
                                        float low_freq, float high_freq) {
    const int num_fft_bins = padded_window / 2;
    const float nyquist = 0.5f * fs;
    if (high_freq <= 0.0f) high_freq += nyquist;
    const float fft_bin_width = fs / padded_window;
    const float mel_low = mel_scale_scalar(low_freq);
    const float mel_high = mel_scale_scalar(high_freq);
    const float mel_delta = (mel_high - mel_low) / (num_bins + 1);
    std::vector<float> left(num_bins), cen(num_bins), right(num_bins);
    for (int b = 0; b < num_bins; ++b) {
        left[b]  = mel_low + (b)      * mel_delta;
        cen[b]   = mel_low + (b + 1)  * mel_delta;
        right[b] = mel_low + (b + 2)  * mel_delta;
    }
    std::vector<float> bins(static_cast<size_t>(num_bins) * num_fft_bins, 0.0f);
    for (int f = 0; f < num_fft_bins; ++f) {
        const float mel = mel_scale_scalar(fft_bin_width * f);
        for (int b = 0; b < num_bins; ++b) {
            float up = (mel - left[b]) / (cen[b] - left[b]);
            float dn = (right[b] - mel) / (right[b] - cen[b]);
            float v = std::min(up, dn);
            if (v < 0.0f) v = 0.0f;
            bins[static_cast<size_t>(b) * num_fft_bins + f] = v;
        }
    }
    return bins;
}

} // namespace

// ---------------------------------------------------------------------------
// sv_model API
// ---------------------------------------------------------------------------

const ggml_tensor * sv_model_tensor(const sv_model & model, const std::string & name) {
    if (model.ctx_w) {
        if (const ggml_tensor * t = ggml_get_tensor(model.ctx_w, name.c_str())) {
            return t;
        }
    }
    // Derived BatchNorm scale/shift tensors live in a separate context.
    if (model.ctx_bn) {
        return ggml_get_tensor(model.ctx_bn, name.c_str());
    }
    return nullptr;
}

bool sv_model_load(const std::string & fname, sv_model & model,
                   ggml_backend_t backend) {
    if (!backend) return false;
    sv_model pending;
    pending.backend = backend;
    const auto cleanup_model = [](sv_model * value) { sv_model_free(*value); };
    std::unique_ptr<sv_model, decltype(cleanup_model)> pending_owner(&pending, cleanup_model);

    // The host GGUF and tensor context have separate ownership.

    struct ggml_context * ctx_host = nullptr;
    struct gguf_init_params host_params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_host,
    };
    struct gguf_context * ctx_gguf_host = gguf_init_from_file(fname.c_str(), host_params);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> host_owner(ctx_host, ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_owner(ctx_gguf_host, gguf_free);
    if (!ctx_gguf_host || !ctx_host) {
        fprintf(stderr, "%s: gguf_init_from_file('%s') failed\n", __func__, fname.c_str());
        return false;
    }
    if (!validate_model(ctx_gguf_host, ctx_host)) return false;

    std::vector<std::pair<std::string, std::vector<float>>> bn_consts;
    if (!fold_batchnorms(ctx_gguf_host, ctx_host, bn_consts)) return false;

    // Duplicate validated tensor descriptors instead of reopening the file.
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_host);
    const ggml_init_params w_params = {
        ggml_tensor_overhead() * (static_cast<size_t>(n_tensors) + 4), nullptr, true,
    };
    pending.ctx_w = ggml_init(w_params);
    if (!pending.ctx_w) return false;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_host, i);
        const ggml_tensor * th = ggml_get_tensor(ctx_host, name);
        ggml_tensor * tw = ggml_dup_tensor(pending.ctx_w, th);
        ggml_set_name(tw, name);
    }
    pending.buf_w = ggml_backend_alloc_ctx_tensors(pending.ctx_w, backend);
    if (!pending.buf_w) {
        fprintf(stderr, "%s: weight allocation failed\n", __func__);
        return false;
    }
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_host, i);
        const ggml_tensor * th = ggml_get_tensor(ctx_host, name);
        ggml_tensor * tw = ggml_get_tensor(pending.ctx_w, name);
        ggml_backend_tensor_set(tw, th->data, 0, ggml_nbytes(th));
    }

    const ggml_init_params bn_params = {
        ggml_tensor_overhead() * (bn_consts.size() + 4), nullptr, true,
    };
    pending.ctx_bn = ggml_init(bn_params);
    if (!pending.ctx_bn) return false;
    for (const auto & kv : bn_consts) {
        ggml_tensor * t = ggml_new_tensor_1d(pending.ctx_bn, GGML_TYPE_F32, kv.second.size());
        ggml_set_name(t, kv.first.c_str());
    }
    pending.buf_bn = ggml_backend_alloc_ctx_tensors(pending.ctx_bn, backend);
    if (!pending.buf_bn) {
        fprintf(stderr, "%s: BatchNorm allocation failed\n", __func__);
        return false;
    }
    for (const auto & kv : bn_consts) {
        ggml_tensor * t = ggml_get_tensor(pending.ctx_bn, kv.first.c_str());
        ggml_backend_tensor_set(t, kv.second.data(), 0, kv.second.size() * sizeof(float));
    }

    sv_model_free(model);
    model = pending;
    pending_owner.release();

    fprintf(stderr, "%s: loaded SV model from '%s' on backend "
                    "(baseWidth=%d scale=%d expansion=%d, %lld BN groups folded)\n",
            __func__, fname.c_str(), model.hparams.base_width, model.hparams.scale,
            model.hparams.expansion, (long long)(bn_consts.size() / 2));
    return true;
}

void sv_model_free(sv_model & model) {
    if (model.buf_bn) { ggml_backend_buffer_free(model.buf_bn); model.buf_bn = nullptr; }
    if (model.ctx_bn) { ggml_free(model.ctx_bn); model.ctx_bn = nullptr; }
    if (model.buf_w)  { ggml_backend_buffer_free(model.buf_w); model.buf_w = nullptr; }
    if (model.ctx_w)  { ggml_free(model.ctx_w); model.ctx_w = nullptr; }
    model.backend = nullptr;
}

// ---------------------------------------------------------------------------
// fbank
// ---------------------------------------------------------------------------

int sv_fbank(const float * audio, int64_t n_samples, std::vector<float> & out_feats) {
    constexpr int kNumMel = 80;
    constexpr float kFrameLen = 25.0f, kFrameShift = 10.0f, kFs = 16000.0f;
    constexpr float kPreemph = 0.97f;
    const int window_shift = static_cast<int>(kFs * kFrameShift * 0.001f);   // 160
    const int window_size  = static_cast<int>(kFs * kFrameLen * 0.001f);     // 400
    int padded = 1;                                  // round to power of two
    while (padded < window_size) padded <<= 1;       // 400 -> 512
    const int n_fft_bins = padded / 2;               // 256

    if (!audio || n_samples < window_size) return 0;
    const int64_t frames = 1 + (n_samples - window_size) / window_shift;
    if (frames > std::numeric_limits<int>::max() ||
        static_cast<uint64_t>(frames) > (out_feats.max_size() - out_feats.size()) / kNumMel) {
        return 0;
    }
    const int m = static_cast<int>(frames);

    // povey window = hann(periodic=False) ** 0.85
    std::vector<float> win(window_size);
    for (int i = 0; i < window_size; ++i) {
        const float hann = 0.5f - 0.5f * std::cos(2.0f * kPi * i / (window_size - 1));
        win[i] = std::pow(hann, 0.85f);
    }
    const auto mel = get_mel_banks(kNumMel, padded, kFs, 20.0f, 0.0f);  // [80, n_fft_bins]

    const size_t output_offset = out_feats.size();
    out_feats.insert(out_feats.end(), static_cast<size_t>(m) * kNumMel, 0.0f);
    std::vector<float> re(padded), im(padded, 0.0f);
    std::vector<float> frame(window_size);

    for (int fr = 0; fr < m; ++fr) {
        const float * base = audio + static_cast<int64_t>(fr) * window_shift;
        // Kaldi _get_window order: copy -> remove_dc_offset -> preemphasis -> window.
        for (int i = 0; i < window_size; ++i) frame[i] = base[i];
        // remove_dc_offset: subtract per-frame mean.
        float mean = 0.0f;
        for (int i = 0; i < window_size; ++i) mean += frame[i];
        mean /= static_cast<float>(window_size);
        for (int i = 0; i < window_size; ++i) frame[i] -= mean;
        // preemphasis (replicate pad: x[-1] == x[0]): x[0] -= p*x[0]; x[i] -= p*x[i-1].
        {
            float prev = frame[0];
            frame[0] = frame[0] * (1.0f - kPreemph);
            for (int i = 1; i < window_size; ++i) {
                const float cur = frame[i];
                frame[i] = cur - kPreemph * prev;
                prev = cur;
            }
        }
        // window + zero-pad to `padded`.
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        for (int i = 0; i < window_size; ++i) re[i] = frame[i] * win[i];
        GGML_ASSERT(audio_fft_radix2(re.data(), im.data(), static_cast<int64_t>(re.size())));
        // power spectrum (first n_fft_bins+1 rfft bins = indices 0..n_fft_bins)
        std::vector<float> spec(n_fft_bins + 1);
        for (int k = 0; k <= n_fft_bins; ++k) {
            spec[k] = re[k] * re[k] + im[k] * im[k];   // |fft|^2 (power)
        }
        // mel: out[b] = sum_k spec[k] * bins[b, k]   (bins uses k in 0..n_fft_bins-1, last padded col 0)
        float * feat = out_feats.data() + output_offset + static_cast<size_t>(fr) * kNumMel;
        for (int b = 0; b < kNumMel; ++b) {
            const float * row = mel.data() + static_cast<size_t>(b) * n_fft_bins;
            double acc = 0.0;
            for (int k = 0; k < n_fft_bins; ++k) acc += static_cast<double>(spec[k]) * row[k];
            // + spec[n_fft_bins]*0 (padded zero col) -> nothing
            float v = static_cast<float>(acc);
            feat[b] = std::log(v < kFbankEps ? kFbankEps : v);
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// forward3 (ggml graph)
// ---------------------------------------------------------------------------

bool sv_forward3(const sv_model & model, const float * fbank, int m_frames, float * out) {
    if (m_frames <= 0 || !fbank || !out || !model.backend ||
        !model.ctx_w || !model.buf_w || !model.ctx_bn || !model.buf_bn) {
        return false;
    }
    const int F       = model.hparams.feat_dim;   // 80
    const int out_dim = model.hparams.out_dim;    // 20480

    // --- Temp graph context. The SV graph is conv-heavy (~1.5-2.5k nodes once
    //     all views/conts/reshapes/elementwise ops are expanded), so size both
    //     the tensor pool and the cgraph generously. ---
    const size_t graph_size      = 8192;
    const size_t n_intermediates = 8192;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_owner(ctx, ggml_free);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() for graph context failed\n", __func__);
        return false;
    }

    // Input [time=m, freq=F, 1, 1] (ne[0]=time=conv width). fbank is row-major
    // [m, F] (freq contiguous); transpose into the ggml layout on upload.
    struct ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m_frames, F, 1, 1);
    ggml_set_name(in, "fbank");
    ggml_set_input(in);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    // conv1 (1->64, 3x3 pad1 stride1) + bn1 + relu
    struct ggml_tensor * h = ggml_relu(ctx, sv_bn2d(ctx,
            sv_conv2d(ctx, in, gw(model, "sv.conv1.weight"), 1, 1),
            model, "sv.bn1"));

    h = sv_run_layer(ctx, model, h, 3, 1, "sv.layer1");
    h = sv_run_layer(ctx, model, h, 4, 2, "sv.layer2");
    struct ggml_tensor * o3 = sv_run_layer(ctx, model, h,   6, 2, "sv.layer3");
    struct ggml_tensor * o4 = sv_run_layer(ctx, model, o3, 3, 2, "sv.layer4");

    // layer3_ds: Conv2d 3x3 stride2 pad1 (no BN / activation).
    struct ggml_tensor * o3ds = sv_conv2d(ctx, o3, gw(model, "sv.layer3_ds.weight"), 2, 1);

    // fuse34: AFF over (o4, o3ds).
    struct ggml_tensor * fuse = sv_aff(ctx, model, o4, o3ds, "sv.fuse34");

    // forward3: mean over time (ne[0]) -> flatten (channel, freq) -> [out_dim].
    const int C  = static_cast<int>(fuse->ne[2]);
    const int Hf = static_cast<int>(fuse->ne[1]);
    if (static_cast<size_t>(C) * static_cast<size_t>(Hf) != static_cast<size_t>(out_dim)) {
        fprintf(stderr, "%s: fuse C*Hf=%d*%d=%d != out_dim=%d\n", __func__,
                C, Hf, C * Hf, out_dim);
        return false;
    }
    struct ggml_tensor * result =
        ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_mean(ctx, fuse)),
                        static_cast<int64_t>(C) * Hf);
    ggml_set_name(result, "sv_emb");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // --- Allocate, upload input, compute, read back. ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model.backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc_owner(alloc, ggml_gallocr_free);
    if (!alloc) {
        fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }

    std::vector<float> in_host(static_cast<size_t>(m_frames) * F);
    for (int t = 0; t < m_frames; ++t) {
        for (int f = 0; f < F; ++f) {
            in_host[static_cast<size_t>(t) + static_cast<size_t>(f) * m_frames] =
                fbank[static_cast<size_t>(t) * F + f];
        }
    }
    ggml_backend_tensor_set(in, in_host.data(), 0,
                            static_cast<size_t>(m_frames) * F * sizeof(float));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }

    ggml_backend_tensor_get(result, out, 0,
                            static_cast<size_t>(out_dim) * sizeof(float));

    return true;
}

} // namespace gpt_sovits
