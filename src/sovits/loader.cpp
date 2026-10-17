#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gpt_sovits {

namespace {

static struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in GGUF\n", __func__, name);
    }
    return t;
}

// ---------------------------------------------------------------------------
// Per-block populate helpers: each looks up one block's tensors (by disjoint
// GGUF name prefix) from a single shared ggml context. Used by the unified
// sovits_model_load to fill all 5 weight sub-structs from one GGUF file.
// ---------------------------------------------------------------------------

static bool populate_weights(
    struct ggml_context * ctx,
    sovits_mel_style_encoder_block_weights & w)
{
    w.spectral_1_w = checked_get_tensor(ctx, "ref_enc.spectral_1_w");
    w.spectral_1_b = checked_get_tensor(ctx, "ref_enc.spectral_1_b");
    w.spectral_2_w = checked_get_tensor(ctx, "ref_enc.spectral_2_w");
    w.spectral_2_b = checked_get_tensor(ctx, "ref_enc.spectral_2_b");
    if (!w.spectral_1_w || !w.spectral_1_b || !w.spectral_2_w || !w.spectral_2_b) {
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "ref_enc.temporal.%d.conv_w", i);
        w.temporal[i].conv_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "ref_enc.temporal.%d.conv_b", i);
        w.temporal[i].conv_b = checked_get_tensor(ctx, name);
        if (!w.temporal[i].conv_w || !w.temporal[i].conv_b) {
            return false;
        }
    }

    w.attention.qkv_w = checked_get_tensor(ctx, "ref_enc.attention.qkv_w");
    w.attention.qkv_b = checked_get_tensor(ctx, "ref_enc.attention.qkv_b");
    w.attention.out_w = checked_get_tensor(ctx, "ref_enc.attention.out_w");
    w.attention.out_b = checked_get_tensor(ctx, "ref_enc.attention.out_b");
    if (!w.attention.qkv_w || !w.attention.qkv_b ||
        !w.attention.out_w || !w.attention.out_b) {
        return false;
    }

    w.fc_w = checked_get_tensor(ctx, "ref_enc.fc_w");
    w.fc_b = checked_get_tensor(ctx, "ref_enc.fc_b");
    if (!w.fc_w || !w.fc_b) {
        return false;
    }

    return true;
}

// Populate v2Pro/v2ProPlus extra weights. Returns false only on a partial
// presence (some-but-not-all tensors, which would indicate a corrupt GGUF);
// a clean all-absent (v2) case returns true with all pointers null.
static bool populate_v2pro_weights(
    struct ggml_context * ctx,
    sovits_v2pro_block_weights & w)
{
    w.sv_emb_w   = ggml_get_tensor(ctx, "v2pro.sv_emb_w");
    w.sv_emb_b   = ggml_get_tensor(ctx, "v2pro.sv_emb_b");
    w.ge_to512_w = ggml_get_tensor(ctx, "v2pro.ge_to512_w");
    w.ge_to512_b = ggml_get_tensor(ctx, "v2pro.ge_to512_b");
    w.prelu_w    = ggml_get_tensor(ctx, "v2pro.prelu_w");

    const bool any = w.sv_emb_w || w.sv_emb_b || w.ge_to512_w ||
                     w.ge_to512_b || w.prelu_w;
    const bool all = w.sv_emb_w && w.sv_emb_b && w.ge_to512_w &&
                     w.ge_to512_b && w.prelu_w;
    if (any && !all) {
        fprintf(stderr, "%s: v2pro tensors partially present in GGUF "
                        "(expected all or none)\n", __func__);
        return false;
    }
    return true;
}

static bool populate_quantizer_weights(
    struct ggml_context * ctx,
    sovits_rvq_decode_block_weights & w)
{
    w.codebook = checked_get_tensor(ctx, "quantizer.codebook");
    if (!w.codebook) {
        return false;
    }

    return true;
}

static bool populate_text_encoder_ssl_weights(
    struct ggml_context * ctx,
    sovits_text_encoder_ssl_block_weights & w)
{
    w.ssl_proj_w = checked_get_tensor(ctx, "text_encoder_ssl.ssl_proj_w");
    w.ssl_proj_b = checked_get_tensor(ctx, "text_encoder_ssl.ssl_proj_b");
    if (!w.ssl_proj_w || !w.ssl_proj_b) {
        return false;
    }

    for (int i = 0; i < kSovitsTextEncoderSslLayers; ++i) {
        auto & layer = w.layers[i];
        char name[96];

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.qkv_w", i);
        layer.qkv_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.qkv_b", i);
        layer.qkv_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.out_w", i);
        layer.out_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.out_b", i);
        layer.out_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.rel_k", i);
        layer.rel_k = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.rel_v_t", i);
        layer.rel_v_t = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ln1_w", i);
        layer.ln1_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ln1_b", i);
        layer.ln1_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ffn_up_w", i);
        layer.ffn_up_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ffn_up_b", i);
        layer.ffn_up_b = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ffn_down_w", i);
        layer.ffn_down_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ffn_down_b", i);
        layer.ffn_down_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ln2_w", i);
        layer.ln2_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_ssl.layers.%d.ln2_b", i);
        layer.ln2_b = checked_get_tensor(ctx, name);

        if (!layer.qkv_w || !layer.qkv_b || !layer.out_w || !layer.out_b ||
            !layer.rel_k || !layer.rel_v_t ||
            !layer.ln1_w || !layer.ln1_b ||
            !layer.ffn_up_w || !layer.ffn_up_b ||
            !layer.ffn_down_w || !layer.ffn_down_b ||
            !layer.ln2_w || !layer.ln2_b) {
            return false;
        }
    }

    return true;
}

static bool populate_text_encoder_text_weights(
    struct ggml_context * ctx,
    sovits_text_encoder_text_block_weights & w)
{
    w.text_embedding = checked_get_tensor(ctx, "text_encoder_text.text_embedding");
    if (!w.text_embedding) {
        return false;
    }

    for (int i = 0; i < kSovitsTextEncoderTextLayers; ++i) {
        auto & layer = w.layers[i];
        char name[96];

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.qkv_w", i);
        layer.qkv_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.qkv_b", i);
        layer.qkv_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.out_w", i);
        layer.out_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.out_b", i);
        layer.out_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.rel_k", i);
        layer.rel_k = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.rel_v_t", i);
        layer.rel_v_t = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ln1_w", i);
        layer.ln1_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ln1_b", i);
        layer.ln1_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ffn_up_w", i);
        layer.ffn_up_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ffn_up_b", i);
        layer.ffn_up_b = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ffn_down_w", i);
        layer.ffn_down_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ffn_down_b", i);
        layer.ffn_down_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ln2_w", i);
        layer.ln2_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_text.layers.%d.ln2_b", i);
        layer.ln2_b = checked_get_tensor(ctx, name);

        if (!layer.qkv_w || !layer.qkv_b || !layer.out_w || !layer.out_b ||
            !layer.rel_k || !layer.rel_v_t ||
            !layer.ln1_w || !layer.ln1_b ||
            !layer.ffn_up_w || !layer.ffn_up_b ||
            !layer.ffn_down_w || !layer.ffn_down_b ||
            !layer.ln2_w || !layer.ln2_b) {
            return false;
        }
    }

    return true;
}

static bool populate_text_encoder_mrte_weights(
    struct ggml_context * ctx,
    sovits_text_encoder_mrte_block_weights & w)
{
    w.ssl_fused_w = checked_get_tensor(ctx, "text_encoder_mrte.ssl_fused_w");
    w.ssl_fused_b = checked_get_tensor(ctx, "text_encoder_mrte.ssl_fused_b");
    w.text_kv_w = checked_get_tensor(ctx, "text_encoder_mrte.text_kv_w");
    w.text_kv_b = checked_get_tensor(ctx, "text_encoder_mrte.text_kv_b");
    w.attn_out_w = checked_get_tensor(ctx, "text_encoder_mrte.attn_out_w");
    w.attn_out_b = checked_get_tensor(ctx, "text_encoder_mrte.attn_out_b");
    w.ge_out_w = checked_get_tensor(ctx, "text_encoder_mrte.ge_out_w");
    w.ge_out_b = checked_get_tensor(ctx, "text_encoder_mrte.ge_out_b");
    if (!w.ssl_fused_w || !w.ssl_fused_b || !w.text_kv_w || !w.text_kv_b ||
        !w.attn_out_w || !w.attn_out_b || !w.ge_out_w || !w.ge_out_b) {
        return false;
    }

    return true;
}

static bool populate_text_encoder_post_weights(
    struct ggml_context * ctx,
    sovits_text_encoder_post_block_weights & w)
{
    for (int i = 0; i < kSovitsTextEncoderPostLayers; ++i) {
        auto & layer = w.layers[i];
        char name[96];

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.qkv_w", i);
        layer.qkv_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.qkv_b", i);
        layer.qkv_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.out_w", i);
        layer.out_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.out_b", i);
        layer.out_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.rel_k", i);
        layer.rel_k = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.rel_v_t", i);
        layer.rel_v_t = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ln1_w", i);
        layer.ln1_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ln1_b", i);
        layer.ln1_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ffn_up_w", i);
        layer.ffn_up_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ffn_up_b", i);
        layer.ffn_up_b = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ffn_down_w", i);
        layer.ffn_down_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ffn_down_b", i);
        layer.ffn_down_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ln2_w", i);
        layer.ln2_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "text_encoder_post.layers.%d.ln2_b", i);
        layer.ln2_b = checked_get_tensor(ctx, name);

        if (!layer.qkv_w || !layer.qkv_b || !layer.out_w || !layer.out_b ||
            !layer.rel_k || !layer.rel_v_t ||
            !layer.ln1_w || !layer.ln1_b ||
            !layer.ffn_up_w || !layer.ffn_up_b ||
            !layer.ffn_down_w || !layer.ffn_down_b ||
            !layer.ln2_w || !layer.ln2_b) {
            return false;
        }
    }

    w.proj_w = checked_get_tensor(ctx, "text_encoder_post.proj_w");
    w.proj_b = checked_get_tensor(ctx, "text_encoder_post.proj_b");
    if (!w.proj_w || !w.proj_b) {
        return false;
    }

    return true;
}

static bool populate_text_encoder_weights(
    struct ggml_context * ctx,
    sovits_text_encoder_block_weights & w)
{
    return populate_text_encoder_ssl_weights(ctx, w.ssl)
        && populate_text_encoder_text_weights(ctx, w.text)
        && populate_text_encoder_mrte_weights(ctx, w.mrte)
        && populate_text_encoder_post_weights(ctx, w.post);
}

static bool populate_flow_weights(
    struct ggml_context * ctx,
    sovits_flow_block_weights & w)
{
    for (int L = 0; L < kSovitsFlowNFlows; ++L) {
        auto & layer = w.layers[L];
        char name[64];

        snprintf(name, sizeof(name), "flow.layers.%d.pre_w", L);
        layer.pre_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "flow.layers.%d.pre_b", L);
        layer.pre_b = checked_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "flow.layers.%d.post_w", L);
        layer.post_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "flow.layers.%d.post_b", L);
        layer.post_b = checked_get_tensor(ctx, name);

        if (!layer.pre_w || !layer.pre_b || !layer.post_w || !layer.post_b) {
            return false;
        }

        auto & enc = layer.enc;
        snprintf(name, sizeof(name), "flow.layers.%d.enc.cond_w", L);
        enc.cond_w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "flow.layers.%d.enc.cond_b", L);
        enc.cond_b = checked_get_tensor(ctx, name);

        if (!enc.cond_w || !enc.cond_b) {
            return false;
        }

        for (int j = 0; j < kSovitsFlowWNLayers; ++j) {
            snprintf(name, sizeof(name), "flow.layers.%d.enc.%d.in_w", L, j);
            enc.layers[j].in_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "flow.layers.%d.enc.%d.in_b", L, j);
            enc.layers[j].in_b = checked_get_tensor(ctx, name);

            snprintf(name, sizeof(name), "flow.layers.%d.enc.%d.rs_w", L, j);
            enc.layers[j].rs_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "flow.layers.%d.enc.%d.rs_b", L, j);
            enc.layers[j].rs_b = checked_get_tensor(ctx, name);

            if (!enc.layers[j].in_w || !enc.layers[j].in_b
                || !enc.layers[j].rs_w || !enc.layers[j].rs_b) {
                return false;
            }
        }
    }
    return true;
}

static bool populate_generator_weights(
    struct ggml_context * ctx,
    sovits_generator_block_weights & w)
{
    w.conv_pre.w = checked_get_tensor(ctx, "generator.conv_pre_w");
    w.conv_pre.b = checked_get_tensor(ctx, "generator.conv_pre_b");
    w.cond.w = checked_get_tensor(ctx, "generator.cond_w");
    w.cond.b = checked_get_tensor(ctx, "generator.cond_b");
    w.conv_post_w = checked_get_tensor(ctx, "generator.conv_post_w");

    if (!w.conv_pre.w || !w.conv_pre.b || !w.cond.w || !w.cond.b || !w.conv_post_w) {
        return false;
    }

    for (int s = 0; s < kSovitsGeneratorStages; ++s) {
        auto & stage = w.stages[s];
        char name[96];

        snprintf(name, sizeof(name), "generator.stages.%d.up_w", s);
        stage.up.w = checked_get_tensor(ctx, name);
        snprintf(name, sizeof(name), "generator.stages.%d.up_b", s);
        stage.up.b = checked_get_tensor(ctx, name);
        if (!stage.up.w || !stage.up.b) {
            return false;
        }

        for (int b = 0; b < kSovitsGeneratorBranches; ++b) {
            auto & block = stage.resblocks[b];
            for (int i = 0; i < kSovitsGeneratorResLayers; ++i) {
                snprintf(name, sizeof(name), "generator.stages.%d.resblocks.%d.convs1.%d.w", s, b, i);
                block.convs1[i].w = checked_get_tensor(ctx, name);
                snprintf(name, sizeof(name), "generator.stages.%d.resblocks.%d.convs1.%d.b", s, b, i);
                block.convs1[i].b = checked_get_tensor(ctx, name);
                snprintf(name, sizeof(name), "generator.stages.%d.resblocks.%d.convs2.%d.w", s, b, i);
                block.convs2[i].w = checked_get_tensor(ctx, name);
                snprintf(name, sizeof(name), "generator.stages.%d.resblocks.%d.convs2.%d.b", s, b, i);
                block.convs2[i].b = checked_get_tensor(ctx, name);

                if (!block.convs1[i].w || !block.convs1[i].b || !block.convs2[i].w || !block.convs2[i].b) {
                    return false;
                }
            }
        }
    }

    return true;
}

// Populate all weight sub-structs from one shared ggml context.
static bool populate_all_weights(
    struct ggml_context * ctx,
    sovits_model & m)
{
    return populate_weights(ctx, m.ref_enc)
        && populate_quantizer_weights(ctx, m.quantizer)
        && populate_text_encoder_weights(ctx, m.text_encoder)
        && populate_flow_weights(ctx, m.flow)
        && populate_generator_weights(ctx, m.generator)
        && populate_v2pro_weights(ctx, m.v2pro);
}

// ---------------------------------------------------------------------------
// Typed GGUF KV readers.
//
// Each returns 1 when the key is present with the expected type (value written
// to `out`), 0 when the key is absent (`out` untouched), and -1 when the key
// is present with the wrong type. Callers turn -1 into a load failure so a
// mistyped value never reaches gguf_get_val_* (undefined behaviour).
// ---------------------------------------------------------------------------

static int read_kv_bool(struct gguf_context * ctx, const char * key, bool & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return 0;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_BOOL) return -1;
    out = gguf_get_val_bool(ctx, id);
    return 1;
}

static int read_kv_u32(struct gguf_context * ctx, const char * key, uint32_t & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return 0;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) return -1;
    out = gguf_get_val_u32(ctx, id);
    return 1;
}

static int read_kv_str(struct gguf_context * ctx, const char * key, const char *& out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return 0;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return -1;
    out = gguf_get_val_str(ctx, id);
    return 1;
}

// ---------------------------------------------------------------------------
// Structural validation of a populated (not yet allocated) model.
//
// Derives gin_channels from the weights themselves (the MelStyleEncoder fc
// output, which all gin-consuming tensors must agree with) and rejects any
// disagreement with the GGUF metadata or with the v2Pro tensor shapes. Runs
// before the backend buffer is allocated so a malformed file fails cheaply
// and deterministically, instead of tripping a GGML_ASSERT deep in block.cpp.
// ---------------------------------------------------------------------------

// Fixed architecture constants (identical for v2 / v2Pro / v2ProPlus).
static constexpr int64_t kMelChannels = 704;   // ref_enc input slice
static constexpr int64_t kStyleHidden = 128;   // MelStyleEncoder hidden / fc input
static constexpr int64_t kRVQDim      = 768;   // quantizer codebook dim
static constexpr int64_t kRVQBins     = 1024;  // quantizer codebook bins
static constexpr int64_t kEncPGeDim   = 512;   // enc_p ge input (ge_to512 out)
static constexpr int64_t kSvEmbedDim  = 20480; // ERes2NetV2 embedding dim
static constexpr int64_t kGinV2       = 512;   // gin_channels for v2
static constexpr int64_t kGinV2Pro    = 1024;  // gin_channels for v2Pro/v2ProPlus

// flow.enc.cond_layer = Conv1d(gin, 2*H*N, k=1): 2*192*4 = 1536.
static constexpr int64_t kFlowCondDim = 2 * kSovitsFlowHidden * kSovitsFlowWNLayers;

// Generator HiFi-GAN layout (matches block.cpp).
static constexpr int64_t kGeneratorPreKernel  = 7;   // Conv1d(192, init, 7)
static constexpr int64_t kGeneratorPostKernel = 7;   // Conv1d(final, 1, 7)
static constexpr int     kGeneratorStrides[kSovitsGeneratorStages] = {10, 8, 2, 2, 2};
static constexpr int     kGeneratorResKernels[kSovitsGeneratorBranches] = {3, 7, 11};
static constexpr int     kGenUpK512[kSovitsGeneratorStages] = {16, 16, 8, 2, 2};
static constexpr int     kGenUpK768[kSovitsGeneratorStages] = {20, 16, 8, 2, 2};

static bool is_1d(const struct ggml_tensor * t) {
    return t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool is_2d(const struct ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1;
}
static bool is_3d(const struct ggml_tensor * t) {
    return t->ne[3] == 1;
}

// Validate the generator channel/kernel plan against the fixed HiFi-GAN
// architecture. gen_init is 512 (v2/v2Pro) or 768 (v2ProPlus); the first
// upsample kernel tracks it (16 vs 20), the remaining kernels are fixed.
static bool validate_generator(const sovits_generator_block_weights & w) {
    auto fail = [](const char * what) {
        fprintf(stderr, "%s: %s\n", __func__, what);
        return false;
    };

    struct ggml_tensor * pre_w  = w.conv_pre.w;
    struct ggml_tensor * pre_b  = w.conv_pre.b;
    struct ggml_tensor * cond_w = w.cond.w;
    struct ggml_tensor * cond_b = w.cond.b;
    struct ggml_tensor * post_w = w.conv_post_w;

    const int64_t gen_init = pre_b->ne[0];
    if (gen_init != 512 && gen_init != 768) {
        return fail("generator initial channels must be 512 or 768");
    }

    // conv_pre: Conv1d(192, gen_init, 7) -> flattened {192*7, gen_init}.
    if (!is_1d(pre_b)) return fail("generator.conv_pre bias is not a vector");
    if (!is_2d(pre_w) ||
        pre_w->ne[0] != kSovitsGeneratorIn * kGeneratorPreKernel ||
        pre_w->ne[1] != gen_init) {
        return fail("generator.conv_pre weight shape/kernel != {192*7, init}");
    }

    // cond: Conv1d(gin, gen_init, 1).
    if (!is_2d(cond_w) || cond_w->ne[1] != gen_init) {
        return fail("generator.cond weight != {gin, init}");
    }
    if (!is_1d(cond_b) || cond_b->ne[0] != gen_init) {
        return fail("generator.cond bias != {init}");
    }

    const int * up_kernels = (gen_init == 768) ? kGenUpK768 : kGenUpK512;

    for (int s = 0; s < kSovitsGeneratorStages; ++s) {
        const int64_t in_ch  = gen_init >> s;
        const int64_t out_ch = gen_init >> (s + 1);
        const auto & stage = w.stages[s];

        // up: ConvTranspose1d(in, out, k) -> {k, out, in}.
        if (!is_3d(stage.up.w) ||
            stage.up.w->ne[0] != up_kernels[s] ||
            stage.up.w->ne[1] != out_ch ||
            stage.up.w->ne[2] != in_ch) {
            return fail("generator stage up weight != {k, out, in}");
        }
        if (!is_1d(stage.up.b) || stage.up.b->ne[0] != out_ch) {
            return fail("generator stage up bias != {out}");
        }
        // HiFi-GAN padding = (k - stride)/2 must be a non-negative integer.
        const int stride = kGeneratorStrides[s];
        if (up_kernels[s] < stride || ((up_kernels[s] - stride) % 2) != 0) {
            return fail("generator stage up kernel/stride gives invalid padding");
        }

        // ResBlock1: 3 branches, each with a fixed kernel and `out_ch` channels.
        for (int b = 0; b < kSovitsGeneratorBranches; ++b) {
            const int64_t rk = kGeneratorResKernels[b];
            const auto & block = stage.resblocks[b];
            for (int i = 0; i < kSovitsGeneratorResLayers; ++i) {
                const auto & c1 = block.convs1[i];
                const auto & c2 = block.convs2[i];
                if (!is_2d(c1.w) || c1.w->ne[0] != out_ch * rk || c1.w->ne[1] != out_ch) {
                    return fail("generator resblock convs1 weight != {ch*k, ch}");
                }
                if (!is_1d(c1.b) || c1.b->ne[0] != out_ch) {
                    return fail("generator resblock convs1 bias != {ch}");
                }
                if (!is_2d(c2.w) || c2.w->ne[0] != out_ch * rk || c2.w->ne[1] != out_ch) {
                    return fail("generator resblock convs2 weight != {ch*k, ch}");
                }
                if (!is_1d(c2.b) || c2.b->ne[0] != out_ch) {
                    return fail("generator resblock convs2 bias != {ch}");
                }
            }
        }
    }

    // conv_post: Conv1d(gen_init/32, 1, 7) -> {final*7, 1}.
    const int64_t final_ch = gen_init >> kSovitsGeneratorStages;
    if (!is_2d(post_w) ||
        post_w->ne[0] != final_ch * kGeneratorPostKernel ||
        post_w->ne[1] != 1) {
        return fail("generator.conv_post weight shape/kernel != {final*7, 1}");
    }

    return true;
}

static bool validate_populated_model(
    sovits_model & m,
    bool     has_v2pro_kv,
    bool     is_v2pro_kv,
    bool     has_out_dim_kv,
    int64_t  out_dim_kv)
{
    auto fail = [](const char * what) {
        fprintf(stderr, "%s: %s\n", __func__, what);
        return false;
    };

    // The v2pro KV flag (when present) must agree with the tensor set.
    const bool has_v2pro_tensors = (m.v2pro.sv_emb_w != nullptr);
    if (has_v2pro_kv && is_v2pro_kv != has_v2pro_tensors) {
        return fail("sovits.v2pro KV disagrees with v2pro.* tensor presence");
    }

    // --- gin_channels: derive from ref_enc.fc output (authoritative) ---
    struct ggml_tensor * fc_w = m.ref_enc.fc_w;
    struct ggml_tensor * fc_b = m.ref_enc.fc_b;
    if (!is_2d(fc_w) || fc_w->ne[0] != kStyleHidden) {
        return fail("ref_enc.fc_w must be {128, gin}");
    }
    if (!is_1d(fc_b)) {
        return fail("ref_enc.fc_b must be a {gin} vector");
    }
    if (fc_w->ne[1] != fc_b->ne[0]) {
        return fail("ref_enc.fc weight/bias output dims disagree");
    }
    const int64_t gin = fc_w->ne[1];
    if (gin <= 0) {
        return fail("derived gin_channels <= 0");
    }

    // Metadata cross-check: sovits.ref_enc.out_dim must equal the fc output.
    if (has_out_dim_kv && out_dim_kv != gin) {
        return fail("sovits.ref_enc.out_dim contradicts ref_enc.fc output dim");
    }

    // --- ref_enc fixed dims (704 -> 128 -> 128) ---
    if (!is_2d(m.ref_enc.spectral_1_w) ||
        m.ref_enc.spectral_1_w->ne[0] != kMelChannels ||
        m.ref_enc.spectral_1_w->ne[1] != kStyleHidden) {
        return fail("ref_enc.spectral_1 weight shape != {704,128}");
    }
    if (!is_1d(m.ref_enc.spectral_1_b) || m.ref_enc.spectral_1_b->ne[0] != kStyleHidden) {
        return fail("ref_enc.spectral_1 bias != {128}");
    }
    if (!is_2d(m.ref_enc.spectral_2_w) ||
        m.ref_enc.spectral_2_w->ne[0] != kStyleHidden ||
        m.ref_enc.spectral_2_w->ne[1] != kStyleHidden) {
        return fail("ref_enc.spectral_2 weight shape != {128,128}");
    }
    if (!is_1d(m.ref_enc.spectral_2_b) || m.ref_enc.spectral_2_b->ne[0] != kStyleHidden) {
        return fail("ref_enc.spectral_2 bias != {128}");
    }

    // --- flow conditioning: cond_layer = Conv1d(gin, 2*H*N, 1) ---
    for (int L = 0; L < kSovitsFlowNFlows; ++L) {
        struct ggml_tensor * cw = m.flow.layers[L].enc.cond_w;
        struct ggml_tensor * cb = m.flow.layers[L].enc.cond_b;
        if (!is_2d(cw) || cw->ne[0] != gin || cw->ne[1] != kFlowCondDim) {
            return fail("flow enc.cond_w shape != {gin, 1536}");
        }
        if (!is_1d(cb) || cb->ne[0] != kFlowCondDim) {
            return fail("flow enc.cond_b shape != {1536}");
        }
    }

    // --- generator: full channel/kernel plan ---
    {
        struct ggml_tensor * gen_cond_w = m.generator.cond.w;
        if (!is_2d(gen_cond_w) || gen_cond_w->ne[0] != gin) {
            return fail("generator.cond in-dim != gin_channels");
        }
        if (!validate_generator(m.generator)) {
            return false;
        }
    }

    // --- quantizer fixed dims ---
    if (!is_2d(m.quantizer.codebook) ||
        m.quantizer.codebook->ne[0] != kRVQDim ||
        m.quantizer.codebook->ne[1] != kRVQBins) {
        return fail("quantizer.codebook shape != {768,1024}");
    }

    // --- version-specific gin + v2Pro tensor shapes ---
    if (has_v2pro_tensors) {
        if (gin != kGinV2Pro) {
            return fail("v2Pro/v2ProPlus model must have gin_channels == 1024");
        }
        if (!is_2d(m.v2pro.sv_emb_w) ||
            m.v2pro.sv_emb_w->ne[0] != kSvEmbedDim ||
            m.v2pro.sv_emb_w->ne[1] != gin ||
            !is_1d(m.v2pro.sv_emb_b) || m.v2pro.sv_emb_b->ne[0] != gin) {
            return fail("v2pro.sv_emb shape != {20480, gin}");
        }
        if (!is_2d(m.v2pro.ge_to512_w) ||
            m.v2pro.ge_to512_w->ne[0] != gin ||
            m.v2pro.ge_to512_w->ne[1] != kEncPGeDim ||
            !is_1d(m.v2pro.ge_to512_b) || m.v2pro.ge_to512_b->ne[0] != kEncPGeDim) {
            return fail("v2pro.ge_to512 shape != {gin, 512}");
        }
        if (!is_1d(m.v2pro.prelu_w) || m.v2pro.prelu_w->ne[0] != gin) {
            return fail("v2pro.prelu_w width != gin_channels");
        }
    } else {
        if (gin != kGinV2) {
            return fail("non-v2Pro model must have gin_channels == 512");
        }
    }

    m.hparams.is_v2pro    = has_v2pro_tensors;
    m.hparams.gin_channels = gin;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// sovits_model_load: load one unified GGUF into all 5 weight sub-structs.
//
// Mirrors t2s_model_load: a single gguf_init_from_file allocates every
// tensor in one backend buffer, populate_all_weights wires them into the
// weight sub-structs, and one pass streams tensor data from disk.
// ---------------------------------------------------------------------------

bool sovits_model_load(
    const std::string & fname,
    sovits_model & model,
    ggml_backend_t backend)
{
    if (backend == nullptr) {
        fprintf(stderr, "%s: backend is null\n", __func__);
        return false;
    }
    model.backend = backend;

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx_w,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: gguf_init_from_file('%s') failed\n", __func__, fname.c_str());
        return false;
    }

    // --- Metadata (typed; a mistyped value is a hard failure) ---
    bool     is_v2pro_kv    = false;
    bool     has_v2pro_kv   = false;
    uint32_t out_dim_u32    = 0;
    bool     has_out_dim_kv = false;
    const char * sfr        = nullptr;

    int rc = read_kv_bool(ctx_gguf, "sovits.v2pro", is_v2pro_kv);
    if (rc < 0) {
        fprintf(stderr, "%s: 'sovits.v2pro' is not a bool\n", __func__);
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }
    has_v2pro_kv = (rc == 1);

    rc = read_kv_u32(ctx_gguf, "sovits.ref_enc.out_dim", out_dim_u32);
    if (rc < 0) {
        fprintf(stderr, "%s: 'sovits.ref_enc.out_dim' is not a uint32\n", __func__);
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }
    has_out_dim_kv = (rc == 1);

    rc = read_kv_str(ctx_gguf, "sovits.semantic_frame_rate", sfr);
    if (rc < 0) {
        fprintf(stderr, "%s: 'sovits.semantic_frame_rate' is not a string\n", __func__);
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }
    // Default (absent) is the shipped v2 25hz checkpoint.
    model.hparams.semantic_frame_rate_25hz =
        !(rc == 1 && sfr != nullptr && std::strcmp(sfr, "50hz") == 0);

    // --- Populate tensor metadata, then validate before allocating. ---
    // gguf_init_from_file(no_alloc) already created every tensor descriptor in
    // model.ctx_w, so lookups and shape checks work with no backend buffer.
    if (!populate_all_weights(model.ctx_w, model)) {
        fprintf(stderr, "%s: populate_all_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }

    if (!validate_populated_model(model, has_v2pro_kv, is_v2pro_kv,
                                  has_out_dim_kv,
                                  has_out_dim_kv ? (int64_t) out_dim_u32 : -1)) {
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }

    // Generator channel plan / first kernel vary by version (v2ProPlus:
    // initial=768, kernel=20); both are derived from the loaded weights at
    // runtime, so no version-specific guard is needed here.

    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        sovits_model_free(model);
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        const size_t nbytes = ggml_nbytes(tensor);

        std::vector<uint8_t> buf(nbytes);
        if (fseek(f, static_cast<long>(offs), SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed for tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            sovits_model_free(model);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            sovits_model_free(model);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s' "
                    "(semantic_frame_rate_25hz=%d, is_v2pro=%d, gin_channels=%lld, "
                    "gen_initial=%lld)\n",
            __func__, (long long) n_tensors, fname.c_str(),
            (int) model.hparams.semantic_frame_rate_25hz,
            (int) model.hparams.is_v2pro,
            (long long) model.hparams.gin_channels,
            (long long) model.generator.conv_pre.b->ne[0]);
    return true;
}

void sovits_model_free(sovits_model & model) {
    if (model.buf_w) {
        ggml_backend_buffer_free(model.buf_w);
        model.buf_w = nullptr;
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
    }
    model.backend = nullptr;
}

} // namespace gpt_sovits
