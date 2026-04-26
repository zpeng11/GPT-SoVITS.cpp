#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
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

    w.attention.q_w = checked_get_tensor(ctx, "ref_enc.attention.q_w");
    w.attention.q_b = checked_get_tensor(ctx, "ref_enc.attention.q_b");
    w.attention.k_w = checked_get_tensor(ctx, "ref_enc.attention.k_w");
    w.attention.k_b = checked_get_tensor(ctx, "ref_enc.attention.k_b");
    w.attention.v_w = checked_get_tensor(ctx, "ref_enc.attention.v_w");
    w.attention.v_b = checked_get_tensor(ctx, "ref_enc.attention.v_b");
    w.attention.out_w = checked_get_tensor(ctx, "ref_enc.attention.out_w");
    w.attention.out_b = checked_get_tensor(ctx, "ref_enc.attention.out_b");
    if (!w.attention.q_w || !w.attention.q_b || !w.attention.k_w || !w.attention.k_b ||
        !w.attention.v_w || !w.attention.v_b || !w.attention.out_w || !w.attention.out_b) {
        return false;
    }

    w.fc_w = checked_get_tensor(ctx, "ref_enc.fc_w");
    w.fc_b = checked_get_tensor(ctx, "ref_enc.fc_b");
    if (!w.fc_w || !w.fc_b) {
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

    return true;
}

template <typename ModelT, typename PopulateFn>
static bool load_model_from_gguf(
    const std::string & fname,
    ModelT & model,
    ggml_backend_t backend,
    PopulateFn populate,
    void (*free_model)(ModelT &))
{
    GGML_ASSERT(backend != nullptr);
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

    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        gguf_free(ctx_gguf);
        free_model(model);
        return false;
    }

    if (!populate(model.ctx_w, model.weights)) {
        fprintf(stderr, "%s: populate_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        free_model(model);
        return false;
    }

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        free_model(model);
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
            free_model(model);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            free_model(model);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s'\n", __func__, (long long) n_tensors, fname.c_str());
    return true;
}

} // namespace

bool sovits_ref_enc_model_load(
    const std::string & fname,
    sovits_ref_enc_model & model,
    ggml_backend_t backend)
{
    return load_model_from_gguf(fname, model, backend, populate_weights, sovits_ref_enc_model_free);
}

bool sovits_quantizer_model_load(
    const std::string & fname,
    sovits_quantizer_model & model,
    ggml_backend_t backend)
{
    return load_model_from_gguf(fname, model, backend, populate_quantizer_weights, sovits_quantizer_model_free);
}

bool sovits_text_encoder_ssl_model_load(
    const std::string & fname,
    sovits_text_encoder_ssl_model & model,
    ggml_backend_t backend)
{
    return load_model_from_gguf(
        fname,
        model,
        backend,
        populate_text_encoder_ssl_weights,
        sovits_text_encoder_ssl_model_free);
}

bool sovits_text_encoder_text_model_load(
    const std::string & fname,
    sovits_text_encoder_text_model & model,
    ggml_backend_t backend)
{
    return load_model_from_gguf(
        fname,
        model,
        backend,
        populate_text_encoder_text_weights,
        sovits_text_encoder_text_model_free);
}

bool sovits_text_encoder_post_model_load(
    const std::string & fname,
    sovits_text_encoder_post_model & model,
    ggml_backend_t backend)
{
    return load_model_from_gguf(
        fname,
        model,
        backend,
        populate_text_encoder_post_weights,
        sovits_text_encoder_post_model_free);
}

void sovits_ref_enc_model_free(sovits_ref_enc_model & model) {
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

void sovits_quantizer_model_free(sovits_quantizer_model & model) {
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

void sovits_text_encoder_ssl_model_free(sovits_text_encoder_ssl_model & model) {
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

void sovits_text_encoder_text_model_free(sovits_text_encoder_text_model & model) {
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

void sovits_text_encoder_post_model_free(sovits_text_encoder_post_model & model) {
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
