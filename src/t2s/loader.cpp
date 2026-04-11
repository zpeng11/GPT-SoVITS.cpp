#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in GGUF\n", __func__, name);
    }
    return t;
}

static uint32_t get_u32_or(struct gguf_context * ctx_gguf, const char * key, uint32_t default_val) {
    int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0) {
        return default_val;
    }
    return gguf_get_val_u32(ctx_gguf, id);
}

// ---------------------------------------------------------------------------
// Read hparams from GGUF KV metadata
// ---------------------------------------------------------------------------

static bool read_hparams(struct gguf_context * ctx_gguf, t2s_hparams & hparams) {
    hparams.embedding_dim      = get_u32_or(ctx_gguf, "t2s.embedding_dim",      512);
    hparams.hidden_dim         = get_u32_or(ctx_gguf, "t2s.hidden_dim",         512);
    hparams.n_head             = get_u32_or(ctx_gguf, "t2s.head",               16);
    hparams.linear_units       = get_u32_or(ctx_gguf, "t2s.linear_units",       2048);
    hparams.n_layer            = get_u32_or(ctx_gguf, "t2s.n_layer",            24);
    hparams.vocab_size         = get_u32_or(ctx_gguf, "t2s.vocab_size",         1025);
    hparams.phoneme_vocab_size = get_u32_or(ctx_gguf, "t2s.phoneme_vocab_size", 732);
    hparams.eos                = get_u32_or(ctx_gguf, "t2s.eos",                1024);
    hparams.inter_channels     = get_u32_or(ctx_gguf, "sovits.inter_channels",  192);
    return true;
}

// ---------------------------------------------------------------------------
// Populate weight structs from the ggml_context
// ---------------------------------------------------------------------------

static bool populate_weights(struct ggml_context * ctx,
                             int n_layer,
                             t2s_model_weights & w) {
    // ---- SoVITS extract-latent -----------------------------------------
    {
        auto & el = w.extract_latent;
        el.ssl_proj_w = checked_get_tensor(ctx, "extract_latent.ssl_proj_w");
        el.ssl_proj_b = checked_get_tensor(ctx, "extract_latent.ssl_proj_b");
        el.codebook   = checked_get_tensor(ctx, "extract_latent.codebook");
        if (!el.ssl_proj_w || !el.ssl_proj_b || !el.codebook) return false;
    }

    // ---- T2S embedding ------------------------------------------------
    {
        auto & enc = w.embed;
        enc.text_embedding = checked_get_tensor(ctx, "encoder.text_embedding");
        enc.bert_proj_w    = checked_get_tensor(ctx, "encoder.bert_proj_w");
        enc.bert_proj_b    = checked_get_tensor(ctx, "encoder.bert_proj_b");
        enc.text_pos_alpha = checked_get_tensor(ctx, "encoder.text_pos_alpha");
        enc.audio_embedding = checked_get_tensor(ctx, "encoder.audio_embedding");
        enc.audio_pos_alpha = checked_get_tensor(ctx, "encoder.audio_pos_alpha");
        if (!enc.text_embedding || !enc.bert_proj_w || !enc.bert_proj_b ||
            !enc.text_pos_alpha || !enc.audio_embedding || !enc.audio_pos_alpha) {
            return false;
        }
    }

    // ---- T2S attention layers -----------------------------------------
    {
        w.attention.resize(n_layer);
        char name[128];
        for (int i = 0; i < n_layer; i++) {
            auto & layer = w.attention[i];

            snprintf(name, sizeof(name), "attention.%d.qkv_w", i);
            layer.qkv_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.qkv_b", i);
            layer.qkv_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.out_proj_w", i);
            layer.out_proj_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.out_proj_b", i);
            layer.out_proj_b = checked_get_tensor(ctx, name);

            snprintf(name, sizeof(name), "attention.%d.ln1_w", i);
            layer.ln1_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.ln1_b", i);
            layer.ln1_b = checked_get_tensor(ctx, name);

            snprintf(name, sizeof(name), "attention.%d.ffn_up_w", i);
            layer.ffn_up_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.ffn_up_b", i);
            layer.ffn_up_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.ffn_down_w", i);
            layer.ffn_down_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.ffn_down_b", i);
            layer.ffn_down_b = checked_get_tensor(ctx, name);

            snprintf(name, sizeof(name), "attention.%d.ln2_w", i);
            layer.ln2_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "attention.%d.ln2_b", i);
            layer.ln2_b = checked_get_tensor(ctx, name);

            if (!layer.qkv_w || !layer.qkv_b || !layer.out_proj_w || !layer.out_proj_b ||
                !layer.ln1_w || !layer.ln1_b ||
                !layer.ffn_up_w || !layer.ffn_up_b || !layer.ffn_down_w || !layer.ffn_down_b ||
                !layer.ln2_w || !layer.ln2_b) {
                return false;
            }
        }
    }

    // ---- T2S sampler --------------------------------------------------
    {
        w.lm_head_w = checked_get_tensor(ctx, "sampler.lm_head_w");
        if (!w.lm_head_w) return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool t2s_model_load(const std::string & fname, t2s_model & model, ggml_backend_t backend) {
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

    // Read hyperparameters from GGUF metadata.
    if (!read_hparams(ctx_gguf, model.hparams)) {
        fprintf(stderr, "%s: read_hparams() failed\n", __func__);
        gguf_free(ctx_gguf);
        t2s_model_free(model);
        return false;
    }

    fprintf(stderr, "%s: hparams: d_model=%u, n_head=%u, n_layer=%u, vocab=%u, eos=%u\n",
            __func__, model.hparams.hidden_dim, model.hparams.n_head,
            model.hparams.n_layer, model.hparams.vocab_size, model.hparams.eos);

    // Allocate backend buffers for all tensors.
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        gguf_free(ctx_gguf);
        t2s_model_free(model);
        return false;
    }

    // Populate the weight struct (look up tensors by name).
    if (!populate_weights(model.ctx_w, (int)model.hparams.n_layer, model.weights)) {
        fprintf(stderr, "%s: populate_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        t2s_model_free(model);
        return false;
    }

    // Load tensor data from disk into backend buffers.
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        t2s_model_free(model);
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            continue;
        }

        const size_t offs   = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        const size_t nbytes = ggml_nbytes(tensor);

        std::vector<uint8_t> buf(nbytes);
        if (fseek(f, static_cast<long>(offs), SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed for tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            t2s_model_free(model);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            t2s_model_free(model);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s'\n", __func__, (long long)n_tensors, fname.c_str());
    return true;
}

void t2s_model_free(t2s_model & model) {
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
