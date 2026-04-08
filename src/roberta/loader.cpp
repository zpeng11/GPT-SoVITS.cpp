#include "gpt_sovits/roberta.h"

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
// Helper: look up a tensor by name, print error on failure.
// ---------------------------------------------------------------------------

static struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in GGUF\n", __func__, name);
    }
    return t;
}

// ---------------------------------------------------------------------------
// Populate the weight struct from the ggml_context that holds the tensor
// metadata created by gguf_init_from_file.
//
// Tensor names in GGUF match the HuggingFace state_dict key names.
// ---------------------------------------------------------------------------

static bool populate_weights(struct ggml_context * ctx, roberta_model_block_weights & w) {
    // ---- Embeddings -----------------------------------------------------
    {
        auto & emb = w.embeddings;
        emb.word_embeddings       = checked_get_tensor(ctx, "bert.embeddings.word_embeddings.weight");
        emb.position_embeddings   = checked_get_tensor(ctx, "bert.embeddings.position_embeddings.weight");
        emb.token_type_embeddings = checked_get_tensor(ctx, "bert.embeddings.token_type_embeddings.weight");
        emb.layer_norm_w          = checked_get_tensor(ctx, "bert.embeddings.LayerNorm.weight");
        emb.layer_norm_b          = checked_get_tensor(ctx, "bert.embeddings.LayerNorm.bias");
        if (!emb.word_embeddings || !emb.position_embeddings || !emb.token_type_embeddings ||
            !emb.layer_norm_w || !emb.layer_norm_b) {
            return false;
        }
    }

    // ---- Encoder: 24 Transformer Layers ---------------------------------
    {
        char name[128];
        for (int i = 0; i < 24; i++) {
            auto & layer = w.encoder.layers[i];

            // Attention Q/K/V + output projection
            auto & attn = layer.attention;
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.query.weight", i);
            attn.q_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.query.bias", i);
            attn.q_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.key.weight", i);
            attn.k_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.key.bias", i);
            attn.k_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.value.weight", i);
            attn.v_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.self.value.bias", i);
            attn.v_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.output.dense.weight", i);
            attn.out_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.output.dense.bias", i);
            attn.out_b = checked_get_tensor(ctx, name);

            if (!attn.q_w || !attn.q_b || !attn.k_w || !attn.k_b ||
                !attn.v_w || !attn.v_b || !attn.out_w || !attn.out_b) {
                return false;
            }

            // Post-attention LayerNorm
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.output.LayerNorm.weight", i);
            layer.attn_ln_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.attention.output.LayerNorm.bias", i);
            layer.attn_ln_b = checked_get_tensor(ctx, name);
            if (!layer.attn_ln_w || !layer.attn_ln_b) return false;

            // FFN: intermediate (up) + output (down)
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.intermediate.dense.weight", i);
            layer.ffn_up_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.intermediate.dense.bias", i);
            layer.ffn_up_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.output.dense.weight", i);
            layer.ffn_down_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.output.dense.bias", i);
            layer.ffn_down_b = checked_get_tensor(ctx, name);
            if (!layer.ffn_up_w || !layer.ffn_up_b || !layer.ffn_down_w || !layer.ffn_down_b) return false;

            // Post-FFN LayerNorm
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.output.LayerNorm.weight", i);
            layer.ffn_ln_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "bert.encoder.layer.%d.output.LayerNorm.bias", i);
            layer.ffn_ln_b = checked_get_tensor(ctx, name);
            if (!layer.ffn_ln_w || !layer.ffn_ln_b) return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool roberta_model_load(const std::string & fname, roberta_model & model, ggml_backend_t backend) {
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
        roberta_model_free(model);
        return false;
    }

    if (!populate_weights(model.ctx_w, model.weights)) {
        fprintf(stderr, "%s: populate_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        roberta_model_free(model);
        return false;
    }

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        roberta_model_free(model);
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
            roberta_model_free(model);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            roberta_model_free(model);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s'\n", __func__, (long long)n_tensors, fname.c_str());
    return true;
}

void roberta_model_free(roberta_model & model) {
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
