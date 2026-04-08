#include "gpt_sovits/hubert.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Helper: look up a tensor by name, abort on failure.
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

static bool populate_weights(struct ggml_context * ctx, hubert_model_block_weights & w) {
    // ---- Feature Encoder (7 conv layers) --------------------------------
    {
        auto & fe = w.feature_encoder;
        char name[128];
        for (int i = 0; i < 7; i++) {
            snprintf(name, sizeof(name), "feature_extractor.conv_layers.%d.conv.weight", i);
            fe.conv_w[i] = checked_get_tensor(ctx, name);
            if (!fe.conv_w[i]) return false;
        }
        fe.conv0_norm_w = checked_get_tensor(ctx, "feature_extractor.conv_layers.0.layer_norm.weight");
        fe.conv0_norm_b = checked_get_tensor(ctx, "feature_extractor.conv_layers.0.layer_norm.bias");
        if (!fe.conv0_norm_w || !fe.conv0_norm_b) return false;
    }

    // ---- Feature Projection ---------------------------------------------
    {
        auto & fp = w.feature_projection;
        fp.layer_norm_w = checked_get_tensor(ctx, "feature_projection.layer_norm.weight");
        fp.layer_norm_b = checked_get_tensor(ctx, "feature_projection.layer_norm.bias");
        fp.projection_w = checked_get_tensor(ctx, "feature_projection.projection.weight");
        fp.projection_b = checked_get_tensor(ctx, "feature_projection.projection.bias");
        if (!fp.layer_norm_w || !fp.layer_norm_b || !fp.projection_w || !fp.projection_b) return false;
    }

    // ---- Encoder: Positional Conv ---------------------------------------
    {
        auto & pc = w.encoder.pos_conv;
        pc.weight_v = checked_get_tensor(ctx, "encoder.pos_conv_embed.conv.weight_v");
        pc.weight_g = checked_get_tensor(ctx, "encoder.pos_conv_embed.conv.weight_g");
        pc.bias     = checked_get_tensor(ctx, "encoder.pos_conv_embed.conv.bias");
        if (!pc.weight_v || !pc.weight_g || !pc.bias) return false;
    }

    // ---- Encoder: LayerNorm ---------------------------------------------
    {
        w.encoder.layer_norm_w = checked_get_tensor(ctx, "encoder.layer_norm.weight");
        w.encoder.layer_norm_b = checked_get_tensor(ctx, "encoder.layer_norm.bias");
        if (!w.encoder.layer_norm_w || !w.encoder.layer_norm_b) return false;
    }

    // ---- Encoder: 12 Transformer Layers ---------------------------------
    {
        char name[128];
        for (int i = 0; i < 12; i++) {
            auto & layer = w.encoder.layers[i];

            // Attention
            auto & attn = layer.attention;
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.q_proj.weight", i);
            attn.q_proj_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.q_proj.bias", i);
            attn.q_proj_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.k_proj.weight", i);
            attn.k_proj_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.k_proj.bias", i);
            attn.k_proj_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.v_proj.weight", i);
            attn.v_proj_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.v_proj.bias", i);
            attn.v_proj_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.out_proj.weight", i);
            attn.out_proj_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.attention.out_proj.bias", i);
            attn.out_proj_b = checked_get_tensor(ctx, name);

            if (!attn.q_proj_w || !attn.q_proj_b || !attn.k_proj_w || !attn.k_proj_b ||
                !attn.v_proj_w || !attn.v_proj_b || !attn.out_proj_w || !attn.out_proj_b) return false;

            // Post-attention LayerNorm
            snprintf(name, sizeof(name), "encoder.layers.%d.layer_norm.weight", i);
            layer.ln1_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.layer_norm.bias", i);
            layer.ln1_b = checked_get_tensor(ctx, name);
            if (!layer.ln1_w || !layer.ln1_b) return false;

            // FFN
            snprintf(name, sizeof(name), "encoder.layers.%d.feed_forward.intermediate_dense.weight", i);
            layer.ffn_up_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.feed_forward.intermediate_dense.bias", i);
            layer.ffn_up_b = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.feed_forward.output_dense.weight", i);
            layer.ffn_down_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.feed_forward.output_dense.bias", i);
            layer.ffn_down_b = checked_get_tensor(ctx, name);
            if (!layer.ffn_up_w || !layer.ffn_up_b || !layer.ffn_down_w || !layer.ffn_down_b) return false;

            // Final LayerNorm
            snprintf(name, sizeof(name), "encoder.layers.%d.final_layer_norm.weight", i);
            layer.ln2_w = checked_get_tensor(ctx, name);
            snprintf(name, sizeof(name), "encoder.layers.%d.final_layer_norm.bias", i);
            layer.ln2_b = checked_get_tensor(ctx, name);
            if (!layer.ln2_w || !layer.ln2_b) return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool hubert_model_load(const std::string & fname, hubert_model & model) {
    // 1. Initialize a CPU backend.
    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // 2. Open the GGUF file -- parse metadata and create tensor descriptors
    //    in a ggml_context (no data allocated yet).
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx_w,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: gguf_init_from_file('%s') failed\n", __func__, fname.c_str());
        return false;
    }

    // 3. Allocate backend buffers for all tensors.
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        gguf_free(ctx_gguf);
        return false;
    }

    // 4. Populate the weight struct (look up tensors by name).
    if (!populate_weights(model.ctx_w, model.weights)) {
        fprintf(stderr, "%s: populate_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        return false;
    }

    // 5. Load tensor data from disk into backend buffers.
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            fprintf(stderr, "%s: warning: tensor '%s' in GGUF but not found in context\n", __func__, name);
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        const size_t nbytes = ggml_nbytes(tensor);

        std::vector<uint8_t> buf(nbytes);
        if (fseek(f, (long)offs, SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed for tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s'\n", __func__, (long long)n_tensors, fname.c_str());
    return true;
}

void hubert_model_free(hubert_model & model) {
    if (model.buf_w) {
        ggml_backend_buffer_free(model.buf_w);
        model.buf_w = nullptr;
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
}

} // namespace gpt_sovits
