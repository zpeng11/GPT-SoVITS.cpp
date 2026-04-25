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

} // namespace

bool sovits_ref_enc_model_load(
    const std::string & fname,
    sovits_ref_enc_model & model,
    ggml_backend_t backend)
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
        sovits_ref_enc_model_free(model);
        return false;
    }

    if (!populate_weights(model.ctx_w, model.weights)) {
        fprintf(stderr, "%s: populate_weights() failed -- missing tensors\n", __func__);
        gguf_free(ctx_gguf);
        sovits_ref_enc_model_free(model);
        return false;
    }

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen('%s') failed\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        sovits_ref_enc_model_free(model);
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
            sovits_ref_enc_model_free(model);
            return false;
        }
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread() failed for tensor '%s' (%zu bytes)\n", __func__, name, nbytes);
            fclose(f);
            gguf_free(ctx_gguf);
            sovits_ref_enc_model_free(model);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    fprintf(stderr, "%s: loaded %lld tensors from '%s'\n", __func__, (long long) n_tensors, fname.c_str());
    return true;
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

} // namespace gpt_sovits
