#include "gpt_sovits/pipeline.h"

#include "synthesis_internal.h"

#include "gpt_sovits/hubert.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpt_sovits {
namespace pipeline_detail {
namespace {

constexpr int64_t kCodeDim = 768;
constexpr int64_t kCodeCount = 1024;
constexpr int64_t kSlotLimit = 1024;

bool fail(std::string * error, const char * reason) noexcept {
    if (error) {
        try {
            *error = reason;
            return false;
        } catch (const std::bad_alloc &) {
        } catch (const std::length_error &) {
        }
    }
    std::fprintf(stderr, "synthesis: %s\n", reason);
    return false;
}

bool pass(std::string * error) {
    if (error) error->clear();
    return true;
}

bool cpu(ggml_backend_t backend) {
    return backend && ggml_backend_is_cpu(backend);
}

template <typename Model>
bool loaded_on(const Model & model, ggml_backend_t backend) {
    return cpu(model.backend) && model.backend == backend &&
           model.ctx_w && model.buf_w;
}

bool checked_add(int64_t & total, int64_t value) {
    if (value > std::numeric_limits<int64_t>::max() - total) return false;
    total += value;
    return true;
}

} // namespace

bool validate_synthesis_config(const synthesis_config & config, std::string * error) {
    try {
        const t2s_sampler_config & sampler = config.sampler;
        if (!std::isfinite(sampler.top_p) || !std::isfinite(sampler.temperature)) {
            return fail(error, "synthesis top_p and temperature must be finite");
        }
        if (!std::isfinite(sampler.repetition_penalty) || sampler.repetition_penalty <= 0.0f) {
            return fail(error, "synthesis repetition_penalty must be positive and finite");
        }
        if (config.generate.max_new_tokens <= 0 || config.generate.max_new_tokens > kSlotLimit) {
            return fail(error, "synthesis max_new_tokens must be in [1, 1024]");
        }
        if (!std::isfinite(config.noise_scale) || config.noise_scale <= 0.0f) {
            return fail(error, "synthesis noise_scale must be positive and finite");
        }
        return pass(error);
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis config validation allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis config validation size exceeded");
    }
}

bool validate_codebooks(const t2s_model & t2s, const sovits_model & sovits,
                        std::string * error) {
    try {
        const ggml_tensor * t2s_book = t2s.weights.extract_latent.codebook;
        const ggml_tensor * sovits_book = sovits.quantizer.codebook;
        if (!t2s_book || !sovits_book) {
            return fail(error, "synthesis requires T2S and SoVITS codebooks");
        }
        if (t2s_book->type != GGML_TYPE_F32 || sovits_book->type != GGML_TYPE_F16) {
            return fail(error, "synthesis requires an F32 T2S codebook and an F16 SoVITS codebook");
        }
        for (const ggml_tensor * book : {t2s_book, sovits_book}) {
            if (book->ne[0] != kCodeDim || book->ne[1] != kCodeCount ||
                ggml_nelements(book) != kCodeDim * kCodeCount ||
                !ggml_is_contiguous(book)) {
                return fail(error, "synthesis codebooks must be contiguous {768, 1024}");
            }
        }

        const size_t n = static_cast<size_t>(kCodeDim * kCodeCount);
        std::vector<float> t2s_values(n);
        std::vector<ggml_fp16_t> sovits_values(n);
        ggml_backend_tensor_get(t2s_book, t2s_values.data(), 0, n * sizeof(float));
        ggml_backend_tensor_get(sovits_book, sovits_values.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) {
            const float value = t2s_values[i];
            if (!std::isfinite(value)) {
                return fail(error, "T2S codebook contains non-finite values");
            }
            if (!std::isfinite(ggml_fp16_to_fp32(sovits_values[i]))) {
                return fail(error, "SoVITS codebook contains non-finite values");
            }
            if (ggml_fp32_to_fp16(value) != sovits_values[i]) {
                char message[128];
                std::snprintf(message, sizeof(message),
                              "codebooks differ after F32-to-F16 rounding at element %zu", i);
                return fail(error, message);
            }
        }
        return pass(error);
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis codebook validation allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis codebook validation size exceeded");
    }
}

bool validate_synthesis_models(const synthesis_models & models, std::string * error) {
    try {
        if (!models.hubert || !models.t2s || !models.sovits || !models.sv) {
            return fail(error, "synthesis requires HuBERT, T2S, SoVITS and SV models");
        }
        const ggml_backend_t backend = models.t2s->backend;
        if (!loaded_on(*models.hubert, backend) ||
            !loaded_on(*models.t2s, backend) ||
            !loaded_on(*models.sovits, backend) ||
            !loaded_on(*models.sv, backend) ||
            !models.sv->ctx_bn || !models.sv->buf_bn ||
            (models.roberta &&
             !loaded_on(*models.roberta, backend))) {
            return fail(error, "synthesis models must be fully loaded on one CPU backend");
        }

        const sovits_hparams & sovits_hp = models.sovits->hparams;
        if (!sovits_hp.is_v2pro || !sovits_hp.semantic_frame_rate_25hz ||
            sovits_hp.gin_channels != 1024 ||
            !models.sovits->generator.conv_pre.w ||
            models.sovits->generator.conv_pre.w->ne[1] != 512) {
            return fail(error, "synthesis supports only v2Pro 25 Hz with gin_channels=1024 "
                               "and conv_pre input 512");
        }

        const t2s_hparams & t2s_hp = models.t2s->hparams;
        if (t2s_hp.phoneme_vocab_size != 732 || t2s_hp.eos != 1024 ||
            t2s_hp.vocab_size != 1025) {
            return fail(error, "synthesis requires the v2 vocabulary 732, EOS 1024 "
                               "and semantic vocabulary 1025");
        }

        const ggml_tensor * t2s_text = models.t2s->weights.embed.text_embedding;
        const ggml_tensor * sovits_text =
            models.sovits->text_encoder.text.text_embedding;
        if (!t2s_text || !sovits_text ||
            t2s_text->ne[1] != 732 || sovits_text->ne[1] != t2s_text->ne[1]) {
            return fail(error, "T2S and SoVITS text embeddings must share the 732-entry "
                               "v2 vocabulary");
        }

        const ggml_tensor * ssl_w = models.t2s->weights.extract_latent.ssl_proj_w;
        const ggml_tensor * ssl_b = models.t2s->weights.extract_latent.ssl_proj_b;
        if (!ssl_w || !ssl_b ||
            ssl_w->ne[0] != 2 || ssl_w->ne[1] != 768 || ssl_w->ne[2] != 768 ||
            ggml_nelements(ssl_b) != 768) {
            return fail(error, "T2S ssl_proj must be Conv1d{2, 768, 768} with a "
                               "768-element bias");
        }

        return validate_codebooks(*models.t2s, *models.sovits, error);
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis model validation allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis model validation size exceeded");
    }
}

bool synthesis_slot_size(int64_t T_ref, int64_t T_target, int64_t T_prompt,
                         int64_t max_new_tokens, uint32_t & out,
                         std::string * error) {
    try {
        if (T_ref <= 0 || T_target <= 0 || T_prompt <= 0 || max_new_tokens <= 0) {
            return fail(error, "synthesis slot budget requires positive "
                               "T_ref, T_target, T_prompt and max_new_tokens");
        }
        int64_t required = T_ref;
        if (!checked_add(required, T_target) ||
            !checked_add(required, T_prompt) ||
            !checked_add(required, max_new_tokens)) {
            return fail(error, "synthesis slot budget overflow");
        }
        --required;
        if (required > kSlotLimit) {
            return fail(error, "synthesis slot budget requires "
                               "T_ref + T_target + T_prompt + max_new_tokens - 1 <= 1024");
        }
        out = static_cast<uint32_t>(required);
        return pass(error);
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis slot budget validation allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis slot budget validation size exceeded");
    }
}

} // namespace pipeline_detail
} // namespace gpt_sovits
