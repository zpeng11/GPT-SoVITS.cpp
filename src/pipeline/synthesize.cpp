#include "gpt_sovits/pipeline.h"
#include "gpt_sovits/hubert.h"
#include "gpt_sovits/sovits.h"
#include "reference_internal.h"
#include "synthesis_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gpt_sovits {
namespace {

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

struct request_sessions {
    t2s_session t2s{};
    sovits_session sovits{};
    request_sessions() = default;
    request_sessions(const request_sessions &) = delete;
    request_sessions & operator=(const request_sessions &) = delete;
    ~request_sessions() {
        sovits_session_free(sovits);
        t2s_session_free(t2s);
    }
};

bool request_budget(int64_t n_samples, int sample_rate, size_t ref_phones,
                    size_t target_phones, int32_t max_new_tokens,
                    uint32_t & slot_size, std::string * error) {
    if (n_samples <= 0 || sample_rate <= 0 || ref_phones > 1024 || target_phones > 1024) {
        return fail(error, "invalid audio length/sample rate or text exceeds 1024-position request limit");
    }
    const int64_t divisor = std::gcd(sample_rate, 16000);
    const int64_t up = 16000 / divisor;
    const int64_t down = sample_rate / divisor;
    if (n_samples > INT64_MAX / up) return fail(error, "reference resampled length overflow");
    const int64_t numerator = n_samples * up;
    const int64_t n16k = numerator / down + (numerator % down != 0);
    if (n16k < 400 || n16k > INT64_MAX - 9600) {
        return fail(error, "invalid reference length for padded HuBERT input");
    }
    const int64_t prompt = hubert_output_frames(n16k + 9600) / 2;
    return pipeline_detail::synthesis_slot_size(static_cast<int64_t>(ref_phones),
        static_cast<int64_t>(target_phones), prompt, max_new_tokens, slot_size, error);
}

} // namespace

bool pipeline_detail::generate_and_decode(
    t2s_session & ts, const t2s_model & t2s, sovits_session & ss,
    const sovits_model & sovits, const std::vector<int32_t> & phones,
    const std::vector<float> & bert, const t2s_generate_config & config,
    synthesis_result & result, std::string * error) {
    if (phones.empty() || phones.size() > 1024 || bert.size() != phones.size() * 1024) {
        return fail(error, "target phones/BERT dimensions are invalid");
    }
    try {
        t2s_generate_result generated;
        if (!t2s_session_generate(ts, t2s, phones.data(), static_cast<int64_t>(phones.size()),
                                   bert.data(), generated, config)) {
            return fail(error, "T2S semantic generation failed");
        }
        if (generated.codes.empty()) {
            return fail(error, "T2S returned immediate EOS with no semantic codes; SoVITS was not called");
        }
        if (!sovits.hparams.semantic_frame_rate_25hz ||
            generated.codes.size() > static_cast<uint64_t>(INT64_MAX / 1280) ||
            generated.codes.size() > std::vector<float>().max_size() / 1280) {
            return fail(error, "unsupported or overflowing waveform length");
        }
        synthesis_result pending;
        pending.samples.resize(generated.codes.size() * 1280);
        if (!sovits_session_forward(ss, sovits, generated.codes.data(),
                static_cast<int64_t>(generated.codes.size()), phones.data(),
                static_cast<int64_t>(phones.size()), pending.samples.data(),
                static_cast<int64_t>(pending.samples.size()))) {
            return fail(error, "SoVITS waveform decoding failed");
        }
        if (!std::all_of(pending.samples.begin(), pending.samples.end(),
                         [](float v) { return std::isfinite(v); })) {
            return fail(error, "SoVITS produced non-finite waveform samples");
        }
        pending.codes = std::move(generated.codes);
        pending.stop_reason = generated.stop_reason;
        result = std::move(pending);
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis allocation size exceeded");
    }
}

bool synthesize_cpu(const synthesis_models & models, const float * pcm,
                    int64_t n_samples, int sample_rate,
                    const text_frontend_result & reference_text,
                    const text_frontend_result & target_text,
                    synthesis_result & result, const synthesis_config & config,
                    std::string * error) {
    try {
        if (!pipeline_detail::validate_synthesis_config(config, error)) return false;
        if (!pcm) return fail(error, "reference PCM must be non-null");
        uint32_t slot_size = 0;
        if (!request_budget(n_samples, sample_rate, reference_text.phoneme_ids.size(),
                            target_text.phoneme_ids.size(), config.generate.max_new_tokens,
                            slot_size, error)) return false;
        if (!pipeline_detail::validate_synthesis_models(models, error)) return false;

        std::vector<float> target_bert;
        if (!text_frontend_build_bert_features(target_text, models.roberta, target_bert, error)) return false;
        pipeline_detail::reference_features reference;
        if (!pipeline_detail::prepare_reference(pcm, n_samples, sample_rate, reference_text,
                *models.hubert, models.roberta, reference, error)) return false;
        uint32_t actual_slot_size = 0;
        if (!pipeline_detail::synthesis_slot_size(reference.T_ref,
                static_cast<int64_t>(target_text.phoneme_ids.size()), reference.T_prompt,
                config.generate.max_new_tokens, actual_slot_size, error)) return false;
        if (actual_slot_size != slot_size) return fail(error, "reference prompt length disagrees with request budget");

        request_sessions sessions;
        if (!t2s_session_init(sessions.t2s, models.t2s->hparams, models.t2s->backend,
                              1, slot_size, config.sampler, GGML_TYPE_F32)) {
            return fail(error, "T2S session initialization failed");
        }
        if (!sovits_session_init(sessions.sovits, models.sovits->backend,
                                  config.noise_scale, config.noise_seed)) {
            return fail(error, "SoVITS session initialization failed");
        }
        if (!pipeline_detail::encode_reference(reference, *models.t2s, *models.sovits,
                *models.sv, sessions.t2s, sessions.sovits, error)) return false;
        return pipeline_detail::generate_and_decode(sessions.t2s, *models.t2s,
            sessions.sovits, *models.sovits, target_text.phoneme_ids, target_bert,
            config.generate, result, error);
    } catch (const std::bad_alloc &) {
        return fail(error, "synthesis allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "synthesis allocation size exceeded");
    }
}

} // namespace gpt_sovits
