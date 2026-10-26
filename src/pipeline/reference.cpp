#include "reference_internal.h"

#include "gpt_sovits/hubert.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gpt_sovits {
namespace pipeline_detail {
namespace {

constexpr int64_t kTailSamples = 9600;
constexpr int64_t kMinSamples = 400;

bool fail(std::string * error, const std::string & reason) {
    if (error) *error = reason;
    else std::fprintf(stderr, "reference: %s\n", reason.c_str());
    return false;
}

bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

bool size_is(size_t size, int64_t channels, int64_t frames) {
    return frames > 0 && frames <= INT64_MAX / channels &&
           static_cast<uint64_t>(frames) <= SIZE_MAX / static_cast<uint64_t>(channels) &&
           size == static_cast<size_t>(frames) * static_cast<size_t>(channels);
}

bool cpu(ggml_backend_t backend) {
    return backend && ggml_backend_is_cpu(backend);
}

} // namespace

bool build_reference_hubert_input(const float * pcm16k, int64_t n_samples,
                                  std::vector<float> & out, std::string * error) {
    if (!pcm16k || n_samples < kMinSamples) {
        return fail(error, "reference requires at least 400 unpadded 16 kHz samples");
    }
    if (n_samples > INT64_MAX - kTailSamples ||
        static_cast<uint64_t>(n_samples) > std::vector<float>().max_size() - kTailSamples) {
        return fail(error, "HuBERT input length overflow");
    }
    if (!std::all_of(pcm16k, pcm16k + n_samples, [](float v) { return std::isfinite(v); })) {
        return fail(error, "reference PCM contains non-finite samples");
    }
    try {
        std::vector<float> pending(pcm16k, pcm16k + n_samples);
        pending.resize(static_cast<size_t>(n_samples + kTailSamples), 0.0f);
        out = std::move(pending);
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, "HuBERT input allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "HuBERT input allocation size exceeded");
    }
}

bool validate_reference_features(const reference_features & f, std::string * error) {
    if (f.T_ref <= 0 || static_cast<uint64_t>(f.T_ref) != f.phones.size() ||
        !std::all_of(f.phones.begin(), f.phones.end(), [](int32_t id) { return id >= 0 && id < 732; })) {
        return fail(error, "reference phones must use the nonempty v2 vocabulary");
    }
    if (!size_is(f.bert.size(), 1024, f.T_ref) || !finite(f.bert)) {
        return fail(error, "reference BERT must be finite {1024, T_ref}");
    }
    if (f.T_hub < 2 || f.T_prompt != f.T_hub / 2 ||
        !size_is(f.hubert.size(), 768, f.T_hub) || !finite(f.hubert)) {
        return fail(error, "reference HuBERT requires finite {768, T_hub}, T_hub >= 2 and T_prompt = T_hub / 2");
    }
    if (f.audio.pcm16k.size() < kMinSamples ||
        f.audio.pcm16k.size() > static_cast<uint64_t>(INT64_MAX - kTailSamples) ||
        !finite(f.audio.pcm16k) ||
        hubert_output_frames(static_cast<int64_t>(f.audio.pcm16k.size()) + kTailSamples) != f.T_hub) {
        return fail(error, "reference audio and padded HuBERT frame count disagree");
    }
    if (!size_is(f.audio.spec704.size(), 704, f.audio.spec_frames) || !finite(f.audio.spec704)) {
        return fail(error, "reference spectrum must be finite {704, T_spec}");
    }
    if (error) error->clear();
    return true;
}

bool prepare_reference(const float * pcm, int64_t n_samples, int sample_rate,
                        const text_frontend_result & text, const hubert_model & hubert,
                        const roberta_model * roberta, reference_features & out,
                        std::string * error) {
    if (!cpu(hubert.backend) || !hubert.ctx_w || !hubert.buf_w ||
        (roberta && (!cpu(roberta->backend) || roberta->backend != hubert.backend))) {
        return fail(error, "reference models require the same loaded CPU backend");
    }
    try {
        reference_features pending;
        if (!audio_prepare_reference(pcm, n_samples, sample_rate, pending.audio)) {
            return fail(error, "reference audio preparation failed");
        }
        if (!text_frontend_build_bert_features(text, roberta, pending.bert, error)) return false;
        pending.phones = text.phoneme_ids;
        pending.T_ref = static_cast<int64_t>(pending.phones.size());
        std::vector<float> input;
        if (!build_reference_hubert_input(pending.audio.pcm16k.data(),
                static_cast<int64_t>(pending.audio.pcm16k.size()), input, error)) return false;
        pending.T_hub = hubert_output_frames(static_cast<int64_t>(input.size()));
        if (pending.T_hub < 2 || static_cast<uint64_t>(pending.T_hub) > pending.hubert.max_size() / 768 ||
            pending.T_hub > INT64_MAX / 768) {
            return fail(error, "invalid or overflowing HuBERT feature length");
        }
        pending.T_prompt = pending.T_hub / 2;
        pending.hubert.resize(static_cast<size_t>(pending.T_hub) * 768);
        int64_t frames = 0;
        if (!hubert_extract_features_from_pcm(hubert, input.data(), static_cast<int64_t>(input.size()),
                pending.hubert.data(), static_cast<int64_t>(pending.hubert.size()), &frames) ||
            frames != pending.T_hub) {
            return fail(error, "reference HuBERT extraction failed");
        }
        if (!validate_reference_features(pending, error)) return false;
        out = std::move(pending);
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, "reference feature allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "reference feature allocation size exceeded");
    }
}

bool encode_reference(const reference_features & f, const t2s_model & t2s,
                       const sovits_model & sovits, const sv_model & sv,
                       t2s_session & ts, sovits_session & ss, std::string * error) {
    if (!validate_reference_features(f, error)) return false;
    if (ts.ctx_ref || ts.buf_ref || ts.ref_text_emb || ts.ref_audio_emb || ts.ref_model ||
        ts.ref_T_ref || ts.ref_T_prompt || ts.ctx_graph || ts.decode_model ||
        ss.ctx_ge || ss.buf_ge || ss.ge || ss.ctx_ge_enc || ss.buf_ge_enc || ss.ge_enc) {
        return fail(error, "reference encoding requires fresh sessions without cached reference/decode state");
    }
    if (!cpu(t2s.backend) || sovits.backend != t2s.backend || sv.backend != t2s.backend ||
        ts.backend != t2s.backend || ss.backend != t2s.backend ||
        !t2s.ctx_w || !t2s.buf_w || !sovits.ctx_w || !sovits.buf_w ||
        !sv.ctx_w || !sv.buf_w || !sv.ctx_bn || !sv.buf_bn ||
        !ts.ctx_kv || !ts.buf_kv || ts.n_batch != 1 || ts.slots.size() != 1 || ts.slot_size == 0 ||
        ts.slots[0].n_pos != 0) {
        return fail(error, "reference encoding requires loaded models and idle initialized sessions on one CPU backend");
    }
    if (!sovits.hparams.is_v2pro || !sovits.hparams.semantic_frame_rate_25hz ||
        sovits.hparams.gin_channels != 1024 || sv.hparams.out_dim != 20480 ||
        t2s.hparams.phoneme_vocab_size != 732 || t2s.hparams.eos != 1024 ||
        !t2s.weights.extract_latent.ssl_proj_w || !t2s.weights.extract_latent.ssl_proj_b ||
        t2s.weights.extract_latent.ssl_proj_w->ne[0] != 2 ||
        t2s.weights.extract_latent.ssl_proj_w->ne[1] != 768 ||
        t2s.weights.extract_latent.ssl_proj_w->ne[2] != 768 ||
        ggml_nelements(t2s.weights.extract_latent.ssl_proj_b) != 768 ||
        !t2s.weights.extract_latent.codebook || !sovits.quantizer.codebook ||
        t2s.weights.extract_latent.codebook->ne[0] != 768 ||
        t2s.weights.extract_latent.codebook->ne[1] != 1024 ||
        sovits.quantizer.codebook->ne[0] != 768 || sovits.quantizer.codebook->ne[1] != 1024) {
        return fail(error, "reference encoding supports the v2Pro/25 Hz v2-vocabulary model configuration");
    }
    if (f.T_ref > ts.slot_size || f.T_prompt > ts.slot_size - f.T_ref) {
        return fail(error, "reference text and prompt exceed session slot_size");
    }
    try {
        if (!t2s_session_compute_ref_emb(ts, t2s, f.phones.data(), f.T_ref,
                                        f.bert.data(), f.hubert.data(), f.T_hub)) {
            return fail(error, "T2S reference embedding failed; dispose of both sessions");
        }
        if (!sovits_session_compute_ge_audio(ss, sovits, sv,
                {{f.audio.spec704.data(), f.audio.spec_frames, f.audio.pcm16k.data(),
                  static_cast<int64_t>(f.audio.pcm16k.size())}})) {
            return fail(error, "SoVITS reference conditioning failed; dispose of both sessions");
        }
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, "reference encoding allocation failed; dispose of both sessions");
    } catch (const std::length_error &) {
        return fail(error, "reference encoding allocation size exceeded; dispose of both sessions");
    }
}

} // namespace pipeline_detail
} // namespace gpt_sovits
