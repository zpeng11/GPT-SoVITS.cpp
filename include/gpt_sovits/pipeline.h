#pragma once

#include "gpt_sovits/t2s.h"
#include "gpt_sovits/text_frontend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

struct hubert_model;
struct roberta_model;
struct sovits_model;
struct sv_model;

struct synthesis_models {
    const hubert_model * hubert = nullptr;
    const roberta_model * roberta = nullptr;
    const t2s_model * t2s = nullptr;
    const sovits_model * sovits = nullptr;
    const sv_model * sv = nullptr;
};

struct synthesis_config {
    t2s_sampler_config sampler;
    t2s_generate_config generate;
    float noise_scale = 0.5f;
    uint64_t noise_seed = 7;
};

struct synthesis_result {
    std::vector<float> samples;
    int sample_rate = 32000;
    std::vector<int32_t> codes;
    t2s_stop_reason stop_reason = t2s_stop_reason::max_new_tokens;
};

// Synchronous, single-reference CPU v2Pro / 25 Hz synthesis with preloaded models.
// All models must share one CPU backend. RoBERTa may be null only when neither
// frontend result needs Chinese BERT inference. Models/backend are borrowed and
// must not be freed, moved, reloaded or used concurrently during this call.
// Per-request sessions are created and freed internally; no reference is retained.
//
// pcm is decoded mono audio (n_samples readable F32 values) at sample_rate.
// Reference conditioning uses unnormalized 16 kHz audio + 9600 trailing zeros
// for HuBERT. Existing C++ resampling is used, not upstream resampler parity.
// Texts must use the v2 frontend contract. Neither text is truncated or split.
// N = reference phones + target phones + reference prompt codes;
// N + max_new_tokens - 1 must fit the initial 1024-position request limit.
//
// Output is raw 32 kHz mono F32, with codes.size() * 1280 samples. There is no
// normalization, clipping, appended silence, int16 conversion or file IO. Codes
// exclude the reference prompt and EOS. Length-limit termination retains all
// accepted codes and reports max_new_tokens. Immediate EOS is an error (no retry)
// and is reachable with generate.min_new_tokens == 0.
// generate.min_new_tokens defaults to 11 and excludes EOS from generation steps
// below it; step 0 is the prediction from prefill. 0 disables the window.
// max_new_tokens is always a hard cap and may be smaller than min_new_tokens; the
// length limit then wins and EOS is never eligible. Repetition history includes
// prompt ids plus accepted codes; other sampler semantics follow
// t2s_session_generate. The EOS window approximates upstream early-EOS suppression
// but does not claim full inference parity.
// Seeds are reproducible only within the same build/backend, not across Python.
//
// Success replaces result and clears error; failure preserves result. Recoverable
// input/allocation/compute failures return false; ggml aborts/corrupt backends are
// not recovery points. Do not alias error with strings in either frontend result.
bool synthesize_cpu(const synthesis_models & models,
                    const float * pcm, int64_t n_samples, int sample_rate,
                    const text_frontend_result & reference_text,
                    const text_frontend_result & target_text,
                    synthesis_result & result,
                    const synthesis_config & config = {},
                    std::string * error = nullptr);

} // namespace gpt_sovits
