#pragma once

#include "gpt_sovits/audio.h"
#include "gpt_sovits/text_frontend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

struct hubert_model;
struct roberta_model;
struct t2s_model;
struct t2s_session;
struct sovits_model;
struct sovits_session;
struct sv_model;

namespace pipeline_detail {

struct reference_features {
    audio_reference_data audio;
    std::vector<int32_t> phones;
    std::vector<float> bert;    // {1024, T_ref}, channel-fast
    std::vector<float> hubert;  // {768, T_hub}, channel-fast
    int64_t T_ref = 0;
    int64_t T_hub = 0;
    int64_t T_prompt = 0;
};

// The verified 32 kHz v2Pro Python path appends 32000 * 0.3 zeros to its
// 16 kHz waveform: 9600 samples, not 0.3 seconds at the input sample rate.
// No normalization is applied. Success replaces out; failure preserves it.
bool build_reference_hubert_input(const float * pcm16k, int64_t n_samples,
                                  std::vector<float> & out,
                                  std::string * error = nullptr);

// Model-free validation, also used before touching session state.
bool validate_reference_features(const reference_features & features,
                                  std::string * error = nullptr);

// Models/backends are borrowed, CPU-only, and must not be used concurrently.
// Audio preparation retains its existing normalization output, but HuBERT uses
// unnormalized pcm16k + tail silence; SV receives unpadded pcm16k. No file IO,
// gain adjustment, trimming or model loading is performed. Success replaces out;
// failure preserves it. This is not a promise of upstream resampler parity.
bool prepare_reference(const float * pcm, int64_t n_samples, int sample_rate,
                        const text_frontend_result & text,
                        const hubert_model & hubert,
                        const roberta_model * roberta,
                        reference_features & out,
                        std::string * error = nullptr);

// Requires loaded, compatible v2Pro/25 Hz models on the same CPU backend and
// fresh initialized single-request sessions with no reference/decode cache.
// One session pair = one reference. Does not own sessions/models/backend or
// retain feature pointers. On compute failure, dispose of BOTH sessions; there
// is no cross-session atomic replacement guarantee. Free sessions before models
// and backend. No target generation or waveform decoding is performed.
bool encode_reference(const reference_features & features,
                       const t2s_model & t2s, const sovits_model & sovits,
                       const sv_model & sv,
                       t2s_session & t2s_state, sovits_session & sovits_state,
                       std::string * error = nullptr);

} // namespace pipeline_detail
} // namespace gpt_sovits
