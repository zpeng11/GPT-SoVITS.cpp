#pragma once

#include "gpt_sovits/pipeline.h"

namespace gpt_sovits {
struct sovits_session;

namespace pipeline_detail {

bool validate_synthesis_config(const synthesis_config & config, std::string * error = nullptr);
bool validate_synthesis_models(const synthesis_models & models, std::string * error = nullptr);

// Limited to the verified F32 T2S / F16 SoVITS conversion pair.
bool validate_codebooks(const t2s_model & t2s, const sovits_model & sovits,
                         std::string * error = nullptr);

bool synthesis_slot_size(int64_t T_ref, int64_t T_target, int64_t T_prompt,
                          int64_t max_new_tokens, uint32_t & out,
                          std::string * error = nullptr);

// Narrow generate/decode composition shared by the runner and contract tests.
// Sessions already contain reference conditioning. Failure preserves result.
bool generate_and_decode(t2s_session & t2s_state, const t2s_model & t2s,
                           sovits_session & sovits_state, const sovits_model & sovits,
                           const std::vector<int32_t> & phones,
                           const std::vector<float> & bert,
                           const t2s_generate_config & config,
                           synthesis_result & result, std::string * error = nullptr);

} // namespace pipeline_detail
} // namespace gpt_sovits
