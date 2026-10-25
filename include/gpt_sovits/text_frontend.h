#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gpt_sovits {

struct text_frontend_tokenizer {
    std::vector<int32_t> input_ids;
    int64_t body_start = 0;
    int64_t body_end = 0;
    // Unicode code point offsets into the segment's normalized_text, not bytes
    // or tokenizer.normalized_text. Special tokens use -1 in both arrays.
    std::vector<int32_t> char_starts;
    std::vector<int32_t> char_ends;
};

struct text_frontend_segment {
    std::string language;
    std::string normalized_text;
    int64_t normalized_start = 0;
    int64_t normalized_end = 0;
    int64_t phoneme_offset = 0;
    int64_t phoneme_count = 0;
    std::optional<std::vector<int32_t>> word2ph;
    std::optional<text_frontend_tokenizer> tokenizer;
};

struct text_frontend_result {
    std::string normalized_text;
    std::vector<int32_t> phoneme_ids;
    std::vector<text_frontend_segment> segments;
};

// Read gpt-sovits-text-frontend/v1 JSON with the v2 phoneme vocabulary.
// Results own their buffers. Success replaces out and clears error, if supplied;
// failure preserves out and reports a diagnostic through error (or stderr when
// error is null). JSON/type/range and recoverable host allocation errors return
// false. Allocation failure in the JSON library's noexcept teardown can terminate
// the process; resource exhaustion there is not a supported recovery point.
// Offsets count Unicode code points. Missing/null segment tokenizers are retained
// as unconfigured; no top-level tokenizer fallback or BERT computation is done.
// Model-specific limits (tokenizer vocabulary and 512 positions) are not checked.
// Do not alias error with the input strings or any string inside out.
bool text_frontend_parse_json(const std::string & json,
                              text_frontend_result & out,
                              std::string * error = nullptr);

bool text_frontend_read_json(const std::string & path,
                             text_frontend_result & out,
                             std::string * error = nullptr);

struct roberta_model;

// Build phoneme-level BERT features from a frontend result (v2 vocabulary).
// Chinese segments use their own tokenizer and RoBERTa hidden_states[-3];
// English phoneme ranges are zero-filled without accessing model/tokenizer.
// Zero-phoneme segments require neither a tokenizer nor model inference.
// A Chinese segment with phonemes requires a loaded model and tokenizer; there is
// no fallback, truncation or chunking. Its CLS/SEP-inclusive length is <= 512
// and token IDs must fit the model vocabulary. Structural input is validated.
//
// out is contiguous F32 ggml {1024, T_phonemes}, out[p * 1024 + channel].
// Success replaces out and clears error; failure preserves out and reports a
// diagnostic through error (or stderr when null). Recoverable host allocation
// and compute failures return false; ggml aborts/corrupt backends do not.
// model/backend are borrowed only for the call and must not be freed, reloaded
// or used concurrently during it. model may be null when no Chinese segment
// has phonemes. Do not alias error with any string inside text.
// This assembles features, not T2S generation or end-to-end synthesis.
bool text_frontend_build_bert_features(const text_frontend_result & text,
                                       const roberta_model * model,
                                       std::vector<float> & out,
                                       std::string * error = nullptr);

} // namespace gpt_sovits
