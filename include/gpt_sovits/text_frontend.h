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

} // namespace gpt_sovits
