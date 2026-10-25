#include "gpt_sovits/text_frontend.h"
#include "frontend_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gpt_sovits {

namespace {

using json = nlohmann::json;

using frontend_detail::contract_error;
using frontend_detail::invalid;
using frontend_detail::indexed;
using frontend_detail::codepoint_count;
using frontend_detail::report_error;

const json & field(const json & object, const char * key, const std::string & path) {
    if (!object.is_object()) invalid(path, "expected an object");
    const auto it = object.find(key);
    if (it == object.end()) invalid(path + "." + key, "required field is missing");
    return *it;
}

const std::string & string_value(const json & value, const std::string & path) {
    if (!value.is_string()) invalid(path, "expected a string");
    return value.get_ref<const std::string &>();
}

int64_t integer(const json & value, const std::string & path,
                int64_t maximum = std::numeric_limits<int64_t>::max()) {
    if (value.is_number_unsigned()) {
        const auto n = value.get<uint64_t>();
        if (n > static_cast<uint64_t>(maximum)) invalid(path, "integer is out of range");
        return static_cast<int64_t>(n);
    }
    if (!value.is_number_integer()) invalid(path, "expected a non-negative integer");
    const auto n = value.get<int64_t>();
    if (n < 0 || n > maximum) invalid(path, "integer is out of range");
    return n;
}

const json & array_value(const json & value, const std::string & path) {
    if (!value.is_array()) invalid(path, "expected an array");
    if (value.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        invalid(path, "array length is out of range");
    }
    return value;
}

std::vector<int32_t> integers(const json & value, const std::string & path, int32_t maximum) {
    array_value(value, path);
    std::vector<int32_t> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result.push_back(static_cast<int32_t>(integer(value[i], indexed(path, i), maximum)));
    }
    return result;
}

text_frontend_tokenizer parse_tokenizer(const json & value, const std::string & path,
                                        int64_t n_chars,
                                        const std::optional<std::vector<int32_t>> & word2ph) {
    if (string_value(field(value, "status", path), path + ".status") != "ok") {
        invalid(path + ".status", "expected ok");
    }
    text_frontend_tokenizer result;
    result.input_ids = integers(field(value, "input_ids", path), path + ".input_ids", INT32_MAX);
    const int64_t n = static_cast<int64_t>(result.input_ids.size());
    if (n < 2) invalid(path + ".input_ids", "expected CLS and SEP tokens");
    result.body_start = integer(field(value, "body_start", path), path + ".body_start", n);
    result.body_end = integer(field(value, "body_end", path), path + ".body_end", n);
    if (result.body_start != 1 || result.body_end != n - 1) {
        invalid(path, "body_start/body_end must exclude exactly one CLS and SEP");
    }
    const auto cls = integer(field(value, "cls_id", path), path + ".cls_id", INT32_MAX);
    const auto sep = integer(field(value, "sep_id", path), path + ".sep_id", INT32_MAX);
    if (cls == sep) invalid(path, "CLS and SEP must have different IDs");
    if (result.input_ids.front() != cls) invalid(path + ".cls_id", "does not match first input ID");
    if (result.input_ids.back() != sep) invalid(path + ".sep_id", "does not match last input ID");

    const auto & spans = array_value(field(value, "token_char_spans", path), path + ".token_char_spans");
    if (spans.size() != result.input_ids.size()) {
        invalid(path + ".token_char_spans", "length does not match input_ids");
    }
    result.char_starts.reserve(spans.size());
    result.char_ends.reserve(spans.size());
    for (size_t t = 0; t < spans.size(); ++t) {
        const auto p = indexed(path + ".token_char_spans", t);
        const bool special = t == 0 || t + 1 == spans.size();
        if (special) {
            if (!spans[t].is_null()) invalid(p, "CLS/SEP span must be null");
            result.char_starts.push_back(-1);
            result.char_ends.push_back(-1);
            continue;
        }
        if (result.input_ids[t] == cls || result.input_ids[t] == sep) {
            invalid(indexed(path + ".input_ids", t), "CLS/SEP IDs cannot occur inside the body");
        }
        array_value(spans[t], p);
        if (spans[t].size() != 2) invalid(p, "expected [start, end]");
        const auto start = integer(spans[t][0], p + "[0]", INT32_MAX);
        const auto end = integer(spans[t][1], p + "[1]", INT32_MAX);
        if (start >= end || end > n_chars) invalid(p, "span is outside segment text or empty");
        result.char_starts.push_back(static_cast<int32_t>(start));
        result.char_ends.push_back(static_cast<int32_t>(end));
    }
    if (word2ph) {
        for (size_t c = 0; c < word2ph->size(); ++c) {
            if ((*word2ph)[c] == 0) continue;
            int matches = 0;
            for (size_t t = static_cast<size_t>(result.body_start);
                 t < static_cast<size_t>(result.body_end); ++t) {
                if (result.char_starts[t] <= static_cast<int64_t>(c) &&
                    static_cast<int64_t>(c) < result.char_ends[t]) ++matches;
                if (matches > 1) break;
            }
            if (matches != 1) {
                invalid(path + ".token_char_spans",
                        "character " + std::to_string(c) + " must have exactly one covering body token");
            }
        }
    }
    return result;
}

text_frontend_result parse_document(const json & root) {
    if (string_value(field(root, "schema", "$"), "schema") != "gpt-sovits-text-frontend/v1") {
        invalid("schema", "unsupported schema (expected gpt-sovits-text-frontend/v1)");
    }
    if (string_value(field(root, "version", "$"), "version") != "v2") {
        invalid("version", "unsupported phoneme vocabulary (expected v2)");
    }
    const auto ok = root.find("ok");
    if ((ok != root.end() && !(*ok == true)) || root.contains("error")) {
        invalid("$", "frontend error payload is not a successful result");
    }

    text_frontend_result result;
    result.normalized_text = string_value(field(root, "normalized_text", "$"), "normalized_text");
    const auto total_chars = codepoint_count(result.normalized_text, "normalized_text");
    result.phoneme_ids = integers(field(root, "phoneme_ids", "$"), "phoneme_ids", 731);
    if (result.phoneme_ids.empty()) invalid("phoneme_ids", "request must contain phonemes");
    const int64_t total_phones = static_cast<int64_t>(result.phoneme_ids.size());
    if (integer(field(root, "phoneme_count", "$"), "phoneme_count") != total_phones) {
        invalid("phoneme_count", "does not match phoneme_ids length");
    }
    const auto & segments = array_value(field(root, "segments", "$"), "segments");
    if (segments.empty()) invalid("segments", "expected at least one segment");
    const auto & offsets = array_value(field(root, "segment_phoneme_offsets", "$"), "segment_phoneme_offsets");
    if (offsets.empty() || offsets.size() - 1 != segments.size()) {
        invalid("segment_phoneme_offsets", "expected segments.size() + 1 entries");
    }
    result.segments.reserve(segments.size());
    int64_t phone_offset = 0;
    int64_t char_offset = 0;
    size_t byte_offset = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto path = indexed("segments", i);
        const auto & value = segments[i];
        if (integer(field(value, "index", path), path + ".index") != static_cast<int64_t>(i)) {
            invalid(path + ".index", "does not match segment order");
        }
        text_frontend_segment segment;
        segment.language = string_value(field(value, "language", path), path + ".language");
        if (segment.language != "zh" && segment.language != "en") {
            invalid(path + ".language", "expected zh or en");
        }
        segment.normalized_text = string_value(field(value, "normalized_text", path), path + ".normalized_text");
        const int64_t n_chars = codepoint_count(segment.normalized_text, path + ".normalized_text");
        if (n_chars > total_chars - char_offset ||
            result.normalized_text.compare(byte_offset, segment.normalized_text.size(), segment.normalized_text) != 0) {
            invalid(path + ".normalized_text", "does not match concatenated normalized_text");
        }
        segment.normalized_start = integer(field(value, "normalized_start", path), path + ".normalized_start");
        segment.normalized_end = integer(field(value, "normalized_end", path), path + ".normalized_end");
        if (segment.normalized_start != char_offset || segment.normalized_end != char_offset + n_chars) {
            invalid(path + ".normalized_start/normalized_end", "does not match Unicode character offsets");
        }
        const auto char_count = value.find("char_count");
        if (char_count != value.end() && integer(*char_count, path + ".char_count") != n_chars) {
            invalid(path + ".char_count", "does not match Unicode character count");
        }
        segment.phoneme_offset = integer(field(value, "phoneme_offset", path), path + ".phoneme_offset");
        if (segment.phoneme_offset != phone_offset ||
            integer(offsets[i], indexed("segment_phoneme_offsets", i)) != phone_offset) {
            invalid(path + ".phoneme_offset", "segments must cover phonemes contiguously in order");
        }
        segment.phoneme_count = integer(field(value, "phoneme_count", path), path + ".phoneme_count");
        const auto & phones = array_value(field(value, "phoneme_ids", path), path + ".phoneme_ids");
        if (segment.phoneme_count > total_phones - phone_offset ||
            static_cast<int64_t>(phones.size()) != segment.phoneme_count) {
            invalid(path + ".phoneme_count", "does not match segment/total phoneme count");
        }
        for (size_t p = 0; p < phones.size(); ++p) {
            if (integer(phones[p], indexed(path + ".phoneme_ids", p), 731) !=
                result.phoneme_ids[static_cast<size_t>(phone_offset) + p]) {
                invalid(indexed(path + ".phoneme_ids", p), "does not match top-level phoneme_ids");
            }
        }
        const auto word2ph = value.find("word2ph");
        if (segment.language == "zh") {
            if (word2ph == value.end()) invalid(path + ".word2ph", "required field is missing");
            segment.word2ph = integers(*word2ph, path + ".word2ph", INT32_MAX);
            if (static_cast<int64_t>(segment.word2ph->size()) != n_chars) {
                invalid(path + ".word2ph", "length does not match Unicode character count");
            }
            int64_t sum = 0;
            for (int32_t count : *segment.word2ph) {
                if (count > segment.phoneme_count - sum) invalid(path + ".word2ph", "sum exceeds phoneme_count");
                sum += count;
            }
            if (sum != segment.phoneme_count) invalid(path + ".word2ph", "sum does not match phoneme_count");
        } else if (word2ph != value.end() && !word2ph->is_null()) {
            invalid(path + ".word2ph", "English segments must omit word2ph or use null");
        }
        const auto tokenizer = value.find("tokenizer");
        if (tokenizer != value.end() && !tokenizer->is_null()) {
            segment.tokenizer = parse_tokenizer(*tokenizer, path + ".tokenizer", n_chars, segment.word2ph);
        }
        phone_offset += segment.phoneme_count;
        char_offset += n_chars;
        byte_offset += segment.normalized_text.size();
        result.segments.push_back(std::move(segment));
    }
    if (phone_offset != total_phones ||
        integer(offsets.back(), indexed("segment_phoneme_offsets", offsets.size() - 1)) != total_phones) {
        invalid("segment_phoneme_offsets", "segments do not cover all phonemes");
    }
    if (byte_offset != result.normalized_text.size() || char_offset != total_chars) {
        invalid("normalized_text", "does not equal concatenated segment texts");
    }
    return result;
}

} // namespace

bool text_frontend_parse_json(const std::string & text, text_frontend_result & out,
                              std::string * error) {
    try {
        // The JSON lexer treats a raw NUL as EOF, even before trailing bytes.
        if (text.find('\0') != std::string::npos) {
            return report_error(error, "JSON text contains a raw NUL byte");
        }
        auto pending = parse_document(json::parse(text));
        out = std::move(pending);
        if (error) error->clear();
        return true;
    } catch (const contract_error & e) {
        return report_error(error, e.what());
    } catch (const json::exception & e) {
        return report_error(error, e.what());
    } catch (const std::bad_alloc &) {
        return report_error(error, "host allocation failed");
    } catch (const std::length_error &) {
        return report_error(error, "host allocation size exceeded");
    }
}

bool text_frontend_read_json(const std::string & path, text_frontend_result & out,
                             std::string * error) {
    try {
        if (path.find('\0') != std::string::npos) {
            return report_error(error, "JSON file path contains a NUL byte");
        }
        struct file_guard {
            FILE * file;
            ~file_guard() { if (file) std::fclose(file); }
        } input{std::fopen(path.c_str(), "rb")};
        if (!input.file) return report_error(error, ("cannot open JSON file: " + path).c_str());
        std::string text;
        char buffer[8192];
        for (;;) {
            const size_t n = std::fread(buffer, 1, sizeof(buffer), input.file);
            if (std::ferror(input.file)) {
                return report_error(error, ("cannot read JSON file: " + path).c_str());
            }
            text.append(buffer, n);
            if (n < sizeof(buffer)) break;
        }
        return text_frontend_parse_json(text, out, error);
    } catch (const std::bad_alloc &) {
        return report_error(error, "host allocation failed");
    } catch (const std::length_error &) {
        return report_error(error, "host allocation size exceeded");
    }
}

} // namespace gpt_sovits
