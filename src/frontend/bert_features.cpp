#include "gpt_sovits/text_frontend.h"
#include "gpt_sovits/roberta.h"
#include "bert_features_internal.h"
#include "frontend_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gpt_sovits {
namespace {

using frontend_detail::codepoint_count;
using frontend_detail::indexed;
using frontend_detail::invalid;

constexpr int64_t kChannels = 1024;
constexpr size_t kMaxTokens = 512;

void validate_tokenizer(const text_frontend_tokenizer & tok, int64_t n_chars,
                        const std::string & path) {
    const size_t n = tok.input_ids.size();
    if (n < 2 || n > kMaxTokens) invalid(path + ".input_ids", "expected 2..512 tokens including CLS/SEP");
    if (tok.char_starts.size() != n || tok.char_ends.size() != n) {
        invalid(path, "ID and span lengths must match");
    }
    if (tok.body_start != 1 || tok.body_end != static_cast<int64_t>(n) - 1) {
        invalid(path, "body_start/body_end must exclude exactly one CLS and SEP");
    }
    if (tok.input_ids.front() == tok.input_ids.back()) invalid(path, "CLS and SEP must have different IDs");
    for (size_t t = 0; t < n; ++t) {
        if (tok.input_ids[t] < 0) invalid(indexed(path + ".input_ids", t), "negative token ID");
        const int32_t start = tok.char_starts[t], end = tok.char_ends[t];
        if (t == 0 || t + 1 == n) {
            if (start != -1 || end != -1) invalid(path, "CLS/SEP spans must be -1/-1");
        } else {
            if (tok.input_ids[t] == tok.input_ids.front() || tok.input_ids[t] == tok.input_ids.back()) {
                invalid(indexed(path + ".input_ids", t), "CLS/SEP IDs cannot occur inside the body");
            }
            if (start < 0 || start >= end || end > n_chars) {
                invalid(indexed(path + ".token_char_spans", t), "span is outside segment text or empty");
            }
        }
    }
}

size_t validate_text(const text_frontend_result & text,
                     detail::frontend_token_feature_provider provider) {
    const size_t phones = text.phoneme_ids.size();
    const std::vector<float> capacity;
    if (phones > static_cast<uint64_t>(INT64_MAX / kChannels) ||
        phones > capacity.max_size() / kChannels ||
        phones > std::numeric_limits<size_t>::max() / sizeof(float) / kChannels) {
        invalid("phoneme_ids", "BERT feature size overflow");
    }
    if (phones == 0) invalid("phoneme_ids", "request must contain phonemes");
    if (text.segments.empty()) invalid("segments", "expected at least one segment");
    for (size_t p = 0; p < phones; ++p) {
        if (text.phoneme_ids[p] < 0 || text.phoneme_ids[p] >= 732) {
            invalid(indexed("phoneme_ids", p), "ID outside v2 vocabulary [0,732)");
        }
    }

    const int64_t total_chars = codepoint_count(text.normalized_text, "normalized_text");
    size_t phone_offset = 0, byte_offset = 0;
    int64_t char_offset = 0;
    for (size_t i = 0; i < text.segments.size(); ++i) {
        const auto & segment = text.segments[i];
        const auto path = indexed("segments", i);
        if (segment.language != "zh" && segment.language != "en") invalid(path + ".language", "expected zh or en");
        if (segment.phoneme_offset < 0 || static_cast<uint64_t>(segment.phoneme_offset) != phone_offset) {
            invalid(path + ".phoneme_offset", "segments must cover phonemes contiguously in order");
        }
        if (segment.phoneme_count < 0 || static_cast<uint64_t>(segment.phoneme_count) > phones - phone_offset) {
            invalid(path + ".phoneme_count", "count outside remaining phoneme range");
        }
        const auto n_chars = codepoint_count(segment.normalized_text, path + ".normalized_text");
        if (n_chars > total_chars - char_offset ||
            text.normalized_text.compare(byte_offset, segment.normalized_text.size(), segment.normalized_text) != 0) {
            invalid(path + ".normalized_text", "does not match concatenated normalized_text");
        }
        if (segment.normalized_start != char_offset || segment.normalized_end != char_offset + n_chars) {
            invalid(path + ".normalized_start/normalized_end", "does not match Unicode character offsets");
        }
        if (segment.language == "zh") {
            if (!segment.word2ph || segment.word2ph->size() != static_cast<uint64_t>(n_chars)) {
                invalid(path + ".word2ph", "expected one count per Unicode character");
            }
            int64_t sum = 0;
            for (int32_t count : *segment.word2ph) {
                if (count < 0 || count > segment.phoneme_count - sum) {
                    invalid(path + ".word2ph", "negative count or sum exceeds phoneme_count");
                }
                sum += count;
            }
            if (sum != segment.phoneme_count) invalid(path + ".word2ph", "sum does not match phoneme_count");
            if (segment.phoneme_count > 0) {
                if (!segment.tokenizer) invalid(path + ".tokenizer", "required for Chinese BERT features");
                validate_tokenizer(*segment.tokenizer, n_chars, path + ".tokenizer");
                if (!provider) invalid(path, "RoBERTa model is required for Chinese BERT features");
            }
        } else if (segment.word2ph) {
            invalid(path + ".word2ph", "English segments must omit word2ph");
        }
        phone_offset += static_cast<size_t>(segment.phoneme_count);
        char_offset += n_chars;
        byte_offset += segment.normalized_text.size();
    }
    if (phone_offset != phones) invalid("segments", "segments do not cover all phonemes");
    if (byte_offset != text.normalized_text.size() || char_offset != total_chars) {
        invalid("normalized_text", "does not equal concatenated segment texts");
    }
    return phones * kChannels;
}

bool extract_tokens(const void * context, const text_frontend_tokenizer & tok,
                    float * out, int64_t out_cap, std::string & error) {
    const auto & model = *static_cast<const roberta_model *>(context);
    if (!model.backend || !model.weights.embeddings.word_embeddings) {
        error = "RoBERTa model is not loaded";
        return false;
    }
    const int64_t vocab = model.weights.embeddings.word_embeddings->ne[1];
    for (size_t t = 0; t < tok.input_ids.size(); ++t) {
        if (tok.input_ids[t] < 0 || tok.input_ids[t] >= vocab) {
            error = "tokenizer.input_ids[" + std::to_string(t) + "] is outside model vocabulary";
            return false;
        }
    }
    if (!roberta_extract_tts_features(model, tok.input_ids.data(),
                                      static_cast<int64_t>(tok.input_ids.size()), out, out_cap)) {
        error = "RoBERTa TTS feature extraction failed";
        return false;
    }
    return true;
}

} // namespace

bool detail::text_frontend_build_bert_features_with_provider(
    const text_frontend_result & text,
    frontend_token_feature_provider provider,
    const void * context,
    std::vector<float> & out,
    std::string * error) {
    try {
        const size_t n_floats = validate_text(text, provider);
        std::vector<float> pending(n_floats, 0.0f);
        for (size_t i = 0; i < text.segments.size(); ++i) {
            const auto & segment = text.segments[i];
            if (segment.language != "zh" || segment.phoneme_count == 0) continue;
            const auto path = indexed("segments", i);
            const auto & tok = *segment.tokenizer;
            std::vector<float> tokens(tok.input_ids.size() * kChannels);
            std::string reason;
            if (!provider(context, tok, tokens.data(), static_cast<int64_t>(tokens.size()), reason)) {
                invalid(path + ".bert", reason.empty() ? "token feature extraction failed" : reason);
            }
            if (!std::all_of(tokens.begin(), tokens.end(), [](float v) { return std::isfinite(v); })) {
                invalid(path + ".bert", "token feature extraction produced non-finite values");
            }
            if (!roberta_expand_bert_features(
                    tokens.data(), static_cast<int64_t>(tok.input_ids.size()),
                    tok.char_starts.data(), tok.char_ends.data(), segment.word2ph->data(),
                    static_cast<int64_t>(segment.word2ph->size()), segment.phoneme_count,
                    pending.data() + static_cast<size_t>(segment.phoneme_offset) * kChannels)) {
                invalid(path + ".bert", "token-to-phoneme expansion failed (invalid span coverage)");
            }
        }
        out = std::move(pending);
        if (error) error->clear();
        return true;
    } catch (const frontend_detail::contract_error & e) {
        return frontend_detail::report_error(error, e.what());
    } catch (const std::bad_alloc &) {
        return frontend_detail::report_error(error, "host allocation failed");
    } catch (const std::length_error &) {
        return frontend_detail::report_error(error, "host allocation size exceeded");
    }
}

bool text_frontend_build_bert_features(const text_frontend_result & text,
                                      const roberta_model * model,
                                      std::vector<float> & out,
                                      std::string * error) {
    return detail::text_frontend_build_bert_features_with_provider(
        text, model ? extract_tokens : nullptr, model, out, error);
}

} // namespace gpt_sovits
