// Test-only v2 generation reference reader. Reuses t2s_reference::detail for
// NPY/JSON/header validation. SHA-256 is syntax-checked only; run
// generate_reference.py --validate before this reader to verify integrity.

#pragma once

#include "decode_reference.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace t2s_generation_reference {

namespace d = t2s_reference::detail;

struct Reference {
    t2s_reference::Reference input;
    int64_t reference_text = 0;
    int64_t target_text = 0;
    int64_t max_new_tokens = 0;
    int64_t min_new_tokens = 0;
    float repetition_penalty = 1.0f;
    std::string stop_reason;
    std::vector<int32_t> codes;
    std::vector<int32_t> selected;
    std::vector<int32_t> context_lengths;
    std::vector<float> logits;
};

Reference load(const std::filesystem::path & directory);
void require_model(const Reference & reference, const gpt_sovits::t2s_model & model);

namespace detail {

using json = nlohmann::json;

constexpr const char * kSchema = "gpt-sovits-t2s-generation-reference/v2";
constexpr const char * kSchemaV1 = "gpt-sovits-t2s-generation-reference/v1";
constexpr const char * kEosPolicy = "suppressed-before-min-new-tokens";
constexpr const char * kLegacyEosPolicy = "unmasked-before-accept";

[[noreturn]] inline void fail(const std::string & where, const std::string & message) {
    throw std::runtime_error("t2s generation reference " + where + ": " + message);
}

inline int64_t nonnegative_integer(const json & value, const std::string & where, uint64_t max_value) {
    uint64_t result = 0;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value < 0) fail(where, "must be a nonnegative integer");
        result = static_cast<uint64_t>(signed_value);
    } else {
        fail(where, "must be a nonnegative integer");
    }
    if (result > max_value) fail(where, "must be <= " + std::to_string(max_value));
    return static_cast<int64_t>(result);
}

struct dimensions {
    int64_t text = 0;
    int64_t reference_text = 0;
    int64_t target_text = 0;
    int64_t prompt = 0;
    int64_t prefill = 0;
    int64_t codes = 0;
    int64_t decisions = 0;
};

struct policy {
    int64_t max_new_tokens = 0;
    int64_t min_new_tokens = 0;
    float repetition_penalty = 1.0f;
};

inline policy validate_policy(const json & value) {
    const std::string where = "manifest.policy";
    d::require_keys(value,
                    {"top_k", "top_p", "temperature", "repetition_penalty", "max_new_tokens",
                     "min_new_tokens", "history", "eos"},
                    where);
    const json & top_k = d::field(value, "top_k", where);
    if (!top_k.is_number_integer() || top_k.get<int64_t>() != 1) fail(where + ".top_k", "must be 1");
    const auto unit = [&](const char * key) {
        const json & item = d::field(value, key, where);
        if (!item.is_number() || item.get<double>() != 1.0) fail(where + "." + key, "must be 1.0");
    };
    unit("top_p");
    unit("temperature");
    const json & penalty = d::field(value, "repetition_penalty", where);
    if (!penalty.is_number()) fail(where + ".repetition_penalty", "must be a positive finite float");
    const double penalty_value = penalty.get<double>();
    if (!std::isfinite(penalty_value) || !(penalty_value > 0.0)) {
        fail(where + ".repetition_penalty", "must be a positive finite float");
    }
    policy out;
    out.max_new_tokens = static_cast<int64_t>(
        d::positive_integer(d::field(value, "max_new_tokens", where), where + ".max_new_tokens", d::kContextLimit));
    out.min_new_tokens = nonnegative_integer(
        d::field(value, "min_new_tokens", where), where + ".min_new_tokens",
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
    out.repetition_penalty = static_cast<float>(penalty_value);
    if (!std::isfinite(out.repetition_penalty) || !(out.repetition_penalty > 0.0f) ||
        !std::isfinite(1.0f / out.repetition_penalty)) {
        fail(where + ".repetition_penalty", "penalty and reciprocal must be positive finite float32");
    }
    d::require_string(d::field(value, "history", where), "prompt+generated-binary", where + ".history");
    const json & eos = d::field(value, "eos", where);
    if (eos.is_string() && eos.get<std::string>() == kLegacyEosPolicy) {
        fail(where + ".eos", "legacy unmasked-before-accept references are unsupported; re-export them");
    }
    d::require_string(eos, kEosPolicy, where + ".eos");
    return out;
}

inline dimensions validate_dimensions(const json & value, int64_t max_new_tokens) {
    const std::string where = "manifest.dimensions";
    d::require_keys(value, {"text", "reference_text", "target_text", "prompt", "prefill", "codes", "decisions"},
                    where);
    const auto positive = [&](const char * key) {
        return static_cast<int64_t>(
            d::positive_integer(d::field(value, key, where), where + "." + key, d::kContextLimit));
    };
    dimensions out;
    out.text = positive("text");
    out.reference_text = positive("reference_text");
    out.target_text = positive("target_text");
    out.prompt = positive("prompt");
    out.prefill = positive("prefill");
    out.codes = nonnegative_integer(d::field(value, "codes", where), where + ".codes", d::kContextLimit);
    out.decisions = positive("decisions");
    if (out.target_text != out.text - out.reference_text) {
        fail(where + ".target_text", "must equal text - reference_text");
    }
    if (out.prefill != out.text + out.prompt) fail(where + ".prefill", "must equal text + prompt");
    if (out.prefill + max_new_tokens - 1 > static_cast<int64_t>(d::kContextLimit)) {
        fail(where, "prefill + max_new_tokens - 1 exceeds the 1024-position context budget");
    }
    if (out.decisions > max_new_tokens) fail(where + ".decisions", "must not exceed max_new_tokens");
    return out;
}

inline std::string validate_stop_reason(const json & value) {
    const std::string where = "manifest.stop_reason";
    if (!value.is_string()) fail(where, "must be a string");
    const std::string out = value.get<std::string>();
    if (out != "eos" && out != "max_new_tokens") fail(where, "must be \"eos\" or \"max_new_tokens\"");
    return out;
}

inline void validate_stop_shape(const dimensions & dims, const policy & pol, const std::string & stop_reason) {
    if (stop_reason == "eos") {
        if (dims.decisions != dims.codes + 1) fail("manifest.dimensions", "eos stop requires decisions = codes + 1");
        if (dims.decisions - 1 < pol.min_new_tokens) {
            fail("manifest.dimensions", "eos stop must occur after the min_new_tokens suppression window");
        }
    } else if (dims.codes != dims.decisions || dims.codes != pol.max_new_tokens) {
        fail("manifest.dimensions", "max_new_tokens stop requires codes = decisions = max_new_tokens");
    }
}

inline void validate_sources(const json & sources) {
    d::require_keys(sources,
                    {"checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256", "inputs", "upstream"},
                    "manifest.sources");
    for (const char * key : {"checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256"}) {
        d::validate_digest(d::field(sources, key, "manifest.sources"), std::string("manifest.sources.") + key);
    }
    const json & inputs = d::field(sources, "inputs", "manifest.sources");
    d::require_keys(inputs, {"t2s_all_phones.npy", "t2s_all_bert.npy", "t2s_prompt.npy"},
                    "manifest.sources.inputs");
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        d::validate_digest(it.value(), "manifest.sources.inputs." + it.key());
    }
    const json & upstream = d::field(sources, "upstream", "manifest.sources");
    if (!upstream.is_object() || upstream.empty()) fail("manifest.sources.upstream", "must be a nonempty object");
    bool has_model = false;
    for (auto it = upstream.begin(); it != upstream.end(); ++it) {
        d::validate_source_name(it.key(), "manifest.sources.upstream." + it.key());
        d::validate_digest(it.value(), "manifest.sources.upstream." + it.key());
        has_model = has_model || it.key() == "AR/models/t2s_model.py";
    }
    if (!has_model) fail("manifest.sources.upstream", "must include AR/models/t2s_model.py");
}

inline std::vector<d::array_spec> make_specs(const dimensions & dims, const gpt_sovits::t2s_hparams & hparams) {
    const d::array_spec codes{"codes", false, {dims.codes}, true};
    return {
        {"phones", false, {dims.text}},
        {"bert", true, {dims.text, 1024}},
        {"prompt", false, {dims.prompt}},
        {"prefill_input", true, {dims.prefill, hparams.hidden_dim}},
        codes,
        {"logits", true, {dims.decisions, hparams.vocab_size}},
        {"selected", false, {dims.decisions}},
        {"context_lengths", false, {dims.decisions}},
    };
}

inline float penalized(float logit, bool seen, float penalty) {
    const float value = !seen ? logit : (logit < 0.0f ? logit * penalty : logit * (1.0f / penalty));
    if (!std::isfinite(value)) fail("logits.npy", "non-finite repetition-adjusted logit");
    return value;
}

inline void validate_selection(const Reference & reference, const gpt_sovits::t2s_hparams & hparams,
                               const dimensions & dims, const std::string & stop_reason) {
    const int64_t eos = hparams.eos;
    const int64_t decisions = dims.decisions;
    if (stop_reason == "eos") {
        if (reference.selected.back() != eos) {
            fail("selected.npy", "eos stop requires the last selected token to be EOS");
        }
        for (int64_t i = 0; i + 1 < decisions; ++i) {
            if (reference.selected[static_cast<size_t>(i)] == eos) {
                fail("selected.npy", "EOS may only appear as the last selected token");
            }
        }
        for (int64_t i = 0; i < dims.codes; ++i) {
            if (reference.selected[static_cast<size_t>(i)] != reference.codes[static_cast<size_t>(i)]) {
                fail("selected.npy", "eos stop requires the selected prefix to equal codes");
            }
        }
    } else {
        for (int32_t id : reference.selected) {
            if (id == eos) fail("selected.npy", "max_new_tokens stop must not select EOS");
        }
        if (reference.selected != reference.codes) {
            fail("selected.npy", "max_new_tokens stop requires selected to equal codes");
        }
    }
    for (int64_t i = 0; i < decisions; ++i) {
        if (reference.context_lengths[static_cast<size_t>(i)] != dims.prefill + i) {
            fail("context_lengths.npy", "invalid value at index " + std::to_string(i));
        }
    }

    const int64_t vocab = hparams.vocab_size;
    // Binary history: the original prompt is seen from the first decision;
    // accepted generated codes join it, and repeated IDs are penalized once.
    std::vector<unsigned char> seen(static_cast<size_t>(vocab), 0);
    for (int32_t id : reference.input.prompt_tokens) {
        if (id < 0 || static_cast<uint64_t>(id) >= static_cast<uint64_t>(vocab)) {
            fail("prompt.npy", "token ID out of range");
        }
        seen[static_cast<size_t>(id)] = 1;
    }
    for (int64_t step = 0; step < decisions; ++step) {
        const size_t base = static_cast<size_t>(step * vocab);
        const bool suppressed = step < reference.min_new_tokens;
        const int64_t limit = suppressed ? eos : vocab;
        if (suppressed && reference.selected[static_cast<size_t>(step)] == eos) {
            fail("selected.npy", "EOS selected inside the min_new_tokens suppression window");
        }
        int64_t best = 0;
        float best_value = penalized(reference.logits[base], seen[0] != 0, reference.repetition_penalty);
        for (int64_t token = 1; token < limit; ++token) {
            const float value = penalized(reference.logits[base + static_cast<size_t>(token)],
                                          seen[static_cast<size_t>(token)] != 0, reference.repetition_penalty);
            if (value > best_value) {
                best_value = value;
                best = token;
            }
        }
        if (reference.selected[static_cast<size_t>(step)] != best) {
            fail("selected.npy", "disagrees with repetition-penalized logits at decision " + std::to_string(step));
        }
        if (reference.selected[static_cast<size_t>(step)] != eos) {
            seen[static_cast<size_t>(reference.selected[static_cast<size_t>(step)])] = 1;
        }
    }
}

} // namespace detail

inline Reference load(const std::filesystem::path & directory) {
    uint16_t probe = 1;
    if (*reinterpret_cast<unsigned char *>(&probe) != 1) detail::fail("host", "big-endian hosts are not supported");

    const d::json manifest = d::read_manifest(directory / "manifest.json");
    d::require_keys(manifest,
                    {"schema", "status", "hparams", "dimensions", "policy", "stop_reason", "sources", "execution",
                     "arrays"},
                    "manifest");
    const detail::json & schema = d::field(manifest, "schema", "manifest");
    if (schema.is_string() && schema.get<std::string>() == detail::kSchemaV1) {
        detail::fail("manifest.schema",
                     "legacy v1 used unmasked EOS; re-export the reference with this reader's schema");
    }
    d::require_string(schema, detail::kSchema, "manifest.schema");
    d::require_string(d::field(manifest, "status", "manifest"), "complete", "manifest.status");
    const gpt_sovits::t2s_hparams hparams = d::validate_hparams(d::field(manifest, "hparams", "manifest"));
    const detail::policy pol = detail::validate_policy(d::field(manifest, "policy", "manifest"));
    const std::string stop_reason = detail::validate_stop_reason(d::field(manifest, "stop_reason", "manifest"));
    const detail::dimensions dims =
        detail::validate_dimensions(d::field(manifest, "dimensions", "manifest"), pol.max_new_tokens);
    detail::validate_stop_shape(dims, pol, stop_reason);
    detail::validate_sources(d::field(manifest, "sources", "manifest"));
    d::validate_execution(d::field(manifest, "execution", "manifest"));

    const std::vector<d::array_spec> specs = detail::make_specs(dims, hparams);
    d::validate_array_metadata(manifest, specs);
    d::check_budget(specs);

    std::set<std::string> expected_files = {"manifest.json"};
    for (const d::array_spec & spec : specs) expected_files.insert(spec.name + ".npy");
    const std::set<std::string> actual_files = d::directory_files(directory);
    if (actual_files != expected_files) {
        std::string detail_text;
        for (const std::string & name : expected_files) {
            if (!actual_files.count(name)) detail_text += " missing=" + name;
        }
        for (const std::string & name : actual_files) {
            if (!expected_files.count(name)) detail_text += " unexpected=" + name;
        }
        detail::fail(directory.string(), "file set mismatch;" + detail_text);
    }

    Reference out;
    out.input.hparams = hparams;
    out.input.text = dims.text;
    out.input.prompt = dims.prompt;
    out.input.prefill = dims.prefill;
    out.reference_text = dims.reference_text;
    out.target_text = dims.target_text;
    out.max_new_tokens = pol.max_new_tokens;
    out.min_new_tokens = pol.min_new_tokens;
    out.repetition_penalty = pol.repetition_penalty;
    out.stop_reason = stop_reason;

    d::load_npy(directory / "phones.npy", false, d::find_spec(specs, "phones").shape, out.input.phones);
    d::load_npy(directory / "bert.npy", true, d::find_spec(specs, "bert").shape, out.input.bert);
    d::load_npy(directory / "prompt.npy", false, d::find_spec(specs, "prompt").shape, out.input.prompt_tokens);
    d::load_npy(directory / "prefill_input.npy", true, d::find_spec(specs, "prefill_input").shape,
                out.input.prefill_input);
    d::load_npy(directory / "codes.npy", false, d::find_spec(specs, "codes").shape, out.codes, true);
    d::load_npy(directory / "logits.npy", true, d::find_spec(specs, "logits").shape, out.logits);
    d::load_npy(directory / "selected.npy", false, d::find_spec(specs, "selected").shape, out.selected);
    d::load_npy(directory / "context_lengths.npy", false, d::find_spec(specs, "context_lengths").shape,
                out.context_lengths);

    d::check_token_range(out.input.phones, hparams.phoneme_vocab_size, "phones.npy");
    d::check_token_range(out.input.prompt_tokens, hparams.eos, "prompt.npy");
    d::check_token_range(out.codes, hparams.eos, "codes.npy");
    for (int32_t id : out.selected) {
        if (id < 0 || static_cast<uint64_t>(id) >= hparams.vocab_size) {
            detail::fail("selected.npy", "token ID out of range");
        }
    }
    detail::validate_selection(out, hparams, dims, stop_reason);
    return out;
}

inline void require_model(const Reference & reference, const gpt_sovits::t2s_model & model) {
    t2s_reference::require_model(reference.input, model);
}

} // namespace t2s_generation_reference
