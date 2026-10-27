// Test-only v1 reference reader. cnpy discards dtype details needed here.
// SHA-256 is checked for syntax only; use generate_reference.py --validate
// before this reader to verify file integrity.

#pragma once

#include "gpt_sovits/t2s.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace t2s_reference {

struct Reference {
    gpt_sovits::t2s_hparams hparams{};
    int64_t text = 0;
    int64_t prompt = 0;
    int64_t steps = 0;
    int64_t prefill = 0;
    int64_t final = 0;

    std::vector<float> prefill_input;
    std::vector<float> prefill_hidden;
    std::vector<float> logits;
    std::vector<float> prefill_k;
    std::vector<float> prefill_v;
    std::vector<float> final_k;
    std::vector<float> final_v;

    std::vector<int32_t> forced_tokens;
    std::vector<int32_t> argmax;
    std::vector<int32_t> context_lengths;
    std::vector<int32_t> audio_positions;

    // Original input arrays, retained for callers that drive the public API.
    std::vector<int32_t> phones;
    std::vector<int32_t> prompt_tokens;
    std::vector<float> bert;
};

// Load and fully validate a reference directory. Throws std::runtime_error.
Reference load(const std::filesystem::path & directory);

// Throw if the reference architecture does not equal the model architecture.
void require_hparams(const Reference & reference, const gpt_sovits::t2s_hparams & model);
void require_model(const Reference & reference, const gpt_sovits::t2s_model & model);

namespace detail {

namespace fs = std::filesystem;
using json = nlohmann::json;

// Reference exports are validated against these local consumption limits:
// hidden/embedding <= 4096, layers <= 128, heads <= 512, vocabularies <= 65536,
// 1024 context positions, and 512 MiB per array / 1 GiB total. These cover the
// verified CPU baseline (H=512, L=24, heads=16) and reject implausible shapes
// before any allocation.
constexpr uint64_t kContextLimit = 1024;
constexpr uint64_t kMaxHeaderBytes = 65536;
constexpr uint64_t kMaxManifestBytes = 1ULL << 20;
constexpr uint64_t kMaxArrayBytes = 512ULL << 20;
constexpr uint64_t kMaxTotalBytes = 1ULL << 30;
constexpr uint64_t kMaxHidden = 4096;
constexpr uint64_t kMaxLayers = 128;
constexpr uint64_t kMaxHead = 512;
constexpr uint64_t kMaxVocab = 65536;
constexpr const char * kSchema = "gpt-sovits-t2s-decode-reference/v1";
constexpr const char * kSemantics =
    "raw-logits; no-sampling; no-EOS-suppression; no-early-stop";

[[noreturn]] inline void fail(const std::string & where, const std::string & message) {
    throw std::runtime_error("t2s decode reference " + where + ": " + message);
}

inline void skip_ws(const std::string & s, size_t & p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) ++p;
}

inline void expect_char(const std::string & s, size_t & p, char c, const std::string & where) {
    if (p >= s.size() || s[p] != c) fail(where, std::string("expected '") + c + "' in header");
    ++p;
}

inline std::string parse_quoted(const std::string & s, size_t & p, const std::string & where) {
    if (p >= s.size() || (s[p] != '\'' && s[p] != '"')) fail(where, "malformed header string");
    const char quote = s[p++];
    std::string out;
    while (p < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[p++]);
        if (c == static_cast<unsigned char>(quote)) return out;
        if (c == '\\' || c >= 128) fail(where, "unsupported escape or non-ASCII header string");
        out.push_back(static_cast<char>(c));
    }
    fail(where, "unterminated header string");
}

inline std::string parse_identifier(const std::string & s, size_t & p, const std::string & where) {
    const size_t start = p;
    while (p < s.size()) {
        const char c = s[p];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')) break;
        ++p;
    }
    if (start == p) fail(where, "malformed header identifier");
    return s.substr(start, p - start);
}

inline std::vector<uint64_t> parse_shape(const std::string & s, size_t & p, const std::string & where) {
    skip_ws(s, p);
    expect_char(s, p, '(', where);
    skip_ws(s, p);
    std::vector<uint64_t> shape;
    if (p < s.size() && s[p] == ')') {
        ++p;
        return shape;
    }
    for (;;) {
        skip_ws(s, p);
        if (p >= s.size() || s[p] < '0' || s[p] > '9') fail(where, "shape dimension must be a nonnegative integer");
        uint64_t value = 0;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            const uint64_t digit = static_cast<uint64_t>(s[p++] - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) fail(where, "shape dimension overflows");
            value = value * 10 + digit;
        }
        if (shape.size() >= 8) fail(where, "array rank exceeds 8");
        shape.push_back(value);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') {
            ++p;
            skip_ws(s, p);
            if (p < s.size() && s[p] == ')') {
                ++p;
                return shape;
            }
            continue;
        }
        if (p < s.size() && s[p] == ')') {
            ++p;
            return shape;
        }
        fail(where, "malformed shape tuple");
    }
}

struct npy_header {
    std::string descr;
    bool float32 = false;
    std::vector<uint64_t> shape;
    uint64_t data_offset = 0;
    uint64_t count = 0;
};

inline npy_header parse_npy_header(const fs::path & path, uint64_t file_size, bool allow_zero = false) {
    const std::string where = path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(where, "cannot open file");
    char magic[6] = {};
    in.read(magic, sizeof(magic));
    const char expected_magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) fail(where, "not a .npy file");
    unsigned char version[2] = {};
    if (!in.read(reinterpret_cast<char *>(version), sizeof(version))) fail(where, "truncated .npy version");
    if ((version[0] != 1 && version[0] != 2) || version[1] != 0) {
        fail(where, "unsupported .npy version (only v1/v2 are accepted)");
    }
    uint64_t header_len = 0;
    const size_t length_bytes = version[0] == 1 ? 2 : 4;
    unsigned char length_raw[4] = {};
    if (!in.read(reinterpret_cast<char *>(length_raw), static_cast<std::streamsize>(length_bytes))) {
        fail(where, "truncated .npy header length");
    }
    for (size_t i = 0; i < length_bytes; ++i) header_len |= static_cast<uint64_t>(length_raw[i]) << (8 * i);
    if (header_len == 0 || header_len > kMaxHeaderBytes) fail(where, "invalid .npy header length");
    const uint64_t data_offset = 8 + length_bytes + header_len;
    if (data_offset > file_size) fail(where, "truncated .npy header");
    std::string text(static_cast<size_t>(header_len), '\0');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) fail(where, "truncated .npy header");

    size_t p = 0;
    skip_ws(text, p);
    expect_char(text, p, '{', where);
    std::string descr;
    bool fortran = false, have_descr = false, have_fortran = false, have_shape = false;
    std::vector<uint64_t> shape;
    for (;;) {
        skip_ws(text, p);
        if (p < text.size() && text[p] == '}') {
            ++p;
            break;
        }
        const std::string key = parse_quoted(text, p, where);
        skip_ws(text, p);
        expect_char(text, p, ':', where);
        skip_ws(text, p);
        if (key == "descr") {
            if (have_descr) fail(where, "duplicate descr in header");
            descr = parse_quoted(text, p, where);
            have_descr = true;
        } else if (key == "fortran_order") {
            if (have_fortran) fail(where, "duplicate fortran_order in header");
            const std::string value = parse_identifier(text, p, where);
            if (value == "True") fortran = true;
            else if (value != "False") fail(where, "fortran_order must be True or False");
            have_fortran = true;
        } else if (key == "shape") {
            if (have_shape) fail(where, "duplicate shape in header");
            shape = parse_shape(text, p, where);
            have_shape = true;
        } else {
            fail(where, "unexpected header key '" + key + "'");
        }
        skip_ws(text, p);
        if (p < text.size() && text[p] == ',') {
            ++p;
            continue;
        }
        if (p < text.size() && text[p] == '}') {
            ++p;
            break;
        }
        fail(where, "malformed header dictionary");
    }
    skip_ws(text, p);
    if (p != text.size()) fail(where, "unexpected trailing header data");
    if (!have_descr || !have_fortran || !have_shape) fail(where, "header is missing descr/fortran_order/shape");
    if (fortran) fail(where, "fortran_order=True is not C order");

    npy_header header;
    header.descr = descr;
    header.shape = shape;
    header.data_offset = data_offset;
    if (descr == "<f4") header.float32 = true;
    else if (descr == "<i4") header.float32 = false;
    else if (descr == ">f4" || descr == ">i4") fail(where, "big-endian dtype is not supported");
    else fail(where, "unsupported dtype '" + descr + "' (expected <f4 or <i4)");

    uint64_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0) {
            if (!allow_zero) fail(where, "zero-sized arrays are not supported");
            count = 0;
            continue;
        }
        if (count > (kMaxArrayBytes / 4) / dim) fail(where, "array is too large");
        count *= dim;
    }
    header.count = count;
    return header;
}

inline void require_regular(const fs::path & path, const std::string & where) {
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(path, ec);
    if (ec) fail(where, "cannot stat file: " + ec.message());
    if (fs::is_symlink(status)) fail(where, "must not be a symbolic link");
    if (!fs::is_regular_file(status)) fail(where, "must be a regular file");
}

template <typename T>
inline void load_npy(const fs::path & path, bool want_float, const std::vector<int64_t> & expected_shape,
                     std::vector<T> & out, bool allow_zero = false) {
    const std::string where = path.string();
    require_regular(path, where);
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) fail(where, "cannot read file size: " + ec.message());
    if (size > kMaxArrayBytes + kMaxHeaderBytes + 12) fail(where, "file is too large");
    const uint64_t file_size = static_cast<uint64_t>(size);
    const npy_header header = parse_npy_header(path, file_size, allow_zero);

    if (header.float32 != want_float) {
        fail(where, "dtype '" + header.descr + "' does not match expected " +
                        (want_float ? "float32" : "int32"));
    }
    if (header.shape.size() != expected_shape.size()) fail(where, "rank does not match manifest shape");
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (header.shape[i] != static_cast<uint64_t>(expected_shape[i])) {
            fail(where, "shape does not match manifest shape");
        }
    }
    const uint64_t bytes = header.count * sizeof(T);
    const uint64_t available = file_size - header.data_offset;
    if (available < bytes) fail(where, "file is truncated: " + std::to_string(available) + " data bytes, expected " +
                                          std::to_string(bytes));
    if (available > bytes) fail(where, "file length has trailing bytes: " + std::to_string(available) +
                                          " data bytes, expected " + std::to_string(bytes));
    if (sizeof(T) != 4) fail(where, "unsupported element size");

    out.resize(static_cast<size_t>(header.count));
    if (bytes != 0) {
        std::ifstream in(path, std::ios::binary);
        if (!in) fail(where, "cannot open file");
        in.seekg(static_cast<std::streamoff>(header.data_offset));
        in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
        if (!in || static_cast<uint64_t>(in.gcount()) != bytes) fail(where, "failed to read array data");
    }
    if constexpr (std::is_same<T, float>::value) {
        for (size_t i = 0; i < out.size(); ++i) {
            if (!std::isfinite(out[i])) fail(where, "contains a non-finite float at index " + std::to_string(i));
        }
    }
}

inline const json & field(const json & object, const std::string & key, const std::string & where) {
    if (!object.is_object()) fail(where, "expected an object");
    const auto it = object.find(key);
    if (it == object.end()) fail(where, "missing field '" + key + "'");
    return *it;
}

inline void require_keys(const json & object, std::initializer_list<const char *> keys, const std::string & where) {
    std::set<std::string> expected;
    for (const char * key : keys) expected.insert(key);
    if (!object.is_object()) fail(where, "expected an object");
    std::set<std::string> actual;
    for (auto it = object.begin(); it != object.end(); ++it) actual.insert(it.key());
    if (actual == expected) return;
    std::string detail;
    for (const std::string & key : expected) {
        if (!actual.count(key)) detail += " missing=" + key;
    }
    for (const std::string & key : actual) {
        if (!expected.count(key)) detail += " unexpected=" + key;
    }
    fail(where, "unexpected or missing fields;" + detail);
}

inline uint64_t positive_integer(const json & value, const std::string & where, uint64_t max_value) {
    uint64_t result = 0;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value <= 0) fail(where, "must be a positive integer");
        result = static_cast<uint64_t>(signed_value);
    } else {
        fail(where, "must be a positive integer");
    }
    if (result == 0 || result > max_value) {
        fail(where, "must be in [1, " + std::to_string(max_value) + "]");
    }
    return result;
}

inline void require_string(const json & value, const std::string & expected, const std::string & where) {
    if (!value.is_string() || value.get<std::string>() != expected) {
        fail(where, "must be \"" + expected + "\"");
    }
}

inline void require_true(const json & value, const std::string & where) {
    if (!value.is_boolean() || !value.get<bool>()) fail(where, "must be true");
}

inline void require_uint(const json & value, uint64_t expected, const std::string & where) {
    if (!value.is_number_integer() || value.get<int64_t>() != static_cast<int64_t>(expected)) {
        fail(where, "must be " + std::to_string(expected));
    }
}

inline void validate_digest(const json & value, const std::string & where) {
    if (!value.is_string()) fail(where, "must be a SHA-256 hex string");
    const std::string digest = value.get<std::string>();
    if (digest.size() != 64) fail(where, "must be a 64-character SHA-256 hex string");
    for (char c : digest) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            fail(where, "must contain only lowercase hexadecimal characters");
        }
    }
}

inline void validate_source_name(const std::string & name, const std::string & where) {
    if (name.find('\\') != std::string::npos) fail(where, "source name must use forward slashes");
    size_t start = 0;
    for (;;) {
        const size_t slash = name.find('/', start);
        const std::string part = name.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (part.empty() || part == "." || part == "..") fail(where, "invalid source-relative filename");
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    if (name.rfind("AR/", 0) != 0) fail(where, "upstream source must be AR-relative");
    if (name.size() < 3 || name.compare(name.size() - 3, 3, ".py") != 0) {
        fail(where, "upstream source must be a .py file");
    }
}

inline void validate_sources(const json & sources) {
    require_keys(sources,
                 {"checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256", "inputs", "upstream"},
                 "manifest.sources");
    for (const char * key : {"checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256"}) {
        validate_digest(field(sources, key, "manifest.sources"), std::string("manifest.sources.") + key);
    }
    const json & inputs = field(sources, "inputs", "manifest.sources");
    require_keys(inputs, {"t2s_all_phones.npy", "t2s_all_bert.npy", "t2s_prompt.npy", "generated_codes.npy"},
                 "manifest.sources.inputs");
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        validate_digest(it.value(), "manifest.sources.inputs." + it.key());
    }
    const json & upstream = field(sources, "upstream", "manifest.sources");
    if (!upstream.is_object() || upstream.empty()) fail("manifest.sources.upstream", "must be a nonempty object");
    bool has_model = false;
    for (auto it = upstream.begin(); it != upstream.end(); ++it) {
        validate_source_name(it.key(), "manifest.sources.upstream." + it.key());
        validate_digest(it.value(), "manifest.sources.upstream." + it.key());
        has_model = has_model || it.key() == "AR/models/t2s_model.py";
    }
    if (!has_model) fail("manifest.sources.upstream", "must include AR/models/t2s_model.py");
}

inline void validate_execution(const json & execution) {
    require_keys(execution,
                 {"device", "dtype", "attention", "eval", "inference_mode", "deterministic_algorithms",
                  "num_threads", "num_interop_threads", "initialization_seed", "python", "numpy", "torch",
                  "system", "machine"},
                 "manifest.execution");
    require_string(field(execution, "device", "manifest.execution"), "cpu", "manifest.execution.device");
    require_string(field(execution, "dtype", "manifest.execution"), "float32", "manifest.execution.dtype");
    require_string(field(execution, "attention", "manifest.execution"), "upstream-manual",
                   "manifest.execution.attention");
    require_true(field(execution, "eval", "manifest.execution"), "manifest.execution.eval");
    require_true(field(execution, "inference_mode", "manifest.execution"), "manifest.execution.inference_mode");
    require_true(field(execution, "deterministic_algorithms", "manifest.execution"),
                 "manifest.execution.deterministic_algorithms");
    require_uint(field(execution, "num_threads", "manifest.execution"), 1, "manifest.execution.num_threads");
    require_uint(field(execution, "num_interop_threads", "manifest.execution"), 1,
                 "manifest.execution.num_interop_threads");
    require_uint(field(execution, "initialization_seed", "manifest.execution"), 0,
                 "manifest.execution.initialization_seed");
    for (const char * key : {"python", "numpy", "torch", "system", "machine"}) {
        const json & value = field(execution, key, "manifest.execution");
        if (!value.is_string()) fail(std::string("manifest.execution.") + key, "must be a string");
        const std::string text = value.get<std::string>();
        if (text.empty() || text.size() >= 256 || text.find_first_of("/\\\r\n") != std::string::npos) {
            fail(std::string("manifest.execution.") + key, "has an invalid version string");
        }
    }
}

inline gpt_sovits::t2s_hparams validate_hparams(const json & hparams) {
    require_keys(hparams,
                 {"embedding_dim", "hidden_dim", "head", "linear_units", "n_layer", "vocab_size",
                  "phoneme_vocab_size", "EOS"},
                 "manifest.hparams");
    gpt_sovits::t2s_hparams out;
    const auto read = [&](const char * key, uint64_t max_value) {
        return static_cast<uint32_t>(
            positive_integer(field(hparams, key, "manifest.hparams"), std::string("manifest.hparams.") + key, max_value));
    };
    out.embedding_dim = read("embedding_dim", kMaxHidden);
    out.hidden_dim = read("hidden_dim", kMaxHidden);
    out.n_head = read("head", kMaxHead);
    out.linear_units = read("linear_units", 4 * kMaxHidden);
    out.n_layer = read("n_layer", kMaxLayers);
    out.vocab_size = read("vocab_size", kMaxVocab);
    out.phoneme_vocab_size = read("phoneme_vocab_size", kMaxVocab);
    out.eos = read("EOS", kMaxVocab - 1);

    if (out.embedding_dim != out.hidden_dim) fail("manifest.hparams.embedding_dim", "must equal hidden_dim");
    if (out.hidden_dim % 2 != 0) fail("manifest.hparams.hidden_dim", "must be even");
    if (out.n_head == 0 || out.n_head > out.hidden_dim || out.hidden_dim % out.n_head != 0) {
        fail("manifest.hparams.head", "must divide hidden_dim");
    }
    if (out.linear_units != 4 * out.hidden_dim) fail("manifest.hparams.linear_units", "must equal 4 * hidden_dim");
    if (out.vocab_size < 2 || out.eos != out.vocab_size - 1) fail("manifest.hparams.EOS", "must equal vocab_size - 1");
    return out;
}

struct dimensions {
    int64_t text = 0;
    int64_t prompt = 0;
    int64_t steps = 0;
    int64_t prefill = 0;
    int64_t final = 0;
};

inline dimensions validate_dimensions(const json & value) {
    require_keys(value, {"text", "prompt", "steps", "prefill", "final"}, "manifest.dimensions");
    dimensions out;
    const auto read = [&](const char * key) {
        return static_cast<int64_t>(positive_integer(field(value, key, "manifest.dimensions"),
                                                     std::string("manifest.dimensions.") + key, kContextLimit));
    };
    out.text = read("text");
    out.prompt = read("prompt");
    out.steps = read("steps");
    out.prefill = read("prefill");
    out.final = read("final");
    if (out.prefill != out.text + out.prompt) fail("manifest.dimensions.prefill", "must equal text + prompt");
    if (out.final != out.prefill + out.steps) fail("manifest.dimensions.final", "must equal prefill + steps");
    if (out.prefill > static_cast<int64_t>(kContextLimit) || out.final > static_cast<int64_t>(kContextLimit)) {
        fail("manifest.dimensions", "prefill/final exceeds the 1024-position context budget");
    }
    return out;
}

struct array_spec {
    std::string name;
    bool float32 = false;
    std::vector<int64_t> shape;
    bool allow_zero = false;
};

inline std::vector<array_spec> make_specs(const dimensions & dims, const gpt_sovits::t2s_hparams & hparams) {
    const int64_t h = hparams.hidden_dim;
    const int64_t l = hparams.n_layer;
    const int64_t v = hparams.vocab_size;
    return {
        {"phones", false, {dims.text}},
        {"bert", true, {dims.text, 1024}},
        {"prompt", false, {dims.prompt}},
        {"forced_tokens", false, {dims.steps}},
        {"prefill_input", true, {dims.prefill, h}},
        {"prefill_hidden", true, {dims.prefill, h}},
        {"logits", true, {dims.steps + 1, v}},
        {"argmax", false, {dims.steps + 1}},
        {"context_lengths", false, {dims.steps + 1}},
        {"audio_positions", false, {dims.steps}},
        {"prefill_k", true, {l, dims.prefill, h}},
        {"prefill_v", true, {l, dims.prefill, h}},
        {"final_k", true, {l, dims.final, h}},
        {"final_v", true, {l, dims.final, h}},
    };
}

inline std::vector<int64_t> json_shape(const json & value, const std::string & where) {
    if (!value.is_array()) fail(where, "must be an array");
    std::vector<int64_t> shape;
    for (const json & item : value) {
        if (item.is_number_unsigned()) {
            const uint64_t n = item.get<uint64_t>();
            if (n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail(where, "dimension overflows");
            shape.push_back(static_cast<int64_t>(n));
        } else if (item.is_number_integer()) {
            const int64_t n = item.get<int64_t>();
            if (n < 0) fail(where, "dimensions must be nonnegative");
            shape.push_back(n);
        } else {
            fail(where, "must contain integers");
        }
    }
    return shape;
}

inline void validate_array_metadata(const json & manifest, const std::vector<array_spec> & specs) {
    const json & entries = field(manifest, "arrays", "manifest");
    if (!entries.is_object()) fail("manifest.arrays", "expected an object");
    std::set<std::string> expected;
    for (const array_spec & spec : specs) expected.insert(spec.name + ".npy");
    std::set<std::string> actual;
    for (auto it = entries.begin(); it != entries.end(); ++it) actual.insert(it.key());
    if (actual != expected) {
        std::string detail;
        for (const std::string & name : expected) if (!actual.count(name)) detail += " missing=" + name;
        for (const std::string & name : actual) if (!expected.count(name)) detail += " unexpected=" + name;
        fail("manifest.arrays", "unexpected or missing entries;" + detail);
    }
    for (const array_spec & spec : specs) {
        const std::string filename = spec.name + ".npy";
        const std::string where = "manifest.arrays." + filename;
        const json & meta = field(entries, filename, "manifest.arrays");
        require_keys(meta, {"dtype", "shape", "order", "ggml_ne", "sha256"}, where);
        require_string(field(meta, "dtype", where), spec.float32 ? "float32" : "int32", where + ".dtype");
        require_string(field(meta, "order", where), "C", where + ".order");
        const std::vector<int64_t> shape = json_shape(field(meta, "shape", where), where + ".shape");
        if (shape != spec.shape) fail(where + ".shape", "does not match the reference dimensions");
        std::vector<int64_t> reversed(spec.shape.rbegin(), spec.shape.rend());
        if (json_shape(field(meta, "ggml_ne", where), where + ".ggml_ne") != reversed) {
            fail(where + ".ggml_ne", "must be the reversed C-order shape");
        }
        validate_digest(field(meta, "sha256", where), where + ".sha256");
    }
}

inline void check_budget(const std::vector<array_spec> & specs) {
    uint64_t total = 0;
    for (const array_spec & spec : specs) {
        uint64_t count = 1;
        for (int64_t dim : spec.shape) {
            if (dim < 0 || (dim == 0 && !spec.allow_zero)) {
                fail("manifest.arrays." + spec.name + ".npy", "shape dimensions must be positive");
            }
            if (dim == 0) {
                count = 0;
                continue;
            }
            const uint64_t d = static_cast<uint64_t>(dim);
            if (count > (kMaxArrayBytes / 4) / d) fail("manifest.arrays." + spec.name + ".npy", "array is too large");
            count *= d;
        }
        const uint64_t bytes = count * 4;
        if (bytes > kMaxTotalBytes - total) fail("manifest", "combined array size exceeds the local limit");
        total += bytes;
    }
}

inline std::set<std::string> directory_files(const fs::path & directory) {
    const std::string where = directory.string();
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(directory, ec);
    if (ec) fail(where, "cannot stat directory: " + ec.message());
    if (fs::is_symlink(status)) fail(where, "must not be a symbolic link");
    if (!fs::is_directory(status)) fail(where, "must be a directory");
    std::set<std::string> files;
    for (fs::directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
        if (ec) fail(where, "cannot list directory: " + ec.message());
        const fs::path & path = it->path();
        const std::string name = path.filename().string();
        std::error_code entry_ec;
        const fs::file_status entry_status = fs::symlink_status(path, entry_ec);
        if (entry_ec) fail(where, "cannot stat entry '" + name + "': " + entry_ec.message());
        if (fs::is_symlink(entry_status) || !fs::is_regular_file(entry_status)) {
            fail(where, "unexpected non-regular entry '" + name + "'");
        }
        files.insert(name);
    }
    if (ec) fail(where, "cannot list directory: " + ec.message());
    return files;
}

inline json read_manifest(const fs::path & path) {
    const std::string where = path.string();
    require_regular(path, where);
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) fail(where, "cannot read file size: " + ec.message());
    if (size > kMaxManifestBytes) fail(where, "manifest is too large");
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(where, "cannot open file");
    std::string text(static_cast<size_t>(size), '\0');
    if (size != 0 && !in.read(text.data(), static_cast<std::streamsize>(text.size()))) fail(where, "cannot read file");
    try {
        return json::parse(text);
    } catch (const json::exception & error) {
        fail(where, std::string("invalid JSON: ") + error.what());
    }
}

inline const array_spec & find_spec(const std::vector<array_spec> & specs, const char * name) {
    for (const array_spec & spec : specs) {
        if (spec.name == name) return spec;
    }
    fail("manifest.arrays", std::string("internal missing spec for ") + name);
}

inline void check_token_range(const std::vector<int32_t> & tokens, int64_t upper_bound, const char * filename) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || static_cast<int64_t>(tokens[i]) >= upper_bound) {
            fail(filename, "token ID out of range at index " + std::to_string(i));
        }
    }
}

} // namespace detail

inline Reference load(const std::filesystem::path & directory) {
    using namespace detail;
    uint16_t endian_probe = 1;
    if (*reinterpret_cast<unsigned char *>(&endian_probe) != 1) fail("host", "big-endian hosts are not supported");

    const json manifest = read_manifest(directory / "manifest.json");
    require_keys(manifest,
                 {"schema", "status", "hparams", "dimensions", "semantics", "sources", "execution", "arrays"},
                 "manifest");
    require_string(field(manifest, "schema", "manifest"), kSchema, "manifest.schema");
    require_string(field(manifest, "status", "manifest"), "complete", "manifest.status");
    const gpt_sovits::t2s_hparams hparams = validate_hparams(field(manifest, "hparams", "manifest"));
    const dimensions dims = validate_dimensions(field(manifest, "dimensions", "manifest"));
    validate_sources(field(manifest, "sources", "manifest"));
    validate_execution(field(manifest, "execution", "manifest"));
    require_string(field(manifest, "semantics", "manifest"), kSemantics, "manifest.semantics");
    const std::vector<array_spec> specs = make_specs(dims, hparams);
    validate_array_metadata(manifest, specs);
    check_budget(specs);

    std::set<std::string> expected_files = {"manifest.json"};
    for (const array_spec & spec : specs) expected_files.insert(spec.name + ".npy");
    const std::set<std::string> actual_files = directory_files(directory);
    if (actual_files != expected_files) {
        std::string detail;
        for (const std::string & name : expected_files) if (!actual_files.count(name)) detail += " missing=" + name;
        for (const std::string & name : actual_files) if (!expected_files.count(name)) detail += " unexpected=" + name;
        fail(directory.string(), "file set mismatch;" + detail);
    }

    Reference out;
    out.hparams = hparams;
    out.text = dims.text;
    out.prompt = dims.prompt;
    out.steps = dims.steps;
    out.prefill = dims.prefill;
    out.final = dims.final;
    load_npy(directory / "phones.npy", false, find_spec(specs, "phones").shape, out.phones);
    load_npy(directory / "bert.npy", true, find_spec(specs, "bert").shape, out.bert);
    load_npy(directory / "prompt.npy", false, find_spec(specs, "prompt").shape, out.prompt_tokens);
    load_npy(directory / "forced_tokens.npy", false, find_spec(specs, "forced_tokens").shape, out.forced_tokens);
    load_npy(directory / "prefill_input.npy", true, find_spec(specs, "prefill_input").shape, out.prefill_input);
    load_npy(directory / "prefill_hidden.npy", true, find_spec(specs, "prefill_hidden").shape, out.prefill_hidden);
    load_npy(directory / "logits.npy", true, find_spec(specs, "logits").shape, out.logits);
    load_npy(directory / "argmax.npy", false, find_spec(specs, "argmax").shape, out.argmax);
    load_npy(directory / "context_lengths.npy", false, find_spec(specs, "context_lengths").shape,
             out.context_lengths);
    load_npy(directory / "audio_positions.npy", false, find_spec(specs, "audio_positions").shape,
             out.audio_positions);
    load_npy(directory / "prefill_k.npy", true, find_spec(specs, "prefill_k").shape, out.prefill_k);
    load_npy(directory / "prefill_v.npy", true, find_spec(specs, "prefill_v").shape, out.prefill_v);
    load_npy(directory / "final_k.npy", true, find_spec(specs, "final_k").shape, out.final_k);
    load_npy(directory / "final_v.npy", true, find_spec(specs, "final_v").shape, out.final_v);

    check_token_range(out.phones, static_cast<int64_t>(hparams.phoneme_vocab_size), "phones.npy");
    check_token_range(out.prompt_tokens, static_cast<int64_t>(hparams.eos), "prompt.npy");
    check_token_range(out.forced_tokens, static_cast<int64_t>(hparams.eos), "forced_tokens.npy");

    const int64_t vocab = hparams.vocab_size;
    for (int64_t row = 0; row <= dims.steps; ++row) {
        int64_t best = 0;
        float best_value = out.logits[static_cast<size_t>(row * vocab)];
        for (int64_t column = 1; column < vocab; ++column) {
            const float value = out.logits[static_cast<size_t>(row * vocab + column)];
            if (value > best_value) {
                best_value = value;
                best = column;
            }
        }
        if (static_cast<int64_t>(out.argmax[static_cast<size_t>(row)]) != best) {
            fail("argmax.npy", "argmax disagrees with logits at row " + std::to_string(row));
        }
    }
    for (int64_t i = 0; i <= dims.steps; ++i) {
        if (static_cast<int64_t>(out.context_lengths[static_cast<size_t>(i)]) != dims.prefill + i) {
            fail("context_lengths.npy", "invalid value at index " + std::to_string(i));
        }
    }
    for (int64_t i = 0; i < dims.steps; ++i) {
        if (static_cast<int64_t>(out.audio_positions[static_cast<size_t>(i)]) != dims.prompt + i) {
            fail("audio_positions.npy", "invalid value at index " + std::to_string(i));
        }
    }
    const int64_t hidden = hparams.hidden_dim;
    const int64_t prefix_values = dims.prefill * hidden;
    for (int64_t layer = 0; layer < static_cast<int64_t>(hparams.n_layer); ++layer) {
        for (int64_t i = 0; i < prefix_values; ++i) {
            const size_t k_index = static_cast<size_t>(layer * dims.final * hidden + i);
            const size_t p_index = static_cast<size_t>(layer * dims.prefill * hidden + i);
            if (out.final_k[k_index] != out.prefill_k[p_index]) {
                fail("final_k.npy", "prefill prefix changed at layer " + std::to_string(layer));
            }
            if (out.final_v[k_index] != out.prefill_v[p_index]) {
                fail("final_v.npy", "prefill prefix changed at layer " + std::to_string(layer));
            }
        }
    }
    return out;
}

inline void require_hparams(const Reference & reference, const gpt_sovits::t2s_hparams & model) {
    const auto check = [](const char * name, uint32_t reference_value, uint32_t model_value) {
        if (reference_value != model_value) {
            detail::fail(std::string("model hparams.") + name,
                         "reference has " + std::to_string(reference_value) + ", model has " +
                             std::to_string(model_value));
        }
    };
    check("embedding_dim", reference.hparams.embedding_dim, model.embedding_dim);
    check("hidden_dim", reference.hparams.hidden_dim, model.hidden_dim);
    check("n_head", reference.hparams.n_head, model.n_head);
    check("linear_units", reference.hparams.linear_units, model.linear_units);
    check("n_layer", reference.hparams.n_layer, model.n_layer);
    check("vocab_size", reference.hparams.vocab_size, model.vocab_size);
    check("phoneme_vocab_size", reference.hparams.phoneme_vocab_size, model.phoneme_vocab_size);
    check("eos", reference.hparams.eos, model.eos);
}

inline void require_model(const Reference & reference, const gpt_sovits::t2s_model & model) {
    require_hparams(reference, model.hparams);
}

} // namespace t2s_reference
