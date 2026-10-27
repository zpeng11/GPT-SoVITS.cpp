// Synthetic reference files keep contract tests independent of Python/models.

#include <gtest/gtest.h>

#include "decode_reference.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using t2s_reference::Reference;
using Json = nlohmann::json;

constexpr int64_t kText = 3;
constexpr int64_t kPrompt = 2;
constexpr int64_t kSteps = 2;
constexpr int64_t kPrefill = kText + kPrompt;
constexpr int64_t kFinal = kPrefill + kSteps;
constexpr int64_t kH = 8;
constexpr int64_t kL = 2;
constexpr int64_t kV = 12;
constexpr int64_t kPhoneVocab = 8;
constexpr int64_t kEOS = 11;

const std::string kDigest(64, 'a');
const char * kSchema = "gpt-sovits-t2s-decode-reference/v1";
const char * kSemantics = "raw-logits; no-sampling; no-EOS-suppression; no-early-stop";

class TempDir {
public:
    TempDir() {
        static std::atomic<unsigned long> sequence{0};
        const auto base = fs::temp_directory_path();
        for (;;) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path candidate =
                base / ("gpt_sovits_t2s_ref_" + std::to_string(stamp) + "_" + std::to_string(++sequence));
            std::error_code ec;
            if (fs::create_directory(candidate, ec)) {
                path_ = candidate;
                return;
            }
            if (ec && ec != std::errc::file_exists) {
                throw std::runtime_error("cannot create test directory: " + ec.message());
            }
        }
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir &) = delete;
    TempDir & operator=(const TempDir &) = delete;
    const fs::path & path() const { return path_; }

private:
    fs::path path_;
};

template <typename T>
std::vector<unsigned char> Raw(const std::vector<T> & values) {
    static_assert(sizeof(T) == 4, "test payloads are 32-bit");
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void PutI32(std::vector<unsigned char> & bytes, size_t index, int32_t value) {
    std::memcpy(bytes.data() + index * 4, &value, sizeof(value));
}

void PutF32(std::vector<unsigned char> & bytes, size_t index, float value) {
    std::memcpy(bytes.data() + index * 4, &value, sizeof(value));
}

struct Array {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<unsigned char> data;
    std::string descr;
    bool fortran = false;
    int version = 1;
};

Array MakeArray(std::string name, std::string dtype, std::vector<int64_t> shape,
                std::vector<unsigned char> data) {
    Array array;
    array.name = std::move(name);
    array.dtype = std::move(dtype);
    array.shape = std::move(shape);
    array.data = std::move(data);
    array.descr = array.dtype == "float32" ? "<f4" : "<i4";
    return array;
}

void AppendLe(std::vector<unsigned char> & out, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) out.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
}

std::vector<unsigned char> EncodeNpy(const Array & array) {
    std::string header = "{'descr': '" + array.descr + "', 'fortran_order': ";
    header += array.fortran ? "True" : "False";
    header += ", 'shape': (";
    for (size_t i = 0; i < array.shape.size(); ++i) {
        if (i != 0) header += ", ";
        header += std::to_string(array.shape[i]);
    }
    header += ",), }";
    const size_t prefix = array.version == 1 ? 10 : 12;
    while ((prefix + header.size() + 1) % 64 != 0) header.push_back(' ');
    header.push_back('\n');

    std::vector<unsigned char> out = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    out.push_back(static_cast<unsigned char>(array.version));
    out.push_back(0);
    AppendLe(out, header.size(), array.version == 1 ? 2 : 4);
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), array.data.begin(), array.data.end());
    return out;
}

void WriteFile(const fs::path & path, const std::vector<unsigned char> & bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    if (!bytes.empty()) out.write(reinterpret_cast<const char *>(bytes.data()),
                                 static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

struct Fixture {
    TempDir temp;
    std::map<std::string, Array> arrays;
    Json manifest;
    std::vector<std::string> extra_files;

    Fixture() {
        std::vector<float> bert(kText * 1024);
        for (size_t i = 0; i < bert.size(); ++i) bert[i] = 0.001f * static_cast<float>(i % 13) - 0.006f;
        std::vector<float> prefill_input(kPrefill * kH);
        std::vector<float> prefill_hidden(kPrefill * kH);
        std::vector<float> prefill_k(kL * kPrefill * kH);
        std::vector<float> prefill_v(kL * kPrefill * kH);
        for (size_t i = 0; i < prefill_input.size(); ++i) {
            prefill_input[i] = 0.05f * static_cast<float>(i + 1);
            prefill_hidden[i] = 0.03f * static_cast<float>(i + 1);
        }
        for (size_t i = 0; i < prefill_k.size(); ++i) {
            prefill_k[i] = 0.01f * static_cast<float>(i + 1);
            prefill_v[i] = -0.01f * static_cast<float>(i + 1);
        }
        std::vector<float> logits(static_cast<size_t>((kSteps + 1) * kV), -1.0f);
        std::vector<int32_t> argmax, context_lengths, audio_positions;
        for (int64_t row = 0; row <= kSteps; ++row) {
            const int32_t best = static_cast<int32_t>((4 + row) % kV);
            logits[static_cast<size_t>(row * kV + best)] = 2.5f;
            argmax.push_back(best);
            context_lengths.push_back(static_cast<int32_t>(kPrefill + row));
        }
        for (int64_t i = 0; i < kSteps; ++i) audio_positions.push_back(static_cast<int32_t>(kPrompt + i));

        std::vector<float> final_k(kL * kFinal * kH, 0.0f);
        std::vector<float> final_v(kL * kFinal * kH, 0.0f);
        for (int64_t layer = 0; layer < kL; ++layer) {
            const int64_t src = layer * kPrefill * kH;
            const int64_t dst = layer * kFinal * kH;
            for (int64_t i = 0; i < kPrefill * kH; ++i) {
                final_k[static_cast<size_t>(dst + i)] = prefill_k[static_cast<size_t>(src + i)];
                final_v[static_cast<size_t>(dst + i)] = prefill_v[static_cast<size_t>(src + i)];
            }
            for (int64_t i = dst + kPrefill * kH; i < (layer + 1) * kFinal * kH; ++i) {
                final_k[static_cast<size_t>(i)] = 0.5f + 0.01f * static_cast<float>(i);
                final_v[static_cast<size_t>(i)] = -0.5f - 0.01f * static_cast<float>(i);
            }
        }

        arrays.emplace("phones.npy", MakeArray("phones.npy", "int32", {kText}, Raw(std::vector<int32_t>{0, 2, 7})));
        arrays.emplace("bert.npy", MakeArray("bert.npy", "float32", {kText, 1024}, Raw(bert)));
        arrays.emplace("prompt.npy",
                       MakeArray("prompt.npy", "int32", {kPrompt}, Raw(std::vector<int32_t>{1, 3})));
        arrays.emplace("forced_tokens.npy",
                       MakeArray("forced_tokens.npy", "int32", {kSteps}, Raw(std::vector<int32_t>{4, 5})));
        arrays.emplace("prefill_input.npy",
                       MakeArray("prefill_input.npy", "float32", {kPrefill, kH}, Raw(prefill_input)));
        arrays.emplace("prefill_hidden.npy",
                       MakeArray("prefill_hidden.npy", "float32", {kPrefill, kH}, Raw(prefill_hidden)));
        arrays.emplace("logits.npy", MakeArray("logits.npy", "float32", {kSteps + 1, kV}, Raw(logits)));
        arrays.emplace("argmax.npy", MakeArray("argmax.npy", "int32", {kSteps + 1}, Raw(argmax)));
        arrays.emplace("context_lengths.npy",
                       MakeArray("context_lengths.npy", "int32", {kSteps + 1}, Raw(context_lengths)));
        arrays.emplace("audio_positions.npy",
                       MakeArray("audio_positions.npy", "int32", {kSteps}, Raw(audio_positions)));
        arrays.emplace("prefill_k.npy", MakeArray("prefill_k.npy", "float32", {kL, kPrefill, kH}, Raw(prefill_k)));
        arrays.emplace("prefill_v.npy", MakeArray("prefill_v.npy", "float32", {kL, kPrefill, kH}, Raw(prefill_v)));
        arrays.emplace("final_k.npy", MakeArray("final_k.npy", "float32", {kL, kFinal, kH}, Raw(final_k)));
        arrays.emplace("final_v.npy", MakeArray("final_v.npy", "float32", {kL, kFinal, kH}, Raw(final_v)));

        manifest = Json{
            {"schema", kSchema},
            {"status", "complete"},
            {"hparams",
             {{"embedding_dim", kH},
              {"hidden_dim", kH},
              {"head", 2},
              {"linear_units", 4 * kH},
              {"n_layer", kL},
              {"vocab_size", kV},
              {"phoneme_vocab_size", kPhoneVocab},
              {"EOS", kEOS}}},
            {"dimensions",
             {{"text", kText}, {"prompt", kPrompt}, {"steps", kSteps},
              {"prefill", kPrefill}, {"final", kFinal}}},
            {"semantics", kSemantics},
            {"sources",
             {{"checkpoint_sha256", kDigest},
              {"exporter_sha256", kDigest},
              {"checkpoint_loader_sha256", kDigest},
              {"inputs",
               {{"t2s_all_phones.npy", kDigest},
                {"t2s_all_bert.npy", kDigest},
                {"t2s_prompt.npy", kDigest},
                {"generated_codes.npy", kDigest}}},
              {"upstream", {{"AR/models/t2s_model.py", kDigest}}}}},
            {"execution",
             {{"device", "cpu"},
              {"dtype", "float32"},
              {"attention", "upstream-manual"},
              {"eval", true},
              {"inference_mode", true},
              {"deterministic_algorithms", true},
              {"num_threads", 1},
              {"num_interop_threads", 1},
              {"initialization_seed", 0},
              {"python", "3.11"},
              {"numpy", "2.0"},
              {"torch", "2.4"},
              {"system", "Darwin"},
              {"machine", "arm64"}}},
            {"arrays", Json::object()},
        };
        for (const auto & entry : arrays) {
            const Array & array = entry.second;
            Json shape = Json::array(), reversed = Json::array();
            for (int64_t dim : array.shape) shape.push_back(dim);
            for (auto it = array.shape.rbegin(); it != array.shape.rend(); ++it) reversed.push_back(*it);
            manifest["arrays"][array.name] = Json{{"dtype", array.dtype},
                                                  {"shape", shape},
                                                  {"order", "C"},
                                                  {"ggml_ne", reversed},
                                                  {"sha256", kDigest}};
        }
    }

    void Write() const {
        for (const auto & entry : arrays) WriteFile(temp.path() / entry.first, EncodeNpy(entry.second));
        for (const std::string & name : extra_files) WriteFile(temp.path() / name, {});
        const std::string text = manifest.dump(2);
        WriteFile(temp.path() / "manifest.json", std::vector<unsigned char>(text.begin(), text.end()));
    }
};

void ExpectLoadError(const fs::path & directory, const std::string & needle) {
    try {
        (void)t2s_reference::load(directory);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(needle), std::string::npos) << message;
        return;
    } catch (const std::exception & error) {
        ADD_FAILURE() << "wrong exception type: " << error.what();
        return;
    }
    ADD_FAILURE() << "load() unexpectedly succeeded";
}

TEST(T2SDecodeReferenceContract, LoadsAndValidatesCompleteV1Export) {
    Fixture fixture;
    fixture.Write();
    const Reference reference = t2s_reference::load(fixture.temp.path());

    EXPECT_EQ(reference.text, kText);
    EXPECT_EQ(reference.prompt, kPrompt);
    EXPECT_EQ(reference.steps, kSteps);
    EXPECT_EQ(reference.prefill, kPrefill);
    EXPECT_EQ(reference.final, kFinal);
    EXPECT_EQ(reference.hparams.hidden_dim, kH);
    EXPECT_EQ(reference.hparams.n_layer, kL);
    EXPECT_EQ(reference.hparams.vocab_size, kV);
    EXPECT_EQ(reference.hparams.phoneme_vocab_size, kPhoneVocab);
    EXPECT_EQ(reference.hparams.eos, kEOS);
    EXPECT_EQ(reference.phones, (std::vector<int32_t>{0, 2, 7}));
    EXPECT_EQ(reference.prompt_tokens, (std::vector<int32_t>{1, 3}));
    EXPECT_EQ(reference.forced_tokens, (std::vector<int32_t>{4, 5}));
    EXPECT_EQ(reference.argmax, (std::vector<int32_t>{4, 5, 6}));
    EXPECT_EQ(reference.context_lengths, (std::vector<int32_t>{5, 6, 7}));
    EXPECT_EQ(reference.audio_positions, (std::vector<int32_t>{2, 3}));
    ASSERT_EQ(reference.bert.size(), static_cast<size_t>(kText * 1024));
    EXPECT_FLOAT_EQ(reference.bert[5], 0.001f * 5.0f - 0.006f);
    ASSERT_EQ(reference.logits.size(), static_cast<size_t>((kSteps + 1) * kV));
    EXPECT_EQ(reference.prefill_k[0], reference.final_k[0]);
    EXPECT_EQ(reference.prefill_v[0], reference.final_v[0]);
    EXPECT_NO_THROW(t2s_reference::require_hparams(reference, reference.hparams));
    gpt_sovits::t2s_model model;
    model.hparams = reference.hparams;
    EXPECT_NO_THROW(t2s_reference::require_model(reference, model));
}

TEST(T2SDecodeReferenceContract, AcceptsVersion2NpyHeaders) {
    Fixture fixture;
    for (auto & entry : fixture.arrays) entry.second.version = 2;
    fixture.Write();
    EXPECT_NO_THROW((void)t2s_reference::load(fixture.temp.path()));
}

TEST(T2SDecodeReferenceContract, RejectsInvalidExportsTableDriven) {
    struct ErrorCase {
        const char * name;
        void (*mutate)(Fixture &);
        const char * needle;
    };
    const ErrorCase cases[] = {
        {"schema", [](Fixture & f) { f.manifest["schema"] = "gpt-sovits-t2s-decode-reference/v2"; },
         "manifest.schema"},
        {"status", [](Fixture & f) { f.manifest["status"] = "partial"; }, "manifest.status"},
        {"semantics", [](Fixture & f) { f.manifest["semantics"] = "sampled"; }, "manifest.semantics"},
        {"dimensions",
         [](Fixture & f) { f.manifest["dimensions"]["prefill"] = kPrefill + 1; }, "manifest.dimensions.prefill"},
        {"hparams", [](Fixture & f) { f.manifest["hparams"]["EOS"] = kEOS - 1; }, "manifest.hparams.EOS"},
        {"execution",
         [](Fixture & f) { f.manifest["execution"]["num_threads"] = 2; },
         "manifest.execution.num_threads"},
        {"dtype", [](Fixture & f) { f.arrays.at("logits.npy").descr = "<f8"; }, "dtype"},
        {"endianness", [](Fixture & f) { f.arrays.at("logits.npy").descr = ">f4"; }, "big-endian"},
        {"order", [](Fixture & f) { f.arrays.at("logits.npy").fortran = true; }, "fortran_order"},
        {"shape",
         [](Fixture & f) {
             Array & array = f.arrays.at("logits.npy");
             array.shape = {kSteps, kV};
             array.data.assign(static_cast<size_t>(kSteps * kV * 4), 0);
         },
         "shape"},
        {"oversized",
         [](Fixture & f) {
             Array & array = f.arrays.at("logits.npy");
             array.shape = {1000000000, kV};
             array.data.clear();
         },
         "too large"},
        {"truncation", [](Fixture & f) { f.arrays.at("logits.npy").data.pop_back(); }, "truncated"},
        {"trailing",
         [](Fixture & f) { f.arrays.at("logits.npy").data.push_back(0); }, "trailing"},
        {"nonfinite",
         [](Fixture & f) {
             PutF32(f.arrays.at("prefill_input.npy").data, 0, std::numeric_limits<float>::quiet_NaN());
         },
         "non-finite"},
        {"phone-id", [](Fixture & f) { PutI32(f.arrays.at("phones.npy").data, 0, kPhoneVocab); }, "phones.npy"},
        {"prompt-id", [](Fixture & f) { PutI32(f.arrays.at("prompt.npy").data, 0, kEOS); }, "prompt.npy"},
        {"context-position",
         [](Fixture & f) { PutI32(f.arrays.at("context_lengths.npy").data, 2, 8); }, "context_lengths"},
        {"audio-position",
         [](Fixture & f) { PutI32(f.arrays.at("audio_positions.npy").data, 0, 1); }, "audio_positions"},
        {"argmax", [](Fixture & f) { PutI32(f.arrays.at("argmax.npy").data, 0, 0); }, "argmax"},
        {"cache-prefix", [](Fixture & f) { PutF32(f.arrays.at("final_k.npy").data, 0, 9.0f); }, "prefix"},
        {"digest",
         [](Fixture & f) { f.manifest["arrays"]["logits.npy"]["sha256"] = std::string(63, 'a'); }, "SHA-256"},
        {"metadata-shape",
         [](Fixture & f) { f.manifest["arrays"]["logits.npy"]["shape"] = Json::array({kSteps + 1, kV + 1}); },
         "manifest.arrays.logits.npy.shape"},
        {"metadata-order",
         [](Fixture & f) { f.manifest["arrays"]["logits.npy"]["order"] = "F"; },
         "manifest.arrays.logits.npy.order"},
        {"missing-file", [](Fixture & f) { f.arrays.erase("bert.npy"); }, "missing="},
        {"extra-file", [](Fixture & f) { f.extra_files.push_back("extra.npy"); }, "unexpected="},
    };
    for (const ErrorCase & error_case : cases) {
        SCOPED_TRACE(error_case.name);
        Fixture fixture;
        error_case.mutate(fixture);
        fixture.Write();
        ExpectLoadError(fixture.temp.path(), error_case.needle);
    }
}

void ExpectHparamsError(const Reference & reference, const gpt_sovits::t2s_hparams & hparams,
                        const std::string & needle) {
    try {
        t2s_reference::require_hparams(reference, hparams);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(needle), std::string::npos) << message;
        return;
    }
    ADD_FAILURE() << "require_hparams() unexpectedly succeeded";
}

TEST(T2SDecodeReferenceContract, RejectsModelArchitectureMismatch) {
    Fixture fixture;
    fixture.Write();
    const Reference reference = t2s_reference::load(fixture.temp.path());
    struct HparamsCase {
        const char * name;
        void (*mutate)(gpt_sovits::t2s_hparams &);
    };
    const HparamsCase cases[] = {
        {"embedding_dim", [](gpt_sovits::t2s_hparams & h) { h.embedding_dim += 1; }},
        {"hidden_dim", [](gpt_sovits::t2s_hparams & h) { h.hidden_dim += 2; }},
        {"head", [](gpt_sovits::t2s_hparams & h) { h.n_head += 1; }},
        {"linear_units", [](gpt_sovits::t2s_hparams & h) { h.linear_units += 4; }},
        {"n_layer", [](gpt_sovits::t2s_hparams & h) { h.n_layer += 1; }},
        {"vocab_size", [](gpt_sovits::t2s_hparams & h) { h.vocab_size += 1; }},
        {"phoneme_vocab_size", [](gpt_sovits::t2s_hparams & h) { h.phoneme_vocab_size += 1; }},
        {"eos", [](gpt_sovits::t2s_hparams & h) { h.eos += 1; }},
    };
    for (const HparamsCase & hparams_case : cases) {
        SCOPED_TRACE(hparams_case.name);
        gpt_sovits::t2s_hparams hparams = reference.hparams;
        hparams_case.mutate(hparams);
        ExpectHparamsError(reference, hparams, hparams_case.name);
    }
    gpt_sovits::t2s_model model;
    model.hparams = reference.hparams;
    model.hparams.n_layer += 1;
    EXPECT_THROW(t2s_reference::require_model(reference, model), std::runtime_error);
}

} // namespace
