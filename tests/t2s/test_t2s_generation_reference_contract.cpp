// Contract tests for the generation reference reader; synthetic files only.

#include <gtest/gtest.h>

#include "generation_reference.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

constexpr int64_t kT = 4, kR = 2, kTarget = 2, kP = 2, kN = 6, kC = 2, kD = 3, kM = 3;
constexpr int64_t kH = 8, kL = 2, kV = 12, kPhoneVocab = 8, kEOS = 11;
const std::string kDigest(64, 'a');
const char * kSchema = "gpt-sovits-t2s-generation-reference/v2";
const char * kSchemaV1 = "gpt-sovits-t2s-generation-reference/v1";

class TempDir {
public:
    TempDir() {
        static std::atomic<unsigned long> sequence{0};
        const auto base = fs::temp_directory_path();
        for (;;) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path candidate =
                base / ("gpt_sovits_t2s_gen_" + std::to_string(stamp) + "_" + std::to_string(++sequence));
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
    std::string header = "{'descr': '" + array.descr + "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < array.shape.size(); ++i) {
        if (i != 0) header += ", ";
        header += std::to_string(array.shape[i]);
    }
    header += ",), }";
    while ((10 + header.size() + 1) % 64 != 0) header.push_back(' ');
    header.push_back('\n');
    std::vector<unsigned char> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    AppendLe(out, header.size(), 2);
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

enum class Mode { Eos, Limit, ImmediateEos };
struct Fixture {
    TempDir temp;
    Mode mode;
    std::map<std::string, Array> arrays;
    Json manifest;
    std::vector<std::string> extra_files;

    explicit Fixture(Mode value = Mode::Eos) : mode(value) {
        const int64_t codes_count = mode == Mode::Eos ? kC : (mode == Mode::Limit ? kD : 0);
        const int64_t decisions = mode == Mode::ImmediateEos ? 1 : kD;
        const int64_t tokens = mode == Mode::ImmediateEos ? 1 : kM;
        std::vector<int32_t> codes, selected, context_lengths;
        std::vector<float> logits(static_cast<size_t>(decisions * kV), -1.0f);
        if (mode == Mode::Eos) {
            codes = {4, 5};
            selected = {4, 5, static_cast<int32_t>(kEOS)};
            logits[4] = 2.0f;
            // Competitor for the prompt-inclusion rejection case: raw 1.8 loses
            // to token 4, but wins once prompt token 4 is penalized.
            logits[6] = 1.8f;
            logits[kV + 4] = 2.1f;
            // 5 is a prompt token; 2.5 / 1.35 still beats 4 / 1.35 at step 1.
            logits[kV + 5] = 2.5f;
            logits[2 * kV + 4] = 1.0f;
            logits[2 * kV + 5] = 1.0f;
            logits[2 * kV + kEOS] = 2.0f;
        } else if (mode == Mode::Limit) {
            codes = {4, 5, 6};
            selected = codes;
            logits[4] = 2.0f;
            logits[kV + 4] = 2.1f;
            // 5 is a prompt token; 2.5 / 1.35 still beats 4 / 1.35 at step 1.
            logits[kV + 5] = 2.5f;
            logits[2 * kV + 4] = 1.0f;
            logits[2 * kV + 5] = 1.0f;
            logits[2 * kV + 6] = 2.0f;
        } else {
            selected = {static_cast<int32_t>(kEOS)};
            logits[kEOS] = 2.0f;
        }
        for (int64_t i = 0; i < decisions; ++i) context_lengths.push_back(static_cast<int32_t>(kN + i));

        std::vector<float> bert(static_cast<size_t>(kT * 1024));
        for (size_t i = 0; i < bert.size(); ++i) bert[i] = 0.001f * static_cast<float>(i % 13) - 0.006f;
        std::vector<float> prefill_input(static_cast<size_t>(kN * kH));
        for (size_t i = 0; i < prefill_input.size(); ++i) prefill_input[i] = 0.05f * static_cast<float>(i + 1);

        arrays.emplace("phones.npy", MakeArray("phones.npy", "int32", {kT}, Raw(std::vector<int32_t>{0, 2, 7, 1})));
        arrays.emplace("bert.npy", MakeArray("bert.npy", "float32", {kT, 1024}, Raw(bert)));
        arrays.emplace("prompt.npy", MakeArray("prompt.npy", "int32", {kP}, Raw(std::vector<int32_t>{1, 5})));
        arrays.emplace("prefill_input.npy", MakeArray("prefill_input.npy", "float32", {kN, kH}, Raw(prefill_input)));
        arrays.emplace("codes.npy", MakeArray("codes.npy", "int32", {codes_count}, Raw(codes)));
        arrays.emplace("logits.npy", MakeArray("logits.npy", "float32", {decisions, kV}, Raw(logits)));
        arrays.emplace("selected.npy", MakeArray("selected.npy", "int32", {decisions}, Raw(selected)));
        arrays.emplace("context_lengths.npy", MakeArray("context_lengths.npy", "int32", {decisions}, Raw(context_lengths)));

        manifest = Json{
            {"schema", kSchema}, {"status", "complete"},
            {"hparams", {{"embedding_dim", kH}, {"hidden_dim", kH}, {"head", 2}, {"linear_units", 4 * kH},
                         {"n_layer", kL}, {"vocab_size", kV}, {"phoneme_vocab_size", kPhoneVocab}, {"EOS", kEOS}}},
            {"dimensions", {{"text", kT}, {"reference_text", kR}, {"target_text", kTarget}, {"prompt", kP},
                            {"prefill", kN}, {"codes", codes_count}, {"decisions", decisions}}},
            {"policy", {{"top_k", 1}, {"top_p", 1.0}, {"temperature", 1.0}, {"repetition_penalty", 1.35},
                        {"max_new_tokens", tokens}, {"min_new_tokens", 0},
                        {"history", "prompt+generated-binary"},
                        {"eos", "suppressed-before-min-new-tokens"}}},
            {"stop_reason", mode == Mode::Limit ? "max_new_tokens" : "eos"},
            {"sources", {{"checkpoint_sha256", kDigest}, {"exporter_sha256", kDigest},
                         {"checkpoint_loader_sha256", kDigest},
                         {"inputs", {{"t2s_all_phones.npy", kDigest}, {"t2s_all_bert.npy", kDigest},
                                     {"t2s_prompt.npy", kDigest}}},
                         {"upstream", {{"AR/models/t2s_model.py", kDigest}}}}},
            {"execution", {{"device", "cpu"}, {"dtype", "float32"}, {"attention", "upstream-manual"},
                           {"eval", true}, {"inference_mode", true}, {"deterministic_algorithms", true},
                           {"num_threads", 1}, {"num_interop_threads", 1}, {"initialization_seed", 0},
                           {"python", "3.11"}, {"numpy", "2.0"}, {"torch", "2.4"}, {"system", "Darwin"},
                           {"machine", "arm64"}}},
            {"arrays", Json::object()},
        };
        for (const auto & entry : arrays) {
            const Array & array = entry.second;
            Json shape = Json::array(), reversed = Json::array();
            for (int64_t dim : array.shape) shape.push_back(dim);
            for (auto it = array.shape.rbegin(); it != array.shape.rend(); ++it) reversed.push_back(*it);
            manifest["arrays"][array.name] = Json{{"dtype", array.dtype}, {"shape", shape}, {"order", "C"},
                                                  {"ggml_ne", reversed}, {"sha256", kDigest}};
        }
    }

    // Replace an array and keep its manifest layout metadata in sync.
    void SetArray(const std::string & name, Array array) {
        arrays[name] = std::move(array);
        const Array & stored = arrays.at(name);
        Json shape = Json::array(), reversed = Json::array();
        for (int64_t dim : stored.shape) shape.push_back(dim);
        for (auto it = stored.shape.rbegin(); it != stored.shape.rend(); ++it) reversed.push_back(*it);
        manifest["arrays"][name] = Json{{"dtype", stored.dtype}, {"shape", shape}, {"order", "C"},
                                        {"ggml_ne", reversed}, {"sha256", kDigest}};
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
        (void)t2s_generation_reference::load(directory);
    } catch (const std::runtime_error & error) {
        EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << error.what();
        return;
    } catch (const std::exception & error) {
        ADD_FAILURE() << "wrong exception type: " << error.what();
        return;
    }
    ADD_FAILURE() << "load() unexpectedly succeeded";
}

TEST(T2SGenerationReferenceContract, LoadsAndValidatesEosExport) {
    Fixture fixture;
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.input.text, kT);
    EXPECT_EQ(reference.input.prompt, kP);
    EXPECT_EQ(reference.input.prefill, kN);
    EXPECT_EQ(reference.input.steps, 0);
    EXPECT_EQ(reference.input.final, 0);
    EXPECT_EQ(reference.reference_text, kR);
    EXPECT_EQ(reference.target_text, kTarget);
    EXPECT_EQ(reference.max_new_tokens, kM);
    EXPECT_EQ(reference.min_new_tokens, 0);
    EXPECT_FLOAT_EQ(reference.repetition_penalty, 1.35f);
    EXPECT_EQ(reference.stop_reason, "eos");
    EXPECT_EQ(reference.input.phones, (std::vector<int32_t>{0, 2, 7, 1}));
    EXPECT_EQ(reference.input.prompt_tokens, (std::vector<int32_t>{1, 5}));
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{4, 5}));
    EXPECT_EQ(reference.selected, (std::vector<int32_t>{4, 5, 11}));
    EXPECT_EQ(reference.context_lengths, (std::vector<int32_t>{6, 7, 8}));
    EXPECT_EQ(reference.logits.size(), static_cast<size_t>(kD * kV));
    EXPECT_EQ(reference.input.prefill_input.size(), static_cast<size_t>(kN * kH));
    EXPECT_EQ(reference.input.bert.size(), static_cast<size_t>(kT * 1024));
    gpt_sovits::t2s_model model;
    model.hparams = reference.input.hparams;
    EXPECT_NO_THROW(t2s_generation_reference::require_model(reference, model));
    model.hparams.n_layer += 1;
    EXPECT_THROW(t2s_generation_reference::require_model(reference, model), std::runtime_error);
}

TEST(T2SGenerationReferenceContract, DuplicatePromptHistoryPenalizesOnce) {
    // {5, 5} must behave exactly like a single prompt token: at step 1 the
    // accepted 4 is penalized to 2.1/1.35 while 5 is penalized once to
    // 2.5/1.35. A second application would drop 5 below 4 and reject the
    // selection, so this fixture fails if repeated IDs are penalized twice.
    Fixture fixture;
    PutI32(fixture.arrays.at("prompt.npy").data, 0, 5);
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.input.prompt_tokens, (std::vector<int32_t>{5, 5}));
    EXPECT_EQ(reference.selected, (std::vector<int32_t>{4, 5, 11}));
}

TEST(T2SGenerationReferenceContract, LoadsImmediateEosWithEmptyCodes) {
    Fixture fixture(Mode::ImmediateEos);
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_TRUE(reference.codes.empty());
    EXPECT_EQ(reference.selected, (std::vector<int32_t>{11}));
    EXPECT_EQ(reference.context_lengths, (std::vector<int32_t>{6}));
    EXPECT_EQ(reference.max_new_tokens, 1);
    EXPECT_EQ(reference.stop_reason, "eos");
}

TEST(T2SGenerationReferenceContract, LoadsMaxNewTokensExport) {
    Fixture fixture(Mode::Limit);
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.stop_reason, "max_new_tokens");
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{4, 5, 6}));
    EXPECT_EQ(reference.selected, reference.codes);
    EXPECT_EQ(reference.context_lengths, (std::vector<int32_t>{6, 7, 8}));
}

// An EOS stop that only happens after a suppression window of the requested
// size.  Raw EOS is the highest logit at every inside-window decision, so
// accepting this export proves the reader excludes it from candidate selection.
void ConfigureLongEos(Fixture & fixture, int64_t min_new_tokens) {
    const int64_t codes_count = min_new_tokens;
    const int64_t decisions = min_new_tokens + 1;
    std::vector<int32_t> codes, selected, context_lengths;
    std::vector<float> logits(static_cast<size_t>(decisions * kV), -1.0f);
    for (int64_t step = 0; step < codes_count; ++step) {
        const int32_t token = static_cast<int32_t>(step);
        codes.push_back(token);
        selected.push_back(token);
        logits[static_cast<size_t>(step * kV + token)] = 2.0f;
        logits[static_cast<size_t>(step * kV + kEOS)] = 5.0f;
    }
    selected.push_back(static_cast<int32_t>(kEOS));
    logits[static_cast<size_t>(codes_count * kV + kEOS)] = 5.0f;
    for (int64_t i = 0; i < decisions; ++i) context_lengths.push_back(static_cast<int32_t>(kN + i));
    fixture.SetArray("codes.npy", MakeArray("codes.npy", "int32", {codes_count}, Raw(codes)));
    fixture.SetArray("logits.npy", MakeArray("logits.npy", "float32", {decisions, kV}, Raw(logits)));
    fixture.SetArray("selected.npy", MakeArray("selected.npy", "int32", {decisions}, Raw(selected)));
    fixture.SetArray("context_lengths.npy",
                     MakeArray("context_lengths.npy", "int32", {decisions}, Raw(context_lengths)));
    fixture.manifest["dimensions"]["codes"] = codes_count;
    fixture.manifest["dimensions"]["decisions"] = decisions;
    fixture.manifest["policy"]["max_new_tokens"] = decisions;
    fixture.manifest["policy"]["min_new_tokens"] = min_new_tokens;
    fixture.manifest["stop_reason"] = "eos";
}

TEST(T2SGenerationReferenceContract, LoadsEosAfterOneStepSuppression) {
    Fixture fixture(Mode::Limit);
    ConfigureLongEos(fixture, 1);
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.min_new_tokens, 1);
    EXPECT_EQ(reference.max_new_tokens, 2);
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{0}));
    // Raw EOS (5.0) beats the accepted token (2.0) at decision 0.
    EXPECT_EQ(reference.selected, (std::vector<int32_t>{0, 11}));
    EXPECT_EQ(reference.stop_reason, "eos");
}

TEST(T2SGenerationReferenceContract, LoadsEosAfterElevenStepSuppression) {
    Fixture fixture(Mode::Limit);
    ConfigureLongEos(fixture, 11);
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.min_new_tokens, 11);
    EXPECT_EQ(reference.max_new_tokens, 12);
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    EXPECT_EQ(reference.selected.size(), 12u);
    EXPECT_EQ(reference.selected.back(), 11);
    EXPECT_EQ(reference.stop_reason, "eos");
}

TEST(T2SGenerationReferenceContract, LoadsMinGreaterThanMaxAsLengthStop) {
    // min > max suppresses EOS for the whole request; the length limit still
    // ends generation normally rather than rejecting a previously legal export.
    Fixture fixture(Mode::Limit);
    fixture.manifest["policy"]["min_new_tokens"] = 4;
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.min_new_tokens, 4);
    EXPECT_EQ(reference.max_new_tokens, 3);
    EXPECT_EQ(reference.stop_reason, "max_new_tokens");
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{4, 5, 6}));
}

TEST(T2SGenerationReferenceContract, LoadsMaxOneWithSuppressedEos) {
    // A one-decision request entirely inside the window: raw EOS is highest
    // but the only accepted code is the best non-EOS token.
    Fixture fixture(Mode::ImmediateEos);
    std::vector<float> logits(static_cast<size_t>(kV), -1.0f);
    logits[kEOS] = 5.0f;
    logits[4] = 2.0f;
    fixture.SetArray("codes.npy", MakeArray("codes.npy", "int32", {1}, Raw(std::vector<int32_t>{4})));
    fixture.SetArray("logits.npy", MakeArray("logits.npy", "float32", {1, kV}, Raw(logits)));
    fixture.SetArray("selected.npy", MakeArray("selected.npy", "int32", {1}, Raw(std::vector<int32_t>{4})));
    fixture.SetArray("context_lengths.npy",
                     MakeArray("context_lengths.npy", "int32", {1}, Raw(std::vector<int32_t>{kN})));
    fixture.manifest["dimensions"]["codes"] = 1;
    fixture.manifest["dimensions"]["decisions"] = 1;
    fixture.manifest["policy"]["min_new_tokens"] = 11;
    fixture.manifest["stop_reason"] = "max_new_tokens";
    fixture.Write();
    const auto reference = t2s_generation_reference::load(fixture.temp.path());
    EXPECT_EQ(reference.min_new_tokens, 11);
    EXPECT_EQ(reference.max_new_tokens, 1);
    EXPECT_EQ(reference.codes, (std::vector<int32_t>{4}));
    EXPECT_EQ(reference.selected, (std::vector<int32_t>{4}));
    EXPECT_EQ(reference.stop_reason, "max_new_tokens");
}

TEST(T2SGenerationReferenceContract, NpyHelperAllowsZeroOnlyWhenRequested) {
    TempDir temp;
    WriteFile(temp.path() / "zero.npy", EncodeNpy(MakeArray("zero.npy", "int32", {0}, {})));
    std::vector<int32_t> out;
    EXPECT_THROW((void)t2s_reference::detail::load_npy(temp.path() / "zero.npy", false, {0}, out),
                 std::runtime_error);
    EXPECT_NO_THROW(t2s_reference::detail::load_npy(temp.path() / "zero.npy", false, {0}, out, true));
    EXPECT_TRUE(out.empty());
}

TEST(T2SGenerationReferenceContract, RejectsInvalidExportsTableDriven) {
    struct ErrorCase {
        const char * name;
        Mode mode;
        void (*mutate)(Fixture &);
        const char * needle;
    };
    const ErrorCase cases[] = {
        {"schema-v1", Mode::Eos, [](Fixture & f) { f.manifest["schema"] = kSchemaV1; }, "re-export"},
        {"status", Mode::Eos, [](Fixture & f) { f.manifest["status"] = "partial"; }, "manifest.status"},
        {"policy-top-k", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["top_k"] = 2; }, "top_k"},
        {"policy-top-k-float", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["top_k"] = 1.0; }, "top_k"},
        {"policy-history", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["history"] = "all"; }, "history"},
        {"policy-history-old", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["history"] = "generated-only-binary"; }, "history"},
        {"policy-eos", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["eos"] = "masked"; }, "policy.eos"},
        {"policy-eos-old", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["eos"] = "unmasked-before-accept"; }, "re-export"},
        {"policy-min-missing", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"].erase("min_new_tokens"); }, "min_new_tokens"},
        {"policy-min-bool", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["min_new_tokens"] = true; },
         "nonnegative"},
        {"policy-min-negative", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["min_new_tokens"] = -1; }, "nonnegative"},
        {"policy-min-float", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["min_new_tokens"] = 1.0; }, "nonnegative"},
        {"policy-min-huge", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["min_new_tokens"] = 2147483648LL; }, "2147483647"},
        {"policy-penalty", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["repetition_penalty"] = 0.0; }, "repetition_penalty"},
        {"policy-reciprocal", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["repetition_penalty"] = 1e-45; }, "repetition_penalty"},
        {"policy-budget", Mode::Eos, [](Fixture & f) { f.manifest["policy"]["max_new_tokens"] = 1024; }, "budget"},
        {"eos-in-window", Mode::Eos,
         [](Fixture & f) { f.manifest["policy"]["min_new_tokens"] = 3; }, "suppression window"},
        {"stop-reason", Mode::Eos, [](Fixture & f) { f.manifest["stop_reason"] = "other"; }, "stop_reason"},
        {"stop-eos-shape", Mode::Eos, [](Fixture & f) { f.manifest["dimensions"]["codes"] = 1; }, "codes"},
        {"stop-limit-shape", Mode::Limit, [](Fixture & f) { f.manifest["dimensions"]["codes"] = 2; }, "codes"},
        {"refsplit", Mode::Eos, [](Fixture & f) { f.manifest["dimensions"]["target_text"] = 3; }, "target_text"},
        {"negative-codes", Mode::Eos, [](Fixture & f) { f.manifest["dimensions"]["codes"] = -1; }, "codes"},
        {"huge-codes", Mode::Eos, [](Fixture & f) { f.manifest["dimensions"]["codes"] = 100000; }, "codes"},
        {"dtype", Mode::Eos, [](Fixture & f) { f.arrays.at("logits.npy").descr = "<f8"; }, "dtype"},
        {"empty-codes-mismatch", Mode::Eos,
         [](Fixture & f) {
             f.arrays.at("codes.npy").shape = {0};
             f.arrays.at("codes.npy").data.clear();
             f.manifest["arrays"]["codes.npy"]["shape"] = Json::array({0});
         },
         "shape"},
        {"notfinite", Mode::Eos,
         [](Fixture & f) { PutF32(f.arrays.at("logits.npy").data, 0, std::numeric_limits<float>::quiet_NaN()); },
         "non-finite"},
        {"adjusted-overflow", Mode::Eos,
         [](Fixture & f) { PutF32(f.arrays.at("logits.npy").data, kV + 4, -std::numeric_limits<float>::max()); },
         "repetition-adjusted"},
        {"phone-id", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("phones.npy").data, 0, kPhoneVocab); },
         "phones.npy"},
        {"code-eos", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("codes.npy").data, 0, kEOS); }, "codes.npy"},
        {"context", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("context_lengths.npy").data, 1, 9); },
         "context_lengths"},
        {"first-eos", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("selected.npy").data, 2, 5); }, "EOS"},
        {"limit-eos", Mode::Limit, [](Fixture & f) { PutI32(f.arrays.at("selected.npy").data, 2, kEOS); }, "EOS"},
        {"limit-last", Mode::Limit, [](Fixture & f) { PutI32(f.arrays.at("selected.npy").data, 2, 5); }, "selected"},
        {"prompt-history", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("prompt.npy").data, 1, 4); },
         "selected"},
        {"seen-accepted", Mode::Eos,
         [](Fixture & f) {
             PutI32(f.arrays.at("codes.npy").data, 1, 4);
             PutI32(f.arrays.at("selected.npy").data, 1, 4);
         },
         "selected"},
        {"codes-prefix", Mode::Eos, [](Fixture & f) { PutI32(f.arrays.at("codes.npy").data, 0, 6); }, "codes"},
        {"sources-inputs", Mode::Eos,
         [](Fixture & f) { f.manifest["sources"]["inputs"]["generated_codes.npy"] = kDigest; }, "inputs"},
    };
    for (const ErrorCase & error_case : cases) {
        SCOPED_TRACE(error_case.name);
        Fixture fixture(error_case.mode);
        error_case.mutate(fixture);
        fixture.Write();
        ExpectLoadError(fixture.temp.path(), error_case.needle);
    }
}

} // namespace
