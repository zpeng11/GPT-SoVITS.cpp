// cli/main.cpp
//
// Single-request offline CPU synthesis command for GPT-SoVITS.cpp.
//
// Loads the five v2Pro GGUF models onto one shared CPU backend, reads the
// reference audio and two text-frontend JSON files, runs synthesize_cpu and
// writes the raw 32 kHz mono F32 result as an IEEE-float WAV. This is a thin
// shell over the public library API: text processing, resampling, clipping,
// streaming and GPU paths do not live here. The synthesis contract, including
// its deliberate non-parity with the Python sampler, is documented in
// include/gpt_sovits/pipeline.h.
//
// Exit codes: 0 success, 1 runtime failure, 2 usage error.

#include "gpt_sovits/audio.h"
#include "gpt_sovits/hubert.h"
#include "gpt_sovits/pipeline.h"
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"
#include "gpt_sovits/t2s.h"
#include "gpt_sovits/text_frontend.h"

#include "ggml-cpu.h"

#include <sys/stat.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage   = 2;

struct cli_options {
    std::string hubert;
    std::string roberta;
    std::string t2s;
    std::string sovits;
    std::string sv;
    std::string ref_audio;
    std::string ref_frontend_json;
    std::string target_frontend_json;
    std::string output;
    bool overwrite = false;

    // Defaults are the library-side defaults from synthesis_config.
    gpt_sovits::synthesis_config config;
    bool have_roberta = false;
};

bool parse_int64(const std::string & text, int64_t & out) {
    if (text.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size()) return false;
    out = static_cast<int64_t>(value);
    return true;
}

bool parse_uint64(const std::string & text, uint64_t & out) {
    if (text.empty() || text[0] == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size()) return false;
    out = static_cast<uint64_t>(value);
    return true;
}

bool parse_float(const std::string & text, float & out) {
    if (text.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size()) return false;
    const float narrowed = static_cast<float>(value);
    if (!std::isfinite(narrowed)) return false;
    out = narrowed;
    return true;
}

void print_usage(std::FILE * out, const char * program) {
    std::fprintf(out,
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  --hubert F                CN-HuBERT GGUF\n"
        "  --t2s F                   T2S GGUF\n"
        "  --sovits F                SoVITS GGUF\n"
        "  --sv F                    speaker-verification GGUF\n"
        "  --ref-audio F             reference audio (.wav/.flac, any rate)\n"
        "  --ref-frontend-json F     reference text frontend JSON (v2 contract)\n"
        "  --target-frontend-json F  target text frontend JSON (v2 contract)\n"
        "  --output F                output mono float32 WAV path (32 kHz)\n"
        "\n"
        "Optional:\n"
        "  --roberta F               chinese-roberta GGUF; needed only when either\n"
        "                            frontend JSON contains Chinese segments\n"
        "  --overwrite               allow replacing an existing --output file\n"
        "  --max-new-tokens N        default 200; N in [1, 1024]\n"
        "  --seed N                  T2S sampling seed (default 42)\n"
        "  --noise-seed N            SoVITS noise seed (default 7)\n"
        "  --noise-scale F           default 0.5; must be positive\n"
        "  --top-k K                 default 15; K <= 0 disables top-k filtering\n"
        "  --top-p P                 default 1.0; P >= 1 disables top-p filtering\n"
        "  --temperature F           default 1.0\n"
        "  --repetition-penalty F    default 1.35; must be positive\n"
        "  -h, --help                print this help\n"
        "\n"
        "Output is raw 32 kHz mono F32 with codes.size() * 1280 samples, written\n"
        "only after synthesis succeeds. Stop reason eos means natural termination;\n"
        "max_new_tokens means the text may be truncated.\n"
        "Exit codes: 0 success, 1 failure, 2 usage error.\n",
        program);
}

enum class parse_result { ok, help, error };

parse_result parse_args(int argc, char ** argv, cli_options & opts, std::string & error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") return parse_result::help;
        if (arg == "--overwrite") {
            opts.overwrite = true;
            continue;
        }
        if (arg.size() <= 2 || arg.compare(0, 2, "--") != 0) {
            error = "unexpected argument '" + arg + "'";
            return parse_result::error;
        }

        const size_t eq = arg.find('=');
        const std::string flag = arg.substr(0, eq == std::string::npos ? arg.size() : eq);
        const bool has_inline = eq != std::string::npos;
        const std::string inline_value = has_inline ? arg.substr(eq + 1) : std::string();

        std::string value;
        auto next_value = [&](std::string & out_value) -> bool {
            if (has_inline) {
                out_value = inline_value;
                return true;
            }
            if (i + 1 >= argc) {
                error = "missing value for " + flag;
                return false;
            }
            out_value = argv[++i];
            return true;
        };
        auto bad = [&](const char * what) {
            error = flag + " expects " + what + ", got '" + value + "'";
            return parse_result::error;
        };

        if (flag == "--hubert") {
            if (!next_value(value)) return parse_result::error;
            opts.hubert = value;
        } else if (flag == "--roberta") {
            if (!next_value(value)) return parse_result::error;
            opts.roberta = value;
            opts.have_roberta = true;
        } else if (flag == "--t2s") {
            if (!next_value(value)) return parse_result::error;
            opts.t2s = value;
        } else if (flag == "--sovits") {
            if (!next_value(value)) return parse_result::error;
            opts.sovits = value;
        } else if (flag == "--sv") {
            if (!next_value(value)) return parse_result::error;
            opts.sv = value;
        } else if (flag == "--ref-audio") {
            if (!next_value(value)) return parse_result::error;
            opts.ref_audio = value;
        } else if (flag == "--ref-frontend-json") {
            if (!next_value(value)) return parse_result::error;
            opts.ref_frontend_json = value;
        } else if (flag == "--target-frontend-json") {
            if (!next_value(value)) return parse_result::error;
            opts.target_frontend_json = value;
        } else if (flag == "--output") {
            if (!next_value(value)) return parse_result::error;
            opts.output = value;
        } else if (flag == "--max-new-tokens") {
            if (!next_value(value)) return parse_result::error;
            int64_t v = 0;
            if (!parse_int64(value, v) || v < 1 || v > 1024)
                return bad("an integer in [1, 1024]");
            opts.config.generate.max_new_tokens = static_cast<int32_t>(v);
        } else if (flag == "--seed") {
            if (!next_value(value)) return parse_result::error;
            int64_t v = 0;
            if (!parse_int64(value, v) || v < 0 || v > 0xffffffffLL)
                return bad("an integer in [0, 4294967295]");
            opts.config.generate.seed = static_cast<uint32_t>(v);
        } else if (flag == "--noise-seed") {
            if (!next_value(value)) return parse_result::error;
            uint64_t v = 0;
            if (!parse_uint64(value, v)) return bad("an integer in [0, 2^64-1]");
            opts.config.noise_seed = v;
        } else if (flag == "--noise-scale") {
            if (!next_value(value)) return parse_result::error;
            float v = 0.0f;
            if (!parse_float(value, v) || v <= 0.0f) return bad("a positive finite number");
            opts.config.noise_scale = v;
        } else if (flag == "--top-k") {
            if (!next_value(value)) return parse_result::error;
            int64_t v = 0;
            if (!parse_int64(value, v) || v < INT32_MIN || v > INT32_MAX)
                return bad("an integer in [-2147483648, 2147483647]");
            opts.config.sampler.top_k = static_cast<int>(v);
        } else if (flag == "--top-p") {
            if (!next_value(value)) return parse_result::error;
            float v = 0.0f;
            if (!parse_float(value, v)) return bad("a finite number");
            opts.config.sampler.top_p = v;
        } else if (flag == "--temperature") {
            if (!next_value(value)) return parse_result::error;
            float v = 0.0f;
            if (!parse_float(value, v)) return bad("a finite number");
            opts.config.sampler.temperature = v;
        } else if (flag == "--repetition-penalty") {
            if (!next_value(value)) return parse_result::error;
            float v = 0.0f;
            if (!parse_float(value, v) || v <= 0.0f) return bad("a positive finite number");
            opts.config.sampler.repetition_penalty = v;
        } else {
            error = "unknown option '" + flag + "'";
            return parse_result::error;
        }
    }

    struct required_flag {
        const char * name;
        const std::string & value;
    };
    const required_flag required[] = {
        {"--hubert", opts.hubert},
        {"--t2s", opts.t2s},
        {"--sovits", opts.sovits},
        {"--sv", opts.sv},
        {"--ref-audio", opts.ref_audio},
        {"--ref-frontend-json", opts.ref_frontend_json},
        {"--target-frontend-json", opts.target_frontend_json},
        {"--output", opts.output},
    };
    for (const required_flag & r : required) {
        if (r.value.empty()) {
            error = std::string("missing required option ") + r.name;
            return parse_result::error;
        }
    }
    return parse_result::ok;
}

bool file_readable(const char * flag, const std::string & path, std::string & error) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = std::string("cannot read ") + flag + " file '" + path + "'";
        return false;
    }
    std::fclose(f);
    return true;
}

// All checks that must run before the CPU backend is created or any model is
// loaded, so bad paths fail in milliseconds instead of after a full load.
bool validate_paths(const cli_options & opts, std::string & error) {
    if (!file_readable("--hubert", opts.hubert, error)) return false;
    if (opts.have_roberta && !file_readable("--roberta", opts.roberta, error)) return false;
    if (!file_readable("--t2s", opts.t2s, error)) return false;
    if (!file_readable("--sovits", opts.sovits, error)) return false;
    if (!file_readable("--sv", opts.sv, error)) return false;
    if (!file_readable("--ref-audio", opts.ref_audio, error)) return false;
    if (!file_readable("--ref-frontend-json", opts.ref_frontend_json, error)) return false;
    if (!file_readable("--target-frontend-json", opts.target_frontend_json, error)) return false;

    std::FILE * existing = std::fopen(opts.output.c_str(), "rb");
    if (existing) {
        std::fclose(existing);
        if (!opts.overwrite) {
            error = "output file '" + opts.output +
                    "' exists (use --overwrite to replace it)";
            return false;
        }
    }
    const size_t slash = opts.output.find_last_of('/');
    const std::string dir = slash == std::string::npos
                                ? "."
                                : opts.output.substr(0, slash == 0 ? 1 : slash);
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        error = "output directory '" + dir + "' does not exist";
        return false;
    }
    return true;
}

int run(const cli_options & opts) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "error: failed to initialize the CPU backend\n");
        return kExitFailure;
    }

    gpt_sovits::hubert_model   hubert;
    gpt_sovits::roberta_model  roberta;
    gpt_sovits::t2s_model      t2s;
    gpt_sovits::sovits_model   sovits;
    gpt_sovits::sv_model       sv;

    // Load in a fixed order; the loaders report their own stderr diagnostics.
    const bool hubert_ok  = gpt_sovits::hubert_model_load(opts.hubert, hubert, backend);
    const bool roberta_ok = hubert_ok &&
                           (!opts.have_roberta ||
                            gpt_sovits::roberta_model_load(opts.roberta, roberta, backend));
    const bool t2s_ok     = roberta_ok && gpt_sovits::t2s_model_load(opts.t2s, t2s, backend);
    const bool sovits_ok  = t2s_ok && gpt_sovits::sovits_model_load(opts.sovits, sovits, backend);
    const bool sv_ok      = sovits_ok && gpt_sovits::sv_model_load(opts.sv, sv, backend);

    auto fail = [&]() -> int {
        const char * stage = !hubert_ok  ? "--hubert"
                           : !roberta_ok ? "--roberta"
                           : !t2s_ok     ? "--t2s"
                           : !sovits_ok  ? "--sovits"
                                         : "--sv";
        std::fprintf(stderr, "error: failed to load the %s model\n", stage);
        if (sovits_ok) gpt_sovits::sovits_model_free(sovits);
        if (t2s_ok) gpt_sovits::t2s_model_free(t2s);
        if (roberta_ok) gpt_sovits::roberta_model_free(roberta);
        if (hubert_ok) gpt_sovits::hubert_model_free(hubert);
        ggml_backend_free(backend);
        return kExitFailure;
    };
    if (!sv_ok) return fail();

    auto free_all = [&]() {
        gpt_sovits::sv_model_free(sv);
        gpt_sovits::sovits_model_free(sovits);
        gpt_sovits::t2s_model_free(t2s);
        if (opts.have_roberta) gpt_sovits::roberta_model_free(roberta);
        gpt_sovits::hubert_model_free(hubert);
        ggml_backend_free(backend);
    };

    std::vector<float> pcm;
    int sample_rate = 0;
    if (!gpt_sovits::audio_read_file(opts.ref_audio, pcm, sample_rate)) {
        std::fprintf(stderr, "error: failed to decode reference audio '%s'\n",
                     opts.ref_audio.c_str());
        free_all();
        return kExitFailure;
    }

    gpt_sovits::text_frontend_result ref_text;
    gpt_sovits::text_frontend_result target_text;
    std::string error;
    if (!gpt_sovits::text_frontend_read_json(opts.ref_frontend_json, ref_text, &error) ||
        !gpt_sovits::text_frontend_read_json(opts.target_frontend_json, target_text, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        free_all();
        return kExitFailure;
    }

    gpt_sovits::synthesis_models models;
    models.hubert  = &hubert;
    models.roberta = opts.have_roberta ? &roberta : nullptr;
    models.t2s     = &t2s;
    models.sovits  = &sovits;
    models.sv      = &sv;

    gpt_sovits::synthesis_result result;
    if (!gpt_sovits::synthesize_cpu(models, pcm.data(), static_cast<int64_t>(pcm.size()),
                                    sample_rate, ref_text, target_text, result,
                                    opts.config, &error)) {
        std::fprintf(stderr, "error: synthesis failed: %s\n", error.c_str());
        free_all();
        return kExitFailure;
    }

    if (!gpt_sovits::audio_write_wav_f32(opts.output, result.samples.data(),
                                         static_cast<int64_t>(result.samples.size()),
                                         result.sample_rate)) {
        std::fprintf(stderr, "error: failed to write output WAV '%s'\n",
                     opts.output.c_str());
        free_all();
        return kExitFailure;
    }

    const bool truncated = result.stop_reason == gpt_sovits::t2s_stop_reason::max_new_tokens;
    if (truncated) {
        std::fprintf(stderr,
                     "warning: stopped at --max-new-tokens=%d; the audio may be "
                     "truncated mid-sentence\n",
                     opts.config.generate.max_new_tokens);
    }
    const double seconds =
        static_cast<double>(result.samples.size()) / static_cast<double>(result.sample_rate);
    std::fprintf(stderr, "wrote '%s': %zu codes (stop: %s), %zu samples (%.2f s) at %d Hz\n",
                 opts.output.c_str(), result.codes.size(),
                 truncated ? "max_new_tokens" : "eos", result.samples.size(), seconds,
                 result.sample_rate);

    free_all();
    return kExitSuccess;
}

const char * program_name(const char * argv0) {
    const char * fallback = "gpt_sovits_cli";
    if (!argv0 || !*argv0) return fallback;
    const char * slash = std::strrchr(argv0, '/');
    return slash && slash[1] ? slash + 1 : argv0;
}

} // namespace

int main(int argc, char ** argv) {
    cli_options opts;
    std::string error;
    const parse_result status = parse_args(argc, argv, opts, error);
    if (status == parse_result::help) {
        print_usage(stdout, program_name(argv[0]));
        return kExitSuccess;
    }
    if (status == parse_result::error) {
        std::fprintf(stderr, "error: %s\n\n", error.c_str());
        print_usage(stderr, program_name(argv[0]));
        return kExitUsage;
    }
    if (!validate_paths(opts, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return kExitFailure;
    }
    return run(opts);
}
