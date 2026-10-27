#include "dr_wav.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

uint64_t count(const std::string & text) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "expected an unsigned integer");
    return std::stoull(text);
}

struct Summary {
    uint64_t codes = 0, samples = 0, sample_rate = 0;
    std::string stop;
};

Summary read_summary(const char * path) {
    std::ifstream input(path);
    require(input.is_open(), "cannot open CLI stderr log");
    const std::regex pattern(
        R"(^wrote '.*': ([0-9]+) codes \(stop: ([a-z_]+)\), ([0-9]+) samples \([^)]*\) at ([0-9]+) Hz\r?$)");
    Summary result;
    size_t matches = 0;
    std::string line;
    while (std::getline(input, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, pattern)) continue;
        ++matches;
        result.codes = count(match[1].str());
        result.stop = match[2].str();
        result.samples = count(match[3].str());
        result.sample_rate = count(match[4].str());
    }
    require(input.eof(), "cannot read CLI stderr log");
    require(matches == 1, "expected exactly one CLI success summary");
    return result;
}

uint32_t read_length(std::ifstream & input, uint64_t offset) {
    require(offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
            "WAV length field offset is out of range");
    std::array<unsigned char, 4> bytes;
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    require(input.good(), "cannot read WAV length field");
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

struct WavReader {
    drwav wav{};

    explicit WavReader(const char * path) {
        require(drwav_init_file(&wav, path, nullptr) != DRWAV_FALSE, "cannot open output WAV");
    }
    ~WavReader() { drwav_uninit(&wav); }
    WavReader(const WavReader &) = delete;
    WavReader & operator=(const WavReader &) = delete;
};

void check(const char * wav_path, const char * log_path, uint64_t max_new_tokens) {
    require(max_new_tokens > 0 && max_new_tokens <= 1024,
            "max_new_tokens must be in [1, 1024]");
    const Summary summary = read_summary(log_path);
    require(summary.codes > 0 && summary.codes <= max_new_tokens,
            "CLI code count is outside the generation budget");
    require((summary.stop == "eos" && summary.codes < max_new_tokens) ||
            (summary.stop == "max_new_tokens" && summary.codes == max_new_tokens),
            "CLI stop reason disagrees with the code count");
    require(summary.samples == summary.codes * 1280,
            "CLI sample count must equal codes * 1280");
    require(summary.sample_rate == 32000, "CLI summary sample rate must be 32000 Hz");

    WavReader reader(wav_path);
    drwav & wav = reader.wav;
    require(wav.container == drwav_container_riff &&
            wav.translatedFormatTag == DR_WAVE_FORMAT_IEEE_FLOAT && wav.bitsPerSample == 32,
            "output must be a RIFF IEEE-float32 WAV");
    require(wav.channels == 1 && wav.sampleRate == 32000,
            "output WAV must be mono at 32000 Hz");
    require(wav.fmt.blockAlign == 4 && wav.fmt.avgBytesPerSec == 128000,
            "output WAV has inconsistent byte rate or block alignment");
    require(wav.totalPCMFrameCount == summary.samples &&
            wav.dataChunkDataSize == summary.samples * sizeof(float),
            "WAV data length disagrees with the CLI summary");
    const uintmax_t file_size = std::filesystem::file_size(wav_path);
    require(wav.dataChunkDataPos >= 4 && wav.dataChunkDataPos <= file_size &&
            wav.dataChunkDataSize == file_size - wav.dataChunkDataPos,
            "WAV data size does not match the file size");
    // dr_wav repairs oversized and unaligned chunks; check the on-disk sizes too.
    std::ifstream raw(wav_path, std::ios::binary);
    require(raw.is_open(), "cannot open WAV for length validation");
    require(static_cast<uint64_t>(read_length(raw, 4)) + 8 == file_size,
            "WAV RIFF length does not match the file size");
    require(read_length(raw, wav.dataChunkDataPos - 4) == summary.samples * sizeof(float),
            "WAV declared data length disagrees with the CLI summary");

    std::array<float, 4096> buffer;
    uint64_t remaining = summary.samples;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo, squared = 0.0;
    while (remaining > 0) {
        const uint64_t requested = std::min<uint64_t>(remaining, buffer.size());
        const uint64_t got = drwav_read_pcm_frames_f32(&wav, requested, buffer.data());
        require(got == requested, "output WAV is truncated or unreadable");
        for (uint64_t i = 0; i < got; ++i) {
            const double value = buffer[i];
            require(std::isfinite(value), "output WAV contains non-finite samples");
            lo = std::min(lo, value);
            hi = std::max(hi, value);
            squared += value * value;
        }
        remaining -= got;
    }
    const double rms = std::sqrt(squared / static_cast<double>(summary.samples));
    require(lo < 0.0 && hi > 0.0 && hi - lo > 1.0e-3 && rms > 1.0e-8,
            "output WAV is silent or degenerate");
    std::cout << "validated CLI WAV: " << summary.codes << " codes (stop: " << summary.stop
              << "), " << summary.samples << " samples at " << summary.sample_rate
              << " Hz, range=[" << lo << ", " << hi << "], rms=" << rms << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--skip") {
        std::cout << "CLI synthesis skipped: GPTSOVITS_REFERENCE_CASE is empty\n";
        return 77;
    }
    if (argc != 4) {
        std::cerr << "usage: cli_synthesis_check <wav> <stderr-log> <max-new-tokens>\n";
        return 2;
    }
    try {
        check(argv[1], argv[2], count(argv[3]));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "CLI synthesis validation: " << error.what() << '\n';
        return 1;
    }
}
