// WAV / FLAC decoding (dr_libs) and float32 WAV writing.
//
// dr_libs are header-only; the two implementations live in this translation
// unit so the rest of the project only includes gpt_sovits/audio.h.

#define DR_WAV_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION

#include "../../third_party/dr_libs/dr_wav.h"
#include "../../third_party/dr_libs/dr_flac.h"

#include "gpt_sovits/audio.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace gpt_sovits {

namespace {

static std::string lower_ascii(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

static std::string extension_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return "";
    }
    return lower_ascii(path.substr(dot));
}

// Average interleaved float frames down to mono.
static bool mix_to_mono(const float * interleaved, uint64_t frames,
                        unsigned channels, std::vector<float> & out) {
    if (channels == 0) {
        fprintf(stderr, "%s: zero channels\n", __func__);
        return false;
    }
    if (frames > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / channels) {
        fprintf(stderr, "%s: decoded size overflow\n", __func__);
        return false;
    }
    const size_t n = static_cast<size_t>(frames);
    out.resize(n);
    if (channels == 1) {
        for (size_t i = 0; i < n; ++i) out[i] = interleaved[i];
    } else {
        const float inv = 1.0f / static_cast<float>(channels);
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            const float * p = interleaved + i * channels;
            for (unsigned c = 0; c < channels; ++c) sum += p[c];
            out[i] = sum * inv;
        }
    }
    return true;
}

static bool read_wav(const std::string & path, std::vector<float> & out, int & sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "%s: failed to open wav '%s'\n", __func__, path.c_str());
        return false;
    }

    const drwav_uint64 frames   = wav.totalPCMFrameCount;
    const unsigned     channels = wav.channels;
    const drwav_uint32 rate     = wav.sampleRate;

    std::vector<float> interleaved;
    drwav_uint64 got = 0;
    if (frames > 0 && channels > 0) {
        if (frames > static_cast<drwav_uint64>(std::numeric_limits<size_t>::max()) / channels) {
            fprintf(stderr, "%s: decoded size overflow in '%s'\n", __func__, path.c_str());
            drwav_uninit(&wav);
            return false;
        }
        interleaved.resize(static_cast<size_t>(frames * channels));
        while (got < frames) {
            const drwav_uint64 r = drwav_read_pcm_frames_f32(
                &wav, frames - got, interleaved.data() + static_cast<size_t>(got) * channels);
            if (r == 0) break;
            got += r;
        }
    }
    drwav_uninit(&wav);

    if (frames > 0 && got == 0) {
        fprintf(stderr, "%s: failed to decode wav '%s'\n", __func__, path.c_str());
        return false;
    }
    if (rate > static_cast<drwav_uint32>(INT_MAX)) {
        fprintf(stderr, "%s: sample rate out of range in '%s'\n", __func__, path.c_str());
        return false;
    }
    if (!mix_to_mono(interleaved.data(), got, channels, out)) return false;
    sample_rate = static_cast<int>(rate);
    return true;
}

static bool read_flac(const std::string & path, std::vector<float> & out, int & sample_rate) {
    drflac * flac = drflac_open_file(path.c_str(), nullptr);
    if (!flac) {
        fprintf(stderr, "%s: failed to open flac '%s'\n", __func__, path.c_str());
        return false;
    }

    const drflac_uint64 frames   = flac->totalPCMFrameCount;
    const unsigned      channels = flac->channels;
    const drflac_uint32 rate     = flac->sampleRate;

    std::vector<float> interleaved;
    drflac_uint64 got = 0;
    if (frames > 0 && channels > 0) {
        if (frames > static_cast<drflac_uint64>(std::numeric_limits<size_t>::max()) / channels) {
            fprintf(stderr, "%s: decoded size overflow in '%s'\n", __func__, path.c_str());
            drflac_close(flac);
            return false;
        }
        interleaved.resize(static_cast<size_t>(frames * channels));
        while (got < frames) {
            const drflac_uint64 r = drflac_read_pcm_frames_f32(
                flac, frames - got, interleaved.data() + static_cast<size_t>(got) * channels);
            if (r == 0) break;
            got += r;
        }
    }
    drflac_close(flac);

    if (frames > 0 && got == 0) {
        fprintf(stderr, "%s: failed to decode flac '%s'\n", __func__, path.c_str());
        return false;
    }
    if (rate > static_cast<drflac_uint32>(INT_MAX)) {
        fprintf(stderr, "%s: sample rate out of range in '%s'\n", __func__, path.c_str());
        return false;
    }
    if (!mix_to_mono(interleaved.data(), got, channels, out)) return false;
    sample_rate = static_cast<int>(rate);
    return true;
}

} // namespace

bool audio_read_file(const std::string & path, std::vector<float> & out, int & sample_rate) {
    out.clear();
    sample_rate = 0;
    if (path.empty()) {
        fprintf(stderr, "%s: empty path\n", __func__);
        return false;
    }

    const std::string ext = extension_of(path);
    if (ext == ".wav") return read_wav(path, out, sample_rate);
    if (ext == ".flac") return read_flac(path, out, sample_rate);

    fprintf(stderr, "%s: unsupported audio extension '%s' in '%s'\n",
            __func__, ext.c_str(), path.c_str());
    return false;
}

bool audio_write_wav_f32(const std::string & path, const float * samples,
                         int64_t n_samples, int sample_rate) {
    if (path.empty() || !samples || n_samples <= 0 || sample_rate <= 0) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return false;
    }
    if (static_cast<uint64_t>(n_samples) >
        (static_cast<uint64_t>(UINT32_MAX) - 36u) / sizeof(float)) {
        fprintf(stderr, "%s: audio too long for a 32-bit RIFF file\n", __func__);
        return false;
    }
    const uint64_t byte_rate64 =
        static_cast<uint64_t>(sample_rate) * 1u * 4u;   // mono, 32-bit samples
    if (byte_rate64 > UINT32_MAX) {
        fprintf(stderr, "%s: sample rate too large\n", __func__);
        return false;
    }

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot open '%s' for writing\n", __func__, path.c_str());
        return false;
    }

    const uint16_t n_chan        = 1;
    const uint16_t bits_per_samp = 32;
    const uint32_t byte_rate     = static_cast<uint32_t>(byte_rate64);
    const uint16_t block_align   = n_chan * (bits_per_samp / 8);
    const uint32_t data_size     = static_cast<uint32_t>(n_samples * sizeof(float));

    #pragma pack(push, 1)
    struct WavHeader {
        char     riff[4];        // "RIFF"
        uint32_t riff_size;      // 36 + data_size
        char     wave[4];        // "WAVE"
        char     fmt_[4];        // "fmt "
        uint32_t fmt_size;       // 16
        uint16_t audio_fmt;      // 3 = IEEE float
        uint16_t n_chan;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_samp;
        char     data[4];        // "data"
        uint32_t data_size;
    } header;
    #pragma pack(pop)

    static_assert(sizeof(header) == 44, "WAV header must be 44 bytes");

    memcpy(header.riff,      "RIFF", 4);
    header.riff_size     = 36 + data_size;
    memcpy(header.wave,      "WAVE", 4);
    memcpy(header.fmt_,      "fmt ", 4);
    header.fmt_size      = 16;
    header.audio_fmt     = 3;
    header.n_chan        = n_chan;
    header.sample_rate   = static_cast<uint32_t>(sample_rate);
    header.byte_rate     = byte_rate;
    header.block_align   = block_align;
    header.bits_per_samp = bits_per_samp;
    memcpy(header.data,      "data", 4);
    header.data_size     = data_size;

    bool ok = (fwrite(&header, sizeof(header), 1, f) == 1) &&
              (fwrite(samples, sizeof(float), static_cast<size_t>(n_samples), f) ==
               static_cast<size_t>(n_samples));
    if (fclose(f) != 0) ok = false;
    if (!ok) {
        fprintf(stderr, "%s: failed writing '%s'\n", __func__, path.c_str());
    }
    return ok;
}

} // namespace gpt_sovits
