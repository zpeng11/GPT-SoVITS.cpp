#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Host-side audio DSP primitives.
//
// This module is plain host code (no ggml dependency) shared by the frontends:
// the SV fbank path and the SoVITS MelStyleEncoder reference spectrogram.
// ---------------------------------------------------------------------------

// In-place radix-2 Cooley-Tukey FFT. `n` must be a positive power of two;
// re/im are both arrays of n values. Invalid n (or null pointers) returns
// false without touching the buffers. Used by audio_stft_magnitude and by
// sv_fbank so both frontends share one implementation.
bool audio_fft_radix2(float * re, float * im, int64_t n);

// Linear magnitude STFT matching upstream GPT-SoVITS `spectrogram_torch`
// (GPT_SoVITS/module/mel_processing.py):
//
//   * periodic Hann window of length win_length
//       w[i] = 0.5 - 0.5 * cos(2*pi*i / win_length)
//     when win_length < n_fft it is zero-padded to n_fft with the window
//     centered (n_fft - win_length) / 2 samples from the left, matching
//     torch.stft's "pad center" behavior
//   * center = false
//   * reflect-pad (n_fft - hop_length) / 2 samples on BOTH sides before
//     framing (integer division, exactly like the upstream int(...))
//   * onesided real FFT of each n_fft frame
//   * magnitude = sqrt(re^2 + im^2 + 1e-6)
//
// The v2/v2Pro/v2ProPlus mel_processing uses 1e-6; later v4 revisions change
// the epsilon to 1e-8. This module follows the v2 value used by the targeted
// SoVITS v2 reference path.
//
// Upstream v2 hyperparameters (verified against GPT_SoVITS/configs/s2.json,
// s2v2Pro.json and s2v2ProPlus.json in the upstream checkout):
//   sampling_rate = 32000, n_fft = filter_length = 2048,
//   win_length = 2048, hop_length = 640.
// The 640 hop is 20 ms at 32 kHz and matches the generator stride product
// (10*8*2*2*2); the stored v2Pro reference spectrum [1,1025,T] is consistent
// with it.
//
// Output uses the ggml {n_bins = n_fft/2 + 1, n_frames} convention with the
// channel-fast layout: out[t * n_bins + k]. `out` is resized (and fully
// overwritten). `out_frames`, when non-null, receives the number of frames.
//
// Returns false on invalid arguments: null pcm, n_samples <= 0, n_fft not a
// positive power of two, hop_length <= 0, win_length <= 0 or > n_fft,
// n_fft < hop_length, or a reflect padding width that is not smaller than
// n_samples (PyTorch/numpy reflect padding requires pad < length).
bool audio_stft_magnitude(const float * pcm, int64_t n_samples,
                          int n_fft, int hop_length, int win_length,
                          std::vector<float> & out, int64_t * out_frames = nullptr);

// Read a .wav or .flac file (decided by the case-insensitive extension),
// convert to mono by averaging all channels, and return float32 samples in
// [-1, 1]-ish. `out` is resized and `sample_rate` set on success. Returns
// false on failure (missing file, unsupported extension, decode error).
bool audio_read_file(const std::string & path, std::vector<float> & out, int & sample_rate);

// Write a mono IEEE-float32 WAV (WAVE_FORMAT_IEEE_FLOAT, 16-byte fmt chunk).
// Returns false on failure (bad arguments, file I/O error, or a data chunk
// that would exceed the 32-bit RIFF size field).
bool audio_write_wav_f32(const std::string & path, const float * samples, int64_t n_samples, int sample_rate);

// Bandlimited resampling for arbitrary rational ratios (polyphase
// windowed-sinc). The low-pass cutoff is min(src_rate, dst_rate)/2, the FIR
// has ~10 zero crossings on each side of the main lobe and uses a Kaiser
// beta=5 window. src_rate == dst_rate is an exact copy. `out` is resized by
// the function. Returns false on invalid arguments: rates <= 0, null input,
// or n_in <= 0.
bool audio_resample(const float * in, int64_t n_in, int src_rate, int dst_rate, std::vector<float> & out);

// Prepared reference inputs for the v2/v2Pro/v2ProPlus synthesis path.
struct audio_reference_data {
    std::vector<float> pcm16k;             // SV: resampled PCM, not normalized
    std::vector<float> normalized_pcm16k;  // HuBERT: same length as pcm16k
    std::vector<float> spec704;            // SoVITS: spec704[t * 704 + c]
    int64_t spec_frames = 0;
};

// Prepare reference inputs from decoded mono PCM. Resample directly from the
// source to 16 kHz (SV / HuBERT) and 32 kHz (SoVITS), without silence padding,
// trimming, gain adjustment or clipping. Samples need not be within [-1, 1],
// but inputs and computed outputs must be finite.
//
// HuBERT normalization uses population variance and (x - mean) / sqrt(var +
// 1e-7). The SoVITS spectrum uses unnormalized 32 kHz PCM, n_fft=win_length=2048,
// hop_length=640 and the first 704 bins of audio_stft_magnitude. On success,
// spec_frames > 0 and spec704.size() == 704 * spec_frames. At least 400 samples
// at 16 kHz are required for HuBERT / SV; this is not an audio-quality minimum.
//
// Input is borrowed only for the call and is not modified before replacement
// of out; it may point into out's existing vectors. Success replaces out and
// invalidates its old buffers. Failure preserves out, including its buffers.
// Returns false with diagnostics on invalid arguments, unsupported resampling,
// non-finite data, size overflow or host allocation failure. No models needed.
// File decoding / channel mixing remain the job of audio_read_file.
// This host-side preparation does not establish end-to-end inference parity.
//
// The caller must supply n_samples readable floats at pcm.
bool audio_prepare_reference(const float * pcm, int64_t n_samples, int sample_rate,
                             audio_reference_data & out);

} // namespace gpt_sovits
