#!/usr/bin/env python3
"""Generate numpy reference fixtures for the C++ audio module tests.

The STFT reference mirrors upstream GPT-SoVITS `spectrogram_torch`
(GPT_SoVITS/module/mel_processing.py):

  * periodic Hann window of length win_length,
  * center=False,
  * reflect padding of (n_fft - hop_length) // 2 on both sides,
  * onesided real FFT,
  * magnitude = sqrt(re^2 + im^2 + eps).

Hyperparameters verified against the upstream configs
(GPT_SoVITS/configs/s2.json, s2v2Pro.json, s2v2ProPlus.json):
sampling_rate=32000, filter_length=n_fft=2048, win_length=2048, hop_length=640.

Both the exact input samples and the expected magnitude are written, so the
C++ test consumes the same signal the reference was computed from and the two
cannot drift apart.

Requires numpy only. If scipy is importable, an additional resample_poly
reference is emitted for the optional resampling parity test.
"""

from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parent / "ref"

# Verified upstream v2 hyperparameters.
SAMPLE_RATE = 32000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 640
STFT_EPS = 1e-6
N_SAMPLES = 16000  # 0.5 s at 32 kHz -> 25 frames


def make_test_signal(n: int) -> np.ndarray:
    """Deterministic sum of sines plus seeded noise, as float32."""
    rng = np.random.default_rng(0x5EED1234)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sig = (
        0.35 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 1234.5 * t + 0.3)
        + 0.08 * np.sin(2.0 * np.pi * 7000.0 * t + 1.1)
        + 0.01 * rng.standard_normal(n)
        + 0.02
    )
    return sig.astype(np.float32)


def periodic_hann(win_length: int) -> np.ndarray:
    i = np.arange(win_length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / win_length)


def stft_magnitude(sig_f32: np.ndarray) -> np.ndarray:
    """Return [n_frames, n_fft//2 + 1] matching audio_stft_magnitude."""
    # Compute on exactly the float32 samples the C++ test will read.
    x = sig_f32.astype(np.float64)
    pad = (N_FFT - HOP_LENGTH) // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    assert n_frames > 0, "input too short for the configured STFT"

    win = periodic_hann(WIN_LENGTH)
    if WIN_LENGTH < N_FFT:
        # torch.stft centers a short window: left pad is floor((n_fft - win)/2).
        left = (N_FFT - WIN_LENGTH) // 2
        win = np.pad(win, (left, N_FFT - WIN_LENGTH - left))

    out = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float64)
    for t in range(n_frames):
        frame = padded[t * HOP_LENGTH : t * HOP_LENGTH + N_FFT] * win
        spec = np.fft.rfft(frame)
        out[t] = np.sqrt(spec.real**2 + spec.imag**2 + STFT_EPS)
    return out


def write_stft_reference() -> None:
    sig = make_test_signal(N_SAMPLES)
    mag = stft_magnitude(sig).astype(np.float32)

    np.save(REF_DIR / "stft_input.npy", sig)
    np.save(REF_DIR / "stft_magnitude.npy", mag)

    print(
        f"stft: n={sig.shape}, frames/bins={mag.shape}, "
        f"max={mag.max():.6f}, min={mag.min():.6g}"
    )


def write_resample_reference() -> None:
    """Optional scipy.signal.resample_poly parity fixture."""
    try:
        from scipy.signal import resample_poly
    except ImportError:
        print("resample: scipy not available, skipping resample_poly reference")
        return

    src_rate, dst_rate = 44100, 16000
    n = src_rate  # 1 second
    t = np.arange(n, dtype=np.float64) / src_rate
    duration = n / src_rate
    f0, f1 = 100.0, 6000.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t * t)
    sig = np.sin(phase).astype(np.float32)

    ref = resample_poly(sig.astype(np.float64), dst_rate, src_rate)
    np.save(REF_DIR / "resample_input_44k.npy", sig)
    np.save(REF_DIR / "resample_poly_16k.npy", ref.astype(np.float32))
    print(
        f"resample: {sig.shape} @ {src_rate} -> {ref.shape} @ {dst_rate} "
        "(scipy.signal.resample_poly)"
    )


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    write_stft_reference()
    write_resample_reference()


if __name__ == "__main__":
    main()
