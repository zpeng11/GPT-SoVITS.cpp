#!/usr/bin/env python3
"""Dump Python reference fixtures for the ERes2NetV2 SV model (forward3 path).

Default outputs (written to the output directory, default tests/sovits/ref/):

    sv_audio16k.npy  deterministic 16 kHz mono waveform fed to fbank, so the
                     C++ frontend gets the exact same samples for parity
    sv_fbank.npy     fbank features {m, 80}  (Kaldi.fbank defaults, 80 mel)
    sv_emb.npy       ERes2NetV2.forward3 output {1, 20480}

With --save-intermediates (developer/debug only; larger files):

    sv_fuse34.npy    fuse_out34 {1, 2048, F', T'}  (pre-flatten-mean)
    sv_out1..4.npy   per-layer outputs for block-level parity

With --stress (regression for the Hardtanh(0, 20) first-layer clamp):

    sv_stress_fbank.npy  synthetic fbank whose first-layer pre-activation
                         exceeds 20 (the upstream forward3 keeps it, since it
                         uses plain F.relu there)
    sv_stress_emb.npy    matching forward3 embedding {1, 20480}

The architecture is FIXED to the one actually implemented and supported by
convert_sv_to_gguf.py and the C++ runtime (src/sv/sv.cpp): ERes2NetV2 with
baseWidth=24, scale=4, expansion=4. No other architecture is offered because
nothing else is wired up downstream.

torchaudio is unavailable in this environment, so the fbank frontend is inlined
(pure torch, replicating the kaldi.fbank defaults used by SV.compute_embedding3).
This is exactly what the C++ sv_fbank port must reproduce.

NOTE: developer-only fixture generator. It imports torch and runs the real
ERes2NetV2 model, so it is not needed at deploy time: end users only need
convert_sv_to_gguf.py (torch-free) plus the C++ runtime.

Usage:
    python python/dump_sv_ref.py \\
        --eres2net-dir /path/to/GPT_SoVITS/eres2net \\
        --ckpt /path/to/pretrained_eres2netv2w24s4ep4.ckpt \\
        [--out-dir tests/sovits/ref] [--save-intermediates] [--stress]

The upstream directory and checkpoint may also come from the environment:
    GPT_SOVITS_ERES2NET_DIR   (or GPT_SOVITS_ROOT, which implies <root>/GPT_SoVITS/eres2net)
    GPT_SOVITS_SV_CKPT
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

# --- Fixed architecture: the only one supported downstream. ----------------
BASE_WIDTH = 24
SCALE = 4
EXPANSION = 4

# --- Fixed audio / fbank parameters (matched by the C++ sv_fbank frontend). -
SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
DURATION_S = 3.0
SEED = 0

# --- Stress fixture: short deterministic audio, scaled fbank. --------------
STRESS_SEED = 7
STRESS_DURATION_S = 1.0
STRESS_GAIN = 4.0   # pushes first-layer pre-activation above 20

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "tests", "sovits", "ref")

EPSILON = torch.finfo(torch.float).eps


# --- inlined kaldi.fbank (pure torch) -------------------------------------
def mel_scale_scalar(freq):
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq):
    return 1127.0 * (1.0 + freq / 700.0).log()


def get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq):
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    b = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + b * mel_freq_delta
    center_mel = mel_low_freq + (b + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (b + 2.0) * mel_freq_delta
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.max(torch.zeros(1), torch.min(up_slope, down_slope))


def fbank(waveform, num_mel_bins=NUM_MEL_BINS, sample_frequency=16000.0, dither=0.0,
          frame_length=25.0, frame_shift=10.0, low_freq=20.0, high_freq=0.0,
          preemphasis_coefficient=0.97, round_to_power_of_two=True,
          snip_edges=True, remove_dc_offset=True, raw_energy=True,
          use_power=True, use_log_fbank=True, window_type="povey"):
    # waveform: (1, n)
    waveform = waveform[0]
    window_shift = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded = (1 if window_size == 0 else 2 ** (window_size - 1).bit_length()) \
        if round_to_power_of_two else window_size

    m = 1 + (waveform.numel() - window_size) // window_shift
    strided = waveform.as_strided((m, window_size),
                                  (window_shift * waveform.stride(0), waveform.stride(0)))
    if remove_dc_offset:
        strided = strided - strided.mean(1, keepdim=True)
    if preemphasis_coefficient != 0.0:
        off = torch.nn.functional.pad(strided.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
        strided = strided - preemphasis_coefficient * off[:, :-1]
    # povey window = hann(periodic=False) ** 0.85
    win = torch.hann_window(window_size, periodic=False).pow(0.85).unsqueeze(0)
    strided = strided * win
    if padded != window_size:
        strided = torch.nn.functional.pad(strided.unsqueeze(0), (0, padded - window_size),
                                           mode="constant", value=0).squeeze(0)
    spectrum = torch.fft.rfft(strided).abs()
    if use_power:
        spectrum = spectrum.pow(2.0)
    mel_energies = get_mel_banks(num_mel_bins, padded, sample_frequency, low_freq, high_freq)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        mel_energies = torch.max(mel_energies, torch.tensor(EPSILON)).log()
    return mel_energies


def synthesize_audio(seed, duration_s):
    """Deterministic 16 kHz mono waveform: a few tones plus reproducible noise."""
    torch.manual_seed(seed)
    n = int(SAMPLE_RATE * duration_s)
    t = torch.arange(n).float() / SAMPLE_RATE
    return (0.3 * torch.sin(2 * math.pi * 220 * t)
            + 0.2 * torch.sin(2 * math.pi * 580 * t)
            + 0.1 * torch.sin(2 * math.pi * 1200 * t)
            + 0.05 * torch.randn(n))


def load_model(eres2net_dir, ckpt):
    if eres2net_dir and eres2net_dir not in sys.path:
        sys.path.insert(0, eres2net_dir)
    from ERes2NetV2 import ERes2NetV2  # noqa: E402 (path injected above)

    model = ERes2NetV2(baseWidth=BASE_WIDTH, scale=SCALE, expansion=EXPANSION)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def save(out_dir, name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    arr = t.astype(np.float32)
    np.save(os.path.join(out_dir, name), arr)
    print(f"  saved {name} {arr.shape} {arr.dtype}")


def dump_main(out_dir, model):
    wav = synthesize_audio(SEED, DURATION_S)
    print(f"audio: sr={SAMPLE_RATE} samples={wav.numel()} ({DURATION_S:.2f}s)")

    feat = fbank(wav.unsqueeze(0))  # {m, 80}
    print(f"fbank: {tuple(feat.shape)}")
    save(out_dir, "sv_audio16k.npy", wav)
    save(out_dir, "sv_fbank.npy", feat)

    # Embedding from the real upstream forward3.
    with torch.no_grad():
        emb = model.forward3(feat.unsqueeze(0))  # {1, 20480}
    print(f"sv_emb: {tuple(emb.shape)}")
    save(out_dir, "sv_emb.npy", emb)


def dump_intermediates(out_dir, model):
    # Re-run the forward3 sub-steps to expose per-layer tensors. Values are
    # identical to forward3 (same weights, same deterministic input).
    feat = fbank(synthesize_audio(SEED, DURATION_S).unsqueeze(0))
    with torch.no_grad():
        x = feat.unsqueeze(0)                    # {1, m, 80}
        x0 = x.permute(0, 2, 1).unsqueeze(1)     # {1, 1, 80, m}
        out = torch.relu(model.bn1(model.conv1(x0)))
        out1 = model.layer1(out)
        out2 = model.layer2(out1)
        out3 = model.layer3(out2)
        out4 = model.layer4(out3)
        out3_ds = model.layer3_ds(out3)
        fuse_out34 = model.fuse34(out4, out3_ds)
    print(f"intermediates: fuse_out34={tuple(fuse_out34.shape)}")
    save(out_dir, "sv_fuse34.npy", fuse_out34)
    save(out_dir, "sv_out1.npy", out1)
    save(out_dir, "sv_out2.npy", out2)
    save(out_dir, "sv_out3.npy", out3)
    save(out_dir, "sv_out4.npy", out4)


def dump_stress(out_dir, model):
    """Synthetic fbank that drives the first-layer pre-activation above 20.

    The upstream forward3 applies plain F.relu to conv1+bn1, so values above 20
    survive there; a gain-scaled fbank reproduces that case. This fixture
    distinguishes plain ReLU from an incorrect first-layer Hardtanh(0, 20).
    """
    base = fbank(synthesize_audio(STRESS_SEED, STRESS_DURATION_S).unsqueeze(0))
    stress = (base * STRESS_GAIN).contiguous()  # {m, 80}
    with torch.no_grad():
        x0 = stress.unsqueeze(0).permute(0, 2, 1).unsqueeze(1)
        pre = model.bn1(model.conv1(x0))
        n_over = int((pre > 20.0).sum().item())
        emb = model.forward3(stress.unsqueeze(0))  # {1, 20480}
    print(f"stress: fbank={tuple(stress.shape)} gain={STRESS_GAIN} "
          f"pre-activation max={pre.max():.3f} (>{20}: {n_over}/{pre.numel()}) "
          f"emb={tuple(emb.shape)} finite={bool(torch.isfinite(emb).all())}")
    save(out_dir, "sv_stress_fbank.npy", stress)
    save(out_dir, "sv_stress_emb.npy", emb)


def resolve_eres2net_dir(cli_value):
    if cli_value:
        return cli_value
    if os.environ.get("GPT_SOVITS_ERES2NET_DIR"):
        return os.environ["GPT_SOVITS_ERES2NET_DIR"]
    root = os.environ.get("GPT_SOVITS_ROOT")
    if root:
        return os.path.join(root, "GPT_SoVITS", "eres2net")
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dump ERes2NetV2 SV reference fixtures (forward3 path).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eres2net-dir", default=None,
                   help="Directory containing ERes2NetV2.py (upstream "
                        "GPT_SoVITS/eres2net). Falls back to "
                        "$GPT_SOVITS_ERES2NET_DIR, then $GPT_SOVITS_ROOT/GPT_SoVITS/eres2net.")
    p.add_argument("--ckpt", default=None,
                   help="Path to pretrained_eres2netv2w24s4ep4.ckpt. "
                        "Falls back to $GPT_SOVITS_SV_CKPT.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--save-intermediates", action="store_true",
                   help="Also write sv_fuse34.npy and sv_out1..4.npy.")
    p.add_argument("--stress", action="store_true",
                   help="Also write sv_stress_fbank.npy / sv_stress_emb.npy "
                        "(first-layer activation > 20 regression).")
    args = p.parse_args()

    eres2net_dir = resolve_eres2net_dir(args.eres2net_dir)
    ckpt = args.ckpt or os.environ.get("GPT_SOVITS_SV_CKPT")
    if not eres2net_dir:
        p.error("--eres2net-dir is required (or set GPT_SOVITS_ERES2NET_DIR / "
                "GPT_SOVITS_ROOT)")
    if not ckpt:
        p.error("--ckpt is required (or set GPT_SOVITS_SV_CKPT)")
    if not os.path.isdir(eres2net_dir):
        p.error(f"--eres2net-dir does not exist: {eres2net_dir}")
    if not os.path.isfile(ckpt):
        p.error(f"--ckpt does not exist: {ckpt}")

    print(f"eres2net dir: {eres2net_dir}")
    print(f"ckpt        : {ckpt}")
    print(f"out dir     : {args.out_dir}")
    print(f"arch        : ERes2NetV2(baseWidth={BASE_WIDTH}, scale={SCALE}, "
          f"expansion={EXPANSION})")

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(eres2net_dir, ckpt)

    dump_main(args.out_dir, model)
    if args.save_intermediates:
        dump_intermediates(args.out_dir, model)
    if args.stress:
        dump_stress(args.out_dir, model)


if __name__ == "__main__":
    main()
