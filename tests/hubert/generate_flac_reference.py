#!/usr/bin/env python3
"""Generate reference outputs for FLAC-based HuBERT parity testing.

Decodes a real FLAC audio file, resamples to 16kHz, normalizes, runs
HuggingFace HubertModel inference, and saves the results as raw f32
binary files for C++ parity comparison.

Usage:
    python generate_flac_reference.py <model_dir> [--flac <path>] [--output-dir <dir>]

Output files (all raw f32, little-endian):
    ref_flac_input.bin          - normalized 16kHz waveform, ggml {N}
    ref_flac_model_output.bin   - HuBERT output, ggml {768, T'}  (memory [T', 768])
    ref_flac_metadata.json      - shapes and info
"""

import argparse
import os

import numpy as np
import torch


def save_f32_bin(path: str, data: np.ndarray) -> None:
    """Save a numpy array as raw little-endian f32 binary."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(data.tobytes())
    size_kb = data.nbytes / 1024
    print(f"  Saved {os.path.basename(path):40s} shape={str(data.shape):20s} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FLAC-based HuBERT reference outputs for C++ parity testing")
    parser.add_argument("model_dir",
                        help="Path to HF chinese-hubert-base directory "
                             "(containing pytorch_model.bin and config.json)")
    parser.add_argument("--flac", "-f",
                        default=os.path.join(os.path.dirname(__file__),
                                             "Narsil.asr_dummy.4.flac"),
                        help="Path to FLAC input file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same as FLAC file directory)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.flac))
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Decode FLAC and resample to 16kHz
    # ------------------------------------------------------------------
    print(f"Decoding FLAC: {args.flac}")
    try:
        import soundfile as sf
        audio, sr = sf.read(args.flac, dtype="float32")
    except ImportError:
        import librosa
        audio, sr = librosa.load(args.flac, sr=None, dtype=np.float32)

    print(f"  Sample rate: {sr} Hz, samples: {len(audio)}, duration: {len(audio)/sr:.2f}s")

    if sr != 16000:
        print(f"  Resampling {sr} Hz -> 16000 Hz")
        try:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000,
                                          res_type="kaiser_best")
        except ImportError:
            # Simple decimation for exact integer ratios
            ratio = sr // 16000
            assert sr == 16000 * ratio, f"Cannot simply decimate {sr} -> 16000"
            audio_16k = audio[::ratio].astype(np.float32)
        audio = np.ascontiguousarray(audio_16k, dtype=np.float32)

    # Normalize: zero mean, unit variance (matching Wav2Vec2FeatureExtractor)
    mean = audio.mean()
    var = audio.var()
    input_values = ((audio - mean) / np.sqrt(var + 1e-7)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Load HuBERT model
    # ------------------------------------------------------------------
    print(f"\nLoading HubertModel from {args.model_dir} ...")
    from transformers import HubertConfig, HubertModel

    config = HubertConfig.from_pretrained(args.model_dir, local_files_only=True)
    config._attn_implementation = "eager"
    config.apply_spec_augment = False
    config.layerdrop = 0.0

    model = HubertModel.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
    )
    model.eval()
    model = model.float()

    # ------------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------------
    print("Running forward pass ...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_values).unsqueeze(0)  # [1, N]
        outputs = model(input_tensor)

    # ------------------------------------------------------------------
    # 4. Save output in ggml memory layout
    # ------------------------------------------------------------------
    print("\nSaving reference outputs ...")

    # Full model output: [1, T', 768] -> [T', 768]
    model_out = outputs.last_hidden_state.squeeze(0)       # [T', 768]
    model_out_np = model_out.contiguous().numpy()           # [T', 768]
    save_f32_bin(os.path.join(args.output_dir, "ref_flac_model_output.bin"), model_out_np)

    T_prime = int(model_out.shape[0])
    print(f"\nSequence length after conv encoder: T' = {T_prime}")
    print("Done!")


if __name__ == "__main__":
    main()
