#!/usr/bin/env python3
"""Generate reference outputs for CN-HuBERT parity testing.

Runs the HuggingFace HubertModel on a deterministic input and saves
intermediate and final outputs as raw f32 binary files whose memory
layout matches the ggml convention (ne[0] contiguous).

Usage:
    python generate_reference.py <model_dir> [--output-dir <dir>] [--length <samples>]

Output files (all raw f32, little-endian):
    ref_input.bin                 - input waveform, ggml {T}
    ref_feature_encoder.bin       - 7-layer conv output, ggml {512, T'}
    ref_feature_projection.bin    - projected features, ggml {768, T'}
    ref_pos_conv.bin              - positional conv output, ggml {768, T'}
    ref_encoder_input.bin         - encoder input after add+LayerNorm, ggml {768, T'}
    ref_attention0.bin            - encoder layer 0 attention output, ggml {768, T'}
    ref_encoder_layer0.bin        - encoder layer 0 output, ggml {768, T'}
    ref_model_output.bin          - final encoder output, ggml {768, T'}
    ref_metadata.json             - shapes, seed, and layout notes
"""

import argparse
import json
import os
import sys

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
        description="Generate HuBERT reference outputs for C++ parity testing")
    parser.add_argument("model_dir",
                        help="Path to HF chinese-hubert-base directory "
                             "(containing pytorch_model.bin and config.json)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Output directory for reference files (default: cwd)")
    parser.add_argument("--length", "-l", type=int, default=16000,
                        help="Input waveform length in samples (default: 16000 = 1s @ 16kHz)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducible input (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate deterministic input waveform
    # ------------------------------------------------------------------
    print(f"Generating input: {args.length} samples, seed={args.seed}")
    rng = np.random.default_rng(args.seed)
    raw = rng.standard_normal(args.length).astype(np.float32)

    # Normalize to zero mean, unit variance.
    # This matches Wav2Vec2FeatureExtractor(do_normalize=True).
    mean = raw.mean()
    var  = raw.var()
    input_values = ((raw - mean) / np.sqrt(var + 1e-7)).astype(np.float32)

    save_f32_bin(os.path.join(args.output_dir, "ref_input.bin"), input_values)

    # ------------------------------------------------------------------
    # 2. Load HuBERT model
    # ------------------------------------------------------------------
    print(f"\nLoading HubertModel from {args.model_dir} ...")
    from transformers import HubertConfig, HubertModel   # noqa: delayed import

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
    model = model.float()  # ensure full f32 precision

    # ------------------------------------------------------------------
    # 3. Register forward hooks for intermediate activations
    # ------------------------------------------------------------------
    intermediates: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_module, _inp, out):
            # HubertFeatureProjection.forward returns (hidden_states, norm_hidden_states)
            intermediates[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    model.feature_extractor.register_forward_hook(_hook("feature_encoder"))
    model.feature_projection.register_forward_hook(_hook("feature_projection"))

    # ------------------------------------------------------------------
    # 4. Forward pass
    # ------------------------------------------------------------------
    print("Running forward pass ...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_values).unsqueeze(0)   # [1, T]
        outputs = model(input_tensor, output_hidden_states=True)

        feat_proj_btc = intermediates["feature_projection"]          # [1, T', 768]
        pos_conv = model.encoder.pos_conv_embed(feat_proj_btc)       # [1, T', 768]
        encoder_input = model.encoder.layer_norm(feat_proj_btc + pos_conv)
        attention0 = model.encoder.layers[0].attention(
            encoder_input,
            attention_mask=None,
            output_attentions=False,
        )[0]
        encoder_layer0 = model.encoder.layers[0](
            encoder_input,
            attention_mask=None,
            output_attentions=False,
        )[0]

    # ------------------------------------------------------------------
    # 5. Extract tensors and convert to ggml memory layout
    #
    #    ggml {ne0, ne1} stores ne1 contiguous blocks of ne0 elements.
    #    In numpy C-order that corresponds to shape (ne1, ne0).
    # ------------------------------------------------------------------
    print("\nSaving reference outputs ...")

    # Feature encoder: hook gives [1, 512, T'] (channels-first from nn.Conv1d)
    #   -> squeeze -> [512, T'] -> transpose to [T', 512] for ggml {512, T'}
    feat_enc = intermediates["feature_encoder"].squeeze(0)           # [512, T']
    feat_enc_np = feat_enc.permute(1, 0).contiguous().numpy()       # [T', 512]
    save_f32_bin(os.path.join(args.output_dir, "ref_feature_encoder.bin"), feat_enc_np)

    # Feature projection: hook gives [1, T', 768]
    #   -> squeeze -> [T', 768]  (already ggml {768, T'} memory layout)
    feat_proj = intermediates["feature_projection"].squeeze(0)       # [T', 768]
    feat_proj_np = feat_proj.contiguous().numpy()                    # [T', 768]
    save_f32_bin(os.path.join(args.output_dir, "ref_feature_projection.bin"), feat_proj_np)

    # Positional conv output: [1, T', 768] -> [T', 768]
    pos_conv_np = pos_conv.squeeze(0).contiguous().numpy()
    save_f32_bin(os.path.join(args.output_dir, "ref_pos_conv.bin"), pos_conv_np)

    # Encoder input after positional add + LayerNorm: [1, T', 768] -> [T', 768]
    encoder_input_np = encoder_input.squeeze(0).contiguous().numpy()
    save_f32_bin(os.path.join(args.output_dir, "ref_encoder_input.bin"), encoder_input_np)

    # Encoder layer 0 attention output: [1, T', 768] -> [T', 768]
    attention0_np = attention0.squeeze(0).contiguous().numpy()
    save_f32_bin(os.path.join(args.output_dir, "ref_attention0.bin"), attention0_np)

    # Encoder layer 0 output: [1, T', 768] -> [T', 768]
    encoder_layer0_np = encoder_layer0.squeeze(0).contiguous().numpy()
    save_f32_bin(os.path.join(args.output_dir, "ref_encoder_layer0.bin"), encoder_layer0_np)

    # Full model output (last_hidden_state): [1, T', 768]
    #   -> squeeze -> [T', 768]
    model_out = outputs.last_hidden_state.squeeze(0)                 # [T', 768]
    model_out_np = model_out.contiguous().numpy()                    # [T', 768]
    save_f32_bin(os.path.join(args.output_dir, "ref_model_output.bin"), model_out_np)

    # ------------------------------------------------------------------
    # 6. Save metadata
    # ------------------------------------------------------------------
    T_prime = int(feat_enc.shape[1])
    metadata = {
        "seed": args.seed,
        "input_length": args.length,
        "sample_rate": 16000,
        "shapes": {
            "ref_input":              {"ggml": [args.length],   "memory": [args.length]},
            "ref_feature_encoder":    {"ggml": [512, T_prime],  "memory": [T_prime, 512]},
            "ref_feature_projection": {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
            "ref_pos_conv":           {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
            "ref_encoder_input":      {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
            "ref_attention0":         {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
            "ref_encoder_layer0":     {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
            "ref_model_output":       {"ggml": [768, T_prime],  "memory": [T_prime, 768]},
        },
        "note": (
            "All .bin files are raw float32 little-endian. "
            "Memory layout matches ggml: ne[0] is the contiguous (innermost) dimension."
        ),
    }

    meta_path = os.path.join(args.output_dir, "ref_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {os.path.basename(meta_path)}")

    print(f"\nSequence length after conv encoder: T' = {T_prime}")
    print("Done!")


if __name__ == "__main__":
    main()
