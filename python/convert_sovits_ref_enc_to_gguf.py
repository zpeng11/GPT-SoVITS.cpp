#!/usr/bin/env python3
"""Convert SoVITS v2 `ref_enc` (MelStyleEncoder) weights to GGUF format.

Usage:
    python convert_sovits_ref_enc_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

Where:
    <sovits_ckpt> - Path to a SoVITS checkpoint (.pth), e.g. s2G2333k.pth

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`
and exports only the tensors needed by `sovits_mel_style_encoder_block_weights`.

Tensor mapping (checkpoint name -> GGUF name):
  ref_enc.spectral.0.fc.weight        -> ref_enc.spectral_1_w
  ref_enc.spectral.0.fc.bias          -> ref_enc.spectral_1_b
  ref_enc.spectral.3.fc.weight        -> ref_enc.spectral_2_w
  ref_enc.spectral.3.fc.bias          -> ref_enc.spectral_2_b

  ref_enc.temporal.0.conv1.conv.weight -> ref_enc.temporal.0.conv_w
  ref_enc.temporal.0.conv1.conv.bias   -> ref_enc.temporal.0.conv_b
  ref_enc.temporal.1.conv1.conv.weight -> ref_enc.temporal.1.conv_w
  ref_enc.temporal.1.conv1.conv.bias   -> ref_enc.temporal.1.conv_b

  ref_enc.slf_attn.w_qs.weight        -> ref_enc.attention.q_w
  ref_enc.slf_attn.w_qs.bias          -> ref_enc.attention.q_b
  ref_enc.slf_attn.w_ks.weight        -> ref_enc.attention.k_w
  ref_enc.slf_attn.w_ks.bias          -> ref_enc.attention.k_b
  ref_enc.slf_attn.w_vs.weight        -> ref_enc.attention.v_w
  ref_enc.slf_attn.w_vs.bias          -> ref_enc.attention.v_b
  ref_enc.slf_attn.fc.weight          -> ref_enc.attention.out_w
  ref_enc.slf_attn.fc.bias            -> ref_enc.attention.out_b

  ref_enc.fc.fc.weight                -> ref_enc.fc_w
  ref_enc.fc.fc.bias                  -> ref_enc.fc_b

All tensors are stored in their PyTorch layout. GGUF reverses dimensions when
loaded into ggml, yielding the conventions expected by `src/sovits/block.cpp`.

Quantization types (applied to 2D Linear weights only; 1D biases and 3D Conv1d
kernels stay unquantized):
    q8  - Q8_0
    q5  - Q5_0
    q4  - Q4_0
"""

from __future__ import annotations

import argparse
import os

import gguf
import numpy as np

from torch_ckpt_utils import load_checkpoint


GGML_TYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8": gguf.GGMLQuantizationType.Q8_0,
    "q5": gguf.GGMLQuantizationType.Q5_0,
    "q4": gguf.GGMLQuantizationType.Q4_0,
}


REF_ENC_MAP = [
    ("ref_enc.spectral_1_w", "ref_enc.spectral.0.fc.weight"),
    ("ref_enc.spectral_1_b", "ref_enc.spectral.0.fc.bias"),
    ("ref_enc.spectral_2_w", "ref_enc.spectral.3.fc.weight"),
    ("ref_enc.spectral_2_b", "ref_enc.spectral.3.fc.bias"),
    ("ref_enc.temporal.0.conv_w", "ref_enc.temporal.0.conv1.conv.weight"),
    ("ref_enc.temporal.0.conv_b", "ref_enc.temporal.0.conv1.conv.bias"),
    ("ref_enc.temporal.1.conv_w", "ref_enc.temporal.1.conv1.conv.weight"),
    ("ref_enc.temporal.1.conv_b", "ref_enc.temporal.1.conv1.conv.bias"),
    ("ref_enc.attention.q_w", "ref_enc.slf_attn.w_qs.weight"),
    ("ref_enc.attention.q_b", "ref_enc.slf_attn.w_qs.bias"),
    ("ref_enc.attention.k_w", "ref_enc.slf_attn.w_ks.weight"),
    ("ref_enc.attention.k_b", "ref_enc.slf_attn.w_ks.bias"),
    ("ref_enc.attention.v_w", "ref_enc.slf_attn.w_vs.weight"),
    ("ref_enc.attention.v_b", "ref_enc.slf_attn.w_vs.bias"),
    ("ref_enc.attention.out_w", "ref_enc.slf_attn.fc.weight"),
    ("ref_enc.attention.out_b", "ref_enc.slf_attn.fc.bias"),
    ("ref_enc.fc_w", "ref_enc.fc.fc.weight"),
    ("ref_enc.fc_b", "ref_enc.fc.fc.bias"),
]


def should_quantize(gguf_name: str, tensor: np.ndarray, block_size: int) -> bool:
    """Return True if this tensor should be quantized."""
    if tensor.ndim != 2:
        return False
    if gguf_name.endswith("_w") and not gguf_name.startswith("ref_enc.temporal."):
        return tensor.shape[1] % block_size == 0
    return False


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    is_quantized = target_type not in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    )
    block_size = gguf.GGML_QUANT_SIZES[target_type][0] if is_quantized else 0
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_ref_enc")

    writer.add_string("sovits.block", "mel_style_encoder")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.ref_enc.in_dim", 704)
    writer.add_uint32("sovits.ref_enc.hidden_dim", 128)
    writer.add_uint32("sovits.ref_enc.out_dim", int(cfg.get("gin_channels", 512)))
    writer.add_uint32("sovits.ref_enc.kernel_size", 5)
    writer.add_uint32("sovits.ref_enc.n_head", 2)
    writer.add_uint32("sovits.ref_enc.temporal_layers", 2)
    writer.add_bool("sovits.ref_enc.full_context", True)

    n_converted = 0
    n_quantized = 0

    for gguf_name, ckpt_name in REF_ENC_MAP:
        if ckpt_name not in weights:
            raise KeyError(
                f"Tensor '{ckpt_name}' not found in checkpoint "
                f"(needed for GGUF tensor '{gguf_name}')"
            )

        tensor_np = weights[ckpt_name]

        if is_quantized and should_quantize(gguf_name, tensor_np, block_size):
            quantized = gguf.quantize(tensor_np, target_type)
            writer.add_tensor(gguf_name, quantized, raw_dtype=target_type)
            n_quantized += 1
            data_type = target_type
        elif ((is_quantized or target_type == gguf.GGMLQuantizationType.F16)
              and tensor_np.ndim >= 2):
            tensor_np = tensor_np.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)
        else:
            tensor_np = tensor_np.astype(np.float32)
            data_type = gguf.GGMLQuantizationType.F32
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)

        n_converted += 1
        print(
            f"  [{n_converted:2d}] {gguf_name:28s} <- {ckpt_name:36s} "
            f"{list(tensor_np.shape)!s:18s} {data_type.name}"
        )

    if n_quantized > 0:
        print(f"\nConverted {n_converted} tensors (quantized {n_quantized})")
    else:
        print(f"\nConverted {n_converted} tensors")
    print(f"Writing GGUF to {output_path}...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"Done! Output: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SoVITS v2 ref_enc (MelStyleEncoder) weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-ref-enc-<type>.gguf)",
    )
    parser.add_argument(
        "--type",
        "-t",
        dest="dtype",
        default="f16",
        choices=list(GGML_TYPES.keys()),
        help="Output data type (default: f16)",
    )
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.sovits_ckpt))[0]
        args.output = f"{base}-ref-enc-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
