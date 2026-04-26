#!/usr/bin/env python3
"""Convert SoVITS v2 quantizer decode weights to GGUF format.

Usage:
    python convert_sovits_quantizer_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16]

This converter is torch-free. It exports only the single-layer codebook needed
by `sovits_rvq_decode_block_forward`.
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
}

QUANTIZER_MAP = [
    ("quantizer.codebook", "quantizer.vq.layers.0._codebook.embed"),
]


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_quantizer")
    writer.add_string("sovits.block", "quantizer_decode")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_string("sovits.semantic_frame_rate", str(cfg.get("semantic_frame_rate", "25hz")))
    writer.add_uint32("sovits.quantizer.dim", 768)
    writer.add_uint32("sovits.quantizer.bins", 1024)
    writer.add_uint32("sovits.quantizer.n_q", 1)

    n_converted = 0
    for gguf_name, ckpt_name in QUANTIZER_MAP:
        if ckpt_name not in weights:
            raise KeyError(
                f"Tensor '{ckpt_name}' not found in checkpoint "
                f"(needed for GGUF tensor '{gguf_name}')"
            )

        tensor_np = weights[ckpt_name]
        if target_type == gguf.GGMLQuantizationType.F16:
            tensor_np = tensor_np.astype(np.float16)
        else:
            tensor_np = tensor_np.astype(np.float32)

        writer.add_tensor(gguf_name, tensor_np, raw_dtype=target_type)
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {gguf_name:20s} <- {ckpt_name:40s} "
            f"{list(tensor_np.shape)!s:16s} {target_type.name}"
        )

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
        description="Convert SoVITS v2 quantizer decode weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-quantizer-<type>.gguf)",
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
        args.output = f"{base}-quantizer-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
