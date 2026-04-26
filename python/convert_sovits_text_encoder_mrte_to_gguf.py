#!/usr/bin/env python3
"""Convert SoVITS v1/v2 `enc_p.mrte` weights to GGUF.

Usage:
    python convert_sovits_text_encoder_mrte_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16]

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`
and exports only the tensors needed by `sovits_text_encoder_mrte_block_weights`.
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


TEXT_ENCODER_MRTE_MAP = [
    ("text_encoder_mrte.c_pre_w", "enc_p.mrte.c_pre.weight"),
    ("text_encoder_mrte.c_pre_b", "enc_p.mrte.c_pre.bias"),
    ("text_encoder_mrte.text_pre_w", "enc_p.mrte.text_pre.weight"),
    ("text_encoder_mrte.text_pre_b", "enc_p.mrte.text_pre.bias"),
    ("text_encoder_mrte.attn.q_w", "enc_p.mrte.cross_attention.conv_q.weight"),
    ("text_encoder_mrte.attn.q_b", "enc_p.mrte.cross_attention.conv_q.bias"),
    ("text_encoder_mrte.attn.k_w", "enc_p.mrte.cross_attention.conv_k.weight"),
    ("text_encoder_mrte.attn.k_b", "enc_p.mrte.cross_attention.conv_k.bias"),
    ("text_encoder_mrte.attn.v_w", "enc_p.mrte.cross_attention.conv_v.weight"),
    ("text_encoder_mrte.attn.v_b", "enc_p.mrte.cross_attention.conv_v.bias"),
    ("text_encoder_mrte.attn.out_w", "enc_p.mrte.cross_attention.conv_o.weight"),
    ("text_encoder_mrte.attn.out_b", "enc_p.mrte.cross_attention.conv_o.bias"),
    ("text_encoder_mrte.c_post_w", "enc_p.mrte.c_post.weight"),
    ("text_encoder_mrte.c_post_b", "enc_p.mrte.c_post.bias"),
]


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_text_encoder_mrte")
    writer.add_string("sovits.block", "text_encoder_mrte")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.text_encoder_mrte.in_dim", 192)
    writer.add_uint32("sovits.text_encoder_mrte.hidden_dim", 512)
    writer.add_uint32("sovits.text_encoder_mrte.out_dim", 192)
    writer.add_uint32("sovits.text_encoder_mrte.n_head", 4)

    n_converted = 0
    for gguf_name, ckpt_name in TEXT_ENCODER_MRTE_MAP:
        if ckpt_name not in weights:
            raise KeyError(
                f"Tensor '{ckpt_name}' not found in checkpoint "
                f"(needed for GGUF tensor '{gguf_name}')"
            )

        tensor_np = weights[ckpt_name]
        if target_type == gguf.GGMLQuantizationType.F16 and tensor_np.ndim >= 2:
            tensor_np = tensor_np.astype(np.float16)
            tensor_type = target_type
        else:
            tensor_np = tensor_np.astype(np.float32)
            tensor_type = gguf.GGMLQuantizationType.F32

        writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {gguf_name:32s} <- {ckpt_name:44s} "
            f"{list(tensor_np.shape)!s:16s} {tensor_np.dtype}"
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
        description="Convert SoVITS text_encoder_mrte weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-text-encoder-mrte-<type>.gguf)",
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
        args.output = f"{base}-text-encoder-mrte-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
