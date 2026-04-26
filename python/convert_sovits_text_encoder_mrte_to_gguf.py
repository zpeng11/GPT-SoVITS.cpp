#!/usr/bin/env python3
"""Convert SoVITS v1/v2 `enc_p.mrte` weights to fused GGUF.

Usage:
    python convert_sovits_text_encoder_mrte_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16]

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`
and exports offline-fused inference weights for `sovits_text_encoder_mrte_block_weights`.
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


def _linearize_conv1x1(weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight.shape[2] != 1:
        raise ValueError(f"Expected Conv1d(kernel=1) weight, got shape {weight.shape}")
    return weight[:, :, 0].astype(np.float32), bias.astype(np.float32)


def _compose_affine(
    w2: np.ndarray,
    b2: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return w2 @ w1, w2 @ b1 + b2


def _stack_conv1x1(weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(weights, axis=0), np.concatenate(biases, axis=0)


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    c_pre_w, c_pre_b = _linearize_conv1x1(weights["enc_p.mrte.c_pre.weight"], weights["enc_p.mrte.c_pre.bias"])
    text_pre_w, text_pre_b = _linearize_conv1x1(weights["enc_p.mrte.text_pre.weight"], weights["enc_p.mrte.text_pre.bias"])
    q_w, q_b = _linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_q.weight"], weights["enc_p.mrte.cross_attention.conv_q.bias"])
    k_w, k_b = _linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_k.weight"], weights["enc_p.mrte.cross_attention.conv_k.bias"])
    v_w, v_b = _linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_v.weight"], weights["enc_p.mrte.cross_attention.conv_v.bias"])
    o_w, o_b = _linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_o.weight"], weights["enc_p.mrte.cross_attention.conv_o.bias"])
    c_post_w, c_post_b = _linearize_conv1x1(weights["enc_p.mrte.c_post.weight"], weights["enc_p.mrte.c_post.bias"])

    q_fused_w, q_fused_b = _compose_affine(q_w, q_b, c_pre_w, c_pre_b)
    k_fused_w, k_fused_b = _compose_affine(k_w, k_b, text_pre_w, text_pre_b)
    v_fused_w, v_fused_b = _compose_affine(v_w, v_b, text_pre_w, text_pre_b)
    skip_from_ssl_w, skip_from_ssl_b = _compose_affine(c_post_w, c_post_b, c_pre_w, c_pre_b)
    attn_out_w, attn_out_b = _compose_affine(c_post_w, c_post_b, o_w, o_b)

    ge_out_w = c_post_w.copy()
    ge_out_b = c_post_b.copy()
    skip_from_ssl_b = skip_from_ssl_b - c_post_b
    attn_out_b = attn_out_b - c_post_b

    ssl_fused_w, ssl_fused_b = _stack_conv1x1(
        [q_fused_w, skip_from_ssl_w],
        [q_fused_b, skip_from_ssl_b],
    )
    text_kv_w, text_kv_b = _stack_conv1x1(
        [k_fused_w, v_fused_w],
        [k_fused_b, v_fused_b],
    )

    tensors: list[tuple[str, np.ndarray]] = [
        ("text_encoder_mrte.ssl_fused_w", ssl_fused_w[:, :, None]),
        ("text_encoder_mrte.ssl_fused_b", ssl_fused_b),
        ("text_encoder_mrte.text_kv_w", text_kv_w[:, :, None]),
        ("text_encoder_mrte.text_kv_b", text_kv_b),
        ("text_encoder_mrte.attn_out_w", attn_out_w[:, :, None]),
        ("text_encoder_mrte.attn_out_b", attn_out_b),
        ("text_encoder_mrte.ge_out_w", ge_out_w[:, :, None]),
        ("text_encoder_mrte.ge_out_b", ge_out_b),
    ]

    writer = gguf.GGUFWriter(output_path, "sovits_text_encoder_mrte")
    writer.add_string("sovits.block", "text_encoder_mrte")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.text_encoder_mrte.in_dim", 192)
    writer.add_uint32("sovits.text_encoder_mrte.hidden_dim", 512)
    writer.add_uint32("sovits.text_encoder_mrte.out_dim", 192)
    writer.add_uint32("sovits.text_encoder_mrte.n_head", 4)

    n_converted = 0
    for gguf_name, tensor_np in tensors:
        if target_type == gguf.GGMLQuantizationType.F16 and tensor_np.ndim >= 2:
            tensor_np = tensor_np.astype(np.float16)
            tensor_type = target_type
        else:
            tensor_np = tensor_np.astype(np.float32)
            tensor_type = gguf.GGMLQuantizationType.F32

        writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {gguf_name:32s} "
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
        description="Convert SoVITS text_encoder_mrte weights to fused GGUF"
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
