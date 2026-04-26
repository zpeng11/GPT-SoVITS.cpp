#!/usr/bin/env python3
"""Convert SoVITS v2 `enc_p.ssl_proj + enc_p.encoder_ssl` weights to GGUF.

Usage:
    python convert_sovits_text_encoder_ssl_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16]

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`
and exports only the tensors needed by `sovits_text_encoder_ssl_block_weights`.
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


TEXT_ENCODER_SSL_MAP = [
    ("text_encoder_ssl.ssl_proj_w", "enc_p.ssl_proj.weight"),
    ("text_encoder_ssl.ssl_proj_b", "enc_p.ssl_proj.bias"),
]

for i in range(3):
    TEXT_ENCODER_SSL_MAP.extend([
        (f"text_encoder_ssl.layers.{i}.out_w", f"enc_p.encoder_ssl.attn_layers.{i}.conv_o.weight"),
        (f"text_encoder_ssl.layers.{i}.out_b", f"enc_p.encoder_ssl.attn_layers.{i}.conv_o.bias"),
        (f"text_encoder_ssl.layers.{i}.rel_k", f"enc_p.encoder_ssl.attn_layers.{i}.emb_rel_k"),
        (f"text_encoder_ssl.layers.{i}.rel_v", f"enc_p.encoder_ssl.attn_layers.{i}.emb_rel_v"),
        (f"text_encoder_ssl.layers.{i}.ln1_w", f"enc_p.encoder_ssl.norm_layers_1.{i}.gamma"),
        (f"text_encoder_ssl.layers.{i}.ln1_b", f"enc_p.encoder_ssl.norm_layers_1.{i}.beta"),
        (f"text_encoder_ssl.layers.{i}.ffn_up_w", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_1.weight"),
        (f"text_encoder_ssl.layers.{i}.ffn_up_b", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_1.bias"),
        (f"text_encoder_ssl.layers.{i}.ffn_down_w", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_2.weight"),
        (f"text_encoder_ssl.layers.{i}.ffn_down_b", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_2.bias"),
        (f"text_encoder_ssl.layers.{i}.ln2_w", f"enc_p.encoder_ssl.norm_layers_2.{i}.gamma"),
        (f"text_encoder_ssl.layers.{i}.ln2_b", f"enc_p.encoder_ssl.norm_layers_2.{i}.beta"),
    ])


def _fused_qkv(weights: dict[str, np.ndarray], layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"enc_p.encoder_ssl.attn_layers.{layer_idx}"
    q_w = weights[f"{prefix}.conv_q.weight"]
    k_w = weights[f"{prefix}.conv_k.weight"]
    v_w = weights[f"{prefix}.conv_v.weight"]
    q_b = weights[f"{prefix}.conv_q.bias"]
    k_b = weights[f"{prefix}.conv_k.bias"]
    v_b = weights[f"{prefix}.conv_v.bias"]
    return np.concatenate([q_w, k_w, v_w], axis=0), np.concatenate([q_b, k_b, v_b], axis=0)


def _packed_rel_k(weights: dict[str, np.ndarray], layer_idx: int) -> np.ndarray:
    rel_k = weights[f"enc_p.encoder_ssl.attn_layers.{layer_idx}.emb_rel_k"]
    return rel_k[0].copy()


def _packed_rel_v_t(weights: dict[str, np.ndarray], layer_idx: int) -> np.ndarray:
    rel_v = weights[f"enc_p.encoder_ssl.attn_layers.{layer_idx}.emb_rel_v"]
    return rel_v[0].transpose(1, 0).copy()


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_text_encoder_ssl")
    writer.add_string("sovits.block", "text_encoder_ssl")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.text_encoder_ssl.in_dim", 768)
    writer.add_uint32("sovits.text_encoder_ssl.hidden_dim", 192)
    writer.add_uint32("sovits.text_encoder_ssl.ffn_dim", 768)
    writer.add_uint32("sovits.text_encoder_ssl.n_head", 2)
    writer.add_uint32("sovits.text_encoder_ssl.n_layer", 3)
    writer.add_uint32("sovits.text_encoder_ssl.kernel_size", 3)
    writer.add_uint32("sovits.text_encoder_ssl.window_size", 4)

    n_converted = 0
    for i in range(3):
        qkv_w, qkv_b = _fused_qkv(weights, i)
        qkv_w = qkv_w.astype(np.float16 if target_type == gguf.GGMLQuantizationType.F16 else np.float32)
        qkv_b = qkv_b.astype(np.float32)
        rel_k = _packed_rel_k(weights, i).astype(np.float32)
        rel_v_t = _packed_rel_v_t(weights, i).astype(np.float32)

        writer.add_tensor(
            f"text_encoder_ssl.layers.{i}.qkv_w",
            qkv_w,
            raw_dtype=target_type if qkv_w.dtype == np.float16 else gguf.GGMLQuantizationType.F32,
        )
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {'text_encoder_ssl.layers.' + str(i) + '.qkv_w':36s} <- fused q/k/v weights{'':18s} "
            f"{list(qkv_w.shape)!s:16s} {qkv_w.dtype}"
        )

        writer.add_tensor(
            f"text_encoder_ssl.layers.{i}.qkv_b",
            qkv_b,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {'text_encoder_ssl.layers.' + str(i) + '.qkv_b':36s} <- fused q/k/v bias{'':21s} "
            f"{list(qkv_b.shape)!s:16s} {qkv_b.dtype}"
        )

        writer.add_tensor(
            f"text_encoder_ssl.layers.{i}.rel_k",
            rel_k,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {'text_encoder_ssl.layers.' + str(i) + '.rel_k':36s} <- packed rel_k{'':27s} "
            f"{list(rel_k.shape)!s:16s} {rel_k.dtype}"
        )

        writer.add_tensor(
            f"text_encoder_ssl.layers.{i}.rel_v_t",
            rel_v_t,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {'text_encoder_ssl.layers.' + str(i) + '.rel_v_t':36s} <- packed rel_v_t{'':25s} "
            f"{list(rel_v_t.shape)!s:16s} {rel_v_t.dtype}"
        )

    for gguf_name, ckpt_name in TEXT_ENCODER_SSL_MAP:
        if gguf_name.endswith(".rel_k") or gguf_name.endswith(".rel_v"):
            continue
        if ckpt_name not in weights:
            raise KeyError(
                f"Tensor '{ckpt_name}' not found in checkpoint "
                f"(needed for GGUF tensor '{gguf_name}')"
            )

        tensor_np = weights[ckpt_name]
        if gguf_name.endswith("ffn_up_w") or gguf_name.endswith("ffn_down_w"):
            tensor_np = tensor_np.astype(np.float32)
        elif target_type == gguf.GGMLQuantizationType.F16 and tensor_np.ndim >= 2:
            tensor_np = tensor_np.astype(np.float16)
        else:
            tensor_np = tensor_np.astype(np.float32)

        writer.add_tensor(gguf_name, tensor_np, raw_dtype=target_type if tensor_np.dtype == np.float16 else gguf.GGMLQuantizationType.F32)
        n_converted += 1
        print(
            f"  [{n_converted:2d}] {gguf_name:36s} <- {ckpt_name:48s} "
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
        description="Convert SoVITS v2 text_encoder_ssl weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-text-encoder-ssl-<type>.gguf)",
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
        args.output = f"{base}-text-encoder-ssl-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
