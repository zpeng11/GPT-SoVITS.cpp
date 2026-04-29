#!/usr/bin/env python3
"""Convert full SoVITS v2 `enc_p` weights to a single GGUF.

Usage:
    python convert_sovits_text_encoder_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

This converter exports the optimized tensor layout consumed by
`src/sovits/block.cpp`:

* all Conv1d(kernel=1) weights are exported as 2D linear weights
* relative-position tensors stay prepacked for inference
* MRTE weights are fused offline into the exact runtime projections
* only 2D weights are quantized for weight-only inference
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


TEXT_ENCODER_SSL_MAP = [
    ("text_encoder_ssl.ssl_proj_w", "enc_p.ssl_proj.weight"),
    ("text_encoder_ssl.ssl_proj_b", "enc_p.ssl_proj.bias"),
]

for i in range(3):
    TEXT_ENCODER_SSL_MAP.extend([
        (f"text_encoder_ssl.layers.{i}.out_w", f"enc_p.encoder_ssl.attn_layers.{i}.conv_o.weight"),
        (f"text_encoder_ssl.layers.{i}.out_b", f"enc_p.encoder_ssl.attn_layers.{i}.conv_o.bias"),
        (f"text_encoder_ssl.layers.{i}.ln1_w", f"enc_p.encoder_ssl.norm_layers_1.{i}.gamma"),
        (f"text_encoder_ssl.layers.{i}.ln1_b", f"enc_p.encoder_ssl.norm_layers_1.{i}.beta"),
        (f"text_encoder_ssl.layers.{i}.ffn_up_w", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_1.weight"),
        (f"text_encoder_ssl.layers.{i}.ffn_up_b", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_1.bias"),
        (f"text_encoder_ssl.layers.{i}.ffn_down_w", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_2.weight"),
        (f"text_encoder_ssl.layers.{i}.ffn_down_b", f"enc_p.encoder_ssl.ffn_layers.{i}.conv_2.bias"),
        (f"text_encoder_ssl.layers.{i}.ln2_w", f"enc_p.encoder_ssl.norm_layers_2.{i}.gamma"),
        (f"text_encoder_ssl.layers.{i}.ln2_b", f"enc_p.encoder_ssl.norm_layers_2.{i}.beta"),
    ])


TEXT_ENCODER_TEXT_MAP = [
    ("text_encoder_text.text_embedding", "enc_p.text_embedding.weight"),
]

for i in range(6):
    TEXT_ENCODER_TEXT_MAP.extend([
        (f"text_encoder_text.layers.{i}.out_w", f"enc_p.encoder_text.attn_layers.{i}.conv_o.weight"),
        (f"text_encoder_text.layers.{i}.out_b", f"enc_p.encoder_text.attn_layers.{i}.conv_o.bias"),
        (f"text_encoder_text.layers.{i}.ln1_w", f"enc_p.encoder_text.norm_layers_1.{i}.gamma"),
        (f"text_encoder_text.layers.{i}.ln1_b", f"enc_p.encoder_text.norm_layers_1.{i}.beta"),
        (f"text_encoder_text.layers.{i}.ffn_up_w", f"enc_p.encoder_text.ffn_layers.{i}.conv_1.weight"),
        (f"text_encoder_text.layers.{i}.ffn_up_b", f"enc_p.encoder_text.ffn_layers.{i}.conv_1.bias"),
        (f"text_encoder_text.layers.{i}.ffn_down_w", f"enc_p.encoder_text.ffn_layers.{i}.conv_2.weight"),
        (f"text_encoder_text.layers.{i}.ffn_down_b", f"enc_p.encoder_text.ffn_layers.{i}.conv_2.bias"),
        (f"text_encoder_text.layers.{i}.ln2_w", f"enc_p.encoder_text.norm_layers_2.{i}.gamma"),
        (f"text_encoder_text.layers.{i}.ln2_b", f"enc_p.encoder_text.norm_layers_2.{i}.beta"),
    ])


TEXT_ENCODER_POST_MAP = [
    ("text_encoder_post.proj_w", "enc_p.proj.weight"),
    ("text_encoder_post.proj_b", "enc_p.proj.bias"),
]

for i in range(3):
    TEXT_ENCODER_POST_MAP.extend([
        (f"text_encoder_post.layers.{i}.out_w", f"enc_p.encoder2.attn_layers.{i}.conv_o.weight"),
        (f"text_encoder_post.layers.{i}.out_b", f"enc_p.encoder2.attn_layers.{i}.conv_o.bias"),
        (f"text_encoder_post.layers.{i}.ln1_w", f"enc_p.encoder2.norm_layers_1.{i}.gamma"),
        (f"text_encoder_post.layers.{i}.ln1_b", f"enc_p.encoder2.norm_layers_1.{i}.beta"),
        (f"text_encoder_post.layers.{i}.ffn_up_w", f"enc_p.encoder2.ffn_layers.{i}.conv_1.weight"),
        (f"text_encoder_post.layers.{i}.ffn_up_b", f"enc_p.encoder2.ffn_layers.{i}.conv_1.bias"),
        (f"text_encoder_post.layers.{i}.ffn_down_w", f"enc_p.encoder2.ffn_layers.{i}.conv_2.weight"),
        (f"text_encoder_post.layers.{i}.ffn_down_b", f"enc_p.encoder2.ffn_layers.{i}.conv_2.bias"),
        (f"text_encoder_post.layers.{i}.ln2_w", f"enc_p.encoder2.norm_layers_2.{i}.gamma"),
        (f"text_encoder_post.layers.{i}.ln2_b", f"enc_p.encoder2.norm_layers_2.{i}.beta"),
    ])


def _linearize_conv1x1(weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 3 or weight.shape[2] != 1:
        raise ValueError(f"Expected Conv1d(kernel=1) weight, got shape {weight.shape}")
    return weight[:, :, 0].astype(np.float32), bias.astype(np.float32)


def _fused_qkv(weights: dict[str, np.ndarray], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    q_w, q_b = _linearize_conv1x1(weights[f"{prefix}.conv_q.weight"], weights[f"{prefix}.conv_q.bias"])
    k_w, k_b = _linearize_conv1x1(weights[f"{prefix}.conv_k.weight"], weights[f"{prefix}.conv_k.bias"])
    v_w, v_b = _linearize_conv1x1(weights[f"{prefix}.conv_v.weight"], weights[f"{prefix}.conv_v.bias"])
    return np.concatenate([q_w, k_w, v_w], axis=0), np.concatenate([q_b, k_b, v_b], axis=0)


def _packed_rel_k(weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return weights[f"{prefix}.emb_rel_k"][0].astype(np.float32).copy()


def _packed_rel_v_t(weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return weights[f"{prefix}.emb_rel_v"][0].transpose(1, 0).astype(np.float32).copy()


def _compose_affine(
    w2: np.ndarray,
    b2: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return w2 @ w1, w2 @ b1 + b2


def _stack_weights(weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(weights, axis=0), np.concatenate(biases, axis=0)


def _convert_tensor(
    gguf_name: str,
    tensor_np: np.ndarray,
    target_type,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    if gguf_name.endswith("text_embedding"):
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32

    if gguf_name.endswith("ffn_up_w") or gguf_name.endswith("ffn_down_w"):
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32

    if tensor_np.ndim <= 1:
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32

    if target_type == gguf.GGMLQuantizationType.F32:
        return tensor_np.astype(np.float32), target_type

    if target_type == gguf.GGMLQuantizationType.F16:
        return tensor_np.astype(np.float16), target_type

    if tensor_np.ndim == 2:
        block_size = gguf.GGML_QUANT_SIZES[target_type][0]
        if tensor_np.shape[1] % block_size == 0:
            quantized = gguf.quantize(tensor_np.astype(np.float32), target_type)
            return quantized, target_type

    return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32


def _prepare_direct_tensor(gguf_name: str, tensor_np: np.ndarray) -> np.ndarray:
    if gguf_name.endswith("text_embedding"):
        return tensor_np.astype(np.float32).copy()

    if gguf_name.endswith("_w") and tensor_np.ndim == 3 and tensor_np.shape[2] == 1:
        return tensor_np[:, :, 0].astype(np.float32).copy()

    return tensor_np.astype(np.float32)


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    version = str(cfg.get("version") or "v2")
    vocab = int(weights["enc_p.text_embedding.weight"].shape[0])

    writer = gguf.GGUFWriter(output_path, "sovits_text_encoder")
    writer.add_string("sovits.block", "text_encoder")
    writer.add_string("sovits.version", version)
    writer.add_string("sovits.text_encoder.layout", "optimized_v2")
    writer.add_uint32("sovits.text_encoder.ssl_in_dim", 768)
    writer.add_uint32("sovits.text_encoder.hidden_dim", 192)
    writer.add_uint32("sovits.text_encoder.ffn_dim", 768)
    writer.add_uint32("sovits.text_encoder.n_head", 2)
    writer.add_uint32("sovits.text_encoder.ssl_n_layer", 3)
    writer.add_uint32("sovits.text_encoder.text_n_layer", 6)
    writer.add_uint32("sovits.text_encoder.post_n_layer", 3)
    writer.add_uint32("sovits.text_encoder.kernel_size", 3)
    writer.add_uint32("sovits.text_encoder.window_size", 4)
    writer.add_uint32("sovits.text_encoder.text_vocab_size", vocab)
    writer.add_uint32("sovits.text_encoder.ge_dim", 512)
    writer.add_uint32("sovits.text_encoder.out_dim", 192)

    n_converted = 0

    for section, n_layers in (("ssl", 3), ("text", 6), ("post", 3)):
        branch = {
            "ssl": "encoder_ssl",
            "text": "encoder_text",
            "post": "encoder2",
        }[section]

        for i in range(n_layers):
            prefix = f"enc_p.{branch}.attn_layers.{i}"
            qkv_w, qkv_b = _fused_qkv(weights, prefix)
            rel_k = _packed_rel_k(weights, prefix)
            rel_v_t = _packed_rel_v_t(weights, prefix)

            name_base = f"text_encoder_{section}.layers.{i}"
            qkv_w, qkv_type = _convert_tensor(f"{name_base}.qkv_w", qkv_w, target_type)
            writer.add_tensor(f"{name_base}.qkv_w", qkv_w, raw_dtype=qkv_type)
            n_converted += 1
            print(f"  [{n_converted:3d}] {name_base + '.qkv_w':36s} <- fused q/k/v weights     {list(qkv_w.shape)!s:16s} {qkv_type.name}")

            writer.add_tensor(f"{name_base}.qkv_b", qkv_b.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
            n_converted += 1
            print(f"  [{n_converted:3d}] {name_base + '.qkv_b':36s} <- fused q/k/v bias        {list(qkv_b.shape)!s:16s} float32")

            writer.add_tensor(f"{name_base}.rel_k", rel_k, raw_dtype=gguf.GGMLQuantizationType.F32)
            n_converted += 1
            print(f"  [{n_converted:3d}] {name_base + '.rel_k':36s} <- packed rel_k            {list(rel_k.shape)!s:16s} float32")

            writer.add_tensor(f"{name_base}.rel_v_t", rel_v_t, raw_dtype=gguf.GGMLQuantizationType.F32)
            n_converted += 1
            print(f"  [{n_converted:3d}] {name_base + '.rel_v_t':36s} <- packed rel_v_t          {list(rel_v_t.shape)!s:16s} float32")

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

    ssl_fused_w, ssl_fused_b = _stack_weights([q_fused_w, skip_from_ssl_w], [q_fused_b, skip_from_ssl_b])
    text_kv_w, text_kv_b = _stack_weights([k_fused_w, v_fused_w], [k_fused_b, v_fused_b])

    mrte_tensors = [
        ("text_encoder_mrte.ssl_fused_w", ssl_fused_w),
        ("text_encoder_mrte.ssl_fused_b", ssl_fused_b),
        ("text_encoder_mrte.text_kv_w", text_kv_w),
        ("text_encoder_mrte.text_kv_b", text_kv_b),
        ("text_encoder_mrte.attn_out_w", attn_out_w),
        ("text_encoder_mrte.attn_out_b", attn_out_b),
        ("text_encoder_mrte.ge_out_w", ge_out_w),
        ("text_encoder_mrte.ge_out_b", ge_out_b),
    ]

    for gguf_name, tensor_np in mrte_tensors:
        tensor_np, tensor_type = _convert_tensor(gguf_name, tensor_np, target_type)
        writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
        n_converted += 1
        print(f"  [{n_converted:3d}] {gguf_name:36s} <- fused MRTE                {list(tensor_np.shape)!s:16s} {tensor_type.name}")

    for mapping in (TEXT_ENCODER_SSL_MAP, TEXT_ENCODER_TEXT_MAP, TEXT_ENCODER_POST_MAP):
        for gguf_name, ckpt_name in mapping:
            if ckpt_name not in weights:
                raise KeyError(
                    f"Tensor '{ckpt_name}' not found in checkpoint "
                    f"(needed for GGUF tensor '{gguf_name}')"
                )

            if gguf_name.endswith("qkv_w") or gguf_name.endswith("qkv_b"):
                continue

            tensor_np = _prepare_direct_tensor(gguf_name, weights[ckpt_name])
            tensor_np, tensor_type = _convert_tensor(gguf_name, tensor_np, target_type)
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
            n_converted += 1
            print(f"  [{n_converted:3d}] {gguf_name:36s} <- {ckpt_name:48s} {list(tensor_np.shape)!s:16s} {tensor_type.name}")

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
        description="Convert full SoVITS text_encoder weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-text-encoder-<type>.gguf)",
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
        args.output = f"{base}-text-encoder-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
