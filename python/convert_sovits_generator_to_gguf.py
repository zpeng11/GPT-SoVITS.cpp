#!/usr/bin/env python3
"""Convert SoVITS v2 Generator weights to GGUF format.

Usage:
    python convert_sovits_generator_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`,
fuses weight_norm'd ConvTranspose1d / Conv1d weights at conversion time, and
exports the tensors needed by `sovits_generator_block_weights`.

All tensors are stored in their PyTorch layout. GGUF reverses dimensions when
loaded into ggml, yielding the conventions expected by `src/sovits/block.cpp`.
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

N_STAGES = 5
N_BRANCHES = 3
N_RES_LAYERS = 3
UPSAMPLE_RATES = [10, 8, 2, 2, 2]
UPSAMPLE_KERNELS = [16, 16, 8, 2, 2]
UPSAMPLE_OUT_CHANNELS = [256, 128, 64, 32, 16]
RESBLOCK_KERNELS = [3, 7, 11]
RESBLOCK_DILATIONS = [1, 3, 5]

DIRECT_MAP = [
    ("generator.conv_pre_w", "dec.conv_pre.weight"),
    ("generator.conv_pre_b", "dec.conv_pre.bias"),
    ("generator.cond_w", "dec.cond.weight"),
    ("generator.cond_b", "dec.cond.bias"),
    ("generator.conv_post_w", "dec.conv_post.weight"),
]


def fuse_weight_norm(
    weight_g: np.ndarray,
    weight_v: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_ch = weight_v.shape[0]
    g = weight_g.squeeze()
    norm = np.linalg.norm(weight_v.reshape(out_ch, -1), axis=1)
    fused = weight_v.astype(np.float32) * (g / (norm + 1e-12)).reshape(-1, 1, 1)
    return fused, bias.astype(np.float32)


def convert_tensor(
    gguf_name: str,
    tensor_np: np.ndarray,
    target_type,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    if target_type == gguf.GGMLQuantizationType.F32:
        return tensor_np.astype(np.float32), target_type

    if tensor_np.ndim <= 1 or tensor_np.shape[0] * tensor_np.shape[-1] < 256:
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32

    if target_type == gguf.GGMLQuantizationType.F16:
        return tensor_np.astype(np.float16), target_type

    block_size = gguf.GGML_QUANT_SIZES[target_type][0]
    if tensor_np.ndim == 2 and tensor_np.shape[1] % block_size == 0:
        quantized = gguf.quantize(tensor_np, target_type)
        return quantized, target_type

    return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32


def add_tensor(writer, gguf_name: str, tensor_np: np.ndarray, target_type, note: str, n_converted: int) -> int:
    tensor_data, tensor_type = convert_tensor(gguf_name, tensor_np, target_type)
    writer.add_tensor(gguf_name, tensor_data, raw_dtype=tensor_type)
    n_converted += 1
    print(f"  [{n_converted:3d}] {gguf_name:40s} <- {note:28s} {list(tensor_data.shape)!s:18s} {tensor_type.name}")
    return n_converted


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_generator")
    writer.add_string("sovits.block", "generator")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.generator.in_dim", 192)
    writer.add_uint32("sovits.generator.gin_dim", int(cfg.get("gin_channels", 512)))
    writer.add_uint32("sovits.generator.n_stages", N_STAGES)
    writer.add_uint32("sovits.generator.n_branches", N_BRANCHES)
    writer.add_uint32("sovits.generator.res_layers", N_RES_LAYERS)
    writer.add_uint32("sovits.generator.upsample_factor", int(np.prod(UPSAMPLE_RATES)))

    n_converted = 0

    for gguf_name, ckpt_name in DIRECT_MAP:
        if ckpt_name not in weights:
            raise KeyError(f"Tensor '{ckpt_name}' not found in checkpoint")
        n_converted = add_tensor(writer, gguf_name, weights[ckpt_name].astype(np.float32), target_type, ckpt_name, n_converted)

    for stage in range(N_STAGES):
        up_prefix = f"dec.ups.{stage}"
        fused_w, fused_b = fuse_weight_norm(
            weights[f"{up_prefix}.weight_g"],
            weights[f"{up_prefix}.weight_v"],
            weights[f"{up_prefix}.bias"],
        )
        n_converted = add_tensor(writer, f"generator.stages.{stage}.up_w", fused_w, target_type, f"fused {up_prefix}", n_converted)
        n_converted = add_tensor(writer, f"generator.stages.{stage}.up_b", fused_b, target_type, f"bias {up_prefix}", n_converted)

        for branch in range(N_BRANCHES):
            res_idx = stage * N_BRANCHES + branch
            for layer in range(N_RES_LAYERS):
                c1_prefix = f"dec.resblocks.{res_idx}.convs1.{layer}"
                c2_prefix = f"dec.resblocks.{res_idx}.convs2.{layer}"

                c1_w, c1_b = fuse_weight_norm(
                    weights[f"{c1_prefix}.weight_g"],
                    weights[f"{c1_prefix}.weight_v"],
                    weights[f"{c1_prefix}.bias"],
                )
                c2_w, c2_b = fuse_weight_norm(
                    weights[f"{c2_prefix}.weight_g"],
                    weights[f"{c2_prefix}.weight_v"],
                    weights[f"{c2_prefix}.bias"],
                )

                base = f"generator.stages.{stage}.resblocks.{branch}"
                n_converted = add_tensor(writer, f"{base}.convs1.{layer}.w", c1_w, target_type, f"fused {c1_prefix}", n_converted)
                n_converted = add_tensor(writer, f"{base}.convs1.{layer}.b", c1_b, target_type, f"bias {c1_prefix}", n_converted)
                n_converted = add_tensor(writer, f"{base}.convs2.{layer}.w", c2_w, target_type, f"fused {c2_prefix}", n_converted)
                n_converted = add_tensor(writer, f"{base}.convs2.{layer}.b", c2_b, target_type, f"bias {c2_prefix}", n_converted)

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
        description="Convert SoVITS v2 Generator weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-generator-<type>.gguf)",
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
        args.output = f"{base}-generator-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
