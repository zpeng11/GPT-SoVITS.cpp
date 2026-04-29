#!/usr/bin/env python3
"""Convert SoVITS v2 flow (ResidualCouplingBlock) weights to GGUF format.

Usage:
    python convert_sovits_flow_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`,
fuses weight_norm'd Conv1d weights (weight_g * weight_v / ||weight_v||) at
conversion time, and exports the tensors needed by `sovits_flow_block_weights`.

All tensors are stored in their PyTorch layout. GGUF reverses dimensions when
loaded into ggml, yielding the conventions expected by `src/sovits/block.cpp`.

GGUF tensor naming (L = coupling layer {0..3}, j = WN layer {0..3}):

  flow.layers.L.pre_w / .pre_b     <- flow.flows.{L*2}.pre.weight / .bias
  flow.layers.L.post_w / .post_b   <- flow.flows.{L*2}.post.weight / .bias
  flow.layers.L.enc.cond_w / .cond_b <- fused(weight_g, weight_v) + bias
  flow.layers.L.enc.j.in_w / .in_b   <- fused(weight_g, weight_v) + bias
  flow.layers.L.enc.j.rs_w / .rs_b   <- fused(weight_g, weight_v) + bias
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
    "q8":  gguf.GGMLQuantizationType.Q8_0,
    "q5":  gguf.GGMLQuantizationType.Q5_0,
    "q4":  gguf.GGMLQuantizationType.Q4_0,
}

N_FLOWS = 4   # n_flows
N_WN = 4      # n_layers in each WN


def fuse_weight_norm(
    weight_g: np.ndarray,
    weight_v: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse weight_norm Conv1d into a single weight tensor.

    PyTorch weight_norm:  weight = weight_g * (weight_v / ||weight_v||)

    Parameters
    ----------
    weight_g : ndarray   shape (out_ch,) — scale per output channel
    weight_v : ndarray   shape (out_ch, in_ch, K) — direction tensor
    bias     : ndarray   shape (out_ch,)

    Returns
    -------
    fused_weight, bias_both as PyTorch-layout arrays.
    """
    out_ch = weight_v.shape[0]
    g = weight_g.squeeze()  # (out_ch,)
    norm = np.linalg.norm(weight_v.reshape(out_ch, -1), axis=1)  # (out_ch,)
    fused = weight_v.astype(np.float32) * (g / (norm + 1e-12)).reshape(-1, 1, 1)
    return fused, bias.astype(np.float32)


def linearize_conv1x1(weight: np.ndarray) -> np.ndarray:
    if weight.ndim == 3 and weight.shape[2] == 1:
        return weight[:, :, 0].astype(np.float32)
    return weight.astype(np.float32)


def flatten_conv1d(weight: np.ndarray) -> np.ndarray:
    if weight.ndim != 3:
        return weight.astype(np.float32)
    out_ch, in_ch, kernel = weight.shape
    return weight.reshape(out_ch, in_ch * kernel).astype(np.float32)


def convert_tensor(
    gguf_name: str,
    tensor_np: np.ndarray,
    target_type,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    """Convert tensor to target type, preferring f32 for small/1D tensors.

    Returns (data, type) tuple.
    """
    if target_type == gguf.GGMLQuantizationType.F32:
        return tensor_np.astype(np.float32), target_type

    if tensor_np.ndim <= 1 or tensor_np.shape[0] * tensor_np.shape[-1] < 256:
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32

    if target_type == gguf.GGMLQuantizationType.F16:
        return tensor_np.astype(np.float16), target_type

    # Quantized types
    block_size = gguf.GGML_QUANT_SIZES[target_type][0]
    if tensor_np.ndim == 2 and tensor_np.shape[1] % block_size == 0:
        quantized = gguf.quantize(tensor_np, target_type)
        return quantized, target_type

    return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32


def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    ckpt_weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(ckpt_weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    writer = gguf.GGUFWriter(output_path, "sovits_flow")

    writer.add_string("sovits.block", "flow")
    writer.add_string("sovits.version", str(cfg.get("version", "v2")))
    writer.add_uint32("sovits.flow.channels", 192)
    writer.add_uint32("sovits.flow.hidden", 192)
    writer.add_uint32("sovits.flow.gin", 512)
    writer.add_uint32("sovits.flow.n_flows", N_FLOWS)
    writer.add_uint32("sovits.flow.wn_layers", N_WN)
    writer.add_uint32("sovits.flow.kernel", 5)

    n_converted = 0

    for L in range(N_FLOWS):
        ckpt_idx = L * 2  # checkpoint uses even indices 0,2,4,6

        # --- pre: regular Conv1d(k=1) ---
        for suffix in ("pre", "post"):
            ckpt_weight_name = f"flow.flows.{ckpt_idx}.{suffix}.weight"
            ckpt_bias_name = f"flow.flows.{ckpt_idx}.{suffix}.bias"
            gguf_w_name = f"flow.layers.{L}.{suffix}_w"
            gguf_b_name = f"flow.layers.{L}.{suffix}_b"

            w = ckpt_weights[ckpt_weight_name].astype(np.float32)
            b = ckpt_weights[ckpt_bias_name].astype(np.float32)
            w = flatten_conv1d(w)

            w_data, w_type = convert_tensor(gguf_w_name, w, target_type)
            writer.add_tensor(gguf_w_name, w_data, raw_dtype=w_type)
            n_converted += 1
            print(f"  [{n_converted:3d}] {gguf_w_name:36s} {list(w_data.shape)!s:18s} {w_type.name}")

            b_data, b_type = convert_tensor(gguf_b_name, b, target_type)
            writer.add_tensor(gguf_b_name, b_data, raw_dtype=b_type)
            n_converted += 1
            print(f"  [{n_converted:3d}] {gguf_b_name:36s} {list(b_data.shape)!s:18s} {b_type.name}")

        # --- enc.cond_layer: weight_norm Conv1d(k=1) ---
        ckpt_prefix = f"flow.flows.{ckpt_idx}.enc.cond_layer"
        cond_g  = ckpt_weights[f"{ckpt_prefix}.weight_g"]
        cond_v  = ckpt_weights[f"{ckpt_prefix}.weight_v"]
        cond_b  = ckpt_weights[f"{ckpt_prefix}.bias"]
        cond_fused_w, cond_fused_b = fuse_weight_norm(cond_g, cond_v, cond_b)
        cond_fused_w = flatten_conv1d(cond_fused_w)

        w_data, w_type = convert_tensor(f"flow.layers.{L}.enc.cond_w", cond_fused_w, target_type)
        writer.add_tensor(f"flow.layers.{L}.enc.cond_w", w_data, raw_dtype=w_type)
        n_converted += 1
        print(f"  [{n_converted:3d}] {'flow.layers.' + str(L) + '.enc.cond_w':36s} {list(w_data.shape)!s:18s} {w_type.name}")

        b_data, b_type = convert_tensor(f"flow.layers.{L}.enc.cond_b", cond_fused_b, target_type)
        writer.add_tensor(f"flow.layers.{L}.enc.cond_b", b_data, raw_dtype=b_type)
        n_converted += 1
        print(f"  [{n_converted:3d}] {'flow.layers.' + str(L) + '.enc.cond_b':36s} {list(b_data.shape)!s:18s} {b_type.name}")

        # --- enc in_layers and res_skip_layers (4 WN layers) ---
        for j in range(N_WN):
            for layer_type in ("in", "rs"):
                ckpt_layer_prefix = f"flow.flows.{ckpt_idx}.enc.{'in' if layer_type == 'in' else 'res_skip'}_layers.{j}"
                g_name = f"{ckpt_layer_prefix}.weight_g"
                v_name = f"{ckpt_layer_prefix}.weight_v"
                b_name = f"{ckpt_layer_prefix}.bias"
                gguf_w_name = f"flow.layers.{L}.enc.{j}.{layer_type}_w"
                gguf_b_name = f"flow.layers.{L}.enc.{j}.{layer_type}_b"

                fused_w, fused_b = fuse_weight_norm(
                    ckpt_weights[g_name],
                    ckpt_weights[v_name],
                    ckpt_weights[b_name],
                )
                fused_w = flatten_conv1d(fused_w)

                w_data, w_type = convert_tensor(gguf_w_name, fused_w, target_type)
                writer.add_tensor(gguf_w_name, w_data, raw_dtype=w_type)
                n_converted += 1
                print(f"  [{n_converted:3d}] {gguf_w_name:36s} {list(w_data.shape)!s:18s} {w_type.name}")

                b_data, b_type = convert_tensor(gguf_b_name, fused_b, target_type)
                writer.add_tensor(gguf_b_name, b_data, raw_dtype=b_type)
                n_converted += 1
                print(f"  [{n_converted:3d}] {gguf_b_name:36s} {list(b_data.shape)!s:18s} {b_type.name}")

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
        description="Convert SoVITS v2 flow (ResidualCouplingBlock) weights to GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-flow-<type>.gguf)",
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
        args.output = f"{base}-flow-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
