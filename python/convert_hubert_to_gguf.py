#!/usr/bin/env python3
"""Convert a HuggingFace chinese-hubert-base checkpoint to GGUF format.

Usage:
    python convert_hubert_to_gguf.py <model_dir> [--output <path>] [--type f32|f16|q8|q5|q4]

Where <model_dir> contains pytorch_model.bin and config.json.

Quantization types (applied to 2D+ tensors only; 1D biases/norms stay f32):
    q8  - Q8_0  (8.5 bits/weight, block size 32)
    q5  - Q5_0  (5.5 bits/weight, block size 32)
    q4  - Q4_0  (4.5 bits/weight, block size 32)
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import gguf


# ggml dtype constants
GGML_TYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8":  gguf.GGMLQuantizationType.Q8_0,
    "q5":  gguf.GGMLQuantizationType.Q5_0,
    "q4":  gguf.GGMLQuantizationType.Q4_0,
}

# Tensors that must stay f32 regardless of the requested quantization type
# (1D tensors are always kept f32; this set covers named exceptions).
_F32_ONLY_NAMES = set()  # currently none beyond 1D tensors


def should_quantize(name: str, tensor: np.ndarray, block_size: int) -> bool:
    """Return True if this tensor should be quantized (2D+ weight, block-aligned).

    Conv1d kernels (3D) are NOT quantized because ggml_conv_1d requires the
    original 3D layout and does not support quantized kernels.  They are
    stored as f16 instead.
    """
    if name in _F32_ONLY_NAMES:
        return False
    if tensor.ndim != 2:
        return False
    return tensor.shape[1] % block_size == 0


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_state_dict(model_dir: str) -> dict:
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {model_dir}")
    return torch.load(bin_path, map_location="cpu", weights_only=True)


def transpose_for_ggml(name: str, tensor: np.ndarray) -> np.ndarray:
    """Transpose a tensor from PyTorch layout to ggml layout.

    Rules:
      - 1D tensors (biases, norms, weight_g flattened): keep as-is
      - 2D Linear (out_features, in_features) -> keep as-is.
        ggml reads numpy C-order and reverses dims, so numpy (out, in) -> ggml
        ne={in, out}, which is what ggml_mul_mat expects for weight tensors.
      - 3D Conv1d (out_channels, in_channels, kernel) -> keep as-is.
        ggml reads numpy (OC, IC, K) C-order and reverses to ne={K, IC, OC},
        which is what the HuBERT conv path expects for its kernel argument.
      - weight_g (1, 1, K) -> flatten to (K,)
    """
    ndim = tensor.ndim

    # Special case: weight_g has shape (1, 1, K) -- flatten to 1D
    if "weight_g" in name:
        return tensor.flatten()

    if ndim == 1:
        return tensor
    elif ndim == 2:
        return tensor
    elif ndim == 3:
        return tensor
    else:
        raise ValueError(f"Unexpected tensor ndim={ndim} for {name}, shape={tensor.shape}")


def convert(model_dir: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading config from {model_dir}...")
    config = load_config(model_dir)

    print(f"Loading state dict from {model_dir}...")
    state_dict = load_state_dict(model_dir)
    print(f"  Found {len(state_dict)} tensors")

    target_type = GGML_TYPES[dtype_str]
    is_quantized = target_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    block_size = gguf.GGML_QUANT_SIZES[target_type][0] if is_quantized else 0
    print(f"  Output type: {dtype_str} ({target_type.name})")

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "hubert")

    # Write model hyperparameters as KV metadata
    gguf_writer.add_uint32("hubert.hidden_size", config["hidden_size"])
    gguf_writer.add_uint32("hubert.intermediate_size", config["intermediate_size"])
    gguf_writer.add_uint32("hubert.num_hidden_layers", config["num_hidden_layers"])
    gguf_writer.add_uint32("hubert.num_attention_heads", config["num_attention_heads"])
    gguf_writer.add_uint32("hubert.num_feat_extract_layers", len(config["conv_dim"]))
    gguf_writer.add_uint32("hubert.num_conv_pos_embeddings", config["num_conv_pos_embeddings"])
    gguf_writer.add_uint32("hubert.num_conv_pos_embedding_groups", config["num_conv_pos_embedding_groups"])

    # Conv feature extractor params (as arrays)
    gguf_writer.add_array("hubert.conv_dim", config["conv_dim"])
    gguf_writer.add_array("hubert.conv_kernel", config["conv_kernel"])
    gguf_writer.add_array("hubert.conv_stride", config["conv_stride"])

    # Process tensors
    n_skipped = 0
    n_converted = 0
    n_quantized = 0

    for name, param in sorted(state_dict.items()):
        tensor_np = param.numpy()

        # Skip masked_spec_embed -- not needed for inference
        if name == "masked_spec_embed":
            print(f"  Skipping {name} (not needed for inference)")
            n_skipped += 1
            continue

        # Transpose to ggml layout
        tensor_ggml = transpose_for_ggml(name, tensor_np)

        if is_quantized and should_quantize(name, tensor_ggml, block_size):
            # 2D weight: quantize row-wise.  The uint8 byte array's shape is
            # the quantized shape; GGUFWriter recovers the float shape via
            # quant_shape_from_byte_shape.
            quantized = gguf.quantize(tensor_ggml, target_type)
            gguf_writer.add_tensor(name, quantized, raw_dtype=target_type)
            n_quantized += 1
            data_type = target_type
            out_shape = tensor_ggml.shape
        elif (is_quantized and tensor_ggml.ndim >= 2 and not should_quantize(name, tensor_ggml, block_size)):
            # 3D+ tensors (Conv1d kernels) that can't be block-quantized:
            # store as f16 to still save space.
            tensor_ggml = tensor_ggml.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            gguf_writer.add_tensor(name, tensor_ggml, raw_dtype=data_type)
            out_shape = tensor_ggml.shape
        elif target_type == gguf.GGMLQuantizationType.F16 and tensor_ggml.ndim >= 2:
            tensor_ggml = tensor_ggml.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            gguf_writer.add_tensor(name, tensor_ggml, raw_dtype=data_type)
            out_shape = tensor_ggml.shape
        else:
            # f32 path (also used for 1D tensors when quantization is requested)
            tensor_ggml = tensor_ggml.astype(np.float32)
            data_type = gguf.GGMLQuantizationType.F32
            gguf_writer.add_tensor(name, tensor_ggml, raw_dtype=data_type)
            out_shape = tensor_ggml.shape

        n_converted += 1

        if n_converted <= 5 or n_converted % 50 == 0:
            print(f"  [{n_converted:3d}] {name:60s} {str(param.shape):20s} -> ggml {str(out_shape):20s} {data_type.name}")

    if n_quantized > 0:
        print(f"\nConverted {n_converted} tensors (quantized {n_quantized}), skipped {n_skipped}")
    else:
        print(f"\nConverted {n_converted} tensors, skipped {n_skipped}")
    print(f"Writing GGUF to {output_path}...")

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    file_size = os.path.getsize(output_path)
    print(f"Done! Output: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace chinese-hubert-base to GGUF")
    parser.add_argument("model_dir", help="Path to HF model directory (containing pytorch_model.bin and config.json)")
    parser.add_argument("--output", "-o", default=None, help="Output GGUF file path (default: <model_dir>/chinese-hubert-base-<type>.gguf)")
    parser.add_argument("--type", "-t", dest="dtype", default="f32", choices=list(GGML_TYPES.keys()), help="Output data type (default: f32)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.model_dir, f"chinese-hubert-base-{args.dtype}.gguf")

    convert(args.model_dir, args.output, args.dtype)


if __name__ == "__main__":
    main()
