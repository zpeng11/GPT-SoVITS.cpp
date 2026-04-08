#!/usr/bin/env python3
"""Convert a HuggingFace chinese-roberta-wwm-ext-large checkpoint to GGUF format.

Usage:
    python convert_roberta_to_gguf.py <model_dir> [--output <path>] [--type f32|f16|q8|q5|q4]

Where <model_dir> contains pytorch_model.bin and config.json.

Quantization types (applied to 2D weight tensors only; 1D biases/norms stay f32):
    q8  - Q8_0  (8.5 bits/weight, block size 32)
    q5  - Q5_0  (5.5 bits/weight, block size 32)
    q4  - Q4_0  (4.5 bits/weight, block size 32)
"""

import argparse
import json
import os
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

# Tensor name prefixes that are not needed for inference.
_SKIP_PREFIXES = ("bert.pooler.", "cls.")


def should_quantize(name: str, tensor: np.ndarray, block_size: int) -> bool:
    """Return True if this tensor should be quantized (2D weight, block-aligned)."""
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
    gguf_writer = gguf.GGUFWriter(output_path, "roberta")

    # Write model hyperparameters as KV metadata
    gguf_writer.add_uint32("roberta.hidden_size", config["hidden_size"])
    gguf_writer.add_uint32("roberta.num_hidden_layers", config["num_hidden_layers"])
    gguf_writer.add_uint32("roberta.num_attention_heads", config["num_attention_heads"])
    gguf_writer.add_uint32("roberta.intermediate_size", config["intermediate_size"])
    gguf_writer.add_uint32("roberta.max_position_embeddings", config["max_position_embeddings"])
    gguf_writer.add_uint32("roberta.vocab_size", config["vocab_size"])
    gguf_writer.add_uint32("roberta.type_vocab_size", config["type_vocab_size"])
    gguf_writer.add_float32("roberta.layer_norm_eps", config["layer_norm_eps"])

    # Process tensors
    n_skipped = 0
    n_converted = 0
    n_quantized = 0

    for name, param in sorted(state_dict.items()):
        # Skip pooler and MLM head — not needed for inference
        if name.startswith(_SKIP_PREFIXES):
            n_skipped += 1
            continue

        tensor_np = param.numpy()

        # PyTorch Linear weight {out_features, in_features} maps directly to
        # ggml layout: ggml reads numpy C-order and reverses dims, so numpy
        # (out, in) -> ggml ne={in, out}, which is correct for ggml_mul_mat.
        # No transposition needed for any RoBERTa tensors (all 1D or 2D).

        if is_quantized and should_quantize(name, tensor_np, block_size):
            quantized = gguf.quantize(tensor_np, target_type)
            gguf_writer.add_tensor(name, quantized, raw_dtype=target_type)
            n_quantized += 1
            data_type = target_type
            out_shape = tensor_np.shape
        elif target_type == gguf.GGMLQuantizationType.F16 and tensor_np.ndim >= 2:
            tensor_np = tensor_np.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            gguf_writer.add_tensor(name, tensor_np, raw_dtype=data_type)
            out_shape = tensor_np.shape
        else:
            tensor_np = tensor_np.astype(np.float32)
            data_type = gguf.GGMLQuantizationType.F32
            gguf_writer.add_tensor(name, tensor_np, raw_dtype=data_type)
            out_shape = tensor_np.shape

        n_converted += 1

        if n_converted <= 5 or n_converted % 50 == 0:
            print(f"  [{n_converted:3d}] {name:70s} {str(list(param.shape)):20s} -> {str(list(out_shape)):20s} {data_type.name}")

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
    parser = argparse.ArgumentParser(description="Convert HuggingFace chinese-roberta-wwm-ext-large to GGUF")
    parser.add_argument("model_dir", help="Path to HF model directory (containing pytorch_model.bin and config.json)")
    parser.add_argument("--output", "-o", default=None, help="Output GGUF file path (default: <model_dir>/chinese-roberta-wwm-ext-large-<type>.gguf)")
    parser.add_argument("--type", "-t", dest="dtype", default="f32", choices=list(GGML_TYPES.keys()), help="Output data type (default: f32)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.model_dir, f"chinese-roberta-wwm-ext-large-{args.dtype}.gguf")

    convert(args.model_dir, args.output, args.dtype)


if __name__ == "__main__":
    main()
