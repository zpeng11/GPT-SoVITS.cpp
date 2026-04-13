#!/usr/bin/env python3
"""Convert T2S + SoVITS extract-latent weights to GGUF format.

Usage:
    python convert_t2s_to_gguf.py <t2s_ckpt> <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

Where:
    <t2s_ckpt>    - Path to T2S checkpoint (.ckpt), e.g. s1v3.ckpt
    <sovits_ckpt> - Path to SoVITS checkpoint (.pth), e.g. s2Gv2ProPlus.pth

Tensor mapping (checkpoint name → GGUF name):
  T2S attention (24 layers):
    model.h.layers.{i}.self_attn.in_proj_weight  → attention.{i}.qkv_w
    model.h.layers.{i}.self_attn.in_proj_bias     → attention.{i}.qkv_b
    model.h.layers.{i}.self_attn.out_proj.weight  → attention.{i}.out_proj_w
    model.h.layers.{i}.self_attn.out_proj.bias    → attention.{i}.out_proj_b
    model.h.layers.{i}.norm1.weight               → attention.{i}.ln1_w
    model.h.layers.{i}.norm1.bias                 → attention.{i}.ln1_b
    model.h.layers.{i}.linear1.weight             → attention.{i}.ffn_up_w
    model.h.layers.{i}.linear1.bias               → attention.{i}.ffn_up_b
    model.h.layers.{i}.linear2.weight             → attention.{i}.ffn_down_w
    model.h.layers.{i}.linear2.bias               → attention.{i}.ffn_down_b
    model.h.layers.{i}.norm2.weight               → attention.{i}.ln2_w
    model.h.layers.{i}.norm2.bias                 → attention.{i}.ln2_b

  T2S encoder:
    model.ar_text_embedding.word_embeddings.weight → encoder.text_embedding
    model.bert_proj.weight                         → encoder.bert_proj_w
    model.bert_proj.bias                           → encoder.bert_proj_b
    model.ar_text_position.alpha                   → encoder.text_pos_alpha
    model.ar_audio_embedding.word_embeddings.weight → encoder.audio_embedding
    model.ar_audio_position.alpha                  → encoder.audio_pos_alpha

  T2S sampler:
    model.ar_predict_layer.weight                  → sampler.lm_head_w

  SoVITS extract latent:
    ssl_proj.weight                                → extract_latent.ssl_proj_w
    ssl_proj.bias                                  → extract_latent.ssl_proj_b
    quantizer.vq.layers.0._codebook.embed          → extract_latent.codebook

All tensors are stored as-is (PyTorch layout); GGUF's dim-reversal produces
the correct ggml convention (ne[0] = innermost dim).

Quantization types (applied to 2D Linear weights only; embeddings, 1D
biases/norms, and 3D Conv1d kernels stay f16):
    q8  - Q8_0  (8.5 bits/weight, block size 32)
    q5  - Q5_0  (5.5 bits/weight, block size 32)
    q4  - Q4_0  (4.5 bits/weight, block size 32)
"""

import argparse
import os
import sys

import numpy as np
import gguf

from torch_ckpt_utils import load_checkpoint


# ggml dtype constants
GGML_TYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8":  gguf.GGMLQuantizationType.Q8_0,
    "q5":  gguf.GGMLQuantizationType.Q5_0,
    "q4":  gguf.GGMLQuantizationType.Q4_0,
}

# Embedding tables must NOT be quantized — they are accessed via
# ggml_get_rows which dequantizes one row at a time, producing
# unacceptable error for the large dynamic range of embedding values.
# The codebook is also kept un-quantized to maintain lookup accuracy.
_EMBEDDING_NAMES = {
    "encoder.text_embedding",
    "encoder.audio_embedding",
    "extract_latent.codebook",
}


def should_quantize(gguf_name: str, tensor: np.ndarray, block_size: int) -> bool:
    """Return True if this tensor should be quantized (2D weight, block-aligned)."""
    if gguf_name in _EMBEDDING_NAMES:
        return False
    if tensor.ndim != 2:
        return False
    return tensor.shape[1] % block_size == 0


# ---------------------------------------------------------------------------
# Weight mapping tables: (GGUF name pattern, checkpoint name pattern)
# ---------------------------------------------------------------------------

T2S_ATTENTION_MAP = [
    ("attention.{i}.qkv_w",       "model.h.layers.{i}.self_attn.in_proj_weight"),
    ("attention.{i}.qkv_b",       "model.h.layers.{i}.self_attn.in_proj_bias"),
    ("attention.{i}.out_proj_w",  "model.h.layers.{i}.self_attn.out_proj.weight"),
    ("attention.{i}.out_proj_b",  "model.h.layers.{i}.self_attn.out_proj.bias"),
    ("attention.{i}.ln1_w",       "model.h.layers.{i}.norm1.weight"),
    ("attention.{i}.ln1_b",       "model.h.layers.{i}.norm1.bias"),
    ("attention.{i}.ffn_up_w",    "model.h.layers.{i}.linear1.weight"),
    ("attention.{i}.ffn_up_b",    "model.h.layers.{i}.linear1.bias"),
    ("attention.{i}.ffn_down_w",  "model.h.layers.{i}.linear2.weight"),
    ("attention.{i}.ffn_down_b",  "model.h.layers.{i}.linear2.bias"),
    ("attention.{i}.ln2_w",       "model.h.layers.{i}.norm2.weight"),
    ("attention.{i}.ln2_b",       "model.h.layers.{i}.norm2.bias"),
]

T2S_ENCODER_MAP = [
    ("encoder.text_embedding",    "model.ar_text_embedding.word_embeddings.weight"),
    ("encoder.bert_proj_w",       "model.bert_proj.weight"),
    ("encoder.bert_proj_b",       "model.bert_proj.bias"),
    ("encoder.text_pos_alpha",    "model.ar_text_position.alpha"),
    ("encoder.audio_embedding",   "model.ar_audio_embedding.word_embeddings.weight"),
    ("encoder.audio_pos_alpha",   "model.ar_audio_position.alpha"),
]

T2S_SAMPLER_MAP = [
    ("sampler.lm_head_w",         "model.ar_predict_layer.weight"),
]

SOVITS_EXTRACT_LATENT_MAP = [
    ("extract_latent.ssl_proj_w", "ssl_proj.weight"),
    ("extract_latent.ssl_proj_b", "ssl_proj.bias"),
    ("extract_latent.codebook",   "quantizer.vq.layers.0._codebook.embed"),
]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(t2s_path: str, sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading T2S checkpoint: {t2s_path}")
    w1, meta1 = load_checkpoint(t2s_path)
    cfg1 = meta1.get("config", {}).get("model", {})
    print(f"  Found {len(w1)} tensors")
    print(f"  Config: d_model={cfg1.get('embedding_dim')}, n_layer={cfg1.get('n_layer')}, "
          f"n_head={cfg1.get('head')}, vocab={cfg1.get('vocab_size')}")

    print(f"Loading SoVITS checkpoint: {sovits_path}")
    w2, meta2 = load_checkpoint(sovits_path)
    cfg2 = meta2.get("config", {}).get("model", {})
    print(f"  Found {len(w2)} tensors")
    print(f"  Config: inter_channels={cfg2.get('inter_channels')}, "
          f"semantic_frame_rate={cfg2.get('semantic_frame_rate')}")

    target_type = GGML_TYPES[dtype_str]
    is_quantized = target_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    block_size = gguf.GGML_QUANT_SIZES[target_type][0] if is_quantized else 0
    print(f"  Output type: {dtype_str} ({target_type.name})")

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "t2s")

    # Write T2S model hyperparameters as KV metadata
    gguf_writer.add_uint32("t2s.embedding_dim",      cfg1.get("embedding_dim", 512))
    gguf_writer.add_uint32("t2s.hidden_dim",         cfg1.get("hidden_dim", 512))
    gguf_writer.add_uint32("t2s.head",               cfg1.get("head", 16))
    gguf_writer.add_uint32("t2s.linear_units",       cfg1.get("linear_units", 2048))
    gguf_writer.add_uint32("t2s.n_layer",            cfg1.get("n_layer", 24))
    gguf_writer.add_uint32("t2s.vocab_size",         cfg1.get("vocab_size", 1025))
    gguf_writer.add_uint32("t2s.phoneme_vocab_size", cfg1.get("phoneme_vocab_size", 732))
    gguf_writer.add_uint32("t2s.eos",                cfg1.get("EOS", 1024))

    # SoVITS extract-latent metadata
    gguf_writer.add_uint32("sovits.inter_channels", cfg2.get("inter_channels", 192))

    # -----------------------------------------------------------------------
    # Collect all (gguf_name, ckpt_name, source_weights) entries
    # -----------------------------------------------------------------------
    n_layer = cfg1.get("n_layer", 24)
    entries = []

    # 24 attention layers
    for i in range(n_layer):
        for gguf_tmpl, ckpt_tmpl in T2S_ATTENTION_MAP:
            entries.append((gguf_tmpl.format(i=i), ckpt_tmpl.format(i=i), w1))

    # Encoder
    for gguf_name, ckpt_name in T2S_ENCODER_MAP:
        entries.append((gguf_name, ckpt_name, w1))

    # Sampler
    for gguf_name, ckpt_name in T2S_SAMPLER_MAP:
        entries.append((gguf_name, ckpt_name, w1))

    # SoVITS extract latent
    for gguf_name, ckpt_name in SOVITS_EXTRACT_LATENT_MAP:
        entries.append((gguf_name, ckpt_name, w2))

    # -----------------------------------------------------------------------
    # Process tensors
    # -----------------------------------------------------------------------
    n_converted = 0
    n_quantized = 0

    for gguf_name, ckpt_name, source in entries:
        if ckpt_name not in source:
            raise KeyError(f"Tensor '{ckpt_name}' not found in checkpoint "
                           f"(needed for GGUF tensor '{gguf_name}')")

        tensor_np = source[ckpt_name]

        # All tensors stored as-is in PyTorch layout; GGUF's automatic dim
        # reversal produces the correct ggml convention.

        if is_quantized and should_quantize(gguf_name, tensor_np, block_size):
            quantized = gguf.quantize(tensor_np, target_type)
            gguf_writer.add_tensor(gguf_name, quantized, raw_dtype=target_type)
            n_quantized += 1
            data_type = target_type
        elif ((is_quantized or target_type == gguf.GGMLQuantizationType.F16)
              and tensor_np.ndim >= 2
              and gguf_name != "extract_latent.codebook"):
            # 2D+ tensors: store as f16 to save space.
            # This covers embedding tables, Linear weights, and Conv1d kernels.
            # Exception: codebook stays F32 because ggml_sum_rows (used for
            # nearest-code distance computation) only supports F32.
            tensor_np = tensor_np.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            gguf_writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)
        else:
            # 1D tensors (biases, norms, alpha scalars): keep as f32
            tensor_np = tensor_np.astype(np.float32)
            data_type = gguf.GGMLQuantizationType.F32
            gguf_writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)

        n_converted += 1

        if n_converted <= 5 or n_converted % 50 == 0:
            print(f"  [{n_converted:3d}] {gguf_name:40s} <- {ckpt_name}")

    if n_quantized > 0:
        print(f"\nConverted {n_converted} tensors (quantized {n_quantized})")
    else:
        print(f"\nConverted {n_converted} tensors")
    print(f"Writing GGUF to {output_path}...")

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    file_size = os.path.getsize(output_path)
    print(f"Done! Output: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert T2S + SoVITS extract-latent weights to GGUF")
    parser.add_argument("t2s_ckpt",
                        help="Path to T2S checkpoint (.ckpt), e.g. s1v3.ckpt")
    parser.add_argument("sovits_ckpt",
                        help="Path to SoVITS checkpoint (.pth), e.g. s2Gv2ProPlus.pth")
    parser.add_argument("--output", "-o", default=None,
                        help="Output GGUF file path (default: <t2s_ckpt_stem>-<type>.gguf)")
    parser.add_argument("--type", "-t", dest="dtype", default="f16",
                        choices=list(GGML_TYPES.keys()),
                        help="Output data type (default: f16)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.t2s_ckpt))[0]
        args.output = f"{base}-{args.dtype}.gguf"

    convert(args.t2s_ckpt, args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
