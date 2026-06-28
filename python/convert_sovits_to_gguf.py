#!/usr/bin/env python3
"""Convert the full SoVITS v2 model (all 5 inference blocks) to a SINGLE GGUF.

This unified converter replaces the 5 per-block scripts
(convert_sovits_ref_enc_to_gguf.py, ..._quantizer_..., ..._text_encoder_...,
..._flow_..., ..._generator_...). It loads the checkpoint once and emits every
block into one GGUF writer, so inference manages a single weights file instead
of 5 spliced together.

The 5 blocks use disjoint GGUF tensor-name prefixes, so they coexist in one
file without collisions:

    ref_enc.*            MelStyleEncoder (reference style embedding ge)
    quantizer.*          single-layer RVQ codebook
    text_encoder_ssl.*   enc_p SSL branch (3 relpos layers)
    text_encoder_text.*  enc_p text branch (6 relpos layers)
    text_encoder_mrte.*  enc_p MRTE branch (fused cross-attention)
    text_encoder_post.*  enc_p post branch (3 relpos layers + proj)
    flow.*               ResidualCouplingBlock inverse flow
    generator.*          HiFi-GAN Generator (decoder)

Each block keeps its exact original conversion + quantization policy (copied
verbatim from the per-block scripts), so the unified output is byte-for-byte
equivalent to concatenating the 5 separate GGUFs and the existing parity
tolerances are preserved.

Usage:
    python convert_sovits_to_gguf.py <sovits_ckpt> [--output <path>] [--type f32|f16|q8|q5|q4]

Where:
    <sovits_ckpt> - Path to a SoVITS generator checkpoint (.pth), e.g. s2G2333k.pth

Reference checkpoint (default local source):
    tests/t2s/models/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth

Default conversion target directory:
    tests/sovits/models/   (e.g. v2-sovits-f16.gguf)

This converter is torch-free. It reads the checkpoint with `torch_ckpt_utils`.
All tensors are stored in their PyTorch layout; GGUF reverses dimensions when
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
    "q8":  gguf.GGMLQuantizationType.Q8_0,
    "q5":  gguf.GGMLQuantizationType.Q5_0,
    "q4":  gguf.GGMLQuantizationType.Q4_0,
}

# Per-block architecture constants (fixed by the shipped v2 checkpoint).
N_FLOWS = 4          # flow coupling layers
N_WN = 4             # WN dilated layers per coupling layer
N_STAGES = 5         # generator upsample stages
N_BRANCHES = 3       # generator ResBlock1 branches per stage
N_RES_LAYERS = 3     # generator residual sublayers per ResBlock1


# ---------------------------------------------------------------------------
# Shared low-level helpers (used by flow + generator)
# ---------------------------------------------------------------------------

def fuse_weight_norm(
    weight_g: np.ndarray,
    weight_v: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse weight_norm Conv1d/ConvTranspose1d into a single weight tensor.

    PyTorch weight_norm:  weight = weight_g * (weight_v / ||weight_v||)
    """
    out_ch = weight_v.shape[0]
    g = weight_g.squeeze()  # (out_ch,)
    norm = np.linalg.norm(weight_v.reshape(out_ch, -1), axis=1)  # (out_ch,)
    fused = weight_v.astype(np.float32) * (g / (norm + 1e-12)).reshape(-1, 1, 1)
    return fused, bias.astype(np.float32)


def flatten_conv1d(weight: np.ndarray) -> np.ndarray:
    """[out_ch, in_ch, K] -> [out_ch, in_ch*K] for runtime im2col + mul_mat."""
    if weight.ndim != 3:
        return weight.astype(np.float32)
    out_ch, in_ch, kernel = weight.shape
    return weight.reshape(out_ch, in_ch * kernel).astype(np.float32)


def linearize_conv1x1(weight: np.ndarray, bias: np.ndarray | None = None):
    """[out, in, 1] -> [out, in] 2D linear weight (drop the singleton kernel)."""
    if weight.ndim == 3 and weight.shape[2] == 1:
        w = weight[:, :, 0].astype(np.float32)
    else:
        w = weight.astype(np.float32)
    if bias is None:
        return w
    return w, bias.astype(np.float32)


# ===========================================================================
# Block: ref_enc (MelStyleEncoder)  -- verbatim policy from the original script
# ===========================================================================

REF_ENC_MAP = [
    ("ref_enc.spectral_1_w", "ref_enc.spectral.0.fc.weight"),
    ("ref_enc.spectral_1_b", "ref_enc.spectral.0.fc.bias"),
    ("ref_enc.spectral_2_w", "ref_enc.spectral.3.fc.weight"),
    ("ref_enc.spectral_2_b", "ref_enc.spectral.3.fc.bias"),
    ("ref_enc.temporal.0.conv_w", "ref_enc.temporal.0.conv1.conv.weight"),
    ("ref_enc.temporal.0.conv_b", "ref_enc.temporal.0.conv1.conv.bias"),
    ("ref_enc.temporal.1.conv_w", "ref_enc.temporal.1.conv1.conv.weight"),
    ("ref_enc.temporal.1.conv_b", "ref_enc.temporal.1.conv1.conv.bias"),
    ("ref_enc.attention.out_w", "ref_enc.slf_attn.fc.weight"),
    ("ref_enc.attention.out_b", "ref_enc.slf_attn.fc.bias"),
    ("ref_enc.fc_w", "ref_enc.fc.fc.weight"),
    ("ref_enc.fc_b", "ref_enc.fc.fc.bias"),
]


def _ref_enc_fused_qkv(weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    q_w = weights["ref_enc.slf_attn.w_qs.weight"].astype(np.float32)
    k_w = weights["ref_enc.slf_attn.w_ks.weight"].astype(np.float32)
    v_w = weights["ref_enc.slf_attn.w_vs.weight"].astype(np.float32)
    q_b = weights["ref_enc.slf_attn.w_qs.bias"].astype(np.float32)
    k_b = weights["ref_enc.slf_attn.w_ks.bias"].astype(np.float32)
    v_b = weights["ref_enc.slf_attn.w_vs.bias"].astype(np.float32)
    return (
        np.concatenate([q_w, k_w, v_w], axis=0),
        np.concatenate([q_b, k_b, v_b], axis=0),
    )


def _ref_enc_should_quantize(gguf_name: str, tensor: np.ndarray, block_size: int) -> bool:
    if tensor.ndim != 2:
        return False
    if gguf_name.endswith("_w") and not gguf_name.startswith("ref_enc.temporal."):
        return tensor.shape[1] % block_size == 0
    return False


def emit_ref_enc(writer, weights, cfg, target_type, state):
    is_quantized = target_type not in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    )
    block_size = gguf.GGML_QUANT_SIZES[target_type][0] if is_quantized else 0

    def emit_tensor(gguf_name: str, tensor_np: np.ndarray) -> None:
        if is_quantized and _ref_enc_should_quantize(gguf_name, tensor_np, block_size):
            quantized = gguf.quantize(tensor_np, target_type)
            writer.add_tensor(gguf_name, quantized, raw_dtype=target_type)
            data_type = target_type
        elif ((is_quantized or target_type == gguf.GGMLQuantizationType.F16)
              and tensor_np.ndim >= 2):
            tensor_np = tensor_np.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)
        else:
            tensor_np = tensor_np.astype(np.float32)
            data_type = gguf.GGMLQuantizationType.F32
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=data_type)
        state["n"] += 1
        print(f"  [ref_enc {state['n']:3d}] {gguf_name:30s} "
              f"{list(tensor_np.shape)!s:18s} {data_type.name}")

    for gguf_name, ckpt_name in REF_ENC_MAP:
        if ckpt_name not in weights:
            raise KeyError(
                f"Tensor '{ckpt_name}' not found in checkpoint "
                f"(needed for GGUF tensor '{gguf_name}')"
            )
        emit_tensor(gguf_name, weights[ckpt_name])

    qkv_w_np, qkv_b_np = _ref_enc_fused_qkv(weights)
    emit_tensor("ref_enc.attention.qkv_w", qkv_w_np)
    writer.add_tensor(
        "ref_enc.attention.qkv_b",
        qkv_b_np.astype(np.float32),
        raw_dtype=gguf.GGMLQuantizationType.F32,
    )
    state["n"] += 1
    print(f"  [ref_enc {state['n']:3d}] {'ref_enc.attention.qkv_b':30s} "
          f"{list(qkv_b_np.shape)!s:18s} F32")

    writer.add_string("sovits.block", "full")
    writer.add_uint32("sovits.ref_enc.in_dim", 704)
    writer.add_uint32("sovits.ref_enc.hidden_dim", 128)
    writer.add_uint32("sovits.ref_enc.out_dim", int(cfg.get("gin_channels", 512)))
    writer.add_uint32("sovits.ref_enc.kernel_size", 5)
    writer.add_uint32("sovits.ref_enc.n_head", 2)
    writer.add_uint32("sovits.ref_enc.temporal_layers", 2)
    writer.add_bool("sovits.ref_enc.full_context", True)


# ===========================================================================
# Block: quantizer (single-layer RVQ decode)  -- verbatim policy
# ===========================================================================

def emit_quantizer(writer, weights, cfg, target_type, state):
    # The RVQ codebook is a lookup table consumed by a gather, so it is never
    # integer-quantized. Only f16 / f32 are valid here; any int-quant target
    # falls back to f32 (matching the precision of the source checkpoint).
    ckpt_name = "quantizer.vq.layers.0._codebook.embed"
    if ckpt_name not in weights:
        raise KeyError(f"Tensor '{ckpt_name}' not found in checkpoint")
    tensor_np = weights[ckpt_name]
    if target_type == gguf.GGMLQuantizationType.F16:
        out_type = gguf.GGMLQuantizationType.F16
        tensor_np = tensor_np.astype(np.float16)
    else:
        out_type = gguf.GGMLQuantizationType.F32
        tensor_np = tensor_np.astype(np.float32)
    writer.add_tensor("quantizer.codebook", tensor_np, raw_dtype=out_type)
    state["n"] += 1
    print(f"  [quant  {state['n']:3d}] quantizer.codebook             "
          f"{list(tensor_np.shape)!s:16s} {out_type.name}")

    writer.add_uint32("sovits.quantizer.dim", 768)
    writer.add_uint32("sovits.quantizer.bins", 1024)
    writer.add_uint32("sovits.quantizer.n_q", 1)


# ===========================================================================
# Block: text_encoder (enc_p: ssl / text / mrte / post)  -- verbatim policy
# ===========================================================================

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


def _te_fused_qkv(weights, prefix):
    q_w, q_b = linearize_conv1x1(weights[f"{prefix}.conv_q.weight"], weights[f"{prefix}.conv_q.bias"])
    k_w, k_b = linearize_conv1x1(weights[f"{prefix}.conv_k.weight"], weights[f"{prefix}.conv_k.bias"])
    v_w, v_b = linearize_conv1x1(weights[f"{prefix}.conv_v.weight"], weights[f"{prefix}.conv_v.bias"])
    return np.concatenate([q_w, k_w, v_w], axis=0), np.concatenate([q_b, k_b, v_b], axis=0)


def _te_packed_rel_k(weights, prefix):
    return weights[f"{prefix}.emb_rel_k"][0].astype(np.float32).copy()


def _te_packed_rel_v_t(weights, prefix):
    return weights[f"{prefix}.emb_rel_v"][0].transpose(1, 0).astype(np.float32).copy()


def _te_compose_affine(w2, b2, w1, b1):
    return w2 @ w1, w2 @ b1 + b2


def _te_stack_weights(ws, bs):
    return np.concatenate(ws, axis=0), np.concatenate(bs, axis=0)


def _te_convert_tensor(gguf_name, tensor_np, target_type):
    if gguf_name.endswith("text_embedding"):
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32
    if tensor_np.ndim <= 1:
        return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32
    if target_type == gguf.GGMLQuantizationType.F32:
        return tensor_np.astype(np.float32), target_type
    if target_type == gguf.GGMLQuantizationType.F16:
        return tensor_np.astype(np.float16), target_type
    # proj_w stays F16 even under int-quant: it produces both m and logs and
    # dominates the m_p parity error under Q4_0; it's tiny (~150 KB).
    if gguf_name == "text_encoder_post.proj_w":
        return tensor_np.astype(np.float16), gguf.GGMLQuantizationType.F16
    if tensor_np.ndim == 2:
        block_size = gguf.GGML_QUANT_SIZES[target_type][0]
        if tensor_np.shape[1] % block_size == 0:
            quantized = gguf.quantize(tensor_np.astype(np.float32), target_type)
            return quantized, target_type
    return tensor_np.astype(np.float32), gguf.GGMLQuantizationType.F32


def _te_prepare_direct_tensor(gguf_name, tensor_np):
    if gguf_name.endswith("text_embedding"):
        return tensor_np.astype(np.float32).copy()
    if gguf_name.endswith("_w") and tensor_np.ndim == 3 and tensor_np.shape[2] == 1:
        return tensor_np[:, :, 0].astype(np.float32).copy()
    if gguf_name.endswith("_w") and tensor_np.ndim == 3 and tensor_np.shape[2] > 1:
        out_c, in_c, k = tensor_np.shape
        return tensor_np.astype(np.float32).reshape(out_c, in_c * k).copy()
    return tensor_np.astype(np.float32)


def emit_text_encoder(writer, weights, cfg, target_type, state):
    vocab = int(weights["enc_p.text_embedding.weight"].shape[0])

    for section, n_layers in (("ssl", 3), ("text", 6), ("post", 3)):
        branch = {"ssl": "encoder_ssl", "text": "encoder_text", "post": "encoder2"}[section]
        for i in range(n_layers):
            prefix = f"enc_p.{branch}.attn_layers.{i}"
            qkv_w, qkv_b = _te_fused_qkv(weights, prefix)
            rel_k = _te_packed_rel_k(weights, prefix)
            rel_v_t = _te_packed_rel_v_t(weights, prefix)
            name_base = f"text_encoder_{section}.layers.{i}"

            qkv_w, qkv_type = _te_convert_tensor(f"{name_base}.qkv_w", qkv_w, target_type)
            writer.add_tensor(f"{name_base}.qkv_w", qkv_w, raw_dtype=qkv_type)
            state["n"] += 1
            print(f"  [te     {state['n']:3d}] {name_base}.qkv_w {'':<18s} {qkv_type.name}")

            writer.add_tensor(f"{name_base}.qkv_b", qkv_b.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            state["n"] += 1
            writer.add_tensor(f"{name_base}.rel_k", rel_k, raw_dtype=gguf.GGMLQuantizationType.F32)
            state["n"] += 1
            writer.add_tensor(f"{name_base}.rel_v_t", rel_v_t, raw_dtype=gguf.GGMLQuantizationType.F32)
            state["n"] += 1

    # MRTE fused projections.
    c_pre_w, c_pre_b = linearize_conv1x1(weights["enc_p.mrte.c_pre.weight"], weights["enc_p.mrte.c_pre.bias"])
    text_pre_w, text_pre_b = linearize_conv1x1(weights["enc_p.mrte.text_pre.weight"], weights["enc_p.mrte.text_pre.bias"])
    q_w, q_b = linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_q.weight"], weights["enc_p.mrte.cross_attention.conv_q.bias"])
    k_w, k_b = linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_k.weight"], weights["enc_p.mrte.cross_attention.conv_k.bias"])
    v_w, v_b = linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_v.weight"], weights["enc_p.mrte.cross_attention.conv_v.bias"])
    o_w, o_b = linearize_conv1x1(weights["enc_p.mrte.cross_attention.conv_o.weight"], weights["enc_p.mrte.cross_attention.conv_o.bias"])
    c_post_w, c_post_b = linearize_conv1x1(weights["enc_p.mrte.c_post.weight"], weights["enc_p.mrte.c_post.bias"])

    q_fused_w, q_fused_b = _te_compose_affine(q_w, q_b, c_pre_w, c_pre_b)
    k_fused_w, k_fused_b = _te_compose_affine(k_w, k_b, text_pre_w, text_pre_b)
    v_fused_w, v_fused_b = _te_compose_affine(v_w, v_b, text_pre_w, text_pre_b)
    skip_from_ssl_w, skip_from_ssl_b = _te_compose_affine(c_post_w, c_post_b, c_pre_w, c_pre_b)
    attn_out_w, attn_out_b = _te_compose_affine(c_post_w, c_post_b, o_w, o_b)

    ge_out_w = c_post_w.copy()
    ge_out_b = c_post_b.copy()
    skip_from_ssl_b = skip_from_ssl_b - c_post_b
    attn_out_b = attn_out_b - c_post_b

    ssl_fused_w, ssl_fused_b = _te_stack_weights([q_fused_w, skip_from_ssl_w], [q_fused_b, skip_from_ssl_b])
    text_kv_w, text_kv_b = _te_stack_weights([k_fused_w, v_fused_w], [k_fused_b, v_fused_b])

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
        tensor_np, tensor_type = _te_convert_tensor(gguf_name, tensor_np, target_type)
        writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
        state["n"] += 1

    for mapping in (TEXT_ENCODER_SSL_MAP, TEXT_ENCODER_TEXT_MAP, TEXT_ENCODER_POST_MAP):
        for gguf_name, ckpt_name in mapping:
            if ckpt_name not in weights:
                raise KeyError(
                    f"Tensor '{ckpt_name}' not found in checkpoint "
                    f"(needed for GGUF tensor '{gguf_name}')"
                )
            if gguf_name.endswith("qkv_w") or gguf_name.endswith("qkv_b"):
                continue
            tensor_np = _te_prepare_direct_tensor(gguf_name, weights[ckpt_name])
            tensor_np, tensor_type = _te_convert_tensor(gguf_name, tensor_np, target_type)
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=tensor_type)
            state["n"] += 1

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


# ===========================================================================
# Block: flow (ResidualCouplingBlock inverse)  -- verbatim policy
# ===========================================================================

def _conv_convert_tensor(gguf_name, tensor_np, target_type):
    """flow/generator shared policy: f32 for small/1D, else f16 or int-quant."""
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


def emit_flow(writer, weights, cfg, target_type, state):
    n = state["n"]
    for L in range(N_FLOWS):
        ckpt_idx = L * 2  # checkpoint uses even indices 0,2,4,6

        for suffix in ("pre", "post"):
            ckpt_weight_name = f"flow.flows.{ckpt_idx}.{suffix}.weight"
            ckpt_bias_name = f"flow.flows.{ckpt_idx}.{suffix}.bias"
            gguf_w_name = f"flow.layers.{L}.{suffix}_w"
            gguf_b_name = f"flow.layers.{L}.{suffix}_b"

            w = flatten_conv1d(weights[ckpt_weight_name].astype(np.float32))
            b = weights[ckpt_bias_name].astype(np.float32)

            w_data, w_type = _conv_convert_tensor(gguf_w_name, w, target_type)
            writer.add_tensor(gguf_w_name, w_data, raw_dtype=w_type)
            n += 1
            print(f"  [flow   {n:3d}] {gguf_w_name:34s} {list(w_data.shape)!s:18s} {w_type.name}")

            b_data, b_type = _conv_convert_tensor(gguf_b_name, b, target_type)
            writer.add_tensor(gguf_b_name, b_data, raw_dtype=b_type)
            n += 1
            print(f"  [flow   {n:3d}] {gguf_b_name:34s} {list(b_data.shape)!s:18s} {b_type.name}")

        ckpt_prefix = f"flow.flows.{ckpt_idx}.enc.cond_layer"
        cond_fused_w, cond_fused_b = fuse_weight_norm(
            weights[f"{ckpt_prefix}.weight_g"],
            weights[f"{ckpt_prefix}.weight_v"],
            weights[f"{ckpt_prefix}.bias"],
        )
        cond_fused_w = flatten_conv1d(cond_fused_w)

        for gguf_name, val in ((f"flow.layers.{L}.enc.cond_w", cond_fused_w),
                               (f"flow.layers.{L}.enc.cond_b", cond_fused_b)):
            v_data, v_type = _conv_convert_tensor(gguf_name, val, target_type)
            writer.add_tensor(gguf_name, v_data, raw_dtype=v_type)
            n += 1
            print(f"  [flow   {n:3d}] {gguf_name:34s} {list(v_data.shape)!s:18s} {v_type.name}")

        for j in range(N_WN):
            for layer_type in ("in", "rs"):
                ckpt_layer_prefix = f"flow.flows.{ckpt_idx}.enc.{'in' if layer_type == 'in' else 'res_skip'}_layers.{j}"
                fused_w, fused_b = fuse_weight_norm(
                    weights[f"{ckpt_layer_prefix}.weight_g"],
                    weights[f"{ckpt_layer_prefix}.weight_v"],
                    weights[f"{ckpt_layer_prefix}.bias"],
                )
                fused_w = flatten_conv1d(fused_w)
                gguf_w_name = f"flow.layers.{L}.enc.{j}.{layer_type}_w"
                gguf_b_name = f"flow.layers.{L}.enc.{j}.{layer_type}_b"

                w_data, w_type = _conv_convert_tensor(gguf_w_name, fused_w, target_type)
                writer.add_tensor(gguf_w_name, w_data, raw_dtype=w_type)
                n += 1
                print(f"  [flow   {n:3d}] {gguf_w_name:34s} {list(w_data.shape)!s:18s} {w_type.name}")

                b_data, b_type = _conv_convert_tensor(gguf_b_name, fused_b, target_type)
                writer.add_tensor(gguf_b_name, b_data, raw_dtype=b_type)
                n += 1
                print(f"  [flow   {n:3d}] {gguf_b_name:34s} {list(b_data.shape)!s:18s} {b_type.name}")

    state["n"] = n
    writer.add_uint32("sovits.flow.channels", 192)
    writer.add_uint32("sovits.flow.hidden", 192)
    writer.add_uint32("sovits.flow.gin", 512)
    writer.add_uint32("sovits.flow.n_flows", N_FLOWS)
    writer.add_uint32("sovits.flow.wn_layers", N_WN)
    writer.add_uint32("sovits.flow.kernel", 5)


# ===========================================================================
# Block: generator (HiFi-GAN)  -- verbatim policy
# ===========================================================================

GENERATOR_DIRECT_MAP = [
    ("generator.conv_pre_w", "dec.conv_pre.weight"),
    ("generator.conv_pre_b", "dec.conv_pre.bias"),
    ("generator.cond_w", "dec.cond.weight"),
    ("generator.cond_b", "dec.cond.bias"),
    ("generator.conv_post_w", "dec.conv_post.weight"),
]


def _gen_add_tensor(writer, gguf_name, tensor_np, target_type, note, state):
    tensor_data, tensor_type = _conv_convert_tensor(gguf_name, tensor_np, target_type)
    writer.add_tensor(gguf_name, tensor_data, raw_dtype=tensor_type)
    state["n"] += 1
    print(f"  [gen    {state['n']:3d}] {gguf_name:40s} <- {note:28s} "
          f"{list(tensor_data.shape)!s:18s} {tensor_type.name}")


def emit_generator(writer, weights, cfg, target_type, state):
    for gguf_name, ckpt_name in GENERATOR_DIRECT_MAP:
        if ckpt_name not in weights:
            raise KeyError(f"Tensor '{ckpt_name}' not found in checkpoint")
        tensor_np = (flatten_conv1d(weights[ckpt_name]) if gguf_name.endswith("_w")
                     else weights[ckpt_name].astype(np.float32))
        _gen_add_tensor(writer, gguf_name, tensor_np, target_type, ckpt_name, state)

    for stage in range(N_STAGES):
        up_prefix = f"dec.ups.{stage}"
        fused_w, fused_b = fuse_weight_norm(
            weights[f"{up_prefix}.weight_g"],
            weights[f"{up_prefix}.weight_v"],
            weights[f"{up_prefix}.bias"],
        )
        _gen_add_tensor(writer, f"generator.stages.{stage}.up_w", fused_w, target_type, f"fused {up_prefix}", state)
        _gen_add_tensor(writer, f"generator.stages.{stage}.up_b", fused_b, target_type, f"bias {up_prefix}", state)

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
                c1_w = flatten_conv1d(c1_w)
                c2_w = flatten_conv1d(c2_w)
                base = f"generator.stages.{stage}.resblocks.{branch}"
                _gen_add_tensor(writer, f"{base}.convs1.{layer}.w", c1_w, target_type, f"fused {c1_prefix}", state)
                _gen_add_tensor(writer, f"{base}.convs1.{layer}.b", c1_b, target_type, f"bias {c1_prefix}", state)
                _gen_add_tensor(writer, f"{base}.convs2.{layer}.w", c2_w, target_type, f"fused {c2_prefix}", state)
                _gen_add_tensor(writer, f"{base}.convs2.{layer}.b", c2_b, target_type, f"bias {c2_prefix}", state)

    writer.add_uint32("sovits.generator.in_dim", 192)
    writer.add_uint32("sovits.generator.gin_dim", int(cfg.get("gin_channels", 512)))
    writer.add_uint32("sovits.generator.n_stages", N_STAGES)
    writer.add_uint32("sovits.generator.n_branches", N_BRANCHES)
    writer.add_uint32("sovits.generator.res_layers", N_RES_LAYERS)


# ===========================================================================
# Unified driver
# ===========================================================================

def convert(sovits_path: str, output_path: str, dtype_str: str) -> None:
    print(f"Loading SoVITS checkpoint: {sovits_path}")
    weights, meta = load_checkpoint(sovits_path)
    cfg = meta.get("config", {}).get("model", {})
    print(f"  Found {len(weights)} tensors")

    target_type = GGML_TYPES[dtype_str]
    print(f"  Output type: {dtype_str} ({target_type.name})")

    version = str(cfg.get("version") or "v2")
    sfr = str(cfg.get("semantic_frame_rate", "25hz"))

    writer = gguf.GGUFWriter(output_path, "sovits")
    # Unified metadata read by the C++ loader.
    writer.add_string("sovits.version", version)
    writer.add_string("sovits.semantic_frame_rate", sfr)

    state = {"n": 0}
    emit_ref_enc(writer, weights, cfg, target_type, state)
    emit_quantizer(writer, weights, cfg, target_type, state)
    emit_text_encoder(writer, weights, cfg, target_type, state)
    emit_flow(writer, weights, cfg, target_type, state)
    emit_generator(writer, weights, cfg, target_type, state)

    n_converted = state["n"]
    print(f"\nConverted {n_converted} tensors into one unified GGUF")
    print(f"Writing GGUF to {output_path}...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"Done! Output: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the full SoVITS v2 model (all 5 blocks) to one unified GGUF"
    )
    parser.add_argument(
        "sovits_ckpt",
        help="Path to SoVITS generator checkpoint (.pth), e.g. s2G2333k.pth",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF file path (default: <ckpt_stem>-sovits-<type>.gguf)",
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
        args.output = f"{base}-sovits-{args.dtype}.gguf"

    convert(args.sovits_ckpt, args.output, args.dtype)


if __name__ == "__main__":
    main()
