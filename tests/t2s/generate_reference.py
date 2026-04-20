#!/usr/bin/env python3
"""Convert .npy reference files to raw .bin files matching ggml memory layout.

The reference .npy files have axis order (batch, seq, d_model) or
(n_layer, batch, seq, d_model), but ggml uses column-major convention
where ne[0] = d_model is the innermost dimension.

Key insight: numpy row-major (seq, d_model) has identical memory layout
to ggml column-major {ne0=d_model, ne1=seq}, so no transpose is needed —
just flatten in C order.

All outputs are stored as float32 for comparison against F32 KV caches.
"""

import os
import numpy as np

REF_DIR = os.path.dirname(os.path.abspath(__file__)) + "/ref"


def convert():
    # xy_pos: (1, 193, 512) -> flatten to (193*512,) float32
    xy_pos = np.load(f"{REF_DIR}/xy_pos.npy")  # (1, 193, 512) fp16
    xy_pos_f32 = xy_pos[0].astype(np.float32)  # (193, 512) fp32
    xy_pos_f32.tofile(f"{REF_DIR}/xy_pos.bin")
    print(f"xy_pos: {xy_pos.shape} -> {xy_pos_f32.shape} -> xy_pos.bin")

    # xy_dec: (1, 193, 512) -> flatten to (193*512,) float32
    xy_dec = np.load(f"{REF_DIR}/xy_dec.npy")
    xy_dec_f32 = xy_dec[0].astype(np.float32)
    xy_dec_f32.tofile(f"{REF_DIR}/xy_dec.bin")
    print(f"xy_dec: {xy_dec.shape} -> {xy_dec_f32.shape} -> xy_dec.bin")

    # k_cache: (24, 1, 193, 512) -> per-layer (193, 512) -> flatten
    k_cache = np.load(f"{REF_DIR}/k_cache.npy")  # (24, 1, 193, 512) fp16
    k_f32 = k_cache[:, 0].astype(np.float32)  # (24, 193, 512) fp32
    # Flatten layer by layer: each layer is (193*512) floats
    k_f32_flat = k_f32.reshape(24, -1)  # (24, 193*512)
    k_f32_flat.tofile(f"{REF_DIR}/k_cache.bin")
    print(f"k_cache: {k_cache.shape} -> (24, {k_f32.shape[1]}, {k_f32.shape[2]}) -> k_cache.bin")

    # v_cache: same as k_cache
    v_cache = np.load(f"{REF_DIR}/v_cache.npy")
    v_f32 = v_cache[:, 0].astype(np.float32)
    v_f32_flat = v_f32.reshape(24, -1)
    v_f32_flat.tofile(f"{REF_DIR}/v_cache.bin")
    print(f"v_cache: {v_cache.shape} -> (24, {v_f32.shape[1]}, {v_f32.shape[2]}) -> v_cache.bin")


if __name__ == "__main__":
    convert()
