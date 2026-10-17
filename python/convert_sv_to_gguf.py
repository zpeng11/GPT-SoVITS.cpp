#!/usr/bin/env python3
"""Convert the ERes2NetV2 speaker-verification model to a GGUF weight container.

Only the forward3 path is needed by GPT-SoVITS (compute_embedding3): conv1+bn1,
layer1-4, layer3_ds, fuse34. The seg_1 / seg_bn_1 / seg_2 embedding head and
num_batches_tracked buffers are dropped.

Supported architecture (fixed): ERes2NetV2(baseWidth=24, scale=4, expansion=4),
feat_dim=80, out_dim=20480 -- i.e. the upstream pretrained_eres2netv2w24s4ep4
checkpoint. It is the only SV architecture the C++ runtime (src/sv/sv.cpp)
implements, so no architecture options are accepted here.

Tensors are stored as float32 with their native PyTorch shape, prefixed "sv."
so the C++ loader can look them up by the same names as the state_dict:

    sv.conv1.weight                       {out, in, kh, kw}
    sv.bn1.{weight,bias,running_mean,running_var}
    sv.layer{L}.{b}.conv1.weight          {out, in, 1, 1}
    sv.layer{L}.{b}.bn1.*  ...
    sv.layer{L}.{b}.convs.{i}.weight      {width, width, 3, 3}
    sv.layer{L}.{b}.bns.{i}.*
    sv.layer{L}.{b}.conv3.weight, sv.layer{L}.{b}.bn3.*
    sv.layer{L}.{b}.shortcut.{0,1}.*      (only when present)
    sv.layer{L}.{b}.fuse_models.{j}.local_att.{0,1,3,4}.*   (layer3/4 only)
    sv.layer3_ds.weight
    sv.fuse34.local_att.{0,1,3,4}.*

The C++ side (src/sv/sv.cpp) runs forward3 as a ggml compute graph on a backend
(Metal on macOS, CPU elsewhere). This GGUF is the weight source: every tensor is
uploaded once into a backend buffer at load time, and all BatchNorm layers are
pre-folded there into scale/shift (out = scale*x + shift; the running stats are
read straight from this GGUF). No numpy/GGUF work happens during inference.

Reference fixtures for the C++ parity tests are produced separately by
python/dump_sv_ref.py (see tests/sovits/README_SV.md).

This converter is torch-free: it reads the checkpoint with torch_ckpt_utils
(stdlib + numpy only), so deploying users do not need PyTorch installed.

Usage:
    python convert_sv_to_gguf.py <sv_ckpt> [--output <path>]
"""

from __future__ import annotations

import argparse
import os

import gguf
import numpy as np

from torch_ckpt_utils import load_checkpoint


def is_forward3_path(key: str) -> bool:
    if key.startswith("seg_"):           # embedding head, unused by forward3
        return False
    if key.endswith("num_batches_tracked"):  # not needed for eval-mode BN
        return False
    return True


def convert(sv_path: str, output_path: str) -> None:
    print(f"Loading SV checkpoint: {sv_path}")
    state, _meta = load_checkpoint(sv_path)   # torch-free (stdlib + numpy)
    # The ckpt is a bare state_dict (OrderedDict of tensors); filter to forward3.
    weights = {k: v for k, v in state.items() if is_forward3_path(k)}
    print(f"  {len(state)} tensors total, {len(weights)} kept (forward3 path)")

    writer = gguf.GGUFWriter(output_path, "sovits-sv")
    writer.add_string("sv.arch", "ERes2NetV2")
    writer.add_uint32("sv.base_width", 24)
    writer.add_uint32("sv.scale", 4)
    writer.add_uint32("sv.expansion", 4)
    writer.add_uint32("sv.feat_dim", 80)
    writer.add_uint32("sv.out_dim", 20480)

    n = 0
    for key, tensor in sorted(weights.items()):
        arr = np.ascontiguousarray(tensor, dtype=np.float32)
        gguf_name = "sv." + key
        writer.add_tensor(gguf_name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        n += 1
        if n <= 8 or n % 100 == 0:
            print(f"  [{n:4d}] {gguf_name:42s} {list(arr.shape)!s}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nConverted {n} tensors. Output: {output_path} "
          f"({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ERes2NetV2 SV model to GGUF")
    p.add_argument("sv_ckpt", help="Path to pretrained_eres2netv2w24s4ep4.ckpt")
    p.add_argument("--output", "-o", default=None,
                   help="Output GGUF (default: <stem>.gguf)")
    args = p.parse_args()
    out = args.output or (os.path.splitext(args.sv_ckpt)[0] + ".gguf")
    convert(args.sv_ckpt, out)


if __name__ == "__main__":
    main()
