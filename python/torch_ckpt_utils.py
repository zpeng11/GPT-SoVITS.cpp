"""Load PyTorch checkpoint files (`.ckpt` / `.pth`) without torch.

Only depends on stdlib (zipfile, pickle) and numpy.

Usage:
    from torch_ckpt_utils import load_checkpoint

    weights, meta = load_checkpoint("s1v3.ckpt")
    # weights: dict[str, np.ndarray]  — all tensors, name → numpy array
    # meta:    dict  — extra keys like "config", "info", etc.

The checkpoint must use PyTorch's standard zip+ pickle serialization format:
    <prefix>/data.pkl       — pickle with persistent_id references
    <prefix>/data/<key>     — raw tensor binary blobs
"""

from __future__ import annotations

import io
import pickle
import zipfile
from typing import Tuple

import numpy as np

__all__ = ["load_checkpoint"]

# ---------------------------------------------------------------------------
# dtype mapping: torch Storage class name → numpy dtype
# ---------------------------------------------------------------------------

_TORCH_STORAGE_TO_NUMPY: dict[str, np.dtype] = {
    "HalfStorage":    np.dtype(np.float16),
    "FloatStorage":   np.dtype(np.float32),
    "DoubleStorage":  np.dtype(np.float64),
    "LongStorage":    np.dtype(np.int64),
    "IntStorage":     np.dtype(np.int32),
    "ShortStorage":   np.dtype(np.int16),
    "ByteStorage":    np.dtype(np.uint8),
    "BoolStorage":    np.dtype(np.bool_),
    "BFloat16Storage": np.dtype(np.float32),  # no native numpy bfloat16
}

# ---------------------------------------------------------------------------
# Custom Unpickler — replaces torch classes with plain-data equivalents
# ---------------------------------------------------------------------------

class _TorchlessUnpickler(pickle.Unpickler):
    """Pickle unpickler that resolves torch serialization types without torch."""

    def find_class(self, module: str, name: str):  # noqa: N802 — pickle API
        # collections.OrderedDict — used for state dicts
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict

        # torch._utils._rebuild_tensor_v2 — reconstructs a tensor from storage
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2

        # torch.<Foo>Storage — represents a typed storage block
        if module == "torch" and name in _TORCH_STORAGE_TO_NUMPY:
            return _StorageTag(_TORCH_STORAGE_TO_NUMPY[name])

        raise pickle.UnpicklingError(
            f"torch_ckpt_utils: unsupported class {module}.{name}"
        )

    def persistent_load(self, pid: tuple):
        # PyTorch format: ('storage', StorageTag, key, device, numel)
        if pid[0] != "storage":
            raise pickle.UnpicklingError(
                f"torch_ckpt_utils: unknown persistent_id type {pid[0]!r}"
            )
        _tag, storage_cls, key, _device, numel = pid
        return _StorageRef(dtype=storage_cls.dtype, key=key, numel=numel)


# -- Lightweight data classes used during unpickling ------------------------

class _StorageTag:
    """Stands in for torch.<Foo>Storage during unpickling."""

    __slots__ = ("dtype",)

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    def __call__(self):
        # Some pickle formats call the storage class as a constructor
        return self


class _StorageRef:
    """Returned by persistent_load — points to a raw binary blob in the ZIP."""

    __slots__ = ("dtype", "key", "numel")

    def __init__(self, dtype: np.dtype, key: str, numel: int):
        self.dtype = dtype
        self.key = key
        self.numel = numel


def _rebuild_tensor_v2(
    storage: _StorageRef,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool = False,
    backward_hooks=None,
) -> dict:
    """Stands in for torch._utils._rebuild_tensor_v2 during unpickling."""
    return {
        "storage_key": storage.key,
        "dtype": storage.dtype,
        "storage_offset": storage_offset,
        "shape": list(size),
        "stride": list(stride),
    }


# ---------------------------------------------------------------------------
# ZIP prefix detection
# ---------------------------------------------------------------------------

def _detect_prefix(namelist: list[str]) -> str:
    """Return the common prefix for `data.pkl` in the ZIP.

    Typical layouts:
        s1v3/data.pkl           → prefix = "s1v3"
        s2Gv2ProPlus/data.pkl   → prefix = "s2Gv2ProPlus"
        archive/data.pkl        → prefix = "archive"
    """
    candidates = [n for n in namelist if n.endswith("/data.pkl")]
    if len(candidates) == 1:
        return candidates[0].rsplit("/data.pkl", 1)[0]
    if not candidates:
        raise ValueError(
            "torch_ckpt_utils: no data.pkl found in checkpoint ZIP"
        )
    raise ValueError(
        f"torch_ckpt_utils: ambiguous data.pkl entries: {candidates}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str,
) -> Tuple[dict[str, np.ndarray], dict]:
    """Load a PyTorch checkpoint without torch.

    Parameters
    ----------
    path : str
        Path to a `.ckpt` or `.pth` file (PyTorch zip-format checkpoint).

    Returns
    -------
    weights : dict[str, np.ndarray]
        All tensors from the ``'weight'`` top-level key, name → numpy array.
    meta : dict
        Remaining top-level keys (``'config'``, ``'info'``, etc.).
    """
    zf = zipfile.ZipFile(path, "r")
    try:
        prefix = _detect_prefix(zf.namelist())
        pkl_bytes = zf.read(f"{prefix}/data.pkl")

        # 1. Parse pickle to get tensor metadata
        raw = _TorchlessUnpickler(io.BytesIO(pkl_bytes)).load()

        if not isinstance(raw, dict) or "weight" not in raw:
            raise ValueError(
                "torch_ckpt_utils: checkpoint has no 'weight' key "
                f"(found: {list(raw.keys()) if isinstance(raw, dict) else type(raw)})"
            )

        weight_meta: dict[str, dict] = raw["weight"]
        meta = {k: v for k, v in raw.items() if k != "weight"}

        # 2. Read each tensor's raw bytes from the ZIP and build numpy arrays
        weights: dict[str, np.ndarray] = {}
        for name, info in weight_meta.items():
            dtype: np.dtype = info["dtype"]
            shape = info["shape"]
            offset = info["storage_offset"]
            numel = 1
            for s in shape:
                numel *= s

            blob = zf.read(f"{prefix}/data/{info['storage_key']}")
            arr = np.frombuffer(blob, dtype=dtype)

            # Slice to the correct offset+range, then reshape
            arr = arr[offset : offset + numel].copy()
            if shape:
                arr = arr.reshape(shape)

            weights[name] = arr

        return weights, meta

    finally:
        zf.close()
