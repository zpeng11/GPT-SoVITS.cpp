"""C++/ggml array-layout helpers.

Contract for BERT features:
  * Python/NumPy arrays are stored as ``(T, 1024)`` C-order arrays;
  * the same buffer is ggml tensor ``{1024, T}`` (ne[0] = 1024);
  * therefore do NOT transpose a ``(T, 1024)`` array before handing its bytes
    to C++/ggml.  A blind ``.T`` would produce ``(1024, T)`` and a wrong
    ggml ``{T, 1024}`` tensor.

Phoneme IDs cross the C++ boundary as flat int32 ``{T}``.
"""

from __future__ import annotations

from typing import Tuple

from .errors import LayoutError

BERT_CHANNELS = 1024


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        from .errors import MissingDependencyError
        raise MissingDependencyError("numpy", "array layout validation and .npz output") from exc
    return np


def validate_bert_features(features, *, expected_t=None, require_float32: bool = False):
    """Validate a ``(T, 1024)`` array and return it as NumPy.

    ``expected_t`` is normally the phoneme count.  The function deliberately
    does not transpose or reshape the data.
    """
    np = _numpy()
    array = np.asarray(features)
    if array.ndim != 2:
        raise LayoutError(f"BERT features must be 2-D (T, 1024); got shape {array.shape}")
    if array.shape[1] != BERT_CHANNELS:
        raise LayoutError(
            f"BERT features must have 1024 channels in axis 1; got shape {array.shape}. "
            "Expected numpy (T, 1024); do not pass (1024, T)."
        )
    if expected_t is not None and array.shape[0] != expected_t:
        raise LayoutError(
            f"BERT feature T={array.shape[0]} does not match phoneme count T={expected_t}"
        )
    if require_float32 and array.dtype != np.float32:
        raise LayoutError(f"BERT features must be float32; got {array.dtype}")
    return array


def ggml_ne_for_bert(features) -> Tuple[int, int]:
    """Return the ggml tensor dimensions for a ``(T, 1024)`` NumPy array."""
    array = validate_bert_features(features)
    return (BERT_CHANNELS, int(array.shape[0]))


def pack_bert_features_for_ggml(features, *, expected_t=None) -> bytes:
    """Return C-order float32 bytes for ggml ``{1024, T}`` without transposing."""
    np = _numpy()
    array = validate_bert_features(features, expected_t=expected_t)
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.tobytes(order="C")


def expand_bert_features(token_features, tokenized, word2ph):
    """Expand token rows to phones using offsets into the normalized input text."""
    np = _numpy()
    array = validate_bert_features(token_features, expected_t=len(tokenized.input_ids))
    if any(not isinstance(count, int) or count < 0 for count in word2ph):
        raise LayoutError("word2ph must contain non-negative integers")
    rows = []
    for char_index, count in enumerate(word2ph):
        matches = [index for index in range(tokenized.body_start, tokenized.body_end)
                   if tokenized.token_char_spans[index] is not None
                   and tokenized.token_char_spans[index][0] <= char_index < tokenized.token_char_spans[index][1]]
        if count and len(matches) != 1:
            raise LayoutError(f"character {char_index} has {len(matches)} matching BERT tokens")
        if count:
            rows.extend([array[matches[0]]] * count)
    return np.ascontiguousarray(rows, dtype=np.float32).reshape(-1, BERT_CHANNELS)


def phoneme_ids_int32(ids) -> "object":
    """Return a NumPy int32 array for the C++ ``{T}`` phoneme contract."""
    np = _numpy()
    return np.asarray(list(ids), dtype=np.int32)
