"""v2 phoneme vocabulary and ID conversion."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .errors import UnknownWordError
from .symbols_v2 import pad, punctuation, symbols as _V2_SYMBOLS

VERSION = "v2"
SYMBOLS: tuple[str, ...] = tuple(_V2_SYMBOLS)
SYMBOL_TO_ID: dict[str, int] = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PUNCTUATION = frozenset(punctuation)
PAD = pad
UNK = "UNK"
UNK_ID = SYMBOL_TO_ID[UNK]
VOCAB_SIZE = len(SYMBOLS)

if len(SYMBOLS) != 732:  # defensive: the C++/GGUF contract is 732
    raise RuntimeError(f"unexpected v2 phoneme vocabulary size: {len(SYMBOLS)}")


def symbol_id(symbol: str) -> int:
    try:
        return SYMBOL_TO_ID[symbol]
    except KeyError:
        raise UnknownWordError(f"phoneme {symbol!r} is not in the v2 vocabulary") from None


def symbol_ids(symbols: Sequence[str], *, strict: bool = False) -> List[int]:
    """Map phone symbols to int32-valued IDs.

    Unknown symbols become ``UNK`` unless ``strict`` is true.  The returned
    Python ints are guaranteed to fit in signed 32-bit integers.
    """
    ids: List[int] = []
    for symbol in symbols:
        value = SYMBOL_TO_ID.get(symbol)
        if value is None:
            if strict:
                raise UnknownWordError(f"phoneme {symbol!r} is not in the v2 vocabulary")
            value = UNK_ID
        ids.append(int(value))
    return ids


def is_symbol(symbol: str) -> bool:
    return symbol in SYMBOL_TO_ID


def unknown_symbols(symbols: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for symbol in symbols:
        if symbol not in SYMBOL_TO_ID and symbol not in seen:
            seen.append(symbol)
    return seen
