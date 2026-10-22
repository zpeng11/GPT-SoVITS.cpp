"""Language modes and deterministic zh/en mixed-script segmentation.

The first slice supports Chinese and English.  ``auto``/``mixed`` splits text
into contiguous script runs with an explicit neutral-character policy:

* punctuation, digits and supported symbols attach to the run on their left
  (or the next run when they are leading);
* whitespace stays inside an English run; at a zh -> en boundary it becomes
  the English run's leading space, so the concatenated normalized text keeps
  a readable separator;
* unsupported scripts (kana, Hangul, Cyrillic, emoji, ...) are a hard error by
  default.  ``allow_unsupported=True`` drops them and reports a warning.

The policy is intentionally data-free: no fasttext/split-lang model, no
network lookup, no locale-dependent heuristic.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .errors import UnsupportedLanguageError, UnsupportedScriptError

LANGUAGE_MODES = ("auto", "mixed", "zh", "all_zh", "en", "all_en")


@dataclass
class SegmentSpec:
    language: str
    text: str


def _ascii_from_fullwidth(ch: str) -> str:
    code = ord(ch)
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if ch == "\u3000":
        return " "
    return ch


def is_han(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
        or 0x30000 <= code <= 0x3134A
        or 0x31350 <= code <= 0x323AF
    )


def is_latin_letter(ch: str) -> bool:
    ch = _ascii_from_fullwidth(ch)
    if ch.isascii():
        return ch.isalpha()
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return name.startswith("LATIN")


def is_digit(ch: str) -> bool:
    ch = _ascii_from_fullwidth(ch)
    return ch.isdigit()


def is_space(ch: str) -> bool:
    return ch.isspace()


# Symbols that the zh/en normalizers understand as neutral text.
_NEUTRAL_SYMBOLS = set("$£€¥￥%‰°℃℉+×÷=~～&@#*/\\|<>^_`")


def classify_char(ch: str) -> str:
    """Return one of han, latin, digit, space, punct, symbol, unsupported."""
    if is_space(ch):
        return "space"
    if is_han(ch):
        return "han"
    if is_digit(ch):
        return "digit"
    if is_latin_letter(ch):
        return "latin"
    if ch in _NEUTRAL_SYMBOLS:
        return "symbol"
    category = unicodedata.category(ch)
    if category.startswith("P"):
        return "punct"
    if category.startswith("M"):
        # A combining mark stays with the surrounding Latin word.
        return "symbol"
    return "unsupported"


def _scan_explicit(text: str, language: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    bad: List[str] = []
    for ch in text:
        kind = classify_char(ch)
        if language == "zh":
            if kind in ("latin",):
                bad.append(ch)
            elif kind == "unsupported":
                bad.append(ch)
        else:  # en
            if kind == "han":
                bad.append(ch)
            elif kind == "unsupported":
                bad.append(ch)
    if bad:
        preview = "".join(dict.fromkeys(bad))[:20]
        warnings.append(
            f"{language} mode will drop characters not handled by the "
            f"{language} normalizer: {preview!r}"
        )
    return warnings, bad


def segment_text(
    text: str,
    language: str = "auto",
    *,
    allow_unsupported: bool = False,
) -> Tuple[List[SegmentSpec], List[str]]:
    """Split ``text`` into language-tagged runs.

    Returns ``(segments, warnings)``.  The concatenation of segment texts is
    exactly the input text (no characters are silently discarded in the
    segmentation stage).
    """
    if language not in LANGUAGE_MODES:
        raise UnsupportedLanguageError(
            f"unknown language mode {language!r}; choose one of {', '.join(LANGUAGE_MODES)}"
        )
    if not isinstance(text, str):
        raise TypeError("text must be str")

    if language in ("zh", "all_zh"):
        warnings, _ = _scan_explicit(text, "zh")
        return ([SegmentSpec("zh", text)] if text else []), warnings
    if language in ("en", "all_en"):
        warnings, _ = _scan_explicit(text, "en")
        return ([SegmentSpec("en", text)] if text else []), warnings

    warnings: List[str] = []
    segments: List[SegmentSpec] = []
    pending_other: List[str] = []
    pending_spaces: List[str] = []

    def append_to_previous(chars: Sequence[str]) -> None:
        if chars:
            segments[-1].text += "".join(chars)

    for ch in text:
        kind = classify_char(ch)
        if kind in ("han", "latin"):
            lang = "zh" if kind == "han" else "en"
            if segments:
                append_to_previous(pending_other)
                pending_other.clear()
                if pending_spaces and lang == "en":
                    prefix = "".join(pending_spaces)
                    pending_spaces.clear()
                    segments.append(SegmentSpec("en", prefix + ch))
                    continue
                if pending_spaces:
                    append_to_previous(pending_spaces)
                    pending_spaces.clear()
                if segments[-1].language == lang:
                    segments[-1].text += ch
                else:
                    segments.append(SegmentSpec(lang, ch))
            else:
                prefix = "".join(pending_other) + "".join(pending_spaces)
                pending_other.clear()
                pending_spaces.clear()
                segments.append(SegmentSpec(lang, prefix + ch))
        elif kind == "space":
            if segments and segments[-1].language == "en":
                segments[-1].text += ch
            else:
                pending_spaces.append(ch)
        elif kind in ("punct", "digit", "symbol"):
            if segments:
                segments[-1].text += ch
            else:
                pending_other.append(ch)
        else:  # unsupported
            if not allow_unsupported:
                raise UnsupportedScriptError(
                    f"unsupported script character {ch!r} (U+{ord(ch):04X}); "
                    "this slice supports Chinese and English only. "
                    "Use --allow-unsupported-scripts to drop it, or choose an explicit zh/en mode."
                )
            warnings.append(
                f"dropped unsupported script character {ch!r} (U+{ord(ch):04X})"
            )

    if pending_other or pending_spaces:
        if segments:
            append_to_previous(pending_other)
            append_to_previous(pending_spaces)
        elif pending_other or pending_spaces:
            # Punctuation/digits only: keep it as Chinese so the zh normalizer
            # can verbalize/retain it.
            segments.append(SegmentSpec("zh", "".join(pending_other) + "".join(pending_spaces)))

    # Merge adjacent same-language runs (defensive; the state machine should
    # already avoid them).
    merged: List[SegmentSpec] = []
    for segment in segments:
        if not segment.text:
            continue
        if merged and merged[-1].language == segment.language:
            merged[-1].text += segment.text
        else:
            merged.append(SegmentSpec(segment.language, segment.text))
    return merged, warnings
