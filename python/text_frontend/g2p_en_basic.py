"""CMUdict-backed English G2P with deterministic OOV fallbacks.

This deliberately avoids g2p_en, nltk and wordsegment.  The fallback chain is:

1. custom overrides in ``engdict-hot.rep``;
2. CMUdict first pronunciation;
3. compound split into known dictionary words;
4. common possessive/plural/verb suffix rules;
5. letter names for short/OOV words.

Every fallback is reported as a warning; nothing is silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .resources import Resources
from .symbols import SYMBOL_TO_ID

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_PUNCT_RE = re.compile(r"[.,?!\-]")
_POSSESSIVE_RE = re.compile(r"^([a-z]+)('s)$")

# ARPA names for single letters.  Values are already in the v2 vocabulary.
_LETTER_NAMES = {
    "a": ["EY1"], "b": ["B", "IY1"], "c": ["S", "IY1"], "d": ["D", "IY1"],
    "e": ["IY1"], "f": ["EH1", "F"], "g": ["JH", "IY1"], "h": ["EY1", "CH"],
    "i": ["AY1"], "j": ["JH", "EY1"], "k": ["K", "EY1"], "l": ["EH1", "L"],
    "m": ["EH1", "M"], "n": ["EH1", "N"], "o": ["OW1"], "p": ["P", "IY1"],
    "q": ["K", "Y", "UW1"], "r": ["AA1", "R"], "s": ["EH1", "S"],
    "t": ["T", "IY1"], "u": ["Y", "UW1"], "v": ["V", "IY1"],
    "w": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"], "x": ["EH1", "K", "S"],
    "y": ["W", "AY1"], "z": ["Z", "IY1"],
}

_SIBILANT_ENDINGS = {"S", "Z", "SH", "ZH", "CH", "JH"}


@dataclass
class EnglishG2PResult:
    phones: List[str]
    warnings: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    # (start, end, phone_start, phone_end) in the normalized English text.
    token_spans: List[Tuple[int, int, int, int]] = field(default_factory=list)


class EnglishG2P:
    def __init__(self, resources: Resources, *, strict: bool = False):
        self.resources = resources
        self.strict = strict
        self._cmu: Optional[Dict[str, List[str]]] = None
        self._memo: Dict[str, Optional[List[str]]] = {}

    @property
    def cmu(self) -> Dict[str, List[str]]:
        if self._cmu is None:
            self._cmu = self.resources.load_cmudict()
        return self._cmu

    def _letter_phones(self, word: str) -> List[str]:
        phones: List[str] = []
        for ch in word.lower():
            if ch in _LETTER_NAMES:
                phones.extend(_LETTER_NAMES[ch])
            else:
                # Keep supported punctuation, otherwise mark UNK.
                phones.append(ch if ch in SYMBOL_TO_ID else "UNK")
        return phones

    def _suffix_phones(self, stem_phones: Sequence[str], suffix: str) -> List[str]:
        if suffix == "s":
            last = stem_phones[-1] if stem_phones else ""
            if last in _SIBILANT_ENDINGS:
                return ["IH0", "Z"]
            if last in {"P", "T", "K", "F", "TH"}:
                return ["S"]
            return ["Z"]
        if suffix == "ed":
            last = stem_phones[-1] if stem_phones else ""
            if last in {"T", "D"}:
                return ["IH0", "D"]
            if last in {"P", "K", "F", "S", "SH", "CH", "TH"}:
                return ["T"]
            return ["D"]
        if suffix == "ing":
            return ["IH0", "NG"]
        return []

    def _compound(self, word: str, depth: int = 0) -> Optional[List[str]]:
        """Split a lower-case word into dictionary/single-letter parts."""
        if depth > 8 or not word:
            return None
        if word in self.cmu:
            return list(self.cmu[word])
        if len(word) == 1:
            return list(_LETTER_NAMES.get(word, ["UNK"]))
        if len(word) <= 4:
            # Very short OOV strings are more predictable spelled as letters
            # than split into a coincidental dictionary word + fragment.
            return [phone for ch in word for phone in _LETTER_NAMES.get(ch, ["UNK"])]
        # Longest prefix first, but leave at least one character for the tail.
        for cut in range(min(len(word) - 1, 24), 0, -1):
            prefix, tail = word[:cut], word[cut:]
            if prefix not in self.cmu and len(prefix) > 1:
                continue
            prefix_phones = list(self.cmu[prefix]) if prefix in self.cmu else list(_LETTER_NAMES.get(prefix, []))
            if not prefix_phones:
                continue
            tail_phones = self._compound(tail, depth + 1)
            if tail_phones is not None:
                return prefix_phones + tail_phones
        return None

    def _lookup(self, word: str) -> Tuple[List[str], bool]:
        """Return (phones, used_fallback) for an English word token."""
        lower = word.lower()
        if lower in self.cmu:
            return list(self.cmu[lower]), False

        if lower in self._memo:
            cached = self._memo[lower]
            return (list(cached) if cached is not None else self._letter_phones(lower), True)

        # Possessive: recurse on the stem and add the appropriate sibilant.
        match = _POSSESSIVE_RE.match(lower)
        if match:
            stem_phones, _ = self._lookup(match.group(1))
            result = stem_phones + self._suffix_phones(stem_phones, "s")
            self._memo[lower] = result
            return result, True

        # Common inflections whose stems are in CMUdict.
        for suffix in ("ing", "ed", "s"):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
                stem = lower[: -len(suffix)]
                if stem in self.cmu:
                    result = list(self.cmu[stem]) + self._suffix_phones(self.cmu[stem], suffix)
                    self._memo[lower] = result
                    return result, True

        # Short OOV tokens are spelled as letter names.
        if len(lower) <= 4:
            result = self._letter_phones(lower)
            self._memo[lower] = result
            return result, True

        # Compound split, e.g. openai -> open + ai, frontend -> front + end.
        compound = self._compound(lower)
        if compound is not None:
            self._memo[lower] = compound
            return compound, True

        # Final fallback: letter names, always in the vocabulary.
        result = self._letter_phones(lower)
        self._memo[lower] = result
        return result, True

    def g2p(self, text: str) -> EnglishG2PResult:
        phones: List[str] = []
        warnings: List[str] = []
        unknown_words: List[str] = []
        token_spans: List[Tuple[int, int, int, int]] = []

        pos = 0
        while pos < len(text):
            word_match = _WORD_RE.match(text, pos)
            if word_match:
                word = word_match.group(0)
                start_phone = len(phones)
                word_phones, used_fallback = self._lookup(word)
                # Guarantee every emitted phone is representable.
                cleaned: List[str] = []
                for phone in word_phones:
                    if phone in SYMBOL_TO_ID:
                        cleaned.append(phone)
                    else:
                        cleaned.append("UNK")
                        warnings.append(f"unknown ARPA phone {phone!r} in word {word!r} -> UNK")
                phones.extend(cleaned)
                token_spans.append((word_match.start(), word_match.end(), start_phone, len(phones)))
                if used_fallback:
                    unknown_words.append(word)
                    warnings.append(
                        f"OOV English word {word!r}: used dictionary-compound/suffix/letter fallback"
                    )
                pos = word_match.end()
                continue

            punct_match = _PUNCT_RE.match(text, pos)
            if punct_match:
                punct = punct_match.group(0)
                start_phone = len(phones)
                phones.append(punct)
                token_spans.append((punct_match.start(), punct_match.end(), start_phone, len(phones)))
                pos = punct_match.end()
                continue

            # Whitespace or an unsupported symbol.
            if not text[pos].isspace():
                warnings.append(f"English G2P skipped unsupported character {text[pos]!r}")
            pos += 1

        if self.strict and unknown_words:
            from .errors import UnknownWordError
            raise UnknownWordError(
                "strict English G2P rejected OOV word(s): " + ", ".join(sorted(set(unknown_words)))
            )
        return EnglishG2PResult(
            phones=phones,
            warnings=warnings,
            unknown_words=unknown_words,
            token_spans=token_spans,
        )
