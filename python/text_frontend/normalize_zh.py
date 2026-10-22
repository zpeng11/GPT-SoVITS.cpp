"""Chinese text normalization and pypinyin-based G2P.

The normalization rules and the pypinyin/opencpop mapping are adapted from
GPT-SoVITS (MIT) and PaddleSpeech zh_normalization (Apache-2.0).  G2PW is not
used and is not a dependency: this is the validated pypinyin branch.

Without ``jieba`` the G2P falls back to per-character segmentation and a
documented third-tone/sandhi heuristic.  Install jieba for POS-aware word
boundaries and neutral tones.
"""

from __future__ import annotations

import importlib
import re
from typing import List, Optional, Sequence, Tuple

from .errors import MissingDependencyError, UnknownWordError
from .resources import Resources
from .symbols import PUNCTUATION

# Same map as GPT-SoVITS chinese2.py.  It is applied before stripping
# non-Chinese characters.
_REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

_REP_PATTERN = re.compile("|".join(re.escape(p) for p in _REP_MAP))
_CONSECUTIVE_PUNCT = re.compile(r"([!?…,.\-])([!?…,.\-])+")

# Maps the pypinyin finals spelling to the compact v2 spelling.
_V_REP_MAP = {"uei": "ui", "iou": "iu", "uen": "un"}
_PINYIN_REP_MAP = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
_SINGLE_REP_MAP = {"v": "yu", "e": "e", "i": "y", "u": "w"}

# The GPT-SoVITS pypinyin branch splits the normalized text after punctuation.
_SENTENCE_SPLIT = re.compile(r"(?<=[!?…,.\-])\s*")

_MUST_ERHUA = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
_NOT_ERHUA = {
    "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
    "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
    "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿",
    "可儿", "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿",
    "鸟儿", "猪儿", "猫儿", "狗儿", "少儿",
}


def _require_pypinyin():
    try:
        module = importlib.import_module("pypinyin")
    except ImportError as exc:  # pragma: no cover - obvious in normal installs
        raise MissingDependencyError(
            "pypinyin", "Chinese text normalization and G2P"
        ) from exc
    return module


_NORMALIZER = None
_TONE_MODIFIER = None
_JIEBA_AVAILABLE = None


def _get_normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        _require_pypinyin()
        try:
            from .zh_normalization.text_normlization import TextNormalizer
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError(
                "pypinyin", "Chinese text normalization", "python -m pip install pypinyin"
            ) from exc
        _NORMALIZER = TextNormalizer()
    return _NORMALIZER


def replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced = _REP_PATTERN.sub(lambda m: _REP_MAP[m.group()], text)
    # Keep Han characters and the symbols that are in the v2 phoneme table.
    from .text import is_han
    replaced = "".join(char for char in replaced if is_han(char) or char in PUNCTUATION)
    return replaced


def replace_consecutive_punctuation(text: str) -> str:
    return _CONSECUTIVE_PUNCT.sub(r"\1", text)


def normalize_zh(text: str) -> str:
    """Normalize Chinese NSW (numbers, dates, phones, ...) and punctuation."""
    if not text:
        return ""
    normalizer = _get_normalizer()
    sentences = normalizer.normalize(text)
    dest = "".join(replace_punctuation(sentence) for sentence in sentences)
    return replace_consecutive_punctuation(dest)


def _tone_sandhi():
    global _TONE_MODIFIER
    if _TONE_MODIFIER is None:
        try:
            from .tone_sandhi import ToneSandhi
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError(
                "pypinyin", "Chinese tone sandhi", "python -m pip install pypinyin"
            ) from exc
        _TONE_MODIFIER = ToneSandhi()
    return _TONE_MODIFIER


def _segment_with_jieba(text: str) -> Optional[List[Tuple[str, str]]]:
    """Return POS-tagged words if jieba is installed, else None."""
    global _JIEBA_AVAILABLE
    if _JIEBA_AVAILABLE is False:
        return None
    try:
        import jieba
        import jieba.posseg as posseg
    except ImportError:
        _JIEBA_AVAILABLE = False
        return None
    _JIEBA_AVAILABLE = True
    try:
        jieba.setLogLevel(60)
    except Exception:
        pass
    return [(word, flag) for word, flag in posseg.lcut(text)]


def _character_segments(text: str) -> List[Tuple[str, str]]:
    """Fallback segmenter: one Han/punctuation char per token.

    Punctuation runs are split into single characters so the v2 symbol table
    never receives a multi-character "phone".  ``pos="x"`` avoids POS-specific
    neutral-tone rules that cannot be inferred without a tagger.
    """
    out: List[Tuple[str, str]] = []
    for ch in text:
        out.append((ch, "x"))
    return out


def _get_initials_finals(word: str):
    pypinyin = _require_pypinyin()
    lazy_pinyin = pypinyin.lazy_pinyin
    Style = pypinyin.Style
    initials = list(lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS))
    finals = list(lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3))
    return initials, finals


def _merge_erhua(initials: List[str], finals: List[str], word: str, pos: str):
    # Fix the common er1 -> er2 case for a trailing 儿.
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and i < len(word) and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    if word not in _MUST_ERHUA and (word in _NOT_ERHUA or pos in {"a", "j", "nr"}):
        return initials, finals
    if len(finals) != len(word):
        return initials, finals

    new_initials: List[str] = []
    new_finals: List[str] = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in _NOT_ERHUA
            and new_finals
            and new_finals[-1]
            and new_finals[-1][-1] in "12345"
        ):
            phn = "er" + new_finals[-1][-1]
        new_initials.append(initials[i])
        new_finals.append(phn)
    return new_initials, new_finals


def _pinyin_to_phones(
    pinyin: str,
    c: str,
    v: str,
    pinyin_to_symbol_map: dict,
):
    """Convert one pypinyin (initial, final-tone3) pair to v2 phones."""
    if c == v:
        if c in PUNCTUATION:
            return [c]
        return None

    if not v:
        return None
    v_without_tone = v[:-1]
    tone = v[-1]
    if tone not in "12345":
        return None

    combined = c + v_without_tone
    if c:
        if v_without_tone in _V_REP_MAP:
            combined = c + _V_REP_MAP[v_without_tone]
    else:
        if combined in _PINYIN_REP_MAP:
            combined = _PINYIN_REP_MAP[combined]
        elif combined and combined[0] in _SINGLE_REP_MAP:
            combined = _SINGLE_REP_MAP[combined[0]] + combined[1:]

    mapped = pinyin_to_symbol_map.get(combined)
    if mapped is None:
        return None
    parts = mapped.split()
    if len(parts) != 2:
        return None
    return [parts[0], parts[1] + tone]


def g2p_zh(
    normalized_text: str,
    resources: Resources,
    *,
    strict: bool = False,
) -> Tuple[List[str], List[int], List[str]]:
    """Return ``(phones, word2ph, warnings)``.

    ``word2ph`` has one entry per character of ``normalized_text`` and sums to
    ``len(phones)``.  Unknown characters become ``UNK`` (or raise in strict
    mode); they never desynchronize the alignment.
    """
    if not normalized_text:
        return [], [], []

    pypinyin = _require_pypinyin()
    lazy_pinyin = pypinyin.lazy_pinyin
    Style = pypinyin.Style
    pinyin_to_symbol_map = resources.load_opencpop()
    tone_modifier = _tone_sandhi()

    warnings: List[str] = []
    phones: List[str] = []
    word2ph: List[int] = []
    used_character_fallback = False

    sentences = [part for part in _SENTENCE_SPLIT.split(normalized_text) if part.strip()]
    if not sentences:
        sentences = [normalized_text]

    for sentence in sentences:
        # Keep the reference behavior: Chinese G2P drops Latin letters inside a
        # Chinese sentence.  Mixed mode splits them out before this function.
        sentence = re.sub("[a-zA-Z]+", "", sentence)
        if not sentence:
            continue

        tagged = _segment_with_jieba(sentence)
        if tagged is None:
            tagged = _character_segments(sentence)
            used_character_fallback = True
        tagged = tone_modifier.pre_merge_for_modify(tagged)

        initials: List[List[str]] = []
        finals: List[List[str]] = []
        for word, pos in tagged:
            if not word:
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
            initials.append(sub_initials)
            finals.append(sub_finals)

        for c, v in zip(
            (item for sub in initials for item in sub),
            (item for sub in finals for item in sub),
        ):
            mapped = _pinyin_to_phones(c + v, c, v, pinyin_to_symbol_map)
            if mapped is None:
                if strict:
                    raise UnknownWordError(
                        f"cannot convert Chinese character to a v2 phoneme: "
                        f"initial={c!r}, final={v!r}"
                    )
                phones.append("UNK")
                word2ph.append(1)
                warnings.append(f"unmapped Chinese pinyin {c + v!r} -> UNK")
            else:
                phones.extend(mapped)
                word2ph.append(len(mapped))

    if used_character_fallback:
        warnings.append(
            "jieba is not installed; Chinese G2P uses character segmentation "
            "and rule-based tone sandhi (POS-aware neutral tones are unavailable)"
        )

    if len(word2ph) != len(normalized_text):
        # This should not happen; surface it rather than returning a silently
        # misaligned word2ph that would corrupt BERT feature repetition.
        raise UnknownWordError(
            "internal Chinese alignment error: "
            f"word2ph has {len(word2ph)} entries for {len(normalized_text)} characters"
        )
    if sum(word2ph) != len(phones):
        raise UnknownWordError(
            f"internal Chinese alignment error: sum(word2ph)={sum(word2ph)} but "
            f"phones={len(phones)}"
        )
    return phones, word2ph, list(dict.fromkeys(warnings))
