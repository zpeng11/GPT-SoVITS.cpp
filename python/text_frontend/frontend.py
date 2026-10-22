"""Orchestrates normalization, G2P, segmentation and tokenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .errors import EmptyTextError, TextFrontendError, UnknownWordError
from .g2p_en import EnglishG2P
from .normalize_en import normalize_en
from .normalize_zh import g2p_zh, normalize_zh
from .resources import Resources
from .symbols import VERSION, symbol_ids, unknown_symbols
from .text import SegmentSpec, segment_text
from .tokenizer import TokenizedText, WordPieceTokenizer

SCHEMA = "gpt-sovits-text-frontend/v1"

# The BERT feature is not computed here.  This metadata states the exact
# contract the C++ side must use when a feature provider is added later.
BERT_LAYOUT = {
    "status": "not_computed",
    "channels": 1024,
    "numpy_shape": None,
    "ggml_ne": None,
    "dtype": "float32",
    "note": (
        "NumPy features are (T, 1024) C-order; the same buffer is ggml "
        "{1024, T}. Do not transpose a (T, 1024) array before writing it."
    ),
}

WORD2PH_NOTE_EN = (
    "English segments do not produce word2ph. The reference pipeline "
    "zero-fills their BERT features; use segment phoneme offsets for slicing."
)


@dataclass
class SegmentResult:
    index: int
    language: str
    source_text: str
    normalized_text: str
    phones: List[str]
    phoneme_ids: List[int]
    word2ph: Optional[List[int]]
    warnings: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    tokenizer: Optional[TokenizedText] = None
    normalized_start: int = 0
    normalized_end: int = 0
    phoneme_offset: int = 0

    @property
    def word2ph_unit(self) -> Optional[str]:
        return "char" if self.word2ph is not None else None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "language": self.language,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "normalized_start": self.normalized_start,
            "normalized_end": self.normalized_end,
            "char_count": len(self.normalized_text),
            "phones": self.phones,
            "phoneme_ids": self.phoneme_ids,
            "phoneme_offset": self.phoneme_offset,
            "phoneme_count": len(self.phoneme_ids),
            "word2ph": self.word2ph,
            "word2ph_unit": self.word2ph_unit,
            "word2ph_note": None if self.word2ph is not None else WORD2PH_NOTE_EN,
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer is not None else None,
            "unknown_words": self.unknown_words,
            "warnings": self.warnings,
        }


@dataclass
class FrontendResult:
    input_text: str
    language_mode: str
    normalized_text: str
    phones: List[str]
    phoneme_ids: List[int]
    segments: List[SegmentResult]
    tokenizer: Optional[TokenizedText]
    warnings: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    word2ph: Optional[List[int]] = None
    tokenizer_requested: bool = False
    english_backend: str = "g2p_en"

    @property
    def phoneme_count(self) -> int:
        return len(self.phoneme_ids)

    def segment_phoneme_offsets(self) -> List[int]:
        offsets = [0]
        for segment in self.segments:
            offsets.append(offsets[-1] + len(segment.phoneme_ids))
        return offsets

    def to_dict(self) -> dict:
        layout = dict(BERT_LAYOUT)
        layout["numpy_shape"] = [self.phoneme_count, 1024]
        layout["ggml_ne"] = [1024, self.phoneme_count]
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "language_mode": self.language_mode,
            "english_backend": self.english_backend,
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "phones": self.phones,
            "phoneme_ids": self.phoneme_ids,
            "phoneme_ids_dtype": "int32",
            "phoneme_count": self.phoneme_count,
            "word2ph": self.word2ph,
            "word2ph_dtype": "int32",
            "word2ph_note": (
                None
                if self.word2ph is not None
                else "word2ph is per-segment for mixed text; see segments[].word2ph"
            ),
            "segments": [segment.to_dict() for segment in self.segments],
            "segment_phoneme_offsets": self.segment_phoneme_offsets(),
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer is not None else None,
            "tokenizer_status": "ok" if self.tokenizer is not None else (
                "not_configured" if not self.tokenizer_requested else "error"
            ),
            "bert_feature_layout": layout,
            "unknown_words": self.unknown_words,
            "warnings": self.warnings,
        }


class TextFrontend:
    """torch-free Chinese/English text frontend.

    Parameters
    ----------
    data_dir:
        Optional override for the package ``data/`` directory.  Also settable
        with ``GPT_SOVITS_TEXT_DATA``.
    tokenizer_path:
        Optional ``tokenizer.json`` or ``vocab.txt``.  Without it the frontend
        still produces phones/phoneme IDs/word2ph, but no token IDs.
    english_backend:
        ``g2p_en`` (default) uses OIE pronunciation rules and the NumPy model.
        ``basic`` explicitly opts into the earlier dictionary/rule-only backend.
    nltk_data_dir:
        Explicit root containing taggers/averaged_perceptron_tagger_eng/ JSON.
        Otherwise GPT_SOVITS_NLTK_DATA or the frontend-specific cache is used.
    english_model_dir:
        Optional directory containing checkpoint20.npz and homographs.en.
        Otherwise data is read from the installed g2p-en==2.1.0 distribution.
    strict:
        Raise ``UnknownWordError`` on unknown Chinese pinyin/English OOV
        instead of emitting ``UNK``/fallback phones.
    allow_unsupported:
        In mixed mode, drop unsupported scripts with warnings instead of
        raising ``UnsupportedScriptError``.
    include_segment_tokenizers:
        Also tokenize each segment independently (each gets exactly one
        CLS/SEP).  The top-level tokenizer always sees the concatenated text.
    """

    def __init__(
        self,
        *,
        data_dir=None,
        tokenizer_path=None,
        strict: bool = False,
        allow_unsupported: bool = False,
        include_segment_tokenizers: bool = True,
        english_backend: str = "g2p_en",
        nltk_data_dir=None,
        english_model_dir=None,
    ):
        if english_backend not in ("g2p_en", "basic"):
            raise ValueError("english_backend must be 'g2p_en' or 'basic'")
        self.english_backend = english_backend
        self.resources = Resources(data_dir)
        if english_backend == "basic":
            from .g2p_en_basic import EnglishG2P as BasicEnglishG2P
            self._english = BasicEnglishG2P(self.resources, strict=strict)
        else:
            self._english = EnglishG2P(
                self.resources, strict=strict, nltk_data_dir=nltk_data_dir,
                model_dir=english_model_dir,
            )
        self.strict = bool(strict)
        self.allow_unsupported = bool(allow_unsupported)
        self.include_segment_tokenizers = bool(include_segment_tokenizers)
        self.tokenizer: Optional[WordPieceTokenizer] = None
        self.tokenizer_requested = tokenizer_path is not None
        if tokenizer_path is not None:
            self.tokenizer = WordPieceTokenizer.from_file(tokenizer_path, required=True)

    # ------------------------------------------------------------------
    # Per-language
    # ------------------------------------------------------------------
    def _process_zh(self, source: str):
        normalized = normalize_zh(source)
        if not normalized:
            return normalized, [], None, [], ["Chinese segment normalized to empty text"]
        phones, word2ph, warnings = g2p_zh(normalized, self.resources, strict=self.strict)
        return normalized, phones, word2ph, [], warnings

    def _process_en(self, source: str, *, preserve_edge_spaces: bool = False):
        normalized = normalize_en(source, preserve_edge_spaces=preserve_edge_spaces)
        if not normalized.strip():
            return normalized, [], None, [], ["English segment normalized to empty text"]
        result = self._english.g2p(normalized)
        if self.english_backend == "basic":
            result.warnings.insert(0, "explicit basic English backend: no POS disambiguation or NumPy prediction")
        return (
            normalized,
            result.phones,
            None,
            result.unknown_words,
            result.warnings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, text: str, language: str = "auto") -> FrontendResult:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not text.strip():
            raise EmptyTextError("input text is empty (or whitespace only)")

        specs, seg_warnings = segment_text(
            text, language, allow_unsupported=self.allow_unsupported
        )
        if not specs:
            raise EmptyTextError("no usable text segment after language segmentation")

        warnings: List[str] = list(seg_warnings)
        unknown_words: List[str] = []
        segments: List[SegmentResult] = []
        mixed_mode = language in ("auto", "mixed")

        for spec in specs:
            if spec.language == "zh":
                normalized, phones, word2ph, seg_unknown, seg_warn = self._process_zh(spec.text)
            elif spec.language == "en":
                normalized, phones, word2ph, seg_unknown, seg_warn = self._process_en(
                    spec.text, preserve_edge_spaces=mixed_mode
                )
            else:  # defensive
                raise TextFrontendError(f"internal error: unknown segment language {spec.language!r}")

            warnings.extend(seg_warn)
            unknown_words.extend(seg_unknown)

            if not normalized.strip():
                warnings.append(
                    f"skipped {spec.language} segment {spec.text!r}: normalization produced empty text"
                )
                continue
            if not phones:
                warnings.append(
                    f"{spec.language} segment {normalized!r} produced no phonemes"
                )

            bad = unknown_symbols(phones)
            if bad:
                if self.strict:
                    raise UnknownWordError(
                        f"segment contains phonemes outside the v2 vocabulary: {bad}"
                    )
                warnings.extend(f"phoneme {symbol!r} mapped to UNK" for symbol in bad)

            ids = symbol_ids(phones, strict=self.strict)
            if word2ph is not None:
                if len(word2ph) != len(normalized):
                    raise UnknownWordError(
                        f"internal alignment error in {spec.language} segment: "
                        f"{len(word2ph)} word2ph entries for {len(normalized)} characters"
                    )
                if sum(word2ph) != len(phones):
                    raise UnknownWordError(
                        f"internal alignment error in {spec.language} segment: "
                        f"sum(word2ph)={sum(word2ph)} != phones={len(phones)}"
                    )

            tokenized: Optional[TokenizedText] = None
            if self.tokenizer is not None and self.include_segment_tokenizers:
                try:
                    tokenized = self.tokenizer.encode(normalized)
                except Exception as exc:  # tokenizer is optional; keep phones usable
                    warnings.append(f"segment tokenization failed: {exc}")

            segments.append(
                SegmentResult(
                    index=len(segments),
                    language=spec.language,
                    source_text=spec.text,
                    normalized_text=normalized,
                    phones=phones,
                    phoneme_ids=ids,
                    word2ph=word2ph,
                    warnings=list(seg_warn),
                    unknown_words=list(seg_unknown),
                    tokenizer=tokenized,
                )
            )

        if not segments:
            raise EmptyTextError(
                "input text produced no usable zh/en segments after normalization"
            )

        normalized_text = ""
        phoneme_offset = 0
        for segment in segments:
            segment.normalized_start = len(normalized_text)
            normalized_text += segment.normalized_text
            segment.normalized_end = len(normalized_text)
            segment.phoneme_offset = phoneme_offset
            phoneme_offset += len(segment.phoneme_ids)

        phones = [phone for segment in segments for phone in segment.phones]
        ids = [value for segment in segments for value in segment.phoneme_ids]

        word2ph: Optional[List[int]] = None
        if all(segment.language == "zh" for segment in segments):
            candidate = [value for segment in segments for value in segment.word2ph or []]
            if len(candidate) == len(normalized_text) and sum(candidate) == len(phones):
                word2ph = candidate

        top_tokenizer: Optional[TokenizedText] = None
        if self.tokenizer is not None:
            try:
                top_tokenizer = self.tokenizer.encode(normalized_text)
            except Exception as exc:
                warnings.append(f"top-level tokenization failed: {exc}")

        # Deduplicate warnings while preserving order.
        warnings = list(dict.fromkeys(warnings))
        unknown_words = list(dict.fromkeys(unknown_words))

        return FrontendResult(
            input_text=text,
            language_mode=language,
            normalized_text=normalized_text,
            phones=phones,
            phoneme_ids=ids,
            segments=segments,
            tokenizer=top_tokenizer,
            warnings=warnings,
            unknown_words=unknown_words,
            word2ph=word2ph,
            tokenizer_requested=self.tokenizer_requested,
            english_backend=self.english_backend,
        )
