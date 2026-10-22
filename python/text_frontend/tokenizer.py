"""Minimal HuggingFace WordPiece tokenizer with explicit special-token policy.

Supported inputs:
  * a HuggingFace ``tokenizer.json`` containing a ``WordPiece`` model and a
    ``BertNormalizer``/``BertPreTokenizer`` (the Chinese RoBERTa format);
  * a plain ``vocab.txt`` (one token per line).

Special-token policy:
  * ``encode(..., add_special_tokens=True)`` emits exactly one ``[CLS]`` at
    the start and one ``[SEP]`` at the end;
  * literal special-token strings in the input are removed first, so
    ``"[CLS]你好[SEP]"`` cannot produce duplicate special tokens;
  * ``body_start/body_end`` and ``token_char_spans`` describe the non-special
    body so callers can align phonemes/word2ph with tokenizer output.

No transformers/tokenizers package is imported.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import ResourceError, TokenizerError

_SPECIAL_NAMES = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


@dataclass
class TokenizedText:
    input_ids: List[int]
    tokens: List[str]
    token_char_spans: List[Optional[Tuple[int, int]]]
    body_start: int
    body_end: int
    normalized_text: str
    unknown_token_count: int
    special_token_count: int
    literal_special_tokens_removed: int
    cls_id: int
    sep_id: int
    unk_id: int

    @property
    def body_token_ids(self) -> List[int]:
        return self.input_ids[self.body_start:self.body_end]

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "input_ids": self.input_ids,
            "tokens": self.tokens,
            "body_start": self.body_start,
            "body_end": self.body_end,
            "body_token_ids": self.body_token_ids,
            "token_char_spans": [
                list(span) if span is not None else None for span in self.token_char_spans
            ],
            "normalized_text": self.normalized_text,
            "unknown_token_count": self.unknown_token_count,
            "special_token_count": self.special_token_count,
            "literal_special_tokens_removed": self.literal_special_tokens_removed,
            "cls_id": self.cls_id,
            "sep_id": self.sep_id,
            "unk_id": self.unk_id,
        }


@dataclass
class _Body:
    tokens: List[str]
    input_ids: List[int]
    spans: List[Tuple[int, int]]
    normalized_text: str


def _is_punctuation(ch: str) -> bool:
    code = ord(ch)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_chinese_char(ch: str) -> bool:
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


class WordPieceTokenizer:
    def __init__(
        self,
        vocab: Dict[str, int],
        *,
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        continuing_subword_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool = True,
        lowercase: bool = True,
        source: Optional[str] = None,
    ):
        self.vocab = dict(vocab)
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.max_input_chars_per_word = int(max_input_chars_per_word)
        self.clean_text = bool(clean_text)
        self.handle_chinese_chars = bool(handle_chinese_chars)
        self.strip_accents = bool(strip_accents)
        self.lowercase = bool(lowercase)
        self.source = source

        missing = [name for name in (unk_token, cls_token, sep_token) if name not in self.vocab]
        if missing:
            raise TokenizerError(
                f"tokenizer vocabulary is missing required special token(s): {', '.join(missing)}"
            )
        self.unk_id = self.vocab[unk_token]
        self.cls_id = self.vocab[cls_token]
        self.sep_id = self.vocab[sep_token]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Optional[Path | str],
        *,
        required: bool = True,
        **overrides,
    ) -> Optional["WordPieceTokenizer"]:
        if path is None:
            if required:
                raise ResourceError(
                    "a tokenizer is required for this operation; pass tokenizer_path "
                    "or --tokenizer pointing to tokenizer.json or vocab.txt"
                )
            return None
        path = Path(path).expanduser()
        if not path.is_file():
            raise ResourceError(f"tokenizer file does not exist: {path}")
        if path.suffix.lower() == ".json":
            return cls.from_json(path, **overrides)
        return cls.from_vocab_file(path, **overrides)

    @classmethod
    def from_json(cls, path: Path | str, **overrides) -> "WordPieceTokenizer":
        path = Path(path)
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot read tokenizer JSON {path}: {exc}") from exc
        model = obj.get("model")
        if not isinstance(model, dict):
            raise TokenizerError(f"tokenizer JSON {path} has no 'model' object")
        model_type = model.get("type")
        if model_type not in (None, "WordPiece"):
            raise TokenizerError(
                f"unsupported tokenizer model type {model_type!r} in {path}; "
                "this frontend supports WordPiece only"
            )
        vocab = model.get("vocab")
        if isinstance(vocab, list):
            vocab = {token: index for index, token in enumerate(vocab)}
        if not isinstance(vocab, dict) or not vocab:
            raise TokenizerError(f"tokenizer JSON {path} has no non-empty model.vocab")
        for added in obj.get("added_tokens") or []:
            content = added.get("content")
            index = added.get("id")
            if content is not None and index is not None:
                vocab.setdefault(content, int(index))

        normalizer = obj.get("normalizer") or {}
        norm_type = normalizer.get("type")
        if norm_type not in (None, "BertNormalizer", "Sequence"):
            raise TokenizerError(
                f"unsupported normalizer {norm_type!r} in {path}; "
                "this frontend supports BertNormalizer"
            )
        # A Sequence is only accepted when the first stage is BertNormalizer.
        if norm_type == "Sequence":
            stages = normalizer.get("normalizers") or []
            if len(stages) != 1 or stages[0].get("type") != "BertNormalizer":
                raise TokenizerError(f"unsupported normalizer sequence in {path}")
            normalizer = stages[0]
        pre_tokenizer = obj.get("pre_tokenizer") or {}
        pre_type = pre_tokenizer.get("type")
        if pre_type not in (None, "BertPreTokenizer"):
            raise TokenizerError(
                f"unsupported pre_tokenizer {pre_type!r} in {path}; "
                "this frontend supports BertPreTokenizer"
            )

        strip_accents = normalizer.get("strip_accents")
        return cls(
            vocab,
            unk_token=model.get("unk_token", "[UNK]"),
            cls_token="[CLS]",
            sep_token="[SEP]",
            continuing_subword_prefix=model.get("continuing_subword_prefix", "##"),
            max_input_chars_per_word=model.get("max_input_chars_per_word", 100),
            clean_text=normalizer.get("clean_text", True),
            handle_chinese_chars=normalizer.get("handle_chinese_chars", True),
            strip_accents=True if strip_accents is None else bool(strip_accents),
            lowercase=normalizer.get("lowercase", True),
            source=str(path),
            **overrides,
        )

    @classmethod
    def from_vocab_file(cls, path: Path | str, **overrides) -> "WordPieceTokenizer":
        path = Path(path)
        try:
            tokens = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ResourceError(f"cannot read tokenizer vocabulary {path}: {exc}") from exc
        vocab = {token: index for index, token in enumerate(tokens) if token}
        if not vocab:
            raise TokenizerError(f"tokenizer vocabulary is empty: {path}")
        return cls(vocab, source=str(path), **overrides)

    # ------------------------------------------------------------------
    # Normalization / pre-tokenization
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        if self.clean_text:
            chars: List[str] = []
            for ch in text:
                if ch in ("\t", "\n", "\r"):
                    chars.append(" ")
                    continue
                category = unicodedata.category(ch)
                if category in ("Cc", "Cf"):
                    continue
                chars.append(ch)
            text = "".join(chars)
        if self.handle_chinese_chars:
            chars = []
            for ch in text:
                if _is_chinese_char(ch):
                    chars.extend([" ", ch, " "])
                else:
                    chars.append(ch)
            text = "".join(chars)
        if self.strip_accents:
            text = "".join(
                ch for ch in unicodedata.normalize("NFD", text)
                if unicodedata.category(ch) != "Mn"
            )
        if self.lowercase:
            text = text.lower()
        return text

    def _pre_tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """Yield (word, start, end), splitting punctuation like BertPreTokenizer."""
        pieces: List[Tuple[str, int, int]] = []
        index = 0
        length = len(text)
        while index < length:
            ch = text[index]
            if ch.isspace():
                index += 1
                continue
            if _is_punctuation(ch):
                pieces.append((ch, index, index + 1))
                index += 1
                continue
            end = index + 1
            while end < length and not text[end].isspace() and not _is_punctuation(text[end]):
                end += 1
            pieces.append((text[index:end], index, end))
            index = end
        return pieces

    def _wordpiece(self, word: str, word_start: int) -> List[Tuple[str, int, int]]:
        if len(word) > self.max_input_chars_per_word:
            return [(self.unk_token, word_start, word_start + len(word))]
        output: List[Tuple[str, int, int]] = []
        start = 0
        while start < len(word):
            end = len(word)
            found: Optional[str] = None
            found_end = end
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = self.continuing_subword_prefix + piece
                if piece in self.vocab:
                    found = piece
                    found_end = end
                    break
                end -= 1
            if found is None:
                return [(self.unk_token, word_start, word_start + len(word))]
            output.append((found, word_start + start, word_start + found_end))
            start = found_end
        return output

    def _encode_body(self, text: str) -> _Body:
        normalized_parts = []
        source_offsets = []
        for index, char in enumerate(text):
            part = self._normalize(char)
            normalized_parts.append(part)
            source_offsets.extend([index] * len(part))
        normalized = "".join(normalized_parts)
        tokens: List[str] = []
        ids: List[int] = []
        spans: List[Tuple[int, int]] = []
        for word, start, end in self._pre_tokenize(normalized):
            for token, char_start, char_end in self._wordpiece(word, start):
                tokens.append(token)
                ids.append(self.vocab[token])
                spans.append((source_offsets[char_start], source_offsets[char_end - 1] + 1))
        return _Body(tokens=tokens, input_ids=ids, spans=spans, normalized_text=normalized)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode_body(self, text: str) -> dict:
        """Tokenize only the body, without [CLS]/[SEP]."""
        body = self._encode_body(text)
        return {
            "input_ids": body.input_ids,
            "tokens": body.tokens,
            "token_char_spans": [list(span) for span in body.spans],
            "normalized_text": body.normalized_text,
            "unknown_token_count": sum(1 for token in body.tokens if token == self.unk_token),
        }

    def _strip_literal_specials(self, text: str) -> Tuple[str, int]:
        removed = 0
        for token in _SPECIAL_NAMES:
            count = text.count(token)
            if count:
                removed += count
                # Preserve length so offsets stay meaningful.
                text = text.replace(token, " " * len(token))
        return text, removed

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenizedText:
        cleaned, removed = self._strip_literal_specials(text)
        body = self._encode_body(cleaned)
        if add_special_tokens:
            input_ids = [self.cls_id] + body.input_ids + [self.sep_id]
            tokens = [self.cls_token] + body.tokens + [self.sep_token]
            spans: List[Optional[Tuple[int, int]]] = [None] + list(body.spans) + [None]
            body_start = 1
            body_end = 1 + len(body.input_ids)
            special_count = 2
        else:
            input_ids = list(body.input_ids)
            tokens = list(body.tokens)
            spans = list(body.spans)
            body_start = 0
            body_end = len(body.input_ids)
            special_count = 0
        unknown_count = sum(1 for token in body.tokens if token == self.unk_token)
        return TokenizedText(
            input_ids=input_ids,
            tokens=tokens,
            token_char_spans=spans,
            body_start=body_start,
            body_end=body_end,
            normalized_text=body.normalized_text,
            unknown_token_count=unknown_count,
            special_token_count=special_count,
            literal_special_tokens_removed=removed,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            unk_id=self.unk_id,
        )
