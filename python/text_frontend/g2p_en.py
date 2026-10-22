"""OIE-compatible English pronunciation without PyTorch or implicit downloads.

Uses POS-dependent homographs, local pronunciation dictionaries, wordsegment
and the g2p-en 2.1.0 NumPy model. The basic backend is an explicit alternative,
never a fallback for missing models or dependencies.
"""

from __future__ import annotations

import re

from .english_model import NumpyG2P, load_homographs, model_directory
from .english_resources import load_pos_tagger
from .errors import MissingDependencyError, ResourceError, UnknownWordError
from .g2p_en_basic import EnglishG2PResult
from .symbols import SYMBOL_TO_ID


class EnglishG2P:
    def __init__(self, resources, *, strict=False, nltk_data_dir=None, model_dir=None):
        self.resources = resources
        self.strict = strict
        self.nltk_data_dir = nltk_data_dir
        self.model_dir = model_dir
        self._ready = False
        self._memo = {}
        self._predictor = None

    def _load(self):
        if self._ready:
            return
        directory = model_directory(self.model_dir)
        tagger = load_pos_tagger(self.nltk_data_dir)
        try:
            from nltk.tokenize import TweetTokenizer
            from nltk.tokenize.casual import HANG_RE
            from wordsegment import Segmenter
        except ImportError as exc:
            raise MissingDependencyError(
                exc.name, "English tokenization and compound segmentation",
                "python -m pip install -r python/text_frontend/requirements.txt",
            ) from exc
        segmenter = Segmenter()
        try:
            segmenter.load()
        except (OSError, ValueError) as exc:
            raise ResourceError(f"cannot load wordsegment package data: {exc}") from exc
        cmu = self.resources.load_english_dictionary()
        names = self.resources.load_names()
        self._homographs = load_homographs(directory)
        self._model_path = directory
        self._tagger = tagger
        self._tokenizer = TweetTokenizer()
        self._hang_pattern = HANG_RE
        self._segmenter = segmenter
        self._cmu = cmu
        self._names = names
        self._ready = True

    def _predict(self, word):
        if self._predictor is None:
            self._predictor = NumpyG2P(self._model_path)
        return self._predictor.predict(word)

    def _query(self, original):
        word = original.lower()
        key = (word, original.istitle())
        if key in self._memo:
            phones, methods = self._memo[key]
            return list(phones), methods
        if len(word) > 1 and word in self._cmu:
            return list(self._cmu[word]), "dictionary"
        if original.istitle() and word in self._names:
            return list(self._names[word]), "name_dictionary"
        if len(word) <= 3:
            phones = []
            for char in word:
                if char == "a":
                    phones.append("EY1")
                elif char.isalpha():
                    phones.extend(self._cmu[char])
                else:
                    phones.append(char)
            return phones, "short_acronym"
        if re.fullmatch(r"[a-z]+'s", word):
            phones, method = self._query(word[:-2])
            last = phones[-1] if phones else ""
            if last in {"P", "T", "K", "F", "TH", "HH"}:
                suffix = ["S"]
            elif last in {"S", "Z", "SH", "ZH", "CH", "JH"}:
                suffix = ["AH0", "Z"]
            else:
                suffix = ["Z"]
            result, method = phones + suffix, "possessive/" + method
        else:
            components = self._segmenter.segment(word)
            if len(components) <= 1:
                result, method = self._predict(word), "numpy_g2p"
            else:
                result = []
                methods = []
                for component in components:
                    phones, component_method = self._query(component)
                    result.extend(phones)
                    methods.append(component_method)
                method = "compound/" + "/".join(dict.fromkeys(methods))
        self._memo[key] = (tuple(result), method)
        return list(result), method

    def _tokenization_view(self, text):
        safe_parts, offsets = [], []
        cursor = 0
        for match in self._hang_pattern.finditer(text):
            safe_parts.append(text[cursor:match.start()])
            offsets.extend((index, index + 1) for index in range(cursor, match.start()))
            safe_parts.append(match.group(1) * 3)
            offsets.extend([
                (match.start(), match.start() + 1),
                (match.start() + 1, match.start() + 2),
                (match.start() + 2, match.end()),
            ])
            cursor = match.end()
        safe_parts.append(text[cursor:])
        offsets.extend((index, index + 1) for index in range(cursor, len(text)))
        return "".join(safe_parts), offsets

    def g2p(self, text: str) -> EnglishG2PResult:
        self._load()
        token_text, offsets = self._tokenization_view(text)
        words = self._tokenizer.tokenize(token_text)
        tagged = self._tagger.tag(words)
        phones, warnings, unknown_words, spans = [], [], [], []
        cursor = 0
        for original, pos in tagged:
            token_start = token_text.find(original, cursor)
            if token_start < 0:
                raise ResourceError(f"English tokenizer cannot align token {original!r} to input")
            token_end = token_start + len(original)
            start = offsets[token_start][0]
            end = offsets[token_end - 1][1]
            cursor = token_end
            word = original.lower()
            method = "dictionary"
            if re.search("[a-z]", word) is None:
                pronunciation = list(word)
            elif len(word) == 1:
                pronunciation = ["EY1"] if original == "A" else list(self._cmu[word])
            elif word in self._homographs:
                first, second, expected_pos = self._homographs[word]
                matches = pos.startswith(expected_pos) or (
                    len(pos) < len(expected_pos) and pos == expected_pos[:len(pos)]
                )
                pronunciation = list(first if matches else second)
            else:
                if self.strict and word not in self._cmu and not (original.istitle() and word in self._names):
                    raise UnknownWordError(f"strict English G2P rejected OOV word: {original}")
                pronunciation, method = self._query(original)
                if word not in self._cmu and method != "name_dictionary":
                    unknown_words.append(original)
                    warnings.append(f"OOV English word {original!r}: used {method}")
            phone_start = len(phones)
            for phone in pronunciation:
                if phone in {"<pad>", "<s>", "</s>", "UW"}:
                    continue
                phone = "UNK" if phone == "<unk>" else "-" if phone == "'" else phone
                if phone not in SYMBOL_TO_ID:
                    if self.strict:
                        raise UnknownWordError(f"unrepresentable English phone {phone!r} in {original!r}")
                    warnings.append(f"unknown English phone {phone!r} in {original!r} -> UNK")
                    phone = "UNK"
                phones.append(phone)
            spans.append((start, end, phone_start, len(phones)))
        return EnglishG2PResult(
            phones=phones, warnings=list(dict.fromkeys(warnings)),
            unknown_words=list(dict.fromkeys(unknown_words)), token_spans=spans,
        )
