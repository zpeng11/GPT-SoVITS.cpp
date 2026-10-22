"""Package-scoped resource loading.

Resources are resolved from this package's ``data/`` directory only, or from
an explicit ``data_dir`` / ``GPT_SOVITS_TEXT_DATA`` override.  Nothing is
looked up in the current working directory, a user model cache, or the old
GPT-SoVITS-OIE runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from .errors import ResourceError

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"

_OPENCPOP = "opencpop-strict.txt"
_CMUDICT = "cmudict.rep"
_HOT_WORDS = "engdict-hot.rep"


class Resources:
    """Lazy, package-local resource container.

    Chinese G2P needs ``opencpop-strict.txt`` and pypinyin.  English G2P needs
    ``cmudict.rep``.  A missing file is reported with the exact expected path
    and the active data directory, never silently replaced by global data.
    """

    def __init__(self, data_dir: Optional[os.PathLike | str] = None):
        override = data_dir if data_dir is not None else os.environ.get("GPT_SOVITS_TEXT_DATA")
        self.data_dir = Path(override).expanduser().resolve() if override else DEFAULT_DATA_DIR
        if not self.data_dir.is_dir():
            raise ResourceError(
                f"text frontend data directory does not exist: {self.data_dir}. "
                "Set GPT_SOVITS_TEXT_DATA to a directory containing opencpop-strict.txt "
                "and cmudict.rep, or leave it unset to use the package data."
            )
        self._opencpop: Optional[Dict[str, str]] = None
        self._cmudict: Optional[Dict[str, List[str]]] = None
        self._hot_words: Optional[Dict[str, List[str]]] = None
        self._english_dictionary: Optional[Dict[str, List[str]]] = None
        self._names: Optional[Dict[str, List[str]]] = None

    def data_path(self, name: str) -> Path:
        path = self.data_dir / name
        if not path.is_file():
            raise ResourceError(
                f"required text frontend resource is missing: {path}. "
                f"Expected file '{name}' in data directory {self.data_dir}."
            )
        return path

    def has(self, name: str) -> bool:
        return (self.data_dir / name).is_file()

    def load_opencpop(self) -> Dict[str, str]:
        """Return pinyin -> "initial final" mapping."""
        if self._opencpop is None:
            path = self.data_path(_OPENCPOP)
            mapping: Dict[str, str] = {}
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2 or not parts[0] or not parts[1].strip():
                        raise ResourceError(
                            f"malformed opencpop mapping at {path}:{lineno}: {line!r}; "
                            "expected '<pinyin>\\t<initial> <final>'"
                        )
                    mapping[parts[0]] = parts[1].strip()
            if not mapping:
                raise ResourceError(f"opencpop mapping is empty: {path}")
            self._opencpop = mapping
        return self._opencpop

    def _read_cmudict_file(self, path: Path, out: Dict[str, List[str]]) -> None:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith(";;;"):
                    continue
                if "  " in line:
                    word, pron = line.split("  ", 1)
                else:
                    fields = line.split()
                    if len(fields) < 2:
                        continue
                    word, pron = fields[0], " ".join(fields[1:])
                word = word.strip().lower()
                # Alternate pronunciations are stored as word(2), word(3), ...
                # Keep the base pronunciation only; the G2P has a deterministic
                # first-pronunciation policy.
                if not word or "(" in word:
                    continue
                phones = pron.split()
                if phones:
                    out.setdefault(word, phones)

    def load_cmudict(self) -> Dict[str, List[str]]:
        """Return lower-case word -> first ARPA pronunciation."""
        if self._cmudict is None:
            path = self.data_path(_CMUDICT)
            cmu: Dict[str, List[str]] = {}
            self._read_cmudict_file(path, cmu)
            if not cmu:
                raise ResourceError(f"CMU dictionary is empty: {path}")
            cmu.update(self.load_hot_words())
            self._cmudict = cmu
        return self._cmudict

    def load_english_dictionary(self) -> Dict[str, List[str]]:
        """OIE dictionary order: CMU, fast additions, then hot-word overrides."""
        if self._english_dictionary is None:
            self.data_path(_HOT_WORDS)
            cmu = dict(self.load_cmudict())
            self._read_cmudict_file(self.data_path("cmudict-fast.rep"), cmu)
            cmu.update(self.load_hot_words())
            for word in ("ae", "ai", "ar", "ios", "hud", "os"):
                cmu.pop(word, None)
            self._english_dictionary = cmu
        return self._english_dictionary

    def load_names(self) -> Dict[str, List[str]]:
        if self._names is None:
            import json
            path = self.data_path("namedict.json")
            try:
                names = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(names, dict) or not all(
                    isinstance(word, str) and isinstance(phones, list) and phones
                    and all(isinstance(phone, str) for phone in phones)
                    for word, phones in names.items()
                ):
                    raise ValueError("expected word -> non-empty list of phones")
            except (OSError, ValueError) as exc:
                raise ResourceError(f"cannot load name dictionary {path}: {exc}") from exc
            self._names = names
        return self._names

    def load_hot_words(self) -> Dict[str, List[str]]:
        """Return custom overrides from engdict-hot.rep (optional file)."""
        if self._hot_words is None:
            hot: Dict[str, List[str]] = {}
            if self.has(_HOT_WORDS):
                path = self.data_path(_HOT_WORDS)
                with path.open("r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        fields = line.split()
                        if len(fields) < 2:
                            raise ResourceError(
                                f"malformed hot-word entry at {path}:{lineno}: {line!r}"
                            )
                        hot[fields[0].lower()] = fields[1:]
            self._hot_words = hot
        return self._hot_words

    def describe(self) -> Dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "opencpop": self.has(_OPENCPOP),
            "cmudict": self.has(_CMUDICT),
            "hot_words": self.has(_HOT_WORDS),
            "opencpop_entries": len(self._opencpop) if self._opencpop is not None else None,
            "cmudict_entries": len(self._cmudict) if self._cmudict is not None else None,
        }
