"""Offline NumPy inference adapted from g2p-en 2.1.0 (Apache-2.0).

By Kyubyong Park and Jongseok Kim (https://github.com/kyubyong/g2p).
See licenses/Apache-2.0.txt. Modified to load validated local data lazily,
without importing g2p_en or depending on its downloading NLTK corpus loader.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from .errors import MissingDependencyError, ResourceError

GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
    "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]


def model_directory(override=None) -> Path:
    if override is not None:
        directory = Path(override).expanduser().resolve()
    else:
        try:
            distribution = metadata.distribution("g2p-en")
        except metadata.PackageNotFoundError as exc:
            raise MissingDependencyError(
                "g2p-en", "English NumPy pronunciation prediction",
                "python -m pip install -r python/text_frontend/requirements.txt",
            ) from exc
        if distribution.version != "2.1.0":
            raise ResourceError(
                f"expected g2p-en==2.1.0 data, found {distribution.version}; "
                "install the pinned requirements or pass english_model_dir explicitly"
            )
        directory = Path(distribution.locate_file("g2p_en")).resolve()
    for name in ("checkpoint20.npz", "homographs.en"):
        if not (directory / name).is_file():
            raise ResourceError(f"English model resource is missing: {directory / name}")
    return directory


def load_homographs(directory: Path) -> dict:
    path = directory / "homographs.en"
    result = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            word, first, second, pos = line.strip().split("|")
            result[word.lower()] = (first.split(), second.split(), pos)
    except (OSError, ValueError) as exc:
        raise ResourceError(f"cannot load homograph data {path}: {exc}") from exc
    result["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
    result["complex"] = (
        ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
        ["K", "AA1", "M", "P", "L", "EH0", "K", "S"], "JJ",
    )
    return result


class NumpyG2P:
    def __init__(self, directory: Path):
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingDependencyError("numpy", "English G2P prediction") from exc
        self.np = np
        self.g2idx = {char: index for index, char in enumerate(GRAPHEMES)}
        shapes = {
            "enc_emb": (29, 256), "dec_emb": (74, 256),
            "fc_w": (74, 256), "fc_b": (74,),
        }
        for prefix in ("enc", "dec"):
            shapes.update({
                f"{prefix}_w_ih": (768, 256), f"{prefix}_w_hh": (768, 256),
                f"{prefix}_b_ih": (768,), f"{prefix}_b_hh": (768,),
            })
        path = directory / "checkpoint20.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                for name, shape in shapes.items():
                    array = data[name]
                    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError(f"invalid weight {name}: {array.shape}, {array.dtype}")
                    setattr(self, name, array)
        except (OSError, ValueError, KeyError) as exc:
            raise ResourceError(f"cannot load English G2P checkpoint {path}: {exc}") from exc

    def _cell(self, x, hidden, prefix):
        np = self.np
        ih = np.matmul(x, getattr(self, prefix + "_w_ih").T) + getattr(self, prefix + "_b_ih")
        hh = np.matmul(hidden, getattr(self, prefix + "_w_hh").T) + getattr(self, prefix + "_b_hh")
        split = ih.shape[-1] * 2 // 3
        reset, update = np.split(1 / (1 + np.exp(-(ih[:, :split] + hh[:, :split]))), 2, axis=-1)
        candidate = np.tanh(ih[:, split:] + reset * hh[:, split:])
        return (1 - update) * candidate + update * hidden

    def predict(self, word: str) -> list[str]:
        np = self.np
        ids = [self.g2idx.get(char, 1) for char in word.lower()] + [2]
        hidden = np.zeros((1, self.enc_w_hh.shape[-1]), dtype=np.float32)
        for index in ids:
            hidden = self._cell(self.enc_emb[[index]], hidden, "enc")
        decoded = self.dec_emb[[2]]
        phones = []
        for _ in range(20):
            hidden = self._cell(decoded, hidden, "dec")
            index = int((np.matmul(hidden, self.fc_w.T) + self.fc_b).argmax())
            if index == 3:
                break
            phones.append(PHONEMES[index])
            decoded = self.dec_emb[[index]]
        return phones
