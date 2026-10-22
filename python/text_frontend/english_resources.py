"""Explicit English POS resource paths; runtime code never downloads data."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import MissingDependencyError, ResourceError

TAGGER_NAME = "averaged_perceptron_tagger_eng"
TAGGER_FILES = [f"{TAGGER_NAME}.{part}.json" for part in ("weights", "tagdict", "classes")]


def nltk_data_directory(override=None) -> Path:
    configured = override if override is not None else os.environ.get("GPT_SOVITS_NLTK_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "gpt-sovits" / "text_frontend" / "nltk_data"


def tagger_directory(override=None) -> Path:
    root = nltk_data_directory(override)
    directory = root / "taggers" / TAGGER_NAME
    missing = [name for name in TAGGER_FILES if not (directory / name).is_file()]
    if missing:
        raise ResourceError(
            f"English POS resources are missing in {directory}: {', '.join(missing)}. "
            "Prepare them explicitly with: PYTHONPATH=python python -m text_frontend "
            f"--prepare-english-resources --nltk-data {root}. "
            "Inference never downloads data or falls back to basic G2P automatically."
        )
    return directory


def load_pos_tagger(override=None):
    directory = tagger_directory(override)
    try:
        from nltk.tag.perceptron import PerceptronTagger
    except ImportError as exc:
        raise MissingDependencyError("nltk", "English POS disambiguation") from exc
    tagger = PerceptronTagger(load=False, lang="eng")
    try:
        import json
        with (directory / TAGGER_FILES[0]).open(encoding="utf-8") as file:
            weights = json.load(file)
        with (directory / TAGGER_FILES[1]).open(encoding="utf-8") as file:
            tagdict = json.load(file)
        with (directory / TAGGER_FILES[2]).open(encoding="utf-8") as file:
            classes = json.load(file)
        import math
        if not isinstance(classes, list) or not classes or not all(isinstance(label, str) and label for label in classes):
            raise ValueError("invalid perceptron classes")
        labels = set(classes)
        if not isinstance(weights, dict) or not weights or not all(
            isinstance(feature, str) and isinstance(scores, dict)
            and all(label in labels and isinstance(score, (int, float)) and math.isfinite(score)
                    for label, score in scores.items())
            for feature, scores in weights.items()
        ) or not any(weights.values()):
            raise ValueError("invalid or empty perceptron weights")
        if not isinstance(tagdict, dict) or not tagdict or not all(
            isinstance(word, str) and isinstance(label, str) and label in labels
            for word, label in tagdict.items()
        ):
            raise ValueError("invalid perceptron tag dictionary")
        tagger.model.weights = weights
        tagger.tagdict = tagdict
        tagger.classes = tagger.model.classes = labels
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ResourceError(f"cannot load English POS resources from {directory}: {exc}") from exc
    return tagger


def prepare_english_resources(override=None, *, model_dir=None) -> dict:
    """Download only when explicitly requested by the caller."""
    from .english_model import model_directory
    model = model_directory(model_dir)
    root = nltk_data_directory(override)
    try:
        directory = tagger_directory(root)
    except ResourceError:
        try:
            from nltk.downloader import Downloader
        except ImportError as exc:
            raise MissingDependencyError("nltk", "English resource preparation") from exc
        root.mkdir(parents=True, exist_ok=True)
        if not Downloader(download_dir=str(root)).download(TAGGER_NAME, quiet=True, raise_on_error=True):
            raise ResourceError(f"failed to download {TAGGER_NAME} into {root}")
        directory = tagger_directory(root)
    load_pos_tagger(root)
    return {"nltk_data_dir": str(root), "tagger_dir": str(directory), "g2p_model_dir": str(model)}
