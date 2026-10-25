#!/usr/bin/env python3
"""Regenerate the checked-in real-tokenizer BERT integration fixture.

The C++ integration test never runs Python.  This script is the only path that
touches the Python text frontend and the real RoBERTa tokenizer; it writes a
single tracked fixture (``fixtures/bert_mixed.json``) and never modifies the
existing mini-vocabulary fixtures or numerical reference binaries.

Requirements
------------
* ``pypinyin==0.55.0`` -- checked exactly, because pinyin/tone output determines
  the phoneme IDs, ``word2ph`` and therefore the BERT alignment in the fixture.
* A local Hugging Face ``tokenizer.json`` (or WordPiece ``vocab.txt``) passed
  explicitly with ``--tokenizer``.  Model assets are not checked in; the local
  checkpoint directory used during development contains, for example::

      tests/t2s/models/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json

The script never downloads resources and never installs packages.  Point a
controlled interpreter at it, for example::

    /tmp/gsv_text_frontend_fixture_venv/bin/python \
        tests/frontend/generate_bert_fixture.py \
        --tokenizer tests/t2s/models/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json

Determinism
-----------
* The English backend is fixed to ``basic``.
* ``jieba`` is blocked at import time so an installed copy cannot change
  segmentation, POS tags, neutral tones or the resulting fixture.
* Only ``pypinyin==0.55.0`` is accepted.
* The input text is fixed to a short zh/en/zh sample.
* The fixture records the tokenizer SHA-256, a fixed model source name and
  format in an unknown ``fixture_provenance`` metadata object (the C++ contract
  ignores unknown metadata).  The source name must not depend on the caller's
  filesystem path, so ``--check`` is stable across machines and local copies.

Usage
-----
::

    python generate_bert_fixture.py --tokenizer PATH            # write
    python generate_bert_fixture.py --tokenizer PATH --check    # compare
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

FIXTURE_NAME = "bert_mixed.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"

# Short mixed text explicitly required by the integration test.  The frontend
# normalizes the full-width punctuation; the fixture stores that normalized form.
MIXED_TEXT = "你好，世界！Hello world!再见。"
EXPECTED_LANGUAGES = ["zh", "en", "zh"]
EXPECTED_SCHEMA = "gpt-sovits-text-frontend/v1"
EXPECTED_VERSION = "v2"
REQUIRED_PYPINYIN = "0.55.0"
CLS_ID = 101
SEP_ID = 102
TOKENIZER_SOURCE_NAME = "chinese-roberta-wwm-ext-large/tokenizer.json"
# Exact per-segment ids produced by the local real tokenizer for MIXED_TEXT.
EXPECTED_INPUT_IDS = [
    [101, 872, 1962, 117, 686, 4518, 106, 102],
    [101, 8701, 8572, 106, 102],
    [101, 1086, 6224, 119, 102],
]


class _BlockJieba:
    """Import hook that makes optional jieba look uninstalled."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jieba" or fullname.startswith("jieba."):
            raise ImportError(
                "generate_bert_fixture.py blocks jieba to keep fixtures deterministic"
            )
        return None


def _require_pypinyin():
    try:
        import pypinyin
    except ImportError as exc:
        raise SystemExit(
            "pypinyin is required to generate this fixture.\n"
            f"The script requires pypinyin=={REQUIRED_PYPINYIN}; use a controlled "
            "interpreter that already has it installed.  This script never "
            "installs packages."
        ) from exc
    version = getattr(pypinyin, "__version__", None)
    if version != REQUIRED_PYPINYIN:
        raise SystemExit(
            f"pypinyin=={REQUIRED_PYPINYIN} is required for deterministic fixtures; "
            f"found {version!r} in {sys.executable}."
        )
    return version


def _load_frontend():
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))
    # Install the hook before importing the package: normalize_zh imports jieba
    # lazily, but tone_sandhi imports it at module import time.
    sys.meta_path.insert(0, _BlockJieba())
    from text_frontend import TextFrontend

    return TextFrontend


def _tokenizer_provenance(tokenizer_path: Path) -> dict:
    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer file does not exist: {tokenizer_path}")
    raw = tokenizer_path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
        is_hf_json = isinstance(parsed, dict) and isinstance(parsed.get("model"), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        is_hf_json = False
    return {
        "source": TOKENIZER_SOURCE_NAME,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": "huggingface-tokenizer-json"
        if is_hf_json
        else "wordpiece-vocab-txt",
    }


def _check_tokenizer(segment: dict, expected_ids: list, *, label: str) -> None:
    tokenizer = segment.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise SystemExit(f"{label}: segment tokenizer is missing")
    if tokenizer.get("cls_id") != CLS_ID or tokenizer.get("sep_id") != SEP_ID:
        raise SystemExit(
            f"{label}: expected real RoBERTa CLS/SEP ids {CLS_ID}/{SEP_ID}, "
            f"got {tokenizer.get('cls_id')}/{tokenizer.get('sep_id')}"
        )
    ids = tokenizer.get("input_ids") or []
    if ids != expected_ids:
        raise SystemExit(
            f"{label}: input_ids do not match the known real-tokenizer fixture: "
            f"{ids}"
        )
    if tokenizer.get("body_start") != 1 or tokenizer.get("body_end") != len(ids) - 1:
        raise SystemExit(f"{label}: body_start/body_end do not exclude CLS/SEP")


def build_fixture(tokenizer_path: Path) -> bytes:
    pypinyin_version = _require_pypinyin()
    tokenizer_info = _tokenizer_provenance(tokenizer_path)

    TextFrontend = _load_frontend()
    frontend = TextFrontend(english_backend="basic", tokenizer_path=tokenizer_path)
    payload = frontend.process(MIXED_TEXT).to_dict()

    # Keep failures local and make the provenance self-describing.
    assert payload["schema"] == EXPECTED_SCHEMA
    assert payload["version"] == EXPECTED_VERSION
    assert payload["tokenizer_status"] == "ok"
    segments = payload["segments"]
    assert [segment["language"] for segment in segments] == EXPECTED_LANGUAGES
    assert payload["normalized_text"] == "你好,世界!Hello world!再见."
    assert payload["phoneme_count"] == sum(
        segment["phoneme_count"] for segment in segments
    )
    assert payload["phoneme_count"] > 0
    for index, segment in enumerate(segments):
        assert segment["phoneme_count"] > 0
        _check_tokenizer(segment, EXPECTED_INPUT_IDS[index], label=f"segments[{index}]")
        if segment["language"] == "zh":
            assert segment["word2ph"] is not None
            assert sum(segment["word2ph"]) == segment["phoneme_count"]
            assert len(segment["word2ph"]) == len(segment["normalized_text"])
            assert len(segment["tokenizer"]["input_ids"]) <= 512
        else:
            assert segment["word2ph"] is None

    payload["fixture_provenance"] = {
        "generator": "tests/frontend/generate_bert_fixture.py",
        "purpose": "real-tokenizer zh/en/zh BERT assembly integration fixture",
        "source_text": MIXED_TEXT,
        "language_mode": "auto",
        "english_backend": "basic",
        "jieba": "blocked at import time",
        "include_segment_tokenizers": True,
        "pypinyin_version": pypinyin_version,
        "tokenizer": tokenizer_info,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _print_diff(name: str, checked_in: bytes, generated: bytes) -> None:
    try:
        checked_in_lines = checked_in.decode("utf-8").splitlines(keepends=True)
        generated_lines = generated.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"  {name}: bytes differ")
        return
    diff = difflib.unified_diff(
        checked_in_lines,
        generated_lines,
        fromfile=f"checked-in/{name}",
        tofile=f"generated/{name}",
    )
    sys.stdout.writelines(diff)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="path to the local real RoBERTa tokenizer.json (or vocab.txt)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated output with the checked-in fixture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIXTURE_DIR,
        help=f"fixture directory (default: {FIXTURE_DIR})",
    )
    args = parser.parse_args(argv)

    tokenizer_path: Path = args.tokenizer
    generated = build_fixture(tokenizer_path)
    output_path: Path = args.output_dir / FIXTURE_NAME

    if args.check:
        if not output_path.is_file():
            print(f"MISSING {output_path}", file=sys.stderr)
            return 1
        actual = output_path.read_bytes()
        if actual != generated:
            print(f"DIFF    {output_path}")
            _print_diff(FIXTURE_NAME, actual, generated)
            print(f"\n{FIXTURE_NAME} needs regeneration.", file=sys.stderr)
            return 1
        print(f"ok      {output_path}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(generated)
    print(f"wrote   {output_path}")
    print(f"tokenizer sha256: {hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
