#!/usr/bin/env python3
"""Regenerate the checked-in text-frontend JSON fixtures.

The C++ tests never run Python.  This script is only needed when the fixture
format changes or the Python frontend output is intentionally refreshed.

Requirements
------------
* The repository's ``python/`` package on ``sys.path`` (done automatically).
* ``pypinyin`` for the Chinese fixtures.  Use a controlled interpreter, for
  example::

      /tmp/gsv-fixture-venv/bin/python tests/frontend/generate_fixtures.py

  The script never downloads resources or installs packages.

Determinism
-----------
``jieba`` is an optional dependency of the Python frontend and changes Chinese
segmentation and tone sandhi.  It is deliberately blocked here so the output
depends only on pypinyin and the checked-in resources.  English uses the
``basic`` backend, so nltk/wordsegment/numpy are not imported.

Usage
-----
::

    python generate_fixtures.py            # rewrite fixtures/
    python generate_fixtures.py --check    # compare, exit 1 on differences
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"

# A deliberately tiny vocabulary.  It contains every code point that appears in
# MIXED_TEXT plus the English words, so each Chinese character has exactly one
# body token.  '𠮷' is supplementary (U+20BB7) and verifies that offsets are
# Unicode code points, not UTF-8 bytes.
MINI_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    ",",
    ".",
    "!",
    "一",
    "七",
    "三",
    "九",
    "二",
    "五",
    "八",
    "六",
    "十",
    "四",
    "天",
    "年",
    "日",
    "月",
    "是",
    "今",
    "好",
    "你",
    "世",
    "界",
    "零",
    "野",
    "家",
    "𠮷",
    "hello",
    "world",
]

MIXED_TEXT = "你好，世界！今天是2026年10月20日。Hello world! 𠮷野家。"
ZH_TEXT = "你好，世界！"
ENGLISH_TEXT = "Hello world!"

# Raw invalid UTF-8: nlohmann::json must reject this at parse time.
INVALID_UTF8 = (
    b'{\n'
    b'  "schema": "gpt-sovits-text-frontend/v1",\n'
    b'  "version": "v2",\n'
    b'  "normalized_text": "invalid \xff byte"\n'
    b'}\n'
)


class _BlockJieba:
    """Import hook that makes optional jieba look uninstalled."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jieba" or fullname.startswith("jieba."):
            raise ImportError(
                "generate_fixtures.py blocks jieba to keep fixtures deterministic"
            )
        return None


def _load_frontend():
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))
    sys.meta_path.insert(0, _BlockJieba())
    from text_frontend import TextFrontend

    return TextFrontend


def _dump(payload: dict, *, escaped: bool) -> bytes:
    text = json.dumps(payload, ensure_ascii=escaped, indent=2) + "\n"
    return text.encode("utf-8")


def build_fixtures() -> dict[str, bytes]:
    if importlib.util.find_spec("pypinyin") is None:
        raise SystemExit(
            "pypinyin is required to regenerate the Chinese fixtures.\n"
            "Run this script with a controlled interpreter that has pypinyin "
            "installed; see the module docstring."
        )

    TextFrontend = _load_frontend()
    vocab_text = "\n".join(MINI_VOCAB) + "\n"

    with tempfile.TemporaryDirectory(prefix="gpt_sovits_frontend_") as tmp:
        vocab_path = Path(tmp) / "mini_vocab.txt"
        vocab_path.write_text(vocab_text, encoding="utf-8")

        frontend = TextFrontend(english_backend="basic", tokenizer_path=vocab_path)
        mixed = frontend.process(MIXED_TEXT).to_dict()
        chinese = frontend.process(ZH_TEXT, "zh").to_dict()
        english = TextFrontend(english_backend="basic").process(ENGLISH_TEXT).to_dict()

    # Sanity checks that keep fixture regeneration failures local and obvious.
    assert mixed["schema"] == "gpt-sovits-text-frontend/v1"
    assert mixed["version"] == "v2"
    assert mixed["tokenizer_status"] == "ok"
    assert [segment["language"] for segment in mixed["segments"]] == [
        "zh",
        "en",
        "zh",
    ]
    assert "𠮷" in mixed["segments"][2]["normalized_text"]
    assert [segment["language"] for segment in chinese["segments"]] == ["zh"]
    assert chinese["word2ph"] is not None
    assert english["tokenizer"] is None
    assert english["segments"][0]["tokenizer"] is None

    return {
        "mini_vocab.txt": vocab_text.encode("utf-8"),
        "valid_mixed.json": _dump(mixed, escaped=False),
        "valid_mixed_escaped.json": _dump(mixed, escaped=True),
        "valid_zh.json": _dump(chinese, escaped=False),
        "valid_en_no_tokenizer.json": _dump(english, escaped=False),
        "invalid_utf8.json": INVALID_UTF8,
    }


def _print_diff(name: str, expected: bytes, actual: bytes) -> None:
    try:
        expected_text = expected.decode("utf-8").splitlines(keepends=True)
        actual_text = actual.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"  {name}: bytes differ")
        return
    diff = difflib.unified_diff(
        expected_text,
        actual_text,
        fromfile=f"checked-in/{name}",
        tofile=f"generated/{name}",
    )
    sys.stdout.writelines(diff)
    if not diff:
        print(f"  {name}: bytes differ but text is identical (line endings?)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated output with the checked-in fixtures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIXTURE_DIR,
        help=f"fixture directory (default: {FIXTURE_DIR})",
    )
    args = parser.parse_args(argv)

    fixtures = build_fixtures()
    output_dir: Path = args.output_dir

    if args.check:
        failures = 0
        for name, expected in fixtures.items():
            path = output_dir / name
            if not path.is_file():
                print(f"MISSING {path}")
                failures += 1
                continue
            actual = path.read_bytes()
            if actual != expected:
                print(f"DIFF    {path}")
                _print_diff(name, expected, actual)
                failures += 1
            else:
                print(f"ok      {path}")
        if failures:
            print(f"\n{failures} fixture(s) need regeneration.", file=sys.stderr)
            return 1
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in fixtures.items():
        path = output_dir / name
        path.write_bytes(data)
        print(f"wrote  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
