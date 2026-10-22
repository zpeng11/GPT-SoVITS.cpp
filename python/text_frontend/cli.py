"""Command-line interface for the text frontend.

Examples::

    PYTHONPATH=python python -m text_frontend --text "你好，世界！" --language auto
    PYTHONPATH=python python -m text_frontend --text-file in.txt \
        --tokenizer /path/tokenizer.json --json out.json --npz out.npz

Structured JSON is the primary C++ integration surface.  ``--npz`` writes
int32 phoneme/token arrays and the (T, 1024) BERT feature layout contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .errors import TextFrontendError
from .frontend import SCHEMA, TextFrontend
from .symbols import SYMBOLS, VERSION


def _read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        try:
            return Path(args.text_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise TextFrontendError(f"cannot read --text-file {args.text_file}: {exc}") from exc
    raise TextFrontendError("one of --text or --text-file is required")


def _write_json(payload: dict, destination: Optional[str], pretty: bool) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=False,
    )
    if destination and destination != "-":
        Path(destination).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


def _save_npz(result, path: str, features=None) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        from .errors import MissingDependencyError
        raise MissingDependencyError("numpy", "--npz output") from exc

    offsets = result.segment_phoneme_offsets()
    norm_offsets = [0]
    for segment in result.segments:
        norm_offsets.append(norm_offsets[-1] + len(segment.normalized_text))

    tokenizer = result.tokenizer
    if tokenizer is not None:
        token_ids = np.asarray(tokenizer.input_ids, dtype=np.int32)
        body_mask = np.zeros(len(token_ids), dtype=np.int32)
        body_mask[tokenizer.body_start:tokenizer.body_end] = 1
        starts = np.asarray(
            [span[0] if span is not None else -1 for span in tokenizer.token_char_spans],
            dtype=np.int32,
        )
        ends = np.asarray(
            [span[1] if span is not None else -1 for span in tokenizer.token_char_spans],
            dtype=np.int32,
        )
    else:
        token_ids = np.zeros((0,), dtype=np.int32)
        body_mask = np.zeros((0,), dtype=np.int32)
        starts = np.zeros((0,), dtype=np.int32)
        ends = np.zeros((0,), dtype=np.int32)

    payload = {
        "phoneme_ids": np.asarray(result.phoneme_ids, dtype=np.int32),
        "word2ph": np.asarray(result.word2ph or [], dtype=np.int32),
        "segment_phoneme_offsets": np.asarray(offsets, dtype=np.int32),
        "segment_normalized_offsets": np.asarray(norm_offsets, dtype=np.int32),
        "tokenizer_input_ids": token_ids,
        "tokenizer_body_mask": body_mask,
        "token_char_starts": starts,
        "token_char_ends": ends,
    }
    if features is not None:
        payload["bert_features"] = features.astype(np.float32, copy=False)
    np.savez(path, **payload)
    return {
        "path": str(Path(path)),
        "phoneme_ids_dtype": "int32",
        "arrays": {
            "phoneme_ids": list(payload["phoneme_ids"].shape),
            "word2ph": list(payload["word2ph"].shape),
            "segment_phoneme_offsets": list(payload["segment_phoneme_offsets"].shape),
            "segment_normalized_offsets": list(payload["segment_normalized_offsets"].shape),
            "tokenizer_input_ids": list(token_ids.shape),
            "bert_features": list(features.shape) if features is not None else None,
        },
    }


def _load_bert_features(path: str, expected_t: int):
    from .layout import validate_bert_features

    try:
        import numpy as np
    except ImportError as exc:
        from .errors import MissingDependencyError
        raise MissingDependencyError("numpy", "--bert-features validation") from exc
    try:
        features = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TextFrontendError(f"cannot load --bert-features {path}: {exc}") from exc
    return validate_bert_features(features, expected_t=expected_t)


def _self_check(args) -> dict:
    import importlib.util

    deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pypinyin", "jieba", "numpy", "nltk", "wordsegment")
    }
    resources = None
    resources_error = None
    try:
        from .resources import Resources
        resources = Resources(args.data_dir).describe()
    except Exception as exc:  # report, do not hide the reason
        resources_error = f"{type(exc).__name__}: {exc}"
    tokenizer = {
        "requested": args.tokenizer is not None,
        "path": args.tokenizer,
        "available": bool(args.tokenizer and Path(args.tokenizer).is_file()),
    }
    english = {"backend": args.english_backend, "ok": True}
    try:
        if args.english_backend == "g2p_en":
            from .g2p_en import EnglishG2P
            from .resources import Resources
            engine = EnglishG2P(Resources(args.data_dir), nltk_data_dir=args.nltk_data, model_dir=args.english_model_dir)
            engine.g2p("I read books and use quux.")
            english["nltk_data_dir"] = str(args.nltk_data) if args.nltk_data else None
    except Exception as exc:
        english.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return {
        "ok": resources_error is None and english["ok"] and deps["pypinyin"],
        "schema": SCHEMA,
        "english": english,
        "version": VERSION,
        "python": sys.version.split()[0],
        "dependencies": deps,
        "resources": resources,
        "resources_error": resources_error,
        "tokenizer": tokenizer,
        "symbols": {"count": len(SYMBOLS), "version": VERSION},
        "notes": [
            "pypinyin is required only for Chinese.",
            "jieba is optional (POS-aware tone sandhi).",
            "English defaults to NumPy g2p-en with NLTK POS and wordsegment.",
            "Inference does not download resources; use --prepare-english-resources explicitly.",
            "torch/transformers/MNN are not used or required.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m text_frontend",
        description="torch-free GPT-SoVITS v2 Chinese/English text frontend",
    )
    parser.add_argument("--text", help="input text")
    parser.add_argument("--text-file", help="read input text from a UTF-8 file")
    parser.add_argument(
        "--language",
        default="auto",
        choices=("auto", "mixed", "zh", "all_zh", "en", "all_en"),
        help="explicit language mode (default: auto/mixed script runs)",
    )
    parser.add_argument("--tokenizer", help="tokenizer.json or vocab.txt (optional)")
    parser.add_argument("--data-dir", help="override package data directory")
    parser.add_argument("--english-backend", choices=("g2p_en", "basic"), default="g2p_en")
    parser.add_argument("--nltk-data", help="explicit root for English POS JSON data")
    parser.add_argument("--english-model-dir", help="directory with g2p-en 2.1.0 checkpoint and homographs")
    parser.add_argument("--prepare-english-resources", action="store_true", help="explicitly download English POS resources, then exit")
    parser.add_argument("--json", dest="json_out", help="write JSON here ('-' = stdout)")
    parser.add_argument("--npz", help="write int32 arrays to this .npz file")
    parser.add_argument("--bert-features", help="optional (T,1024) .npy to validate/attach")
    parser.add_argument("--bert-bin", help="write ggml {1024,T} F32 bytes from --bert-features")
    parser.add_argument("--strict", action="store_true", help="raise on OOV instead of fallback")
    parser.add_argument(
        "--allow-unsupported-scripts",
        action="store_true",
        help="drop unsupported scripts with warnings in mixed mode",
    )
    parser.add_argument("--no-segment-tokenizers", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not print JSON to stdout")
    parser.add_argument("--self-check", action="store_true", help="report dependencies/resources")
    parser.add_argument("--list-symbols", action="store_true", help="print the v2 symbol table")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_symbols:
        payload = {"version": VERSION, "count": len(SYMBOLS), "symbols": list(SYMBOLS)}
        _write_json(payload, args.json_out, args.pretty)
        return 0

    if args.self_check:
        payload = _self_check(args)
        _write_json(payload, args.json_out, args.pretty)
        return 0 if payload["ok"] else 1

    try:
        if args.prepare_english_resources:
            from .english_resources import prepare_english_resources
            payload = prepare_english_resources(args.nltk_data, model_dir=args.english_model_dir)
            _write_json({"ok": True, **payload}, args.json_out, args.pretty)
            return 0
        text = _read_text(args)
        frontend = TextFrontend(
            data_dir=args.data_dir,
            tokenizer_path=args.tokenizer,
            strict=args.strict,
            allow_unsupported=args.allow_unsupported_scripts,
            include_segment_tokenizers=not args.no_segment_tokenizers,
            english_backend=args.english_backend,
            nltk_data_dir=args.nltk_data,
            english_model_dir=args.english_model_dir,
        )
        result = frontend.process(text, args.language)
        payload = result.to_dict()

        features = None
        if args.bert_features:
            features = _load_bert_features(args.bert_features, result.phoneme_count)
            layout = payload["bert_feature_layout"]
            layout["status"] = "provided_by_caller"
            layout["numpy_shape"] = list(features.shape)
            layout["ggml_ne"] = [1024, int(features.shape[0])]
            layout["dtype"] = str(features.dtype)
            if args.bert_bin:
                from .layout import pack_bert_features_for_ggml
                Path(args.bert_bin).write_bytes(
                    pack_bert_features_for_ggml(features, expected_t=result.phoneme_count)
                )
                layout["ggml_bin"] = str(Path(args.bert_bin))
        elif args.bert_bin:
            raise TextFrontendError("--bert-bin requires --bert-features")

        if args.npz:
            payload["npz"] = _save_npz(result, args.npz, features)

        if not args.quiet:
            _write_json(payload, args.json_out, args.pretty)
        elif args.json_out and args.json_out != "-":
            _write_json(payload, args.json_out, args.pretty)
        return 0
    except Exception as exc:
        error = {
            "ok": False,
            "schema": SCHEMA,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        # JSON errors go to stdout when JSON output was requested or implied.
        if args.json_out is not None or args.npz is None:
            _write_json(error, "-", args.pretty)
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
