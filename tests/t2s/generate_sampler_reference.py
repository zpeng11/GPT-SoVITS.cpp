#!/usr/bin/env python3
"""Export synthetic CPU/F32 sampler fixtures from a trusted upstream checkout.

Only export imports PyTorch and executes AR/models/utils.py. Help and validation
need only NumPy and the standard library. No models, downloads or RNG are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import types

import numpy as np

SCHEMA = "gpt-sovits-t2s-sampler-reference/v1"
UPSTREAM_SOURCE = "AR/models/utils.py"
MIN_RELATIVE_MARGIN = 0.01
SEMANTICS = {
    "probabilities": "vocabulary-indexed;upstream-logits_to_probs",
    "noise": "vocabulary-indexed;fixed-positive-f32;no-rng",
    "sampled": "argmax(probabilities/noise)",
    "greedy": "all-maxima-of-adjusted-logits-before-filtering",
    "eos": "last-id;optional-prefix-exclusion-before-upstream",
    "disabled_top_k": "nonpositive-maps-to-None",
}
INPUT_KEYS = {"name", "logits", "previous_tokens", "parameters", "noise"}
PARAMETER_KEYS = {"top_k", "top_p", "temperature", "repetition_penalty", "suppress_eos"}
EXPECTED_KEYS = {"adjusted_logits", "probabilities", "greedy_candidates", "sampled", "relative_margin"}
EXECUTION_KEYS = {"device", "dtype", "num_threads", "num_interop_threads",
                  "deterministic_algorithms", "inference_mode", "python", "numpy",
                  "torch", "system", "machine"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), f"required regular file is missing: {path}")
    return path


def canonical_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def content_digest(document):
    return hashlib.sha256(canonical_bytes(
        {key: value for key, value in document.items() if key != "content_sha256"})).hexdigest()


def case_catalog():
    """Public synthetic inputs only; all floating-point inputs round-trip as F32."""
    cases = []

    def add(name, logits, noise, *, history=(), top_k=0, top_p=1.0,
            temperature=1.0, repetition_penalty=1.0, suppress_eos=False):
        cases.append({
            "name": name, "logits": [float(np.float32(x)) for x in logits],
            "previous_tokens": list(history), "noise": [float(np.float32(x)) for x in noise],
            "parameters": {"top_k": top_k, "top_p": float(np.float32(top_p)),
                           "temperature": float(np.float32(temperature)),
                           "repetition_penalty": float(np.float32(repetition_penalty)),
                           "suppress_eos": suppress_eos},
        })

    logits = [6, 20, 1, 17, 9, 3, 14, 7, 19, 2, 12, 5, 18, 10, 4, 16, 8, 15, 11, 13]
    noise = [1.0] * len(logits)
    noise[15], noise[2] = 0.01, 0.00001
    add("default_policy", [x * 0.25 for x in logits], noise,
        history=[1, 1, 8, 12, 4], top_k=15, repetition_penalty=1.35)
    add("top_p_removes_crossing", [0, -1, -2], [1, 0.01, 1], top_p=0.8)
    add("top_p_low_keeps_leader", [0, -1, -2], [1, 0.01, 1], top_p=0.1)
    add("top_p_zero_keeps_leader", [0, -1, -2], [1, 0.01, 1], top_p=0.0)
    add("top_p_before_temperature", [1, 3, 2, 0], [1, 1, 0.01, 1],
        top_p=0.9, temperature=0.5)
    add("top_p_before_top_k", [1, 3, 2, 0], [1, 1, 0.01, 1], top_p=0.9, top_k=2)
    add("temperature_after_filtering", [0, 2, 1, -1], [1, 1, 0.2, 1],
        top_p=0.95, temperature=0.5)
    for name, history, penalty in (
        ("signed_penalty_duplicate_history", [0, 1, 3, 0, 1], 2.0),
        ("signed_penalty_unique_history", [0, 1, 3], 2.0),
        ("penalty_disabled", [0, 1, 3, 0, 1], 1.0),
    ):
        add(name, [-1, 3, -0.5, 0, 2.5], [1, 0.2, 0.1, 1, 1],
            history=history, repetition_penalty=penalty)
    for name, k in (("top_k_zero", 0), ("top_k_negative", -1),
                    ("top_k_at_vocab", 4), ("top_k_above_vocab", 5)):
        add(name, [1, 3, 0, 2], [1, 1, 0.0001, 1], top_k=k)
    add("top_k_pivot_ties", [2, 3, 0, 2], [0.01, 1, 1, 0.02], top_k=2)
    add("top_k_one_leading_ties", [2, 0, 2, 1], [1, 1, 0.01, 1], top_k=1)
    for suppress in (False, True):
        add("eos_prefix_suppressed" if suppress else "eos_unmasked",
            [1, 0, 3], [1, 0.01, 1], top_p=0.8, suppress_eos=suppress)
    add("single_token_vocabulary", [0], [0.5], top_k=15)
    return cases


def exact_keys(value, keys, name):
    require(isinstance(value, dict) and set(value) == keys, f"invalid {name} fields")


def integer(value, name, low, high):
    require(type(value) is int and low <= value <= high, f"invalid {name}")
    return value


def f32_number(value, name):
    require(type(value) in (int, float), f"{name} must be a number, not a boolean")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.float32(value)
    require(np.isfinite(rounded) and float(rounded) == value, f"{name} must be finite F32")
    return rounded


def f32_vector(value, name, length=None):
    require(isinstance(value, list) and 0 < len(value) <= 64, f"invalid {name} length")
    if length is not None:
        require(len(value) == length, f"invalid {name} length")
    return np.asarray([f32_number(x, name) for x in value], dtype=np.float32)


def greedy_candidates(adjusted):
    return np.flatnonzero(adjusted == adjusted.max()).tolist()


def replay(probabilities, noise):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = probabilities / noise
    require(np.isfinite(scores).all(), "non-finite replay scores")
    order = np.sort(scores)
    first = order[-1]
    second = order[-2] if len(order) > 1 else np.float32(0)
    require(first > 0, "replay must have a positive winning score")
    margin = float((first - second) / first)
    require(margin >= MIN_RELATIVE_MARGIN, "sampled winner has an ambiguous relative margin")
    return int(np.argmax(scores)), margin


def validate_case(case):
    exact_keys(case, INPUT_KEYS | {"expected"}, "case")
    require(isinstance(case["name"], str) and case["name"], "invalid case name")
    logits = f32_vector(case["logits"], "logits")
    vocab = len(logits)
    params = case["parameters"]
    exact_keys(params, PARAMETER_KEYS, "parameters")
    integer(params["top_k"], "top_k", -(2**31), 2**31 - 1)
    f32_number(params["top_p"], "top_p")
    f32_number(params["temperature"], "temperature")
    penalty = f32_number(params["repetition_penalty"], "repetition_penalty")
    with np.errstate(over="ignore", divide="ignore"):
        require(penalty > 0 and np.isfinite(np.float32(1) / penalty),
                "repetition_penalty and its reciprocal must be finite and positive")
    require(type(params["suppress_eos"]) is bool, "suppress_eos must be boolean")
    active = vocab - int(params["suppress_eos"])
    require(active > 0, "EOS suppression requires a nonempty prefix")
    history = case["previous_tokens"]
    require(isinstance(history, list), "previous_tokens must be a list")
    for token in history:
        integer(token, "previous token ID", 0, active - 1)
    noise = f32_vector(case["noise"], "noise", vocab)
    require(np.all(noise > 0), "noise must be positive")
    expected = case["expected"]
    exact_keys(expected, EXPECTED_KEYS, "expected")
    adjusted = f32_vector(expected["adjusted_logits"], "adjusted_logits", active)
    probabilities = f32_vector(expected["probabilities"], "probabilities", vocab)
    require(np.all(probabilities >= 0) and np.all(probabilities <= 1), "invalid probabilities")
    require(abs(float(probabilities.sum(dtype=np.float64)) - 1.0) <= 1e-6,
            "probabilities must sum to one")
    if params["suppress_eos"]:
        require(probabilities[-1] == 0, "suppressed EOS must have zero probability")
    candidates = expected["greedy_candidates"]
    require(isinstance(candidates, list) and candidates, "invalid greedy_candidates")
    for token in candidates:
        integer(token, "greedy token ID", 0, active - 1)
    require(candidates == greedy_candidates(adjusted), "incorrect greedy_candidates")
    require(np.all(probabilities[candidates] > 0), "greedy candidates must remain eligible")
    sampled = integer(expected["sampled"], "sampled token ID", 0, active - 1)
    selected, margin = replay(probabilities, noise)
    require(sampled == selected, "incorrect sampled token")
    recorded_margin = f32_number(expected["relative_margin"], "relative_margin")
    require(abs(float(recorded_margin) - margin) <= 1e-7, "incorrect relative_margin")


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_document(document):
    exact_keys(document, {"schema", "status", "semantics", "sources", "execution",
                          "cases", "content_sha256"}, "reference")
    require(document["schema"] == SCHEMA and document["status"] == "complete", "invalid schema/status")
    require(document["semantics"] == SEMANTICS, "invalid semantics")
    sources = document["sources"]
    exact_keys(sources, {"exporter_sha256", "upstream"}, "sources")
    require(valid_digest(sources["exporter_sha256"]), "invalid exporter SHA-256")
    exact_keys(sources["upstream"], {UPSTREAM_SOURCE}, "upstream sources")
    require(valid_digest(sources["upstream"][UPSTREAM_SOURCE]), "invalid upstream SHA-256")
    execution = document["execution"]
    exact_keys(execution, EXECUTION_KEYS, "execution")
    require(execution["device"] == "cpu" and execution["dtype"] == "float32", "CPU/F32 is required")
    for key in ("num_threads", "num_interop_threads"):
        integer(execution[key], key, 1, 1)
    for key in ("deterministic_algorithms", "inference_mode"):
        require(execution[key] is True, f"{key} must be true")
    for key in ("python", "numpy", "torch", "system", "machine"):
        require(isinstance(execution[key], str) and 0 < len(execution[key]) <= 200,
                f"invalid execution.{key}")
    cases = document["cases"]
    catalog = case_catalog()
    require(isinstance(cases, list) and len(cases) == len(catalog), "incorrect case count")
    names = set()
    for case, inputs in zip(cases, catalog):
        validate_case(case)
        require(case["name"] not in names, "duplicate case name")
        names.add(case["name"])
        require({key: case[key] for key in INPUT_KEYS} == inputs, "case inputs differ from catalog")
    require(valid_digest(document["content_sha256"]) and
            document["content_sha256"] == content_digest(document), "content SHA-256 mismatch")
    return document


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def validate_reference(path):
    document = json.loads(regular_file(path).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
    return validate_document(document)


def publish_reference(output, document):
    output = Path(output).absolute()
    require(not os.path.lexists(output), f"output already exists; refusing to overwrite: {output}")
    validate_document(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        stage = Path(directory) / "reference.json"
        stage.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
        validate_reference(stage)
        # A hard link publishes the complete file without replacing a raced path.
        os.link(stage, output)
    return document


def load_upstream(root):
    root = Path(root).resolve()
    if (root / "GPT_SoVITS" / "AR").is_dir():
        root = root / "GPT_SoVITS"
    path = regular_file(root / UPSTREAM_SOURCE)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    module = types.ModuleType("t2s_sampler_upstream")
    module.__file__ = str(path)
    # Compile the hashed bytes directly: no package import, sys.path change or pyc.
    exec(compile(source, str(path), "exec"), module.__dict__)
    require(callable(getattr(module, "logits_to_probs", None)), "upstream logits_to_probs is missing")
    require(sha256(path) == digest, "upstream source changed while importing")
    return module.logits_to_probs, path, digest


def configure_torch():
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return {"device": "cpu", "dtype": "float32", "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": True, "inference_mode": True,
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "system": platform.system(), "machine": platform.machine()}


def run_cases(logits_to_probs):
    import torch

    cases = case_catalog()
    with torch.inference_mode():
        for case in cases:
            params = case["parameters"]
            active = len(case["logits"]) - int(params["suppress_eos"])
            raw = torch.tensor([case["logits"]], dtype=torch.float32, device="cpu")
            working = raw[:, :active].clone()
            history = torch.tensor([case["previous_tokens"]], dtype=torch.int64, device="cpu")
            probabilities = logits_to_probs(
                working, previous_tokens=history,
                top_k=params["top_k"] if params["top_k"] > 0 else None,
                top_p=params["top_p"], temperature=params["temperature"],
                repetition_penalty=params["repetition_penalty"])
            require(probabilities.dtype == torch.float32 and probabilities.device.type == "cpu"
                    and tuple(probabilities.shape) == (1, active), "invalid upstream probability tensor")
            # Upstream mutates working for repetition penalty only; filtering and
            # temperature return new tensors. Keep the effective-vocabulary logits.
            adjusted = working[0].tolist()
            full_probs = probabilities[0].tolist() + ([0.0] if params["suppress_eos"] else [])
            scores = torch.tensor(full_probs, dtype=torch.float32, device="cpu") / torch.tensor(
                case["noise"], dtype=torch.float32, device="cpu")
            ranked = scores.sort(descending=True).values
            second = ranked[1] if len(ranked) > 1 else torch.zeros_like(ranked[0])
            case["expected"] = {
                "adjusted_logits": adjusted, "probabilities": full_probs,
                "greedy_candidates": greedy_candidates(np.asarray(adjusted, dtype=np.float32)),
                "sampled": int(scores.argmax()),
                "relative_margin": float((ranked[0] - second) / ranked[0]),
            }
            validate_case(case)
    return cases


def export_reference(upstream_root, output):
    require(not os.path.lexists(output), f"output already exists; refusing to overwrite: {output}")
    exporter_hash = sha256(__file__)
    execution = configure_torch()
    upstream, path, upstream_hash = load_upstream(upstream_root)
    cases = run_cases(upstream)
    require(sha256(path) == upstream_hash, "upstream source changed during export")
    require(sha256(__file__) == exporter_hash, "exporter changed during export")
    document = {"schema": SCHEMA, "status": "complete", "semantics": dict(SEMANTICS),
                "sources": {"exporter_sha256": exporter_hash,
                            "upstream": {UPSTREAM_SOURCE: upstream_hash}},
                "execution": execution, "cases": cases}
    document["content_sha256"] = content_digest(document)
    return publish_reference(output, document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="FILE", help="validate a fixture without PyTorch or upstream")
    parser.add_argument("--upstream-root", help="trusted GPT-SoVITS checkout; executes AR/models/utils.py")
    parser.add_argument("--output", help="new output JSON file; existing paths are never overwritten")
    args = parser.parse_args(argv)
    if args.validate:
        if args.upstream_root is not None or args.output is not None:
            parser.error("--validate cannot be combined with export arguments")
    elif args.upstream_root is None or args.output is None:
        parser.error("export requires --upstream-root and --output")
    try:
        document = (validate_reference(args.validate) if args.validate
                    else export_reference(args.upstream_root, args.output))
        print(f"Validated {SCHEMA}: {len(document['cases'])} cases")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, OverflowError) as exc:
        print(f"t2s sampler reference: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
