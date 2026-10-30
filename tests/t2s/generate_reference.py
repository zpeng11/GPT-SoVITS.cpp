#!/usr/bin/env python3
"""Export local upstream CPU/F32 T2S decode or matched-policy greedy references.

No sampling, audio processing, downloads or production-runtime changes. Export
requires PyTorch and a trusted upstream GPT-SoVITS checkout and its dependencies;
--help and --validate need neither PyTorch nor the upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

SCHEMA = "gpt-sovits-t2s-decode-reference/v1"
GEN_SCHEMA = "gpt-sovits-t2s-generation-reference/v2"
GEN_SCHEMA_V1 = "gpt-sovits-t2s-generation-reference/v1"
EOS_POLICY = "suppressed-before-min-new-tokens"
EOS_POLICY_V1 = "unmasked-before-accept"
CONTEXT_LIMIT = 1024
INPUT_FILES = (
    "t2s_all_phones.npy", "t2s_all_bert.npy", "t2s_prompt.npy", "generated_codes.npy",
)
HPARAM_KEYS = (
    "embedding_dim", "hidden_dim", "head", "linear_units", "n_layer",
    "vocab_size", "phoneme_vocab_size", "EOS",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), f"required regular file is missing: {path}")
    return path


def positive_int(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def nonnegative_int32(value, name):
    require(type(value) is int and 0 <= value <= np.iinfo(np.int32).max,
            f"{name} must be a nonnegative int32")
    return value


def validate_hparams(hparams):
    require(isinstance(hparams, dict), "hparams must be an object")
    for key in HPARAM_KEYS:
        positive_int(hparams.get(key), f"hparams.{key}")
    hidden = hparams["hidden_dim"]
    require(hparams["embedding_dim"] == hidden and hidden % 2 == 0,
            "equal, even embedding_dim/hidden_dim are required")
    require(hidden % hparams["head"] == 0, "hidden_dim must be divisible by head")
    require(hparams["linear_units"] == 4 * hidden, "upstream requires linear_units == 4 * hidden_dim")
    require(hparams["EOS"] == hparams["vocab_size"] - 1, "EOS must be the last vocabulary entry")
    require(max(hparams.values()) <= np.iinfo(np.int32).max, "hparams exceed int32 range")


def load_npy(path, *, mmap_mode=None):
    array = np.load(regular_file(path), allow_pickle=False, mmap_mode=mmap_mode)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"expected a single .npy array: {path}")
    return array


def token_vector(array, name, *, generated=False):
    require(array.dtype.kind == "i" and array.dtype.itemsize in (4, 8),
            f"{name} must have int32 or int64 dtype")
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    elif generated and array.ndim == 3 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    require(array.ndim == 1 and array.size > 0, f"{name} must be a nonempty batch-one token vector")
    require(np.all((array >= 0) & (array <= np.iinfo(np.int32).max)), f"{name} contains invalid token IDs")
    return np.ascontiguousarray(array, dtype="<i4")


def text_inputs(directory, hparams):
    validate_hparams(hparams)
    directory = Path(directory)
    phones = token_vector(load_npy(directory / INPUT_FILES[0]), "phones")
    bert = load_npy(directory / INPUT_FILES[1])
    if bert.ndim == 3 and bert.shape[0] == 1:
        bert = bert[0]
    require(bert.dtype.kind == "f" and bert.dtype.itemsize == 4, "BERT must have float32 dtype")
    require(bert.shape == (1024, phones.size), "BERT must have shape [1024,T] or [1,1024,T]")
    require(np.isfinite(bert).all(), "BERT contains non-finite values")
    prompt = token_vector(load_npy(directory / INPUT_FILES[2]), "prompt")
    require(np.all(phones < hparams["phoneme_vocab_size"]), "phones exceed the phoneme vocabulary")
    require(np.all(prompt < hparams["EOS"]), "prompt must contain semantic IDs below EOS")
    return {"phones": phones, "bert": np.ascontiguousarray(bert.T, dtype="<f4"), "prompt": prompt}


def load_inputs(directory, steps, hparams):
    positive_int(steps, "steps")
    require(steps < CONTEXT_LIMIT, "steps exceeds the 1024-position context budget")
    inputs = text_inputs(directory, hparams)
    codes = token_vector(load_npy(Path(directory) / INPUT_FILES[3]), "generated_codes", generated=True)
    require(codes.size >= steps, "generated_codes contains fewer tokens than --steps")
    require(np.all(codes < hparams["EOS"]), "generated_codes must contain semantic IDs below EOS")
    require(len(inputs["phones"]) + len(inputs["prompt"]) + steps <= CONTEXT_LIMIT,
            "prefill + teacher-forced steps exceeds the 1024-position context budget")
    inputs["forced_tokens"] = codes[:steps].copy()
    return inputs


def generation_inputs(directory, reference_phones, max_new_tokens, hparams):
    positive_int(reference_phones, "reference_phones")
    positive_int(max_new_tokens, "max_new_tokens")
    inputs = text_inputs(directory, hparams)
    require(reference_phones < len(inputs["phones"]), "reference/target text split must leave nonempty target phones")
    require(len(inputs["phones"]) + len(inputs["prompt"]) + max_new_tokens - 1 <= CONTEXT_LIMIT,
            "generation exceeds the 1024-position context budget")
    return inputs


def run_teacher_forced(model, inputs):
    import torch

    def array(tensor):
        return np.array(tensor.detach().cpu().numpy(), dtype="<f4", order="C", copy=True)

    def caches(tensors):
        return array(torch.stack(tensors, dim=0).squeeze(1))

    phones = torch.from_numpy(inputs["phones"].astype(np.int64)).unsqueeze(0)
    bert = torch.from_numpy(inputs["bert"]).unsqueeze(0)
    prompt = torch.from_numpy(inputs["prompt"].astype(np.int64)).unsqueeze(0)
    forced = torch.from_numpy(inputs["forced_tokens"].astype(np.int64)).unsqueeze(0)
    text_len, prompt_len, steps = phones.shape[1], prompt.shape[1], forced.shape[1]
    prefill_len = text_len + prompt_len
    model.cpu().float().eval()
    with torch.inference_mode():
        text = model.ar_text_position(model.ar_text_embedding(phones) + model.bert_proj(bert))
        # Position the complete fixed history once; each decode input retains its
        # audio-only position, independent of the number of text tokens.
        audio = model.ar_audio_position(model.ar_audio_embedding(torch.cat((prompt, forced), dim=1)))
        prefill = torch.cat((text, audio[:, :prompt_len]), dim=1)
        mask = torch.zeros((prefill_len, prefill_len), dtype=torch.bool)
        mask[:text_len, text_len:] = True
        mask[text_len:, text_len:] = torch.triu(torch.ones((prompt_len, prompt_len), dtype=torch.bool), 1)
        hidden, k, v = model.t2s_transformer.process_prompt(prefill, mask[None, None], None, False)
        outputs = dict(inputs)
        outputs.update(prefill_input=array(prefill[0]), prefill_hidden=array(hidden[0]),
                       prefill_k=caches(k), prefill_v=caches(v))
        logits = [array(model.ar_predict_layer(hidden[:, -1])[0])]
        for step in range(steps):
            token = audio[:, prompt_len + step:prompt_len + step + 1]
            hidden, k, v = model.t2s_transformer.decode_next_token(token, k, v, None, False)
            logits.append(array(model.ar_predict_layer(hidden[:, -1])[0]))
        outputs.update(logits=np.stack(logits), final_k=caches(k), final_v=caches(v))
    outputs["argmax"] = outputs["logits"].argmax(axis=1).astype("<i4")
    outputs["context_lengths"] = np.arange(prefill_len, prefill_len + steps + 1, dtype="<i4")
    outputs["audio_positions"] = np.arange(prompt_len, prompt_len + steps, dtype="<i4")
    return outputs


def array_specs(hparams, dimensions):
    t, p, s = (dimensions[k] for k in ("text", "prompt", "steps"))
    n, h, layers, vocab = t + p, hparams["hidden_dim"], hparams["n_layer"], hparams["vocab_size"]
    return {
        "phones": ("int32", (t,)), "bert": ("float32", (t, 1024)),
        "prompt": ("int32", (p,)), "forced_tokens": ("int32", (s,)),
        "prefill_input": ("float32", (n, h)), "prefill_hidden": ("float32", (n, h)),
        "logits": ("float32", (s + 1, vocab)), "argmax": ("int32", (s + 1,)),
        "context_lengths": ("int32", (s + 1,)), "audio_positions": ("int32", (s,)),
        "prefill_k": ("float32", (layers, n, h)), "prefill_v": ("float32", (layers, n, h)),
        "final_k": ("float32", (layers, n + s, h)), "final_v": ("float32", (layers, n + s, h)),
    }


def validate_digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"{name} must be a SHA-256 digest")


def validate_provenance(sources, input_files=INPUT_FILES):
    require(isinstance(sources, dict) and set(sources) == {
        "checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256", "inputs", "upstream",
    }, "unexpected sources fields")
    for name in ("checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256"):
        validate_digest(sources.get(name), f"sources.{name}")
    require(isinstance(sources.get("inputs"), dict) and set(sources["inputs"]) == set(input_files),
            "sources.inputs must list the original input files")
    upstream = sources.get("upstream")
    require(isinstance(upstream, dict) and "AR/models/t2s_model.py" in upstream,
            "sources.upstream must include AR/models/t2s_model.py")
    for names in (sources["inputs"], upstream):
        for name, digest in names.items():
            require(isinstance(name, str) and "\\" not in name
                    and all(part not in ("", ".", "..") for part in name.split("/")),
                    "invalid source-relative filename")
            validate_digest(digest, f"sources.{name}")
    require(all(name.startswith("AR/") and name.endswith(".py") for name in upstream),
            "upstream sources must be AR-relative Python files")


def validate_execution(execution):
    require(isinstance(execution, dict), "execution must be an object")
    expected = {
        "device": "cpu", "dtype": "float32", "attention": "upstream-manual",
        "eval": True, "inference_mode": True, "deterministic_algorithms": True,
        "num_threads": 1, "num_interop_threads": 1, "initialization_seed": 0,
    }
    for key, value in expected.items():
        require(type(execution.get(key)) is type(value) and execution[key] == value,
                f"unsupported execution.{key}")
    versions = {"python", "numpy", "torch", "system", "machine"}
    require(set(execution) == set(expected) | versions, "unexpected execution fields")
    for key in versions:
        require(isinstance(execution.get(key), str) and bool(execution[key]) and len(execution[key]) < 256
                and not any(c in execution[key] for c in ("/", "\\", "\n", "\r")), f"invalid execution.{key}")


def validate_reference(directory):
    directory = Path(directory)
    require(not directory.is_symlink() and directory.is_dir(), "reference must be a regular directory")
    manifest_path = regular_file(directory / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "unexpected reference schema")
    require(manifest.get("status") == "complete", "reference is incomplete")
    hparams = manifest.get("hparams")
    validate_hparams(hparams)
    require(set(hparams) == set(HPARAM_KEYS), "unexpected hparams keys")
    require(set(manifest) == {"schema", "status", "hparams", "dimensions", "semantics", "sources", "execution", "arrays"},
            "unexpected manifest fields")
    dimensions = manifest.get("dimensions")
    require(isinstance(dimensions, dict) and set(dimensions) == {"text", "prompt", "steps", "prefill", "final"},
            "unexpected dimensions")
    for name, value in dimensions.items():
        positive_int(value, f"dimensions.{name}")
    t, p, s, n, final = (dimensions[k] for k in ("text", "prompt", "steps", "prefill", "final"))
    require(n == t + p and final == n + s and final <= CONTEXT_LIMIT, "inconsistent context dimensions")
    validate_provenance(manifest.get("sources"))
    validate_execution(manifest.get("execution"))
    require(manifest.get("semantics") == "raw-logits; no-sampling; no-EOS-suppression; no-early-stop",
            "unexpected reference semantics")
    specs = array_specs(hparams, dimensions)
    arrays = validate_arrays(directory, manifest, specs)
    require(np.all((arrays["phones"] >= 0) & (arrays["phones"] < hparams["phoneme_vocab_size"])),
            "phones exceed the phoneme vocabulary")
    for name in ("prompt", "forced_tokens"):
        require(np.all((arrays[name] >= 0) & (arrays[name] < hparams["EOS"])), f"invalid semantic IDs in {name}")
    require(np.array_equal(arrays["argmax"], arrays["logits"].argmax(axis=1)), "argmax disagrees with logits")
    require(np.array_equal(arrays["context_lengths"], np.arange(n, final + 1)), "invalid context_lengths")
    require(np.array_equal(arrays["audio_positions"], np.arange(p, p + s)), "invalid audio_positions")
    for cache in ("k", "v"):
        require(np.array_equal(arrays[f"prefill_{cache}"], arrays[f"final_{cache}"][:, :n]),
                f"final {cache} cache does not preserve prefill prefix")
    return manifest


def validate_arrays(directory, manifest, specs):
    entries = manifest.get("arrays")
    require(isinstance(entries, dict) and set(entries) == {name + ".npy" for name in specs},
            "unexpected array file list")
    require({path.name for path in directory.iterdir()} == set(entries) | {"manifest.json"},
            "unexpected or missing files in reference directory")
    arrays = {}
    for name, (dtype, shape) in specs.items():
        filename = name + ".npy"
        meta = entries[filename]
        require(isinstance(meta, dict) and meta.get("shape") == list(shape) and meta.get("dtype") == dtype
                and meta.get("ggml_ne") == list(reversed(shape)) and meta.get("order") == "C",
                f"invalid layout metadata for {filename}")
        validate_digest(meta.get("sha256"), filename)
        path = regular_file(directory / filename)
        require(sha256(path) == meta["sha256"], f"SHA-256 mismatch for {filename}")
        a = load_npy(path, mmap_mode="r")
        require(a.dtype == np.dtype("<f4" if dtype == "float32" else "<i4") and a.shape == shape
                and a.flags.c_contiguous, f"invalid dtype/shape/order for {filename}")
        require(np.isfinite(a).all(), f"non-finite values in {filename}")
        arrays[name] = a
    return arrays


def write_reference(output_dir, arrays, hparams, provenance, execution):
    output = Path(output_dir).absolute()
    require(not os.path.lexists(output), f"output already exists; refusing to overwrite: {output}")
    validate_hparams(hparams)
    dimensions = {"text": len(arrays["phones"]), "prompt": len(arrays["prompt"]),
                  "steps": len(arrays["forced_tokens"])}
    dimensions["prefill"] = dimensions["text"] + dimensions["prompt"]
    dimensions["final"] = dimensions["prefill"] + dimensions["steps"]
    specs = array_specs(hparams, dimensions)
    manifest = {
        "schema": SCHEMA, "status": "complete", "hparams": hparams, "dimensions": dimensions,
        "semantics": "raw-logits; no-sampling; no-EOS-suppression; no-early-stop",
        "sources": provenance, "execution": execution, "arrays": {},
    }
    return publish_reference(output, arrays, specs, manifest, validate_reference)


def publish_reference(output, arrays, specs, manifest, validate):
    output = Path(output).absolute()
    require(not os.path.lexists(output), f"output already exists; refusing to overwrite: {output}")
    require(set(arrays) == set(specs), "unexpected array keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp)
        for name in specs:
            filename = name + ".npy"
            a = arrays[name]
            path = stage / filename
            np.save(path, a, allow_pickle=False)
            manifest["arrays"][filename] = {
                "dtype": str(a.dtype), "shape": list(a.shape), "order": "C",
                "ggml_ne": list(reversed(a.shape)), "sha256": sha256(path),
            }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate(stage)
        # mkdir reserves the destination without replacing even an empty directory
        # created by another process. The completion marker is published last.
        output.mkdir()
        published = []
        try:
            for filename in [name + ".npy" for name in specs] + ["manifest.json"]:
                os.link(stage / filename, output / filename)
                published.append(output / filename)
        except BaseException:
            for path in reversed(published):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                output.rmdir()
            except OSError:
                pass
            raise
    return manifest


def generation_policy(max_new_tokens, repetition_penalty, min_new_tokens=11):
    positive_int(max_new_tokens, "max_new_tokens")
    nonnegative_int32(min_new_tokens, "min_new_tokens")
    require(type(repetition_penalty) in (int, float) and np.isfinite(repetition_penalty)
            and repetition_penalty > 0, "repetition_penalty must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        penalty = np.float32(repetition_penalty)
        reciprocal = np.float32(1.0) / penalty
    require(np.isfinite(penalty) and penalty > 0 and np.isfinite(reciprocal),
            "repetition_penalty and its reciprocal must be finite and positive in float32")
    return {"top_k": 1, "top_p": 1.0, "temperature": 1.0,
            "repetition_penalty": float(repetition_penalty), "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "history": "prompt+generated-binary", "eos": EOS_POLICY}


def adjusted_logits(logits, codes, repetition_penalty, prompt=()):
    """Penalize the binary history of the original prompt plus accepted codes."""
    adjusted = np.array(logits, dtype=np.float32, copy=True)
    history = [int(token) for token in prompt]
    history.extend(int(token) for token in codes)
    ids = np.asarray(sorted(set(history)), dtype=np.int64)
    values = adjusted[ids]
    penalty = np.float32(repetition_penalty)
    # C++ applies a precomputed reciprocal to nonnegative logits.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        adjusted[ids] = np.where(values < 0, values * penalty, values * np.float32(1.0 / penalty))
    require(np.isfinite(adjusted).all(), "non-finite repetition-adjusted logits")
    return adjusted


def run_greedy(model, inputs, reference_phones, max_new_tokens, repetition_penalty,
               min_new_tokens=11):
    import torch

    generation_policy(max_new_tokens, repetition_penalty, min_new_tokens)
    positive_int(reference_phones, "reference_phones")
    require(reference_phones < len(inputs["phones"]), "reference/target text split must leave nonempty target phones")
    t, p = len(inputs["phones"]), len(inputs["prompt"])
    require(t + p + max_new_tokens - 1 <= CONTEXT_LIMIT, "generation exceeds the 1024-position context budget")
    eos = int(model.EOS)
    model.cpu().float().eval()
    phones = torch.from_numpy(inputs["phones"].astype(np.int64))[None]
    bert = torch.from_numpy(inputs["bert"])[None]
    prompt = torch.from_numpy(inputs["prompt"].astype(np.int64))[None]
    codes, rows, selected = [], [], []
    reason = "max_new_tokens"
    with torch.inference_mode():
        text = model.ar_text_position(model.ar_text_embedding(phones) + model.bert_proj(bert))
        audio = model.ar_audio_position(model.ar_audio_embedding(prompt))
        prefill = torch.cat((text, audio), dim=1)
        mask = torch.zeros((t + p, t + p), dtype=torch.bool)
        mask[:t, t:] = True
        mask[t:, t:] = torch.triu(torch.ones((p, p), dtype=torch.bool), 1)
        hidden, k, v = model.t2s_transformer.process_prompt(prefill, mask[None, None], None, False)
        for step in range(max_new_tokens):
            logits = model.ar_predict_layer(hidden[:, -1])[0].cpu().numpy().copy()
            require(np.isfinite(logits).all(), f"non-finite generation logits at step {step}")
            scores = adjusted_logits(logits, codes, repetition_penalty, inputs["prompt"])
            # The last vocabulary entry is EOS; suppress it inside the window.
            token = int(scores[:eos].argmax() if step < min_new_tokens else scores.argmax())
            rows.append(logits)
            selected.append(token)
            if token == eos:
                reason = "eos"
                break
            codes.append(token)
            if len(codes) == max_new_tokens:
                break
            history = torch.tensor([list(inputs["prompt"]) + codes], dtype=torch.int64)
            positioned = model.ar_audio_position(model.ar_audio_embedding(history))[:, -1:]
            hidden, k, v = model.t2s_transformer.decode_next_token(positioned, k, v, None, False)
        result = dict(inputs)
        result.update(prefill_input=np.array(prefill[0].cpu().numpy(), dtype="<f4", order="C", copy=True),
                      codes=np.asarray(codes, dtype="<i4"), logits=np.asarray(rows, dtype="<f4"),
                      selected=np.asarray(selected, dtype="<i4"),
                      context_lengths=np.arange(t + p, t + p + len(rows), dtype="<i4"))
    return result, reason


def generation_specs(hparams, dimensions):
    t, p, n, c, d = (dimensions[k] for k in ("text", "prompt", "prefill", "codes", "decisions"))
    return {"phones": ("int32", (t,)), "bert": ("float32", (t, 1024)),
            "prompt": ("int32", (p,)), "prefill_input": ("float32", (n, hparams["hidden_dim"])),
            "codes": ("int32", (c,)), "logits": ("float32", (d, hparams["vocab_size"])),
            "selected": ("int32", (d,)), "context_lengths": ("int32", (d,))}


def validate_generation_reference(directory):
    directory = Path(directory)
    require(not directory.is_symlink() and directory.is_dir(), "reference must be a regular directory")
    manifest = json.loads(regular_file(directory / "manifest.json").read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and set(manifest) == {
        "schema", "status", "hparams", "dimensions", "policy", "stop_reason", "sources", "execution", "arrays",
    }, "unexpected generation manifest fields")
    require(manifest["schema"] == GEN_SCHEMA and manifest["status"] == "complete", "unexpected generation schema/status")
    hp, dims, policy = (manifest[k] for k in ("hparams", "dimensions", "policy"))
    validate_hparams(hp)
    require(set(hp) == set(HPARAM_KEYS), "unexpected hparams keys")
    require(isinstance(dims, dict) and set(dims) == {
        "text", "reference_text", "target_text", "prompt", "prefill", "codes", "decisions",
    }, "unexpected generation dimensions")
    for key, value in dims.items():
        require(type(value) is int and 0 <= value <= CONTEXT_LIMIT and (key == "codes" or value > 0),
                f"invalid dimensions.{key}")
    t, r, target, p, n, c, d = (dims[k] for k in ("text", "reference_text", "target_text", "prompt", "prefill", "codes", "decisions"))
    require(t == r + target and n == t + p, "inconsistent text split/prefill dimensions")
    require(isinstance(policy, dict) and set(policy) == {
        "top_k", "top_p", "temperature", "repetition_penalty", "max_new_tokens", "min_new_tokens",
        "history", "eos",
    }, "unexpected generation policy fields")
    require(policy["eos"] != EOS_POLICY_V1,
            "legacy unmasked-before-accept generation references are unsupported; re-export")
    minimum = policy["min_new_tokens"]
    require(type(minimum) is int and 0 <= minimum <= np.iinfo(np.int32).max,
            "generation policy min_new_tokens must be a nonnegative int32")
    expected = generation_policy(policy["max_new_tokens"], policy["repetition_penalty"], minimum)
    require(type(policy["top_k"]) is int and all(type(policy[k]) in (int, float) for k in ("top_p", "temperature"))
            and policy == expected, "unsupported generation policy")
    maximum = policy["max_new_tokens"]
    require(n + maximum - 1 <= CONTEXT_LIMIT, "generation exceeds the 1024-position context budget")
    reason = manifest["stop_reason"]
    if reason == "eos":
        require(d == c + 1 and d <= maximum, "inconsistent stop reason/codes/decisions")
        require(d - 1 >= minimum, "EOS stop must occur after the min_new_tokens suppression window")
    elif reason == "max_new_tokens":
        require(c == d == maximum, "inconsistent stop reason/codes/decisions")
    else:
        require(False, "unexpected generation stop reason")
    validate_provenance(manifest["sources"], INPUT_FILES[:3])
    validate_execution(manifest["execution"])
    arrays = validate_arrays(directory, manifest, generation_specs(hp, dims))
    require(np.all((arrays["phones"] >= 0) & (arrays["phones"] < hp["phoneme_vocab_size"])), "invalid phones")
    for name in ("prompt", "codes"):
        require(np.all((arrays[name] >= 0) & (arrays[name] < hp["EOS"])), f"invalid semantic IDs in {name}")
    require(np.array_equal(arrays["context_lengths"], np.arange(n, n + d)), "invalid context_lengths")
    require(np.array_equal(arrays["selected"][:c], arrays["codes"]), "selected candidates disagree with codes")
    if reason == "eos":
        require(arrays["selected"][-1] == hp["EOS"], "EOS stop must include an unaccepted EOS decision")
    for step, row in enumerate(arrays["logits"]):
        scores = adjusted_logits(row, arrays["codes"][:step], policy["repetition_penalty"],
                                 arrays["prompt"])
        expected_token = int(scores[:hp["EOS"]].argmax() if step < minimum else scores.argmax())
        require(expected_token == arrays["selected"][step], f"selected token disagrees with policy at step {step}")
    return manifest


def write_generation_reference(output_dir, arrays, hparams, provenance, execution,
                               reference_phones, max_new_tokens, repetition_penalty, stop_reason,
                               min_new_tokens=11):
    policy = generation_policy(max_new_tokens, repetition_penalty, min_new_tokens)
    dimensions = {"text": len(arrays["phones"]), "reference_text": reference_phones,
                  "target_text": len(arrays["phones"]) - reference_phones, "prompt": len(arrays["prompt"]),
                  "prefill": len(arrays["phones"]) + len(arrays["prompt"]),
                  "codes": len(arrays["codes"]), "decisions": len(arrays["selected"])}
    manifest = {"schema": GEN_SCHEMA, "status": "complete", "hparams": hparams, "dimensions": dimensions,
                "policy": policy, "stop_reason": stop_reason, "sources": provenance, "execution": execution, "arrays": {}}
    return publish_reference(output_dir, arrays, generation_specs(hparams, dimensions), manifest, validate_generation_reference)


class SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Timestamp-based pyc validation can accept stale, same-size source edits.
        # Provenance must describe the source actually executed, not cached code.
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class UpstreamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "AR" and not fullname.startswith("AR."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        require(spec is not None, f"upstream module is missing: {fullname}")
        if spec.loader is not None:
            require(spec.origin and spec.origin.endswith(".py"), f"upstream module must be Python source: {fullname}")
            spec.loader = SourceOnlyLoader(fullname, spec.origin)
        return spec


def upstream_module(root):
    root = Path(root).resolve()
    if (root / "GPT_SoVITS" / "AR").is_dir():
        root = root / "GPT_SoVITS"
    regular_file(root / "AR/models/t2s_model.py")
    before = {path.resolve(): sha256(path) for path in (root / "AR").rglob("*.py")}
    require(not any(name == "AR" or name.startswith("AR.") for name in sys.modules),
            "AR is already imported; run the exporter in a fresh process")
    sys.path.insert(0, str(root))
    finder = UpstreamFinder()
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module("AR.models.t2s_model")
    except ImportError as exc:
        raise ValueError(f"cannot import upstream T2S; use an environment with its dependencies: {exc}") from exc
    finally:
        sys.meta_path.remove(finder)
        sys.path.pop(0)
    sources = {}
    for name, imported in list(sys.modules.items()):
        if name == "AR" or name.startswith("AR."):
            source = getattr(imported, "__file__", None)
            if source and source.endswith(".py"):
                path = Path(source).resolve()
                require(path.is_relative_to(root), f"upstream module {name} came from outside --upstream-root")
                digest = sha256(path)
                require(before.get(path) == digest, f"upstream source changed while importing {name}")
                sources[path.relative_to(root).as_posix()] = digest
    return module, root, sources


def configure_torch():
    import torch

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    return {
        "device": "cpu", "dtype": "float32", "attention": "upstream-manual", "eval": True,
        "inference_mode": True, "deterministic_algorithms": True,
        "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
        "initialization_seed": 0, "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "system": platform.system(), "machine": platform.machine(),
    }


def export_reference(checkpoint, upstream_root, inputs_dir, output_dir, steps=8, *,
                     generation=False, reference_phones=None, max_new_tokens=200, repetition_penalty=1.0,
                     min_new_tokens=11):
    require(not os.path.lexists(output_dir), f"output already exists; refusing to overwrite: {output_dir}")
    if generation:
        positive_int(reference_phones, "reference_phones")
        generation_policy(max_new_tokens, repetition_penalty, min_new_tokens)
    else:
        positive_int(steps, "steps")
    input_files = INPUT_FILES[:3] if generation else INPUT_FILES
    checkpoint = regular_file(checkpoint)
    inputs_dir = Path(inputs_dir)
    loader_path = REPO_ROOT / "python/torch_ckpt_utils.py"
    exporter_hash, loader_hash = sha256(__file__), sha256(loader_path)
    checkpoint_hash = sha256(checkpoint)
    input_hashes = {name: sha256(regular_file(inputs_dir / name)) for name in input_files}
    loader_name = "t2s_reference_checkpoint_loader"
    spec = importlib.util.spec_from_file_location(loader_name, loader_path,
                                                loader=SourceOnlyLoader(loader_name, str(loader_path)))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    weights, meta = loader.load_checkpoint(str(checkpoint))
    config = meta.get("config", {})
    require(isinstance(config, dict) and isinstance(config.get("model"), dict), "checkpoint is missing model config")
    hparams = {key: config["model"].get(key) for key in HPARAM_KEYS}
    validate_hparams(hparams)
    inputs = (generation_inputs(inputs_dir, reference_phones, max_new_tokens, hparams) if generation
              else load_inputs(inputs_dir, steps, hparams))
    require(weights and all(name.startswith("model.") for name in weights), "expected model.* checkpoint weights")
    execution = configure_torch()
    import torch

    module, root, upstream = upstream_module(upstream_root)
    with torch.device("cpu"):
        model = module.Text2SemanticDecoder(config).cpu().float().eval()
    state = {name[len("model."):]: torch.from_numpy(value) for name, value in weights.items()}
    model.load_state_dict(state, strict=True)
    del state, weights
    if generation:
        arrays, reason = run_greedy(model, inputs, reference_phones, max_new_tokens,
                                    repetition_penalty, min_new_tokens)
    else:
        arrays = run_teacher_forced(model, inputs)
    require(sha256(checkpoint) == checkpoint_hash, "checkpoint changed during export")
    require(all(sha256(inputs_dir / name) == digest for name, digest in input_hashes.items()),
            "inputs changed during export")
    require(all(sha256(root / name) == digest for name, digest in upstream.items()),
            "upstream source changed during export")
    require(sha256(__file__) == exporter_hash and sha256(loader_path) == loader_hash,
            "exporter or checkpoint loader changed during export")
    provenance = {
        "checkpoint_sha256": checkpoint_hash, "inputs": input_hashes, "upstream": upstream,
        "exporter_sha256": exporter_hash, "checkpoint_loader_sha256": loader_hash,
    }
    if generation:
        return write_generation_reference(output_dir, arrays, hparams, provenance, execution,
                                          reference_phones, max_new_tokens, repetition_penalty,
                                          reason, min_new_tokens)
    return write_reference(output_dir, arrays, hparams, provenance, execution)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="DIRECTORY", help="validate an existing export without PyTorch")
    parser.add_argument("--checkpoint", help="local T2S checkpoint")
    parser.add_argument("--upstream-root", help="trusted local GPT-SoVITS checkout (imports and executes its Python code)")
    parser.add_argument("--inputs-dir", help="directory containing four precomputed inference-dump .npy inputs")
    parser.add_argument("--output-dir", help="new output directory; existing paths are never overwritten")
    parser.add_argument("--steps", type=int, default=None, help="teacher-forced decode inputs (default: 8)")
    parser.add_argument("--generate", action="store_true", help="export closed-loop top-k 1 under the C++ generation policy")
    parser.add_argument("--reference-phones", type=int, help="reference phone count in concatenated all-phones input")
    parser.add_argument("--max-new-tokens", type=int, help="generation budget (default: 200)")
    parser.add_argument("--repetition-penalty", type=float, help="prompt+generated repetition penalty (default: 1.0)")
    parser.add_argument("--min-new-tokens", type=int,
                        help="EOS suppression window in predictions; 0 disables it (default: 11)")
    args = parser.parse_args(argv)
    exporting = (args.checkpoint, args.upstream_root, args.inputs_dir, args.output_dir, args.steps,
                 args.reference_phones, args.max_new_tokens, args.repetition_penalty,
                 args.min_new_tokens)
    if args.validate:
        if args.generate or any(value is not None for value in exporting):
            parser.error("--validate cannot be combined with export arguments")
    elif not all(value is not None for value in exporting[:4]):
        parser.error("export requires --checkpoint, --upstream-root, --inputs-dir and --output-dir")
    elif args.generate:
        if args.steps is not None or args.reference_phones is None:
            parser.error("--generate requires --reference-phones and cannot use --steps")
    elif any(value is not None for value in exporting[5:]):
        parser.error("generation policy arguments require --generate")
    try:
        if args.validate:
            path = regular_file(Path(args.validate) / "manifest.json")
            header = json.loads(path.read_text(encoding="utf-8"))
            require(isinstance(header, dict), "reference manifest must be an object")
            schema = header.get("schema")
            require(schema != GEN_SCHEMA_V1,
                    "generation reference schema v1 used unmasked EOS; re-export it with this exporter")
            manifest = (validate_generation_reference(args.validate) if schema == GEN_SCHEMA
                        else validate_reference(args.validate))
        else:
            manifest = export_reference(args.checkpoint, args.upstream_root, args.inputs_dir,
                                        args.output_dir, 8 if args.steps is None else args.steps,
                                        generation=args.generate, reference_phones=args.reference_phones,
                                        max_new_tokens=200 if args.max_new_tokens is None else args.max_new_tokens,
                                        repetition_penalty=1.0 if args.repetition_penalty is None else args.repetition_penalty,
                                        min_new_tokens=11 if args.min_new_tokens is None else args.min_new_tokens)
        d = manifest["dimensions"]
        if manifest["schema"] == GEN_SCHEMA:
            print(f"Validated {GEN_SCHEMA}: codes={d['codes']}, decisions={d['decisions']}, stop={manifest['stop_reason']}")
        else:
            print(f"Validated {SCHEMA}: prefill={d['prefill']}, decode_steps={d['steps']}, final={d['final']}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"t2s reference export: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
