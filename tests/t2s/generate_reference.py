#!/usr/bin/env python3
"""Export local upstream CPU/F32 T2S teacher-forced decode references.

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


def load_inputs(directory, steps, hparams):
    validate_hparams(hparams)
    positive_int(steps, "steps")
    require(steps < CONTEXT_LIMIT, "steps exceeds the 1024-position context budget")
    directory = Path(directory)
    phones = token_vector(load_npy(directory / INPUT_FILES[0]), "phones")
    bert = load_npy(directory / INPUT_FILES[1])
    if bert.ndim == 3 and bert.shape[0] == 1:
        bert = bert[0]
    require(bert.dtype.kind == "f" and bert.dtype.itemsize == 4, "BERT must have float32 dtype")
    require(bert.shape == (1024, phones.size), "BERT must have shape [1024,T] or [1,1024,T]")
    require(np.isfinite(bert).all(), "BERT contains non-finite values")
    prompt = token_vector(load_npy(directory / INPUT_FILES[2]), "prompt")
    codes = token_vector(load_npy(directory / INPUT_FILES[3]), "generated_codes", generated=True)
    require(codes.size >= steps, "generated_codes contains fewer tokens than --steps")
    require(np.all(phones < hparams["phoneme_vocab_size"]), "phones exceed the phoneme vocabulary")
    for name, tokens in (("prompt", prompt), ("generated_codes", codes)):
        require(np.all(tokens < hparams["EOS"]), f"{name} must contain semantic IDs below EOS")
    require(phones.size + prompt.size + steps <= CONTEXT_LIMIT,
            "prefill + teacher-forced steps exceeds the 1024-position context budget")
    return {
        "phones": phones,
        "bert": np.ascontiguousarray(bert.T, dtype="<f4"),
        "prompt": prompt,
        "forced_tokens": codes[:steps].copy(),
    }


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


def validate_provenance(sources):
    require(isinstance(sources, dict) and set(sources) == {
        "checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256", "inputs", "upstream",
    }, "unexpected sources fields")
    for name in ("checkpoint_sha256", "exporter_sha256", "checkpoint_loader_sha256"):
        validate_digest(sources.get(name), f"sources.{name}")
    require(isinstance(sources.get("inputs"), dict) and set(sources["inputs"]) == set(INPUT_FILES),
            "sources.inputs must list the four original input files")
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


def write_reference(output_dir, arrays, hparams, provenance, execution):
    output = Path(output_dir).absolute()
    require(not os.path.lexists(output), f"output already exists; refusing to overwrite: {output}")
    validate_hparams(hparams)
    dimensions = {"text": len(arrays["phones"]), "prompt": len(arrays["prompt"]),
                  "steps": len(arrays["forced_tokens"])}
    dimensions["prefill"] = dimensions["text"] + dimensions["prompt"]
    dimensions["final"] = dimensions["prefill"] + dimensions["steps"]
    specs = array_specs(hparams, dimensions)
    require(set(arrays) == set(specs), "unexpected array keys")
    manifest = {
        "schema": SCHEMA, "status": "complete", "hparams": hparams, "dimensions": dimensions,
        "semantics": "raw-logits; no-sampling; no-EOS-suppression; no-early-stop",
        "sources": provenance, "execution": execution, "arrays": {},
    }
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
        validate_reference(stage)
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


def export_reference(checkpoint, upstream_root, inputs_dir, output_dir, steps=8):
    require(not os.path.lexists(output_dir), f"output already exists; refusing to overwrite: {output_dir}")
    positive_int(steps, "steps")
    checkpoint = regular_file(checkpoint)
    inputs_dir = Path(inputs_dir)
    loader_path = REPO_ROOT / "python/torch_ckpt_utils.py"
    exporter_hash, loader_hash = sha256(__file__), sha256(loader_path)
    checkpoint_hash = sha256(checkpoint)
    input_hashes = {name: sha256(regular_file(inputs_dir / name)) for name in INPUT_FILES}
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
    inputs = load_inputs(inputs_dir, steps, hparams)
    require(weights and all(name.startswith("model.") for name in weights), "expected model.* checkpoint weights")
    execution = configure_torch()
    import torch

    module, root, upstream = upstream_module(upstream_root)
    with torch.device("cpu"):
        model = module.Text2SemanticDecoder(config).cpu().float().eval()
    state = {name[len("model."):]: torch.from_numpy(value) for name, value in weights.items()}
    model.load_state_dict(state, strict=True)
    del state, weights
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
    return write_reference(output_dir, arrays, hparams, provenance, execution)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="DIRECTORY", help="validate an existing export without PyTorch")
    parser.add_argument("--checkpoint", help="local T2S checkpoint")
    parser.add_argument("--upstream-root", help="trusted local GPT-SoVITS checkout (imports and executes its Python code)")
    parser.add_argument("--inputs-dir", help="directory containing four precomputed inference-dump .npy inputs")
    parser.add_argument("--output-dir", help="new output directory; existing paths are never overwritten")
    parser.add_argument("--steps", type=int, default=None, help="teacher-forced decode inputs (default: 8)")
    args = parser.parse_args(argv)
    exporting = (args.checkpoint, args.upstream_root, args.inputs_dir, args.output_dir, args.steps)
    if args.validate:
        if any(value is not None for value in exporting):
            parser.error("--validate cannot be combined with export arguments")
    elif not all(value is not None for value in exporting[:4]):
        parser.error("export requires --checkpoint, --upstream-root, --inputs-dir and --output-dir")
    try:
        if args.validate:
            manifest = validate_reference(args.validate)
        else:
            manifest = export_reference(args.checkpoint, args.upstream_root, args.inputs_dir,
                                        args.output_dir, 8 if args.steps is None else args.steps)
        d = manifest["dimensions"]
        print(f"Validated {SCHEMA}: prefill={d['prefill']}, decode_steps={d['steps']}, final={d['final']}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"t2s reference export: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
