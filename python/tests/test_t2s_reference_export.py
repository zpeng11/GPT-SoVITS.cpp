"""Tests for tests/t2s/generate_reference.py.

Torch-free tests cover input normalization, the strict manifest/validator
contract, atomic writes and lazy import paths. Optional PyTorch tests use a
tiny T2S test double (not a real GPT-SoVITS oracle) for full-sequence parity
and CLI export plumbing, including a temporary tiny upstream AR package.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tests/t2s/generate_reference.py"
LOADER = REPO_ROOT / "python/torch_ckpt_utils.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("t2s_export_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()

HPARAMS = {"embedding_dim": 8, "hidden_dim": 8, "head": 2, "linear_units": 32,
           "n_layer": 2, "vocab_size": 12, "phoneme_vocab_size": 8, "EOS": 11}
SEMANTICS = "raw-logits; no-sampling; no-EOS-suppression; no-early-stop"
EXECUTION = {"device": "cpu", "dtype": "float32", "attention": "upstream-manual",
             "eval": True, "inference_mode": True, "deterministic_algorithms": True,
             "num_threads": 1, "num_interop_threads": 1, "initialization_seed": 0,
             "python": platform.python_version(), "numpy": np.__version__,
             "torch": "2.11.0", "system": platform.system(), "machine": platform.machine()}
EXPECTED = {"phones", "bert", "prompt", "forced_tokens", "prefill_input", "prefill_hidden",
            "logits", "argmax", "context_lengths", "audio_positions",
            "prefill_k", "prefill_v", "final_k", "final_v"}
TINY_SOURCE = r'''"""Tiny T2S test double (not real GPT-SoVITS)."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Pos(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.x_scale = math.sqrt(float(dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        p = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        d = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        t = torch.zeros(max_len, dim)
        t[:, 0::2] = torch.sin(p * d)
        t[:, 1::2] = torch.cos(p * d)
        self.register_buffer("pe", t)
    def forward(self, x):
        return x * self.x_scale + self.alpha * self.pe[:x.shape[1]].unsqueeze(0)

class Block(nn.Module):
    def __init__(self, h, heads, ff):
        super().__init__()
        self.heads = heads
        self.head_dim = h // heads
        self.in_proj = nn.Linear(h, 3 * h)
        self.out_proj = nn.Linear(h, h)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.linear1 = nn.Linear(h, ff)
        self.linear2 = nn.Linear(ff, h)
    def _qkv(self, x):
        b, l, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, -1)
        f = lambda t: t.view(b, l, self.heads, self.head_dim).transpose(1, 2)
        return f(q), f(k), f(v)
    def _att(self, q, k, v, mask):
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            s = s.masked_fill(mask, float("-inf"))
        return torch.matmul(torch.softmax(s, -1), v)
    def _res(self, x, y):
        b, l, h = x.shape
        y = y.transpose(1, 2).reshape(b, l, h)
        x = self.norm1(x + self.out_proj(y))
        return self.norm2(x + self.linear2(F.gelu(self.linear1(x))))
    def prefill(self, x, mask):
        b, l, _ = x.shape
        q, k, v = self._qkv(x)
        if mask is not None:
            mask = mask.expand(b, self.heads, l, l)
        y = self._att(q, k, v, mask)
        return (self._res(x, y), k.transpose(1, 2).reshape(b, l, -1),
                v.transpose(1, 2).reshape(b, l, -1))
    def decode(self, x, kc, vc, mask):
        b, _, _ = x.shape
        q, k, v = self._qkv(x)
        kc = torch.cat((kc, k.transpose(1, 2).reshape(b, 1, -1)), 1)
        vc = torch.cat((vc, v.transpose(1, 2).reshape(b, 1, -1)), 1)
        n = kc.shape[1]
        kk = kc.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        vv = vc.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.expand(b, self.heads, 1, n)
        return self._res(x, self._att(q, kk, vv, mask)), kc, vc

class Transformer(nn.Module):
    def __init__(self, layers, h, heads, ff):
        super().__init__()
        self.blocks = nn.ModuleList([Block(h, heads, ff) for _ in range(layers)])
        self.last_process_mask = None
        self.last_process_padding_mask = None
        self.last_process_sdpa = None
        self.last_decode_mask = None
        self.last_decode_sdpa = None
        self.decode_calls = 0
    def process_prompt(self, x, attn_mask, padding_mask=None, torch_sdpa=True):
        self.last_process_mask = attn_mask
        self.last_process_padding_mask = padding_mask
        self.last_process_sdpa = torch_sdpa
        h = x
        ks = []
        vs = []
        for block in self.blocks:
            h, k, v = block.prefill(h, attn_mask)
            ks.append(k)
            vs.append(v)
        return h, ks, vs
    def decode_next_token(self, x, k_cache, v_cache, attn_mask=None, torch_sdpa=True):
        self.last_decode_mask = attn_mask
        self.last_decode_sdpa = torch_sdpa
        self.decode_calls += 1
        h = x
        ks = []
        vs = []
        for block, k, v in zip(self.blocks, k_cache, v_cache):
            h, k, v = block.decode(h, k, v, attn_mask)
            ks.append(k)
            vs.append(v)
        return h, ks, vs

class Text2SemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config["model"] if isinstance(config, dict) and "model" in config else config
        self.embedding_dim = int(m["embedding_dim"])
        self.hidden_dim = int(m["hidden_dim"])
        self.head = int(m["head"])
        self.n_layer = int(m["n_layer"])
        self.linear_units = int(m["linear_units"])
        self.vocab_size = int(m["vocab_size"])
        self.phoneme_vocab_size = int(m["phoneme_vocab_size"])
        self.EOS = int(m["EOS"])
        self.ar_text_embedding = nn.Embedding(self.phoneme_vocab_size, self.embedding_dim)
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_position = Pos(self.embedding_dim)
        self.ar_audio_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ar_audio_position = Pos(self.embedding_dim)
        self.t2s_transformer = Transformer(
            self.n_layer, self.hidden_dim, self.head, self.linear_units)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
'''


def default_inputs(text=5, prompt=3, codes=6):
    phones = (np.arange(text, dtype=np.int64) % 7).astype(np.int32)
    bert = (((np.arange(1024 * text, dtype=np.float64) % 257) / 257.0)
            .astype(np.float32).reshape(1024, text))
    prompt_tokens = ((np.arange(prompt, dtype=np.int64) + 1) % 11).astype(np.int32)
    generated = ((np.arange(codes, dtype=np.int64) + 2) % 11).astype(np.int32)
    return phones, bert, prompt_tokens, generated


def write_inputs(directory, arrays=None, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = default_inputs(**kwargs) if arrays is None else arrays
    for name, array in zip(TOOL.INPUT_FILES, arrays):
        np.save(directory / name, array, allow_pickle=False)
    return directory


def provenance():
    def digest(value):
        return hashlib.sha256(value).hexdigest()
    return {
        "checkpoint_sha256": digest(b"checkpoint"),
        "inputs": {name: digest(name.encode()) for name in TOOL.INPUT_FILES},
        "upstream": {"AR/models/t2s_model.py": digest(b"upstream")},
        "exporter_sha256": TOOL.sha256(TOOL_PATH),
        "checkpoint_loader_sha256": TOOL.sha256(LOADER),
    }


def make_arrays(hparams=HPARAMS, text=4, prompt=3, steps=5):
    layers, hidden, vocab = hparams["n_layer"], hparams["hidden_dim"], hparams["vocab_size"]
    n = text + prompt
    rng = np.random.default_rng(0x5EED)
    phones = ((np.arange(text, dtype=np.int64) + 1) % hparams["phoneme_vocab_size"]).astype(np.int32)
    bert = rng.standard_normal((text, 1024)).astype(np.float32)
    prompt_tokens = ((np.arange(prompt, dtype=np.int64) + 1) % hparams["EOS"]).astype(np.int32)
    forced = ((np.arange(steps, dtype=np.int64) + 2) % hparams["EOS"]).astype(np.int32)
    logits = rng.standard_normal((steps + 1, vocab)).astype(np.float32)
    pre_k = rng.standard_normal((layers, n, hidden)).astype(np.float32)
    pre_v = rng.standard_normal((layers, n, hidden)).astype(np.float32)
    fin_k = rng.standard_normal((layers, n + steps, hidden)).astype(np.float32)
    fin_v = rng.standard_normal((layers, n + steps, hidden)).astype(np.float32)
    fin_k[:, :n] = pre_k
    fin_v[:, :n] = pre_v
    return {
        "phones": phones, "bert": bert, "prompt": prompt_tokens, "forced_tokens": forced,
        "prefill_input": rng.standard_normal((n, hidden)).astype(np.float32),
        "prefill_hidden": rng.standard_normal((n, hidden)).astype(np.float32),
        "logits": logits, "argmax": logits.argmax(1).astype(np.int32),
        "context_lengths": np.arange(n, n + steps + 1, dtype=np.int32),
        "audio_positions": np.arange(prompt, prompt + steps, dtype=np.int32),
        "prefill_k": pre_k, "prefill_v": pre_v, "final_k": fin_k, "final_v": fin_v,
    }


def make_reference(parent, name="reference", arrays=None):
    output = Path(parent) / name
    manifest = TOOL.write_reference(
        output, make_arrays() if arrays is None else arrays,
        HPARAMS, provenance(), EXECUTION)
    return output, manifest


def read_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(directory, manifest):
    (Path(directory) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mutate_manifest(directory, mutate):
    manifest = read_manifest(directory)
    mutate(manifest)
    write_manifest(directory, manifest)


def rewrite_array(directory, manifest, name, array):
    filename = name + ".npy"
    path = Path(directory) / filename
    np.save(path, array, allow_pickle=False)
    manifest["arrays"][filename]["sha256"] = TOOL.sha256(path)
    write_manifest(directory, manifest)


def with_value(array, value, index=0):
    result = array.copy()
    result[index] = value
    return result


def tiny_namespace():
    import torch  # noqa: F401
    namespace = {"__name__": "tiny_t2s_test_double"}
    exec(compile(TINY_SOURCE, "<tiny_t2s_test_double>", "exec"), namespace)
    return namespace


def build_model():
    import torch
    namespace = tiny_namespace()
    torch.manual_seed(0)
    model = namespace["Text2SemanticDecoder"]({"model": dict(HPARAMS)})
    return model.cpu().float().eval()


def write_upstream(root, suffix=""):
    root = Path(root)
    package = root / "GPT_SoVITS/AR"
    models = package / "models"
    models.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (models / "__init__.py").write_text("", encoding="utf-8")
    (models / "t2s_model.py").write_text(TINY_SOURCE + suffix, encoding="utf-8")
    return root / "GPT_SoVITS"


def write_checkpoint(path, prefix="model.", drop=None, extra=None):
    import torch
    model = build_model()
    state = {prefix + name: value.detach().clone()
             for name, value in model.state_dict().items()}
    if drop is not None:
        state.pop(prefix + drop)
    if extra is not None:
        state[prefix + extra] = torch.zeros(1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": state, "config": {"model": dict(HPARAMS)}}, path)
    return model


BLOCKER = r'''
import importlib.abc
import runpy
import socket
import sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "transformers", "torchmetrics"} or root == "AR" or fullname.startswith("GPT_SoVITS"):
            raise AssertionError("forbidden import: " + fullname)
        return None
sys.meta_path.insert(0, _Block())
def _no_network(*args, **kwargs):
    raise AssertionError("network access attempted")
socket.socket = _no_network
socket.create_connection = _no_network
def _load(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_reference_blocked", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
'''


def run_blocked(code, *args, timeout=60):
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + "\n" + code, *map(str, args)],
        capture_output=True, text=True, timeout=timeout)


def run_cli(*args, timeout=240, env=None):
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *map(str, args)],
        capture_output=True, text=True, timeout=timeout, env=env)


class LoadInputsTests(unittest.TestCase):
    def test_normalizes_layouts_and_prefix(self):
        phones, bert, prompt, codes = default_inputs(codes=8)
        variants = {
            "base": (phones, bert, prompt, codes),
            "batch_one_int64": (phones[None, :].astype(np.int64),
                                bert[None, :, :].astype(np.float32),
                                prompt[None, :].astype(np.int64),
                                codes[None, None, :].astype(np.int64)),
        }
        for name, arrays in variants.items():
            with self.subTest(layout=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=arrays)
                inputs = TOOL.load_inputs(directory, 4, HPARAMS)
                self.assertEqual(set(inputs), {"phones", "bert", "prompt", "forced_tokens"})
                self.assertEqual((inputs["phones"].dtype, inputs["bert"].dtype,
                                  inputs["prompt"].dtype, inputs["forced_tokens"].dtype),
                                 (np.int32, np.float32, np.int32, np.int32))
                np.testing.assert_array_equal(inputs["phones"], phones.astype(np.int32))
                np.testing.assert_array_equal(inputs["bert"], bert.T.astype(np.float32))
                np.testing.assert_array_equal(inputs["prompt"], prompt.astype(np.int32))
                np.testing.assert_array_equal(inputs["forced_tokens"], codes[:4].astype(np.int32))
                self.assertTrue(inputs["phones"].flags.c_contiguous)
                self.assertTrue(inputs["bert"].flags.c_contiguous)
        with tempfile.TemporaryDirectory() as directory:
            write_inputs(directory, arrays=(phones, bert, prompt, codes))
            with self.assertRaisesRegex(ValueError, "fewer tokens"):
                TOOL.load_inputs(directory, 9, HPARAMS)
            bad_tail = codes.copy()
            bad_tail[-1] = HPARAMS["EOS"]
            write_inputs(directory, arrays=(phones, bert, prompt, bad_tail))
            with self.assertRaises(ValueError):
                TOOL.load_inputs(directory, 4, HPARAMS)

    def test_context_and_steps_boundaries(self):
        phones = np.zeros(1, dtype=np.int32)
        bert = np.zeros((1024, 1), dtype=np.float32)
        prompt = np.zeros(1, dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            write_inputs(directory, arrays=(phones, bert, prompt, np.zeros(1022, np.int32)))
            self.assertEqual(TOOL.load_inputs(directory, 1022, HPARAMS)["forced_tokens"].shape, (1022,))
        for codes, steps, message in ((1023, 1023, "context budget"), (1024, 1024, "steps exceeds")):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=(phones, bert, prompt, np.zeros(codes, np.int32)))
                with self.assertRaisesRegex(ValueError, message):
                    TOOL.load_inputs(directory, steps, HPARAMS)

    def test_rejects_bad_token_arrays_and_ranges(self):
        phones, bert, prompt, codes = default_inputs(codes=8)
        cases = {
            "float32 phones": (phones.astype(np.float32), bert, prompt, codes),
            "int16 phones": (phones.astype(np.int16), bert, prompt, codes),
            "float64 codes": (phones, bert, prompt, codes.astype(np.float64)),
            "uint32 prompt": (phones, bert, prompt.astype(np.uint32), codes),
            "phone >= vocab": (with_value(phones, HPARAMS["phoneme_vocab_size"]), bert, prompt, codes),
            "negative phone": (with_value(phones, -1), bert, prompt, codes),
            "prompt == EOS": (phones, bert, with_value(prompt, HPARAMS["EOS"]), codes),
            "negative prompt": (phones, bert, with_value(prompt, -1), codes),
            "code == EOS": (phones, bert, prompt, with_value(codes, HPARAMS["EOS"])),
            "negative code": (phones, bert, prompt, with_value(codes, -1)),
            "empty phones": (phones[:0], bert, prompt, codes),
            "empty prompt": (phones, bert, prompt[:0], codes),
            "empty codes": (phones, bert, prompt, codes[:0]),
        }
        for name, arrays in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=arrays)
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, 4, HPARAMS)

    def test_rejects_bad_bert_steps_hparams(self):
        phones, bert, prompt, codes = default_inputs()
        nan_bert = bert.copy()
        nan_bert[0, 0] = np.nan
        inf_bert = bert.copy()
        inf_bert[1, 1] = np.inf
        bad_bert = {"float64": bert.astype(np.float64), "float16": bert.astype(np.float16),
                    "nan": nan_bert, "inf": inf_bert, "transposed": bert.T.copy(),
                    "wrong length": np.concatenate([bert, bert[:, :1]], axis=1)}
        for name, bad in bad_bert.items():
            with self.subTest(bert=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=(phones, bad, prompt, codes))
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, 4, HPARAMS)
        for steps in (0, -1, True, 4.0):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=(phones, bert, prompt, codes))
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, steps, HPARAMS)
        bad_hparams = {
            "missing": {k: v for k, v in HPARAMS.items() if k != "EOS"},
            "zero": dict(HPARAMS, embedding_dim=0),
            "unequal": dict(HPARAMS, embedding_dim=16),
            "odd": dict(HPARAMS, embedding_dim=7, hidden_dim=7, head=1),
            "head": dict(HPARAMS, head=3),
            "ffn": dict(HPARAMS, linear_units=16),
            "eos": dict(HPARAMS, EOS=10),
            "bool": dict(HPARAMS, n_layer=True),
        }
        for name, hparams in bad_hparams.items():
            with self.subTest(hparams=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory, arrays=(phones, bert, prompt, codes))
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, 4, hparams)

    def test_missing_symlink_and_npz_inputs(self):
        for name in TOOL.INPUT_FILES:
            with self.subTest(missing=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory)
                (Path(directory) / name).unlink()
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, 4, HPARAMS)
            with self.subTest(symlink=name), tempfile.TemporaryDirectory() as directory:
                write_inputs(directory)
                path = Path(directory) / name
                target = path.with_name("real-" + name)
                path.rename(target)
                path.symlink_to(target.name)
                with self.assertRaises(ValueError):
                    TOOL.load_inputs(directory, 4, HPARAMS)
        with tempfile.TemporaryDirectory() as directory:
            write_inputs(directory)
            path = Path(directory) / TOOL.INPUT_FILES[0]
            with path.open("wb") as stream:
                np.savez(stream, phones=np.zeros(2, np.int32))
            with self.assertRaisesRegex(ValueError, "single .npy"):
                TOOL.load_inputs(directory, 4, HPARAMS)


class WriteReferenceTests(unittest.TestCase):
    def test_writes_complete_self_validating_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_reference(directory)
            self.assertEqual((manifest["schema"], manifest["status"], manifest["semantics"]),
                             (TOOL.SCHEMA, "complete", SEMANTICS))
            self.assertEqual(manifest["hparams"], HPARAMS)
            self.assertEqual(manifest["dimensions"],
                             {"text": 4, "prompt": 3, "steps": 5, "prefill": 7, "final": 12})
            self.assertEqual(set(manifest["sources"]), set(provenance()))
            self.assertEqual(manifest["execution"], EXECUTION)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"manifest.json"} | {n + ".npy" for n in EXPECTED})
            specs = TOOL.array_specs(HPARAMS, manifest["dimensions"])
            for name, (dtype, shape) in specs.items():
                with self.subTest(array=name):
                    meta = manifest["arrays"][name + ".npy"]
                    self.assertEqual((meta["dtype"], tuple(meta["shape"]), tuple(meta["ggml_ne"]),
                                      meta["order"]), (dtype, shape, shape[::-1], "C"))
                    self.assertEqual(meta["sha256"], TOOL.sha256(output / (name + ".npy")))
                    loaded = np.load(output / (name + ".npy"))
                    self.assertEqual((loaded.dtype, loaded.shape),
                                     (np.float32 if dtype == "float32" else np.int32, shape))
            self.assertEqual(TOOL.validate_reference(output), manifest)

    def test_refuses_existing_paths_and_second_write(self):
        arrays = make_arrays()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.write_text("sentinel", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                TOOL.write_reference(existing, arrays, HPARAMS, provenance(), EXECUTION)
            self.assertEqual(existing.read_text(encoding="utf-8"), "sentinel")
            for name in ("empty", "nonempty"):
                path = root / name
                path.mkdir()
                if name == "nonempty":
                    (path / "sentinel").write_text("kept", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    TOOL.write_reference(path, arrays, HPARAMS, provenance(), EXECUTION)
                self.assertEqual([p.name for p in path.iterdir()],
                                 [] if name == "empty" else ["sentinel"])
            output, manifest = make_reference(root)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                TOOL.write_reference(output, arrays, HPARAMS, provenance(), EXECUTION)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(read_manifest(output), manifest)

    def test_rejects_invalid_arrays_without_output(self):
        wrong = make_arrays()
        wrong["phones"] = wrong["phones"].astype(np.int64)
        wrong_logits = make_arrays()
        wrong_logits["logits"] = wrong_logits["logits"].astype(np.float64)
        wrong_shape = make_arrays()
        wrong_shape["prefill_k"] = np.zeros((2, 6, 8), np.float32)
        nonfinite = make_arrays()
        nonfinite["prefill_hidden"][0, 0] = np.nan
        bad_argmax = make_arrays()
        bad_argmax["argmax"][0] = (bad_argmax["argmax"][0] + 1) % HPARAMS["vocab_size"]
        bad_context = make_arrays()
        bad_context["context_lengths"][0] += 1
        bad_audio = make_arrays()
        bad_audio["audio_positions"][0] += 1
        bad_prefix = make_arrays()
        bad_prefix["final_k"][0, 0, 0] += 1
        bad_phone = make_arrays()
        bad_phone["phones"][0] = HPARAMS["phoneme_vocab_size"]
        bad_prompt = make_arrays()
        bad_prompt["prompt"][0] = HPARAMS["EOS"]
        cases = {"phones int64": wrong, "logits float64": wrong_logits,
                 "cache shape": wrong_shape, "nonfinite": nonfinite, "argmax": bad_argmax,
                 "context": bad_context, "audio": bad_audio, "cache prefix": bad_prefix,
                 "phone range": bad_phone, "prompt range": bad_prompt,
                 "missing key": {k: v for k, v in make_arrays().items() if k != "logits"},
                 "extra key": dict(make_arrays(), extra=np.zeros(1, np.float32))}
        for name, arrays in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "out"
                with self.assertRaises(ValueError):
                    TOOL.write_reference(output, arrays, HPARAMS, provenance(), EXECUTION)
                self.assertFalse(output.exists())
                self.assertEqual([p.name for p in Path(directory).iterdir()
                                  if p.name.startswith(".out-")], [])

    def test_rejects_invalid_hparams_provenance_execution(self):
        bad_sources = provenance()
        bad_sources["extra"] = "a" * 64
        bad_execution = dict(EXECUTION, extra="not-allowed")
        cases = {
            "hparams": (HPARAMS | {"EOS": 10}, provenance(), EXECUTION),
            "sources digest": (HPARAMS, dict(provenance(), checkpoint_sha256="x"), EXECUTION),
            "sources extra": (HPARAMS, bad_sources, EXECUTION),
            "sources upstream": (HPARAMS, dict(provenance(), upstream={}), EXECUTION),
            "execution threads": (HPARAMS, provenance(), dict(EXECUTION, num_threads=2)),
            "execution extra": (HPARAMS, provenance(), bad_execution),
            "execution path": (HPARAMS, provenance(), dict(EXECUTION, python="/usr/bin/python3")),
            "execution missing": (HPARAMS, provenance(),
                                  {k: v for k, v in EXECUTION.items() if k != "torch"}),
        }
        for name, (hparams, sources, execution) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "out"
                with self.assertRaises(ValueError):
                    TOOL.write_reference(output, make_arrays(), hparams, sources, execution)
                self.assertFalse(output.exists())

    def test_publish_link_is_no_replace_and_preserves_foreign_file(self):
        arrays = make_arrays()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            real_link = os.link
            injected = [False]

            def link(src, dst):
                if not injected[0]:
                    (Path(dst).parent / "logits.npy").write_bytes(b"foreign")
                    injected[0] = True
                return real_link(src, dst)

            with mock.patch.object(TOOL.os, "link", side_effect=link):
                with self.assertRaises(OSError):
                    TOOL.write_reference(output, arrays, HPARAMS, provenance(), EXECUTION)
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual((output / "logits.npy").read_bytes(), b"foreign")
            self.assertEqual({p.name for p in output.iterdir()}, {"logits.npy"})

    def test_output_is_deterministic_and_path_free(self):
        with tempfile.TemporaryDirectory() as directory:
            first, first_manifest = make_reference(directory, "first")
            second, second_manifest = make_reference(directory, "second")
            self.assertEqual(first_manifest, second_manifest)
            for path in first.iterdir():
                with self.subTest(file=path.name):
                    self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            serialized = json.dumps(first_manifest, sort_keys=True)
            self.assertNotIn(str(directory), serialized)
            self.assertNotIn("timestamp", serialized.lower())
            self.assertNotIn("cwd", serialized.lower())


class ValidateReferenceTests(unittest.TestCase):
    def test_requires_regular_complete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                TOOL.validate_reference(root / "missing")
            regular = root / "file"
            regular.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                TOOL.validate_reference(regular)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                TOOL.validate_reference(empty)
            output, _ = make_reference(root)
            link = root / "link"
            link.symlink_to(output.name, target_is_directory=True)
            with self.assertRaises(ValueError):
                TOOL.validate_reference(link)

    def test_rejects_file_set_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            (output / "logits.npy").unlink()
            with self.assertRaises(ValueError):
                TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            (output / "extra.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected or missing files"):
                TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            (output / "nested").mkdir()
            with self.assertRaisesRegex(ValueError, "unexpected or missing files"):
                TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            target = Path(directory) / "real-logits.npy"
            (output / "logits.npy").rename(target)
            (output / "logits.npy").symlink_to(target)
            with self.assertRaises(ValueError):
                TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            array = np.load(output / "logits.npy")
            array[0, 0] += 1
            np.save(output / "logits.npy", array, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TOOL.validate_reference(output)

    def test_rejects_layout_and_value_tampering(self):
        metadata = {
            "shape": lambda m: m["arrays"]["logits.npy"].update(shape=[5, 12]),
            "dtype": lambda m: m["arrays"]["logits.npy"].update(dtype="float64"),
            "ggml_ne": lambda m: m["arrays"]["logits.npy"].update(ggml_ne=[6, 12]),
            "order": lambda m: m["arrays"]["logits.npy"].update(order="F"),
        }
        for name, mutate in metadata.items():
            with self.subTest(metadata=name), tempfile.TemporaryDirectory() as directory:
                output, _ = make_reference(directory)
                mutate_manifest(output, mutate)
                with self.assertRaisesRegex(ValueError, "invalid layout metadata"):
                    TOOL.validate_reference(output)

        def rewrite(name, change):
            def apply(output):
                manifest = read_manifest(output)
                array = np.load(output / (name + ".npy"))
                result = change(array)
                rewrite_array(output, manifest, name, array if result is None else result)
            return apply

        cases = {
            "truncated": rewrite("logits", lambda a: a[:-1].copy()),
            "float64": rewrite("logits", lambda a: a.astype(np.float64)),
            "nonfinite": rewrite("logits", lambda a: a.__setitem__((0, 0), np.nan)),
            "argmax": rewrite("argmax", lambda a: a.__setitem__(0, (a[0] + 1) % HPARAMS["vocab_size"])),
            "context": rewrite("context_lengths", lambda a: a.__setitem__(0, a[0] + 1)),
            "audio": rewrite("audio_positions", lambda a: a.__setitem__(0, a[0] + 1)),
            "cache prefix": rewrite("final_k", lambda a: a.__setitem__((0, 0, 0), a[0, 0, 0] + 1)),
            "negative prompt": rewrite("prompt", lambda a: a.__setitem__(0, -1)),
            "EOS prompt": rewrite("prompt", lambda a: a.__setitem__(0, HPARAMS["EOS"])),
            "negative forced": rewrite("forced_tokens", lambda a: a.__setitem__(0, -1)),
            "EOS forced": rewrite("forced_tokens", lambda a: a.__setitem__(0, HPARAMS["EOS"])),
            "negative phone": rewrite("phones", lambda a: a.__setitem__(0, -1)),
            "phone range": rewrite("phones", lambda a: a.__setitem__(0, HPARAMS["phoneme_vocab_size"])),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output, _ = make_reference(directory)
                mutate(output)
                with self.assertRaises(ValueError):
                    TOOL.validate_reference(output)

    def test_rejects_manifest_source_execution_tampering(self):
        def set_extra(m):
            m["extra"] = 1

        def sources_extra(m):
            m["sources"]["extra"] = "a" * 64

        def execution_extra(m):
            m["execution"]["extra"] = "x"

        def execution_path(m):
            m["execution"]["python"] = "/usr/bin/python3"

        mutations = {
            "top-level extra": set_extra,
            "schema": lambda m: m.update(schema="other"),
            "status": lambda m: m.update(status="partial"),
            "semantics": lambda m: m.update(semantics="sampled"),
            "hparams keys": lambda m: m["hparams"].update(extra=1),
            "hparams value": lambda m: m["hparams"].update(EOS=10),
            "dimensions keys": lambda m: m["dimensions"].pop("final"),
            "dimensions sum": lambda m: m["dimensions"].update(prefill=99),
            "dimensions budget": lambda m: m["dimensions"].update(final=1025),
            "sources extra": sources_extra,
            "sources digest": lambda m: m["sources"].update(checkpoint_sha256="x"),
            "sources inputs": lambda m: m["sources"].update(inputs={"x": "a" * 64}),
            "sources upstream": lambda m: m["sources"].update(upstream={}),
            "sources traversal": lambda m: m["sources"]["upstream"].update(
                {"AR/../t2s_model.py": "a" * 64}),
            "execution extra": execution_extra,
            "execution device": lambda m: m["execution"].update(device="cuda"),
            "execution type": lambda m: m["execution"].update(num_threads=True),
            "execution path": execution_path,
            "execution missing": lambda m: m["execution"].pop("system"),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output, _ = make_reference(directory)
                mutate_manifest(output, mutate)
                with self.assertRaises(ValueError):
                    TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            mutate_manifest(output, lambda m: m["arrays"].__setitem__(
                "../escape.npy", copy.deepcopy(m["arrays"]["logits.npy"])))
            with self.assertRaises(ValueError):
                TOOL.validate_reference(output)
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            mutate_manifest(output, lambda m: m["arrays"].pop("argmax.npy"))
            with self.assertRaises(ValueError):
                TOOL.validate_reference(output)


class LazyImportTests(unittest.TestCase):
    def test_import_help_and_validate_are_torch_network_free(self):
        import_code = r'''
module = _load(sys.argv[1])
assert module.SCHEMA == "gpt-sovits-t2s-decode-reference/v1"
assert "torch" not in sys.modules
assert not any(n == "AR" or n.startswith("AR.") or n.startswith("GPT_SoVITS") for n in sys.modules)
'''
        self.assertEqual(run_blocked(import_code, TOOL_PATH).returncode, 0)
        help_code = r'''
script = sys.argv[1]
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
'''
        result = run_blocked(help_code, TOOL_PATH)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            validate_code = r'''
script, directory = sys.argv[1], sys.argv[2]
sys.argv = [script, "--validate", directory]
runpy.run_path(script, run_name="__main__")
'''
            result = run_blocked(validate_code, TOOL_PATH, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated gpt-sovits-t2s-decode-reference/v1", result.stdout)

    def test_validate_failure_is_torch_free(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = make_reference(directory)
            array = np.load(output / "logits.npy")
            array[0, 0] += 1
            np.save(output / "logits.npy", array, allow_pickle=False)
            code = r'''
script, directory = sys.argv[1], sys.argv[2]
sys.argv = [script, "--validate", directory]
runpy.run_path(script, run_name="__main__")
'''
            result = run_blocked(code, TOOL_PATH, output)
            self.assertEqual(result.returncode, 1)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertNotIn("forbidden import", result.stderr)

    def test_export_checks_output_and_steps_before_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            code = r'''
module = _load(sys.argv[1])
try:
    module.export_reference("missing.ckpt", "missing", "missing", sys.argv[2], 1)
except ValueError as exc:
    assert "refusing to overwrite" in str(exc), str(exc)
else:
    raise AssertionError("expected ValueError")
assert "torch" not in sys.modules
'''
            self.assertEqual(run_blocked(code, TOOL_PATH, existing).returncode, 0)
            code = r'''
module = _load(sys.argv[1])
try:
    module.export_reference("missing.ckpt", "missing", "missing", sys.argv[2], 0)
except ValueError as exc:
    assert "positive integer" in str(exc), str(exc)
else:
    raise AssertionError("expected ValueError")
assert "torch" not in sys.modules
'''
            self.assertEqual(run_blocked(code, TOOL_PATH, Path(directory) / "new").returncode, 0)


class ExportPreconditionTests(unittest.TestCase):
    def test_preconditions_and_cli_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            with self.assertRaises(ValueError):
                TOOL.export_reference(root / "missing.ckpt", root, root, output, 4)
            self.assertFalse(output.exists())
            checkpoint = root / "checkpoint.ckpt"
            checkpoint.write_bytes(b"not a checkpoint")
            link = root / "link.ckpt"
            link.symlink_to(checkpoint.name)
            with self.assertRaises(ValueError):
                TOOL.export_reference(link, root, root, output, 4)
            self.assertFalse(output.exists())
        self.assertEqual(run_cli("--steps", "3").returncode, 2)
        self.assertEqual(run_cli("--validate", "x", "--checkpoint", "y").returncode, 2)


class PycRegressionTests(unittest.TestCase):
    def stale_source(self, root, name="stub.py"):
        source = Path(root) / name
        old = b'MARKER = "old"\n'
        new = b'MARKER = "new"\n'
        source.write_bytes(old)
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        py_compile.compile(str(source), cfile=str(cache), doraise=True)
        source.write_bytes(new)
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(source.stat().st_size, len(old))
        self.assertTrue(cache.exists())
        return source, cache

    def test_source_only_loader_ignores_same_size_same_mtime_pyc(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.stale_source(directory)
            namespace = {}
            exec(importlib.machinery.SourceFileLoader("normal_stub", str(source))
                 .get_code("normal_stub"), namespace)
            self.assertEqual(namespace["MARKER"], "old")
            namespace = {}
            exec(TOOL.SourceOnlyLoader("source_stub", str(source))
                 .get_code("source_stub"), namespace)
            self.assertEqual(namespace["MARKER"], "new")

    def test_checkpoint_loader_spec_uses_source_only_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.stale_source(directory)
            name = "t2s_reference_checkpoint_loader"
            spec = importlib.util.spec_from_file_location(
                name, source, loader=TOOL.SourceOnlyLoader(name, str(source)))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.MARKER, "new")

    def test_upstream_module_fresh_process_ignores_stale_pyc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "AR/models"
            package.mkdir(parents=True)
            (root / "AR/__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            self.stale_source(package, "t2s_model.py")
            code = r'''
import hashlib, importlib.util, sys
spec = importlib.util.spec_from_file_location("tool", sys.argv[1])
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
module, root, sources = tool.upstream_module(sys.argv[2])
assert module.MARKER == "new", module.MARKER
digest = hashlib.sha256(open(sys.argv[2] + "/AR/models/t2s_model.py", "rb").read()).hexdigest()
assert sources["AR/models/t2s_model.py"] == digest, (sources, digest)
print("OK")
'''
            result = subprocess.run(
                [sys.executable, "-c", code, str(TOOL_PATH), str(root)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OK", result.stdout)


def full_forward(model, inputs):
    """Independent full-sequence forward with the exporter's mask semantics."""
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        phones = torch.from_numpy(inputs["phones"].astype(np.int64)).unsqueeze(0)
        bert = torch.from_numpy(inputs["bert"]).unsqueeze(0)
        prompt = torch.from_numpy(inputs["prompt"].astype(np.int64)).unsqueeze(0)
        forced = torch.from_numpy(inputs["forced_tokens"].astype(np.int64)).unsqueeze(0)
        t, p, s = phones.shape[1], prompt.shape[1], forced.shape[1]
        text = model.ar_text_position(model.ar_text_embedding(phones) + model.bert_proj(bert))
        audio = model.ar_audio_position(model.ar_audio_embedding(torch.cat((prompt, forced), 1)))
        sequence = torch.cat((text, audio), 1)
        total = t + p + s
        mask = torch.zeros((total, total), dtype=torch.bool)
        mask[:t, t:] = True
        mask[t:, t:] = torch.triu(torch.ones((total - t, total - t), dtype=torch.bool), 1)
        hidden = sequence
        ks, vs = [], []
        for block in model.t2s_transformer.blocks:
            b, length, width = hidden.shape
            q, k, v = block.in_proj(hidden).chunk(3, -1)
            q = q.view(b, length, block.heads, block.head_dim).transpose(1, 2)
            k = k.view(b, length, block.heads, block.head_dim).transpose(1, 2)
            v = v.view(b, length, block.heads, block.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(block.head_dim)
            scores = scores.masked_fill(mask[None, None], float("-inf"))
            attended = torch.matmul(torch.softmax(scores, -1), v)
            attended = attended.transpose(1, 2).reshape(b, length, width)
            hidden = block.norm1(hidden + block.out_proj(attended))
            hidden = block.norm2(hidden + block.linear2(F.gelu(block.linear1(hidden))))
            ks.append(k.transpose(1, 2).reshape(b, length, width))
            vs.append(v.transpose(1, 2).reshape(b, length, width))
        return hidden, ks, vs, sequence


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class TeacherForcedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def load(self, directory, steps=5, codes=6):
        write_inputs(directory, text=5, prompt=3, codes=codes)
        return TOOL.load_inputs(directory, steps, HPARAMS)

    def test_contract_positions_and_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            outputs = TOOL.run_teacher_forced(model, inputs)
            self.assertEqual(set(outputs), EXPECTED)
            t, p = len(inputs["phones"]), len(inputs["prompt"])
            n, s = t + p, len(inputs["forced_tokens"])
            h, layers, vocab = HPARAMS["hidden_dim"], HPARAMS["n_layer"], HPARAMS["vocab_size"]
            shapes = {"prefill_input": (n, h), "prefill_hidden": (n, h),
                      "logits": (s + 1, vocab), "argmax": (s + 1,),
                      "context_lengths": (s + 1,), "audio_positions": (s,),
                      "prefill_k": (layers, n, h), "prefill_v": (layers, n, h),
                      "final_k": (layers, n + s, h), "final_v": (layers, n + s, h)}
            for name, shape in shapes.items():
                with self.subTest(array=name):
                    self.assertEqual(outputs[name].shape, shape)
                    self.assertEqual(outputs[name].dtype,
                                     np.float32 if name not in
                                     {"argmax", "context_lengths", "audio_positions"} else np.int32)
            for name in ("phones", "prompt", "forced_tokens"):
                self.assertEqual(outputs[name].dtype, np.int32)
            np.testing.assert_array_equal(outputs["context_lengths"],
                                          np.arange(n, n + s + 1, dtype=np.int32))
            np.testing.assert_array_equal(outputs["audio_positions"],
                                          np.arange(p, p + s, dtype=np.int32))
            np.testing.assert_array_equal(outputs["argmax"],
                                          outputs["logits"].argmax(1).astype(np.int32))
            np.testing.assert_array_equal(outputs["forced_tokens"], inputs["forced_tokens"])
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory, steps=1, codes=4)
            outputs = TOOL.run_teacher_forced(model, inputs)
            self.assertEqual(outputs["logits"].shape, (2, HPARAMS["vocab_size"]))
            self.assertEqual(model.t2s_transformer.decode_calls, 1)
            n = len(inputs["phones"]) + len(inputs["prompt"])
            self.assertEqual(outputs["final_k"].shape[1], n + 1)

    def test_matches_full_sequence_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            outputs = TOOL.run_teacher_forced(model, inputs)
            hidden, ks, vs, sequence = full_forward(model, inputs)
            t, p, s = len(inputs["phones"]), len(inputs["prompt"]), len(inputs["forced_tokens"])
            n = t + p
            np.testing.assert_allclose(outputs["prefill_input"], sequence[0, :n].numpy(),
                                       rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(outputs["prefill_hidden"], hidden[0, :n].numpy(),
                                       rtol=1e-4, atol=1e-4)
            for layer in range(HPARAMS["n_layer"]):
                np.testing.assert_allclose(outputs["prefill_k"][layer],
                                           ks[layer][0, :n].numpy(), rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(outputs["prefill_v"][layer],
                                           vs[layer][0, :n].numpy(), rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(outputs["final_k"][layer],
                                           ks[layer][0].numpy(), rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(outputs["final_v"][layer],
                                           vs[layer][0].numpy(), rtol=1e-4, atol=1e-4)
            positions = np.arange(n - 1, n + s)
            expected = model.ar_predict_layer(hidden[0, positions].clone()).detach().numpy()
            np.testing.assert_allclose(outputs["logits"], expected, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(outputs["final_k"][:, :n], outputs["prefill_k"])
            np.testing.assert_array_equal(outputs["final_v"][:, :n], outputs["prefill_v"])

    def test_uses_upstream_mask_and_manual_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            TOOL.run_teacher_forced(model, inputs)
            t, p = len(inputs["phones"]), len(inputs["prompt"])
            n, s = t + p, len(inputs["forced_tokens"])
            transformer = model.t2s_transformer
            self.assertIs(transformer.last_process_sdpa, False)
            self.assertIsNone(transformer.last_process_padding_mask)
            self.assertEqual(transformer.last_process_mask.shape, (1, 1, n, n))
            mask = transformer.last_process_mask[0, 0].numpy()
            np.testing.assert_array_equal(mask[:t, :t], False)
            np.testing.assert_array_equal(mask[:t, t:], True)
            np.testing.assert_array_equal(mask[t:, :t], False)
            np.testing.assert_array_equal(mask[t:, t:],
                                          np.triu(np.ones((p, p), dtype=bool), 1))
            self.assertIsNone(transformer.last_decode_mask)
            self.assertIs(transformer.last_decode_sdpa, False)
            self.assertEqual(transformer.decode_calls, s)

    def test_determinism_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            originals = {k: v.copy() for k, v in inputs.items()}
            first = TOOL.run_teacher_forced(model, inputs)
            second = TOOL.run_teacher_forced(model, inputs)
            for name in EXPECTED:
                with self.subTest(array=name):
                    np.testing.assert_array_equal(first[name], second[name])
            for name, value in originals.items():
                np.testing.assert_array_equal(inputs[name], value)

    def test_no_eos_early_stop_and_writer_roundtrip(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            steps, vocab, eos = len(inputs["forced_tokens"]), HPARAMS["vocab_size"], HPARAMS["EOS"]

            class EosHead(torch.nn.Module):
                def forward(self, x):
                    logits = torch.full((x.shape[0], vocab), -1000.0, dtype=x.dtype)
                    logits[:, eos] = 1000.0
                    return logits

            model.ar_predict_layer = EosHead()
            outputs = TOOL.run_teacher_forced(model, inputs)
            self.assertEqual(outputs["logits"].shape, (steps + 1, vocab))
            np.testing.assert_array_equal(outputs["argmax"], np.full(steps + 1, eos, np.int32))
        with tempfile.TemporaryDirectory() as directory:
            model = build_model()
            inputs = self.load(directory)
            outputs = TOOL.run_teacher_forced(model, inputs)
            output = Path(directory) / "reference"
            manifest = TOOL.write_reference(output, outputs, HPARAMS, provenance(), EXECUTION)
            self.assertEqual(TOOL.validate_reference(output), manifest)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class CliExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def files(self, root, **kwargs):
        root = Path(root)
        upstream = write_upstream(root / "upstream")
        checkpoint = root / "tiny.ckpt"
        model = write_checkpoint(checkpoint, **kwargs)
        inputs = write_inputs(root / "inputs")
        outputs = root / "outputs"
        outputs.mkdir()
        return upstream, checkpoint, inputs, outputs, model

    def export(self, upstream, checkpoint, inputs, output, steps=4, env=None):
        return run_cli("--checkpoint", checkpoint, "--upstream-root", upstream,
                       "--inputs-dir", inputs, "--output-dir", output,
                       "--steps", steps, env=env)

    def assert_outputs_match(self, actual, expected):
        for name, value in expected.items():
            with self.subTest(array=name):
                loaded = np.load(actual / (name + ".npy"), allow_pickle=False)
                if value.dtype.kind == "f":
                    np.testing.assert_allclose(loaded, value, rtol=1e-4, atol=1e-4)
                else:
                    np.testing.assert_array_equal(loaded, value)

    def test_weights_provenance_no_pyc_and_absolute_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream, checkpoint, inputs, outputs, model = self.files(directory)
            malicious = Path(directory) / "malicious"
            malicious.mkdir()
            (malicious / "torch_ckpt_utils.py").write_text(
                "raise RuntimeError('relative loader used')\n", encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(malicious))
            output = outputs / "ref"
            result = self.export(upstream, checkpoint, inputs, output, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Validated gpt-sovits-t2s-decode-reference/v1", result.stdout)
            manifest = TOOL.validate_reference(output)
            self.assertEqual(manifest["dimensions"],
                             {"text": 5, "prompt": 3, "steps": 4, "prefill": 8, "final": 12})
            self.assertEqual(manifest["hparams"], HPARAMS)
            self.assertEqual(manifest["execution"]["attention"], "upstream-manual")
            self.assertEqual(manifest["execution"]["num_threads"], 1)
            self.assertEqual(manifest["execution"]["num_interop_threads"], 1)
            self.assertEqual(manifest["execution"]["initialization_seed"], 0)
            sources = manifest["sources"]
            self.assertEqual(sources["checkpoint_sha256"], TOOL.sha256(checkpoint))
            for name in TOOL.INPUT_FILES:
                self.assertEqual(sources["inputs"][name], TOOL.sha256(inputs / name))
            self.assertEqual(sources["exporter_sha256"], TOOL.sha256(TOOL_PATH))
            self.assertEqual(sources["checkpoint_loader_sha256"], TOOL.sha256(LOADER))
            self.assertEqual(sources["upstream"]["AR/models/t2s_model.py"],
                             TOOL.sha256(upstream / "AR/models/t2s_model.py"))
            self.assertNotIn(str(directory), json.dumps(manifest))
            for path in upstream.rglob("*"):
                self.assertNotEqual(path.name, "__pycache__")
                self.assertNotEqual(path.suffix, ".pyc")
            expected = TOOL.run_teacher_forced(model, TOOL.load_inputs(inputs, 4, HPARAMS))
            self.assert_outputs_match(output, expected)

    def test_root_selection_and_cross_process_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = write_upstream(root / "wrapper")
            checkpoint = root / "tiny.ckpt"
            write_checkpoint(checkpoint)
            inputs = write_inputs(root / "inputs")
            direct_a, direct_b, wrapped = (root / name for name in ("a", "b", "wrapped"))
            runs = [self.export(upstream, checkpoint, inputs, direct_a),
                    self.export(upstream, checkpoint, inputs, direct_b),
                    self.export(upstream.parent, checkpoint, inputs, wrapped)]
            for result in runs:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((direct_a / "manifest.json").read_bytes(),
                             (direct_b / "manifest.json").read_bytes())
            self.assertEqual((direct_a / "manifest.json").read_bytes(),
                             (wrapped / "manifest.json").read_bytes())
            for name in sorted(EXPECTED):
                with self.subTest(array=name):
                    self.assertEqual((direct_a / (name + ".npy")).read_bytes(),
                                     (wrapped / (name + ".npy")).read_bytes())

    def test_checkpoint_prefix_and_strict_mismatch(self):
        cases = {
            "prefix": ({"prefix": ""}, "expected model"),
            "missing": ({"drop": "ar_predict_layer.weight"}, "Missing key"),
            "unexpected": ({"extra": "bogus.weight"}, "Unexpected key"),
        }
        for name, (kwargs, message) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                upstream, checkpoint, inputs, outputs, _ = self.files(directory, **kwargs)
                output = outputs / "ref"
                result = self.export(upstream, checkpoint, inputs, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertFalse(output.exists())

    def test_source_mutation_during_import_fails_without_output(self):
        mutate = ('\nwith open(__file__, "a", encoding="utf-8") as _f:\n'
                  '    _f.write("\\n# mutated during import\\n")\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = write_upstream(root / "upstream", suffix=mutate)
            checkpoint = root / "tiny.ckpt"
            write_checkpoint(checkpoint)
            inputs = write_inputs(root / "inputs")
            outputs = root / "out"
            outputs.mkdir()
            output = outputs / "ref"
            result = self.export(upstream, checkpoint, inputs, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("changed while importing", result.stderr)
            self.assertFalse(output.exists())

    def test_existing_output_exits_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream, checkpoint, inputs, outputs, _ = self.files(directory)
            output = outputs / "existing"
            output.mkdir()
            code = r'''
script = sys.argv[1]
sys.argv = [script, "--checkpoint", sys.argv[2], "--upstream-root", sys.argv[3],
            "--inputs-dir", sys.argv[4], "--output-dir", sys.argv[5], "--steps", "2"]
runpy.run_path(script, run_name="__main__")
'''
            result = run_blocked(code, TOOL_PATH, checkpoint, upstream, inputs, output)
            self.assertEqual(result.returncode, 1)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertNotIn("forbidden import", result.stderr)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
