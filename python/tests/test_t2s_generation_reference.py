"""Tests for the T2S greedy generation reference exporter.

Fixtures (tiny model, checkpoint/upstream/input writers, full-sequence oracle,
blockers, CLI runner) are reused from test_t2s_reference_export.py.
"""

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import test_t2s_reference_export as old  # noqa: E402

TOOL = old.TOOL
HPARAMS = old.HPARAMS
EOS, VOCAB, HIDDEN = HPARAMS["EOS"], HPARAMS["vocab_size"], HPARAMS["hidden_dim"]
GEN_INPUTS = ("t2s_all_phones.npy", "t2s_all_bert.npy", "t2s_prompt.npy")
ARRAY_KEYS = {"phones", "bert", "prompt", "prefill_input", "codes", "logits",
              "selected", "context_lengths"}
TOP_KEYS = {"schema", "status", "hparams", "dimensions", "policy", "stop_reason",
            "sources", "execution", "arrays"}
POLICY_KEYS = {"top_k", "top_p", "temperature", "repetition_penalty", "max_new_tokens",
               "history", "eos"}
EOS_KW = {"eos": True, "codes": 2, "penalty": 1.5}


def write_three(directory, phones, bert, prompt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in zip(GEN_INPUTS, (phones, bert, prompt)):
        np.save(directory / name, array, allow_pickle=False)
    return directory

def gen_provenance():
    sources = old.provenance()
    return dict(sources, inputs={name: sources["inputs"][name] for name in GEN_INPUTS})


def generation_arrays(reference_text=2, target_text=3, prompt=2, codes=3, *,
                      eos=False, penalty=1.0, max_new_tokens=None):
    text, decisions = reference_text + target_text, codes + (1 if eos else 0)
    budget = (codes + 1 if eos else codes) if max_new_tokens is None else max_new_tokens
    rng = np.random.default_rng(0x6E4)
    phones = (np.arange(text) % HPARAMS["phoneme_vocab_size"]).astype(np.int32)
    bert = rng.standard_normal((text, 1024)).astype(np.float32)
    prompt_tokens = ((np.arange(prompt) + 1) % EOS).astype(np.int32)
    chosen = [1 + index % (EOS - 1) for index in range(decisions)]
    if eos: chosen[-1] = EOS
    logits = np.full((decisions, VOCAB), -0.5, dtype=np.float32)
    for index, token in enumerate(chosen): logits[index, token] = 2.0 + 0.25 * index
    # Chosen positives stay ahead of all -0.5 entries even after /penalty.
    prefill = text + prompt
    arrays = {"phones": phones, "bert": bert, "prompt": prompt_tokens,
              "prefill_input": rng.standard_normal((prefill, HIDDEN)).astype(np.float32),
              "codes": np.asarray(chosen[:codes], dtype=np.int32), "logits": logits,
              "selected": np.asarray(chosen, dtype=np.int32),
              "context_lengths": np.arange(prefill, prefill + decisions, dtype=np.int32)}
    manifest = {
        "schema": TOOL.GEN_SCHEMA, "status": "complete", "hparams": dict(HPARAMS),
        "dimensions": {"text": text, "reference_text": reference_text,
                       "target_text": target_text, "prompt": prompt, "prefill": prefill,
                       "codes": codes, "decisions": decisions},
        "policy": dict(top_k=1, top_p=1.0, temperature=1.0,
                       repetition_penalty=float(penalty), max_new_tokens=budget,
                       history="generated-only-binary", eos="unmasked-before-accept"),
        "stop_reason": "eos" if eos else "max_new_tokens", "sources": gen_provenance(),
        "execution": dict(old.EXECUTION), "arrays": {}}
    return arrays, manifest

def write_generation(root, name="ref", **kwargs):
    arrays, manifest = generation_arrays(**kwargs)
    output = Path(root) / name
    output.mkdir()
    for array_name, array in arrays.items():
        np.save(output / (array_name + ".npy"), array, allow_pickle=False)
    for array_name, (dtype, shape) in TOOL.generation_specs(
            HPARAMS, manifest["dimensions"]).items():
        manifest["arrays"][array_name + ".npy"] = {
            "dtype": dtype, "shape": list(shape), "order": "C",
            "ggml_ne": list(reversed(shape)),
            "sha256": TOOL.sha256(output / (array_name + ".npy"))}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest

def mutate(change):
    return lambda output: old.mutate_manifest(output, change)

def rewrite(name, change):
    def apply(output):
        manifest = old.read_manifest(output)
        old.rewrite_array(output, manifest, name,
                          change(np.load(output / (name + ".npy"))))
    return apply


def overflow(output):
    manifest = old.read_manifest(output)
    logits = np.load(output / "logits.npy")
    logits[1, int(np.load(output / "codes.npy")[0])] = -np.finfo(np.float32).max
    old.rewrite_array(output, manifest, "logits", logits)
    old.mutate_manifest(output, lambda m: m["policy"].update(repetition_penalty=1.35))

class GenerationInputsTests(unittest.TestCase):
    def test_normalizes_layouts_and_ignores_generated_codes(self):
        phones, bert, prompt, _ = old.default_inputs(text=5, prompt=3, codes=6)
        with tempfile.TemporaryDirectory() as directory:
            write_three(directory, phones[None].astype(np.int64),
                        bert[None].astype(np.float32), prompt[None].astype(np.int64))
            self.assertFalse((Path(directory) / "generated_codes.npy").exists())
            inputs = TOOL.generation_inputs(directory, 2, 4, HPARAMS)
            self.assertEqual((set(inputs), [inputs[k].dtype for k in ("phones", "bert", "prompt")]),
                             ({"phones", "bert", "prompt"}, [np.int32, np.float32, np.int32]))
            np.testing.assert_array_equal(inputs["phones"], phones.astype(np.int32))
            np.testing.assert_array_equal(inputs["bert"], bert.T.astype(np.float32))
            np.testing.assert_array_equal(inputs["prompt"], prompt.astype(np.int32))

    def test_split_and_budget_boundaries(self):
        phones, bert, prompt, _ = old.default_inputs(text=5, prompt=3, codes=6)
        with tempfile.TemporaryDirectory() as directory:
            write_three(directory, phones, bert, prompt)
            self.assertEqual(TOOL.generation_inputs(directory, 2, 1017, HPARAMS)["phones"].size, 5)
            cases = {"reference zero": (0, 4, "positive integer"),
                     "reference bool": (True, 4, "positive integer"),
                     "reference equals text": (5, 4, "nonempty target"),
                     "reference beyond text": (6, 4, "nonempty target"),
                     "budget zero": (2, 0, "positive integer"),
                     "budget bool": (2, True, "positive integer"),
                     "budget overflow": (2, 1018, "context budget")}
            for name, (reference, budget, message) in cases.items():
                with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                    TOOL.generation_inputs(directory, reference, budget, HPARAMS)

@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class GreedyPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    @staticmethod
    def scripted_model(rows):
        import torch

        class Head(torch.nn.Module):
            calls = 0

            def forward(self, hidden):
                row = rows[min(self.calls, len(rows) - 1)]
                self.calls += 1
                return torch.as_tensor(row, dtype=torch.float32).reshape(1, -1).expand(
                    hidden.shape[0], -1)

        model = old.build_model()
        model.ar_predict_layer = Head()
        return model

    def test_scripted_binary_penalty_and_stop_policy(self):
        def row(high):
            values = np.full(VOCAB, -10.0, dtype=np.float32)
            values[list(high)] = list(high.values())
            return values

        cases = {
            "positive divide": ([1], [row({5: 3.0, 6: 2.5})] * 2, 2, 2.0, [5, 6]),
            "negative multiply": ([1], [row({5: -2.0, 6: -2.5})] * 2, 2, 2.0, [5, 6]),
            # Binary history: the second sighting applies the same penalty.
            "penalty once": ([1], [row({5: 3.0, 6: 2.5})] * 3, 3, 2.0, [5, 6, 5]),
            # Prompt IDs never enter the generated-only history.
            "prompt excluded": ([5], [row({5: 3.0, 6: 2.5})], 1, 2.0, [5]),
            # EOS is unmasked and stops before being appended.
            "eos unmasked": ([1], [row({EOS: 5.0, 5: 4.0})], 3, 2.0, [EOS]),
            # At the budget the last accepted code is retained, no extra probe.
            "limit retains last": ([1], [row({5: 3.0, 6: 2.0}), row({5: 2.0, 6: 3.0})],
                                   2, 1.0, [5, 6])}
        for name, (prompt, rows, budget, penalty, selected) in cases.items():
            with self.subTest(case=name):
                inputs = {"phones": np.arange(3, dtype=np.int32),
                          "bert": np.zeros((3, 1024), dtype=np.float32),
                          "prompt": np.asarray(prompt, dtype=np.int32)}
                arrays, reason = TOOL.run_greedy(
                    self.scripted_model(rows), inputs, 1, budget, penalty)
                np.testing.assert_array_equal(arrays["selected"], np.asarray(selected, np.int32))
                expected_codes = selected[:-1] if selected[-1] == EOS else selected
                self.assertEqual(reason, "eos" if selected[-1] == EOS else "max_new_tokens")
                np.testing.assert_array_equal(arrays["codes"],
                                              np.asarray(expected_codes, np.int32))
                np.testing.assert_array_equal(
                    arrays["context_lengths"],
                    np.arange(3 + len(prompt), 3 + len(prompt) + len(selected)))

    def test_closed_loop_feedback_matches_full_forward(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            phones, bert, prompt, _ = old.default_inputs(text=4, prompt=2, codes=5)
            write_three(directory, phones, bert, prompt)
            inputs = TOOL.generation_inputs(directory, 2, 3, HPARAMS)
            model = old.build_model()
            arrays, reason = TOOL.run_greedy(model, inputs, 2, 3, 1.5)
            codes, decisions = arrays["codes"], arrays["logits"].shape[0]
            self.assertEqual(decisions, len(codes) + (1 if reason == "eos" else 0))
            self.assertGreaterEqual(decisions, 2)
            prefill = len(inputs["phones"]) + len(inputs["prompt"])
            for step in range(decisions):
                hidden, _, _, sequence = old.full_forward(
                    model, dict(inputs, forced_tokens=np.asarray(codes[:step], np.int32)))
                with torch.inference_mode():
                    raw = model.ar_predict_layer(hidden[0, -1:])[0].numpy()
                np.testing.assert_allclose(arrays["logits"][step], raw, rtol=1e-4, atol=1e-4)
                expected = int(np.argmax(TOOL.adjusted_logits(
                    raw, codes[:step].tolist(), 1.5)))
                self.assertEqual(int(arrays["selected"][step]), expected)
                if step == 0:
                    np.testing.assert_allclose(arrays["prefill_input"],
                                               sequence[0, :prefill].numpy(),
                                               rtol=1e-4, atol=1e-4)

class ManifestTests(unittest.TestCase):
    def test_valid_roundtrip_covers_limit_eos_and_zero_codes(self):
        for name, kwargs in (("limit", {}), ("eos", EOS_KW),
                             ("immediate eos", {"eos": True, "codes": 0, "penalty": 1.5})):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output, manifest = write_generation(directory, **kwargs)
                codes = kwargs.get("codes", 3)
                dimensions = {"text": 5, "reference_text": 2, "target_text": 3,
                              "prompt": 2, "prefill": 7, "codes": codes,
                              "decisions": codes + (1 if kwargs.get("eos") else 0)}
                self.assertEqual(TOOL.validate_generation_reference(output), manifest)
                self.assertEqual((manifest["schema"], set(manifest), set(manifest["policy"]),
                                  manifest["dimensions"], set(manifest["sources"]["inputs"])),
                                 (TOOL.GEN_SCHEMA, TOP_KEYS, POLICY_KEYS, dimensions,
                                  set(GEN_INPUTS)))
                self.assertEqual({path.name for path in output.iterdir()},
                                 {"manifest.json"} | {key + ".npy" for key in ARRAY_KEYS})
                dtypes = {name: np.load(output / (name + ".npy")).dtype.name
                          for name in ARRAY_KEYS}
                self.assertEqual(dtypes, {
                    name: "float32" if name in {"bert", "prefill_input", "logits"}
                    else "int32" for name in ARRAY_KEYS})

    def test_tampering_and_policy_violations_are_rejected(self):
        cases = {
            "stop eos on limit": ({}, mutate(lambda m: m.update(stop_reason="eos"))),
            "stop limit on eos": (EOS_KW, mutate(lambda m: m.update(
                stop_reason="max_new_tokens"))),
            "eos codes": (EOS_KW, mutate(lambda m: m["dimensions"].update(codes=3))),
            "limit budget": ({}, mutate(lambda m: m["policy"].update(max_new_tokens=4))),
            "split sum": ({}, mutate(lambda m: m["dimensions"].update(target_text=4))),
            "reference zero": ({}, mutate(lambda m: m["dimensions"].update(
                reference_text=0, target_text=5))),
            "top_k": ({}, mutate(lambda m: m["policy"].update(top_k=2))),
            "penalty zero": ({}, mutate(lambda m: m["policy"].update(repetition_penalty=0.0))),
            "penalty nan": ({}, mutate(
                lambda m: m["policy"].update(repetition_penalty=float("nan")))),
            "eos policy": ({}, mutate(lambda m: m["policy"].update(eos="masked"))),
            "policy missing": ({}, mutate(lambda m: m["policy"].pop("top_k"))),
            "four inputs": ({}, mutate(lambda m: m["sources"]["inputs"].update(
                {"generated_codes.npy": "a" * 64}))),
            "upstream traversal": ({}, mutate(lambda m: m["sources"]["upstream"].update(
                {"AR/../x.py": "a" * 64}))),
            "layout dtype": ({}, mutate(lambda m: m["arrays"]["logits.npy"].update(
                dtype="float64"))),
            "logits float64": ({}, rewrite("logits", lambda a: a.astype(np.float64))),
            "selected int64": ({}, rewrite("selected", lambda a: a.astype(np.int64))),
            "logits rows": ({}, rewrite("logits", lambda a: a[:-1].copy())),
            "nonfinite logits": ({}, rewrite(
                "logits", lambda a: a.__setitem__((0, 0), np.nan) or a)),
            "f32 overflow": ({}, overflow),
            "selected value": ({}, rewrite("selected", lambda a: a.__setitem__(0, 2) or a)),
            "phone range": ({}, rewrite("phones", lambda a: a.__setitem__(
                0, HPARAMS["phoneme_vocab_size"]) or a)),
            "context": ({}, rewrite(
                "context_lengths", lambda a: a.__setitem__(0, a[0] + 1) or a))}
        for name, (kwargs, apply) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                output, _ = write_generation(directory, **kwargs)
                apply(output)
                with self.assertRaises(ValueError):
                    TOOL.validate_generation_reference(output)

    def test_policy_and_overflow_are_rejected_before_publish(self):
        boundary = float(np.float32(1.0 / np.finfo(np.float32).max))
        for rp in (1e-45, boundary):
            with self.assertRaisesRegex(ValueError, "reciprocal"):
                TOOL.generation_policy(1, rp)
        raw = np.array([-np.finfo(np.float32).max, 1.0, 0.0], np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            TOOL.adjusted_logits(raw, [0], 1.35)
        arrays, _ = generation_arrays()
        arrays["logits"][1, int(arrays["codes"][0])] = -np.finfo(np.float32).max
        eos, _ = generation_arrays(eos=True, codes=0, penalty=1.0)
        with tempfile.TemporaryDirectory() as directory:
            for name, data, rp, budget, reason in (
                    ("overflow", arrays, 1.35, 3, "max_new_tokens"),
                    ("eos", eos, boundary, 1, "eos")):
                output = Path(directory) / name
                with self.assertRaises(ValueError):
                    TOOL.write_generation_reference(output, data, HPARAMS, gen_provenance(),
                                                     old.EXECUTION, 2, budget, rp, reason)
                self.assertFalse(output.exists())

    def test_no_overwrite_cli_usage_and_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                TOOL.export_reference("missing.ckpt", "missing", "missing", existing,
                                      generation=True, reference_phones=1, max_new_tokens=2)
            self.assertEqual(list(existing.iterdir()), [])
        paths = ("--checkpoint", "x", "--upstream-root", "x", "--inputs-dir", "x", "--output-dir", "x")
        for args in (("--generate", "--steps", "3", "--reference-phones", "1", *paths),
                     ("--generate", *paths),
                     ("--generate", "--reference-phones", "1", "--validate", "x"),
                     ("--validate", "x", "--generate"),
                     (*paths, "--max-new-tokens", "2")):
            with self.subTest(args=args):
                self.assertEqual(old.run_cli(*args).returncode, 2)
        params = inspect.signature(TOOL.export_reference).parameters
        self.assertEqual(list(params)[:5],
                         ["checkpoint", "upstream_root", "inputs_dir", "output_dir", "steps"])
        self.assertEqual((params["steps"].default, params["steps"].kind),
                         (8, inspect.Parameter.POSITIONAL_OR_KEYWORD))
        for name, default in (("generation", False), ("reference_phones", None),
                              ("max_new_tokens", 200), ("repetition_penalty", 1.0)):
            self.assertEqual((params[name].kind, params[name].default),
                             (inspect.Parameter.KEYWORD_ONLY, default))
        for name in ("generation_inputs", "run_greedy", "validate_generation_reference"):
            self.assertTrue(callable(getattr(TOOL, name)))

BLOCKED_IMPORT = r'''
module = _load(sys.argv[1])
assert module.GEN_SCHEMA == "gpt-sovits-t2s-generation-reference/v1"
assert "torch" not in sys.modules
assert not any(n == "AR" or n.startswith("AR.") or n.startswith("GPT_SoVITS")
               for n in sys.modules)
'''

BLOCKED_RUN = r'''
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''

class LazyGenerationTests(unittest.TestCase):
    def test_import_help_and_validate_are_torch_network_free(self):
        self.assertEqual(old.run_blocked(BLOCKED_IMPORT, old.TOOL_PATH).returncode, 0)
        result = old.run_blocked(BLOCKED_RUN, old.TOOL_PATH, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--generate", "--reference-phones", "--max-new-tokens",
                     "--repetition-penalty"):
            self.assertIn(flag, result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            generation, _ = write_generation(directory, name="generation")
            decode, _ = old.make_reference(directory, name="decode")
            for path, schema in ((generation, TOOL.GEN_SCHEMA), (decode, old.TOOL.SCHEMA)):
                with self.subTest(schema=schema):
                    result = old.run_blocked(BLOCKED_RUN, old.TOOL_PATH, "--validate", path)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"Validated {schema}", result.stdout)

    def test_validate_rejects_non_object_manifest_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(("null", '"x"', "[]")):
                path = Path(directory) / f"m{index}"
                path.mkdir()
                (path / "manifest.json").write_text(text, encoding="utf-8")
                result = old.run_blocked(BLOCKED_RUN, old.TOOL_PATH, "--validate", path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("forbidden import", result.stderr)

    def test_generation_export_refuses_before_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            code = r'''
module = _load(sys.argv[1])
try:
    module.export_reference("missing.ckpt", "missing", "missing", sys.argv[2],
                            generation=True, reference_phones=1, max_new_tokens=2)
except ValueError as exc:
    assert "refusing to overwrite" in str(exc), str(exc)
else:
    raise AssertionError("expected ValueError")
assert "torch" not in sys.modules
'''
            result = old.run_blocked(code, old.TOOL_PATH, existing)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(existing.iterdir()), [])

@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class GenerationCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_export_matches_greedy_and_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = old.write_upstream(root / "upstream")
            checkpoint = root / "tiny.ckpt"
            model = old.write_checkpoint(checkpoint)
            inputs = old.write_inputs(root / "inputs", text=5, prompt=3, codes=6)
            outputs = []
            for name in ("first", "second"):
                output = root / name
                result = old.run_cli("--checkpoint", checkpoint, "--upstream-root", upstream,
                                     "--inputs-dir", inputs, "--output-dir", output,
                                     "--generate", "--reference-phones", 2,
                                     "--max-new-tokens", 3, "--repetition-penalty", 1.35)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"Validated {TOOL.GEN_SCHEMA}", result.stdout)
                outputs.append(output)
            first, second = outputs
            self.assertEqual((first / "manifest.json").read_bytes(),
                             (second / "manifest.json").read_bytes())
            for name in ARRAY_KEYS:
                with self.subTest(array=name):
                    self.assertEqual((first / (name + ".npy")).read_bytes(),
                                     (second / (name + ".npy")).read_bytes())
            expected, reason = TOOL.run_greedy(
                model, TOOL.generation_inputs(inputs, 2, 3, HPARAMS), 2, 3, 1.35)
            for name, value in expected.items():
                with self.subTest(expected=name):
                    loaded = np.load(first / (name + ".npy"), allow_pickle=False)
                    if value.dtype.kind == "f":
                        np.testing.assert_allclose(loaded, value, rtol=1e-4, atol=1e-4)
                    else:
                        np.testing.assert_array_equal(loaded, value)
            manifest = TOOL.validate_generation_reference(first)
            self.assertEqual(manifest["dimensions"],
                             {"text": 5, "reference_text": 2, "target_text": 3,
                              "prompt": 3, "prefill": 8, "codes": len(expected["codes"]),
                              "decisions": len(expected["logits"])})
            self.assertEqual(manifest["policy"],
                             {"top_k": 1, "top_p": 1.0, "temperature": 1.0,
                              "repetition_penalty": 1.35, "max_new_tokens": 3,
                              "history": "generated-only-binary",
                              "eos": "unmasked-before-accept"})
            self.assertEqual(manifest["stop_reason"], reason)
            self.assertEqual(manifest["execution"], old.EXECUTION)
            self.assertEqual(set(manifest["sources"]["inputs"]), set(GEN_INPUTS))
            for name in GEN_INPUTS:
                self.assertEqual(manifest["sources"]["inputs"][name],
                                 TOOL.sha256(inputs / name))
            self.assertNotIn(str(root), json.dumps(manifest))

if __name__ == "__main__":
    unittest.main()
