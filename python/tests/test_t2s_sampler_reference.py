"""Sampler fixture integrity and exporter plumbing, without model dependencies.

The checked-in oracle comes from real upstream code. Optional tests below use an
explicit test double only to exercise loading, parameter adaptation and output.
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tests/t2s/generate_sampler_reference.py"
FIXTURE = ROOT / "tests/t2s/fixtures/sampler_reference.json"
spec = importlib.util.spec_from_file_location("sampler_reference_under_test", TOOL_PATH)
TOOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOOL)

# This intentionally is not a sampler implementation or a numerical oracle.
# Returning a uniform distribution makes its use distinguishable from upstream.
TEST_DOUBLE = '''import torch

def logits_to_probs(logits, previous_tokens=None, temperature=1.0, top_k=None,
                    top_p=None, repetition_penalty=1.0):
    assert top_k is None or top_k > 0
    if previous_tokens.numel():
        assert int(previous_tokens.max()) < logits.shape[1]
    logits.add_(0.25)
    return torch.full_like(logits, 1.0 / logits.shape[1])
'''


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def reseal(document):
    document["content_sha256"] = TOOL.content_digest(document)
    return document


def check_analytic_probabilities(document):
    # These three cases leave exactly tokens 1 and 2. Check their two-way
    # softmax directly, not with a second implementation of the filtering policy.
    cases = {case["name"]: case for case in document["cases"]}
    for name, gap in (("top_p_before_temperature", 2.0), ("top_p_before_top_k", 1.0),
                      ("temperature_after_filtering", 2.0)):
        lower = 1.0 / (1.0 + np.exp(gap))
        np.testing.assert_allclose(cases[name]["expected"]["probabilities"],
                                   [0.0, 1.0 - lower, lower, 0.0], rtol=2e-6, atol=1e-7,
                                   err_msg=name)


def cli(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL_PATH), *map(str, args)],
                          text=True, capture_output=True, timeout=60)


def write_upstream(root, source=TEST_DOUBLE):
    path = Path(root) / "GPT_SoVITS" / TOOL.UPSTREAM_SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    # Loading the single source must not execute package initializers.
    for package in (path.parents[0], path.parents[1], path.parents[2]):
        (package / "__init__.py").write_text("raise AssertionError('package imported')\n", encoding="utf-8")
    return path


class SamplerFixtureTests(unittest.TestCase):
    def test_checked_in_fixture_matches_catalog_and_exporter(self):
        document = TOOL.validate_reference(FIXTURE)
        self.assertEqual(document["sources"]["exporter_sha256"], TOOL.sha256(TOOL_PATH))
        self.assertEqual(len(document["cases"]), 19)
        self.assertEqual([c["name"] for c in document["cases"]],
                         [c["name"] for c in TOOL.case_catalog()])
        self.assertEqual(len({c["name"] for c in document["cases"]}), 19)
        self.assertEqual(FIXTURE.read_text(encoding="utf-8"),
                         json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        text = FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn("timestamp", text)

    def test_analytic_probabilities_and_resealed_numeric_error(self):
        document = TOOL.validate_reference(FIXTURE)
        check_analytic_probabilities(document)
        case = next(c for c in document["cases"] if c["name"] == "temperature_after_filtering")
        case["expected"]["probabilities"] = [0.0, 0.875, 0.125, 0.0]
        selected, margin = TOOL.replay(np.asarray(case["expected"]["probabilities"], np.float32),
                                       np.asarray(case["noise"], np.float32))
        case["expected"].update(sampled=selected, relative_margin=margin)
        # Integrity/replay validation cannot prove numerical correctness, even
        # with real source hashes. The independent analytic check catches this.
        TOOL.validate_document(reseal(document))
        with self.assertRaises(AssertionError):
            check_analytic_probabilities(document)

    def test_known_case_relationships(self):
        cases = {c["name"]: c for c in TOOL.validate_reference(FIXTURE)["cases"]}
        default = cases["default_policy"]
        self.assertEqual(default["parameters"], {
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "repetition_penalty": float(np.float32(1.35)), "suppress_eos": False})
        self.assertEqual(sum(p > 0 for p in default["expected"]["probabilities"]), 15)
        self.assertEqual(default["expected"]["sampled"], 15)
        for name in ("top_p_removes_crossing", "top_p_low_keeps_leader", "top_p_zero_keeps_leader"):
            self.assertEqual(cases[name]["expected"]["probabilities"], [1.0, 0.0, 0.0])
        for name in ("top_p_before_temperature", "top_p_before_top_k", "temperature_after_filtering"):
            self.assertEqual(sum(p > 0 for p in cases[name]["expected"]["probabilities"]), 2)
        duplicate = cases["signed_penalty_duplicate_history"]["expected"]
        self.assertEqual(duplicate, cases["signed_penalty_unique_history"]["expected"])
        self.assertEqual(duplicate["adjusted_logits"], [-2.0, 1.5, -0.5, 0.0, 2.5])
        self.assertEqual(cases["penalty_disabled"]["expected"]["adjusted_logits"],
                         cases["penalty_disabled"]["logits"])
        for name in ("top_k_negative", "top_k_at_vocab", "top_k_above_vocab"):
            self.assertEqual(cases[name]["expected"], cases["top_k_zero"]["expected"])
        self.assertEqual(sum(p > 0 for p in cases["top_k_pivot_ties"]["expected"]["probabilities"]), 3)
        self.assertEqual(cases["top_k_one_leading_ties"]["expected"]["greedy_candidates"], [0, 2])
        self.assertEqual(cases["eos_unmasked"]["expected"]["probabilities"], [0.0, 0.0, 1.0])
        self.assertEqual(cases["eos_prefix_suppressed"]["expected"]["probabilities"], [1.0, 0.0, 0.0])
        self.assertEqual(len(cases["eos_prefix_suppressed"]["expected"]["adjusted_logits"]), 2)

    def test_invalid_documents_are_rejected_even_with_recomputed_digest(self):
        def change_case(key, value):
            return lambda d: d["cases"][0].__setitem__(key, value)

        def change_expected(key, value):
            return lambda d: d["cases"][0]["expected"].__setitem__(key, value)

        def change_parameter(key, value):
            return lambda d: d["cases"][0]["parameters"].__setitem__(key, value)

        changes = {
            "schema": lambda d: d.update(schema="old"),
            "status": lambda d: d.update(status="partial"),
            "extra field": lambda d: d.update(unexpected=True),
            "semantics": lambda d: d["semantics"].update(noise="rank-indexed"),
            "missing cases": lambda d: d["cases"].pop(),
            "duplicate case": lambda d: d["cases"].__setitem__(1, copy.deepcopy(d["cases"][0])),
            "wrong upstream name": lambda d: d["sources"].update(upstream={"/private/utils.py": "0" * 64}),
            "bad source digest": lambda d: d["sources"].update(exporter_sha256="x" * 64),
            "execution backend": lambda d: d["execution"].update(device="cuda"),
            "execution boolean thread": lambda d: d["execution"].update(num_threads=True),
            "execution dtype": lambda d: d["execution"].update(dtype="float64"),
            "execution flag": lambda d: d["execution"].update(inference_mode=False),
            "empty version": lambda d: d["execution"].update(torch=""),
            "empty logits": change_case("logits", []),
            "short noise": change_case("noise", [1.0]),
            "zero noise": change_case("noise", [0.0] * 20),
            "negative noise": change_case("noise", [-1.0] * 20),
            "history out of range": change_case("previous_tokens", [20]),
            "history boolean": change_case("previous_tokens", [True]),
            "history not a list": change_case("previous_tokens", "0"),
            "boolean k": change_parameter("top_k", True),
            "noninteger k": change_parameter("top_k", 1.5),
            "overflow k": change_parameter("top_k", 2**31),
            "invalid suppression": change_parameter("suppress_eos", 1),
            "zero penalty": change_parameter("repetition_penalty", 0.0),
            "negative penalty": change_parameter("repetition_penalty", -1.0),
            "invalid F32 parameter": change_parameter("top_p", 0.1),
            "boolean parameter": change_parameter("temperature", True),
            "probability length": change_expected("probabilities", [1.0]),
            "negative probability": change_expected("probabilities", [-0.5] + [0.0] * 19),
            "probability sum": change_expected("probabilities", [0.0] * 20),
            "adjusted length": change_expected("adjusted_logits", [1.0]),
            "incorrect greedy": change_expected("greedy_candidates", [0]),
            "boolean greedy": change_expected("greedy_candidates", [True]),
            "incorrect sampled": change_expected("sampled", 2),
            "sampled range": change_expected("sampled", 20),
            "boolean sampled": change_expected("sampled", True),
            "incorrect margin": change_expected("relative_margin", 0.0),
        }
        for name, change in changes.items():
            with self.subTest(case=name):
                document = load_fixture()
                change(document)
                with self.assertRaises(ValueError):
                    TOOL.validate_document(reseal(document))

    def test_nonfinite_and_non_f32_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1e100, 0.1, True):
            with self.subTest(value=value):
                document = load_fixture()
                document["cases"][0]["logits"][0] = value
                with self.assertRaises(ValueError):
                    TOOL.validate_document(document)

    def test_digest_tampering_is_rejected(self):
        document = load_fixture()
        document["sources"]["exporter_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content SHA-256"):
            TOOL.validate_document(document)

    def test_ambiguous_sampled_oracle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            TOOL.replay(np.asarray([0.5, 0.5], dtype=np.float32), np.ones(2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            TOOL.replay(np.ones(2, dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32))

    def test_json_duplicates_constants_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for text in ('{"schema":1,"schema":2}', '{"value":NaN}', '{"value":Infinity}', '[]'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    TOOL.validate_reference(path)
            path.unlink()
            path.symlink_to(FIXTURE)
            with self.assertRaises(ValueError):
                TOOL.validate_reference(path)


class SamplerPublishingTests(unittest.TestCase):
    def test_publish_and_no_overwrite(self):
        document = load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new" / "reference.json"
            TOOL.publish_reference(output, document)
            self.assertEqual(TOOL.validate_reference(output), document)
            existing_dir = root / "existing-directory"
            existing_dir.mkdir()
            symlink = root / "link.json"
            symlink.symlink_to(output)
            broken = root / "broken.json"
            broken.symlink_to(root / "missing")
            before = output.read_bytes()
            for existing in (output, existing_dir, symlink, broken):
                with self.subTest(path=existing), self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    TOOL.publish_reference(existing, document)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(existing_dir.iterdir()), [])

    def test_failed_publication_and_raced_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reference.json"
            with mock.patch.object(TOOL.os, "link", side_effect=OSError("no hard links")):
                with self.assertRaises(OSError):
                    TOOL.publish_reference(output, load_fixture())
            self.assertEqual(list(root.iterdir()), [])
            real_link = os.link

            def raced_link(source, destination):
                Path(destination).write_text("other writer", encoding="utf-8")
                real_link(source, destination)

            with mock.patch.object(TOOL.os, "link", side_effect=raced_link):
                with self.assertRaises(FileExistsError):
                    TOOL.publish_reference(output, load_fixture())
            self.assertEqual(output.read_text(encoding="utf-8"), "other writer")
            self.assertEqual(list(root.iterdir()), [output])

    def test_invalid_document_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            document = load_fixture()
            document["cases"].pop()
            with self.assertRaises(ValueError):
                TOOL.publish_reference(output, document)
            self.assertEqual(list(Path(directory).iterdir()), [])


class SamplerCliTests(unittest.TestCase):
    def test_help_and_validate_without_torch_or_network(self):
        # Bootstrap the trusted tool by absolute path; isolated Python never adds
        # the temporary cwd to sys.path. Block imports/sockets before runpy.
        bootstrap = '''import importlib.abc, runpy, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("unexpected torch import")
sys.meta_path.insert(0, BlockTorch())
def audit(event, args):
    if event.startswith("socket."):
        raise AssertionError("unexpected network access")
sys.addaudithook(audit)
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        for arguments in (("--help",), ("--validate", str(FIXTURE))):
            result = subprocess.run([sys.executable, "-I", "-c", bootstrap, str(TOOL_PATH), *arguments],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_invalid_modes_and_missing_inputs(self):
        for arguments in ((), ("--output", "unused.json"),
                          ("--validate", str(FIXTURE), "--upstream-root", "unused")):
            with self.subTest(arguments=arguments):
                result = cli(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
        result = cli("--validate", FIXTURE.with_name("does-not-exist.json"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("required regular file", result.stderr)

    def test_existing_export_output_rejected_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            output.write_text("sentinel", encoding="utf-8")
            result = cli("--upstream-root", "unused", "--output", output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")

    def test_source_loading_ignores_pyc_and_package_initializers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_upstream(directory, 'def logits_to_probs(*args, **kwargs): return "old"\n')
            py_compile.compile(str(path), doraise=True)
            stamp = path.stat()
            path.write_text('def logits_to_probs(*args, **kwargs): return "new"\n', encoding="utf-8")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            before = list(sys.path)
            function, loaded, digest = TOOL.load_upstream(directory)
            self.assertEqual(function(), "new")
            self.assertEqual(loaded, path.resolve())
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(sys.path, before)
            function, _, _ = TOOL.load_upstream(path.parents[2])
            self.assertEqual(function(), "new")

    def test_missing_and_symlink_upstream_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TOOL.load_upstream(directory)
            path = write_upstream(directory, "logits_to_probs = None\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                TOOL.load_upstream(directory)
            target = path.with_name("real.py")
            path.rename(target)
            path.symlink_to(target.name)
            with self.assertRaises(ValueError):
                TOOL.load_upstream(directory)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class SamplerTorchTests(unittest.TestCase):
    def test_parameter_adaptation_and_input_clone(self):
        import torch

        catalog = TOOL.case_catalog()
        before = copy.deepcopy(catalog)
        calls = []

        def double(logits, **kwargs):
            calls.append((logits.shape[1], kwargs))
            self.assertEqual(logits.dtype, torch.float32)
            self.assertEqual(logits.device.type, "cpu")
            self.assertFalse(torch.is_grad_enabled())
            logits.add_(0.25)
            return torch.full_like(logits, 1.0 / logits.shape[1])

        with mock.patch.object(TOOL, "case_catalog", return_value=catalog):
            cases = TOOL.run_cases(double)
        self.assertEqual([{k: case[k] for k in TOOL.INPUT_KEYS} for case in cases], before)
        for case, (active, kwargs) in zip(cases, calls):
            params = case["parameters"]
            self.assertEqual(active, len(case["logits"]) - int(params["suppress_eos"]))
            self.assertEqual(kwargs["top_k"], params["top_k"] if params["top_k"] > 0 else None)
            self.assertEqual(kwargs["previous_tokens"].tolist(), [case["previous_tokens"]])
            self.assertEqual(kwargs["top_p"], params["top_p"])
            self.assertEqual(kwargs["temperature"], params["temperature"])
            self.assertEqual(kwargs["repetition_penalty"], params["repetition_penalty"])
            np.testing.assert_array_equal(case["expected"]["adjusted_logits"],
                                          np.asarray(case["logits"][:active], np.float32) + np.float32(0.25))

    def test_cross_process_export_identity_with_explicit_test_double(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_upstream(root / "upstream")
            outputs = [root / "a.json", root / "b.json"]
            for output in outputs:
                result = cli("--upstream-root", root / "upstream", "--output", output)
                self.assertEqual(result.returncode, 0, result.stderr)
                document = TOOL.validate_reference(output)
                self.assertEqual(document["sources"]["upstream"][TOOL.UPSTREAM_SOURCE], TOOL.sha256(source))
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            # A test-double export is not allowed to replace the real oracle.
            self.assertNotEqual(TOOL.validate_reference(outputs[0])["cases"][0]["expected"],
                                load_fixture()["cases"][0]["expected"])


if __name__ == "__main__":
    unittest.main()
