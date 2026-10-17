import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))
from torch_ckpt_utils import load_checkpoint

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def expected_state():
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    return {
        "weight": base,
        "bias": np.arange(4, dtype=np.float32),
        "transposed": base.T,
        "sliced": base[1::2, 1::2],
        "broadcast": np.broadcast_to(base[1, :4], (3, 4)),
        "scalar": np.array(7.5, dtype=np.float32),
        "empty": np.empty((2, 0, 3), dtype=np.float32),
        "half": np.array([0.5, -1.25], dtype=np.float16),
        "double": np.array([1.125, -2.5], dtype=np.float64),
        "long": np.array([2**40, -3], dtype=np.int64),
        "int": np.array([123, -45], dtype=np.int32),
        "short": np.array([12, -3], dtype=np.int16),
        "byte": np.array([0, 255], dtype=np.uint8),
        "bool": np.array([True, False]),
    }


class StateAssertions:
    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            with self.subTest(tensor=name):
                self.assertEqual(actual[name].shape, value.shape)
                self.assertEqual(actual[name].dtype, value.dtype)
                np.testing.assert_array_equal(actual[name], value)
                self.assertTrue(actual[name].flags.c_contiguous)
                self.assertTrue(actual[name].flags.writeable)


class CheckpointTests(StateAssertions, unittest.TestCase):
    def test_bare_state_dict_with_weight_key(self):
        weights, meta = load_checkpoint(FIXTURES / "bare.pt")
        self.assert_state_equal(weights, expected_state())
        self.assertEqual(meta, {})

    def test_wrapped_state_dict_preserves_metadata(self):
        weights, meta = load_checkpoint(FIXTURES / "wrapped.pt")
        self.assert_state_equal(weights, expected_state())
        self.assertEqual(meta, {"config": {"version": 2}, "info": "fixture"})

    def test_empty_state_dicts(self):
        for filename, metadata in [("empty.pt", {}),
                                   ("empty_wrapped.pt", {"config": {"version": 2}})]:
            with self.subTest(filename=filename):
                weights, meta = load_checkpoint(FIXTURES / filename)
                self.assertEqual(weights, {})
                self.assertEqual(meta, metadata)

    def test_shared_storage_results_are_independent(self):
        weights, _ = load_checkpoint(FIXTURES / "bare.pt")
        weights["weight"][0, 0] = -999
        self.assertEqual(weights["transposed"][0, 0], 0)
        weights["weight"][1, 0] = -999
        self.assertEqual(weights["broadcast"][0, 0], 6)

    def test_load_does_not_import_torch(self):
        code = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('loader attempted to import torch')
sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, sys.argv[1])
from torch_ckpt_utils import load_checkpoint
weights, meta = load_checkpoint(sys.argv[2])
assert weights['scalar'].shape == ()
assert 'torch' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(PYTHON_DIR), str(FIXTURES / "bare.pt")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_layouts(self):
        for value in [[], 42, {"config": {}}, {"weight": 123}]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.pt"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("archive/data.pkl", pickle.dumps(value, protocol=2))
                with self.assertRaises(ValueError):
                    load_checkpoint(path)

    def test_missing_or_ambiguous_pickle(self):
        for entries, message in [(["archive/other"], "no data.pkl"),
                                 (["a/data.pkl", "b/data.pkl"], "ambiguous")]:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.pt"
                with zipfile.ZipFile(path, "w") as archive:
                    for entry in entries:
                        archive.writestr(entry, pickle.dumps({}))
                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint(path)

    def test_non_zip_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            path.write_bytes(pickle.dumps({}))
            with self.assertRaises(zipfile.BadZipFile):
                load_checkpoint(path)

    def test_missing_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.pt"
            with zipfile.ZipFile(FIXTURES / "bare.pt") as source, \
                    zipfile.ZipFile(path, "w") as dest:
                for name in source.namelist():
                    if "/data/" not in name:
                        dest.writestr(name, source.read(name))
            with self.assertRaises(KeyError):
                load_checkpoint(path)

    def test_rejects_unknown_pickle_globals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsupported.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", pickle.dumps(Path("unused")))
            with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported class"):
                load_checkpoint(path)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional: requires PyTorch")
class TorchParityTests(StateAssertions, unittest.TestCase):
    def test_torch_save_round_trip(self):
        import torch
        base = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        state = {"weight": base.t(), "offset": base[1::2, 2::3],
                 "scalar": torch.tensor(2.5), "empty": torch.empty(0)}
        with tempfile.TemporaryDirectory() as tmp:
            for wrapped in [False, True]:
                with self.subTest(wrapped=wrapped):
                    path = Path(tmp) / "round_trip.pt"
                    payload = {"weight": state, "config": {"n": 1}} if wrapped else state
                    torch.save(payload, path)
                    actual, meta = load_checkpoint(path)
                    reference = torch.load(path, map_location="cpu", weights_only=True)
                    expected = reference["weight"] if wrapped else reference
                    self.assert_state_equal(actual, {k: v.numpy() for k, v in expected.items()})
                    self.assertEqual(meta, {"config": {"n": 1}} if wrapped else {})

    def check_model(self, variable):
        import torch
        model_dir = os.environ.get(variable)
        if not model_dir:
            self.skipTest(f"optional: set {variable} to a local HF model directory")
        path = Path(model_dir) / "pytorch_model.bin"
        actual, meta = load_checkpoint(path)
        reference = torch.load(path, map_location="cpu", weights_only=True)
        self.assert_state_equal(actual, {k: v.numpy() for k, v in reference.items()})
        self.assertEqual(meta, {})

    def test_real_hubert(self):
        self.check_model("HUBERT_MODEL_DIR")

    def test_real_roberta(self):
        self.check_model("ROBERTA_MODEL_DIR")


if __name__ == "__main__":
    unittest.main()
