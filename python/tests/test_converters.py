import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

CONFIG = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
    "num_attention_heads": 4, "conv_dim": [2], "conv_kernel": [4],
    "conv_stride": [2], "num_conv_pos_embeddings": 4,
    "num_conv_pos_embedding_groups": 1, "max_position_embeddings": 32,
    "vocab_size": 2, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
}


def expected_tensors(architecture):
    linear = (np.arange(192, dtype=np.float32) - 96).reshape(64, 3).T / 32
    bias = np.array([0.5, -1.25, 2.0], dtype=np.float32)
    if architecture == "hubert":
        return {
            "encoder.layers.0.attention.q_proj.weight": linear,
            "encoder.layers.0.attention.q_proj.bias": bias,
            "feature_extractor.conv_layers.0.conv.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "encoder.pos_conv_embed.conv.weight_g": np.arange(4, dtype=np.float32),
        }
    return {
        "bert.encoder.layer.0.attention.self.query.weight": linear,
        "bert.encoder.layer.0.attention.self.query.bias": bias,
        "bert.embeddings.word_embeddings.weight": np.arange(64, dtype=np.float32).reshape(2, 32) / 16,
    }


@unittest.skipUnless(importlib.util.find_spec("gguf"), "converter tests require gguf")
class ConverterTests(unittest.TestCase):
    def check_conversion(self, architecture, output_type):
        import gguf
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            shutil.copyfile(FIXTURES / f"{architecture}.pt", model_dir / "pytorch_model.bin")
            (model_dir / "config.json").write_text(json.dumps(CONFIG))
            output = model_dir / "output.gguf"
            # Run the actual CLI from another directory and reject torch imports,
            # even when this test suite is running in a PyTorch-enabled environment.
            code = """
import importlib.abc
import runpy
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('converter attempted to import torch')
sys.meta_path.insert(0, BlockTorch())
script = sys.argv[1]
sys.path.insert(0, sys.argv[2])
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name='__main__')
"""
            result = subprocess.run(
                [sys.executable, "-c", code,
                 str(PYTHON_DIR / f"convert_{architecture}_to_gguf.py"),
                 str(PYTHON_DIR), str(model_dir), "--output", str(output), "--type", output_type],
                cwd=tmp, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reader = gguf.GGUFReader(output)
            actual = {tensor.name: tensor for tensor in reader.tensors}
            expected = expected_tensors(architecture)
            self.assertEqual(set(actual), set(expected))
            self.assertEqual(reader.get_field("general.architecture").contents(), architecture)
            self.assertEqual(reader.get_field(f"{architecture}.hidden_size").contents(), 32)
            for name, value in expected.items():
                with self.subTest(tensor=name):
                    tensor = actual[name]
                    np.testing.assert_array_equal(tensor.shape, value.shape[::-1])
                    self.assertEqual(tensor.n_elements, value.size)
                    self.assertTrue(np.isfinite(tensor.data).all())
                    if output_type == "q8" and value.ndim == 2 and "embeddings" not in name:
                        self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.Q8_0)
                        decoded = gguf.dequantize(tensor.data, tensor.tensor_type)
                        np.testing.assert_allclose(decoded, value, rtol=0, atol=0.013)
                    else:
                        half = (output_type == "f16" and value.ndim >= 2) or (
                            output_type == "q8" and architecture == "hubert" and value.ndim == 3)
                        dtype = np.float16 if half else np.float32
                        self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.F16
                                         if half else gguf.GGMLQuantizationType.F32)
                        self.assertEqual(tensor.data.dtype, np.dtype(dtype))
                        np.testing.assert_array_equal(tensor.data, value.astype(dtype))

    def test_hubert_f32(self):
        self.check_conversion("hubert", "f32")

    def test_hubert_f16(self):
        self.check_conversion("hubert", "f16")

    def test_hubert_q8(self):
        self.check_conversion("hubert", "q8")

    def test_roberta_f32(self):
        self.check_conversion("roberta", "f32")

    def test_roberta_f16(self):
        self.check_conversion("roberta", "f16")

    def test_roberta_q8(self):
        self.check_conversion("roberta", "q8")


if __name__ == "__main__":
    unittest.main()
