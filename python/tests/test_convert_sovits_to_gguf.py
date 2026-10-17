"""Converter regressions for the unified SoVITS GGUF entry point.

Covers the v2Pro/v2ProPlus wrap-up:
  * gin_channels derivation from the weights (all four flow coupling layers,
    generator cond, ref_enc fc, sv_emb) and rejection of a contradictory cfg;
  * all-or-none v2Pro module detection and v2Pro tensor-shape checks;
  * complete generator structure validation (kernels / channels / strides);
  * _derive_version, including explicit v2Pro/v2ProPlus vs initial-channel
    conflicts;
  * a full synthetic conversion through convert_from_weights (positive
    control) plus single-fault synthetic conversions that must be rejected.

An optional end-to-end positive control converts a real v2Pro checkpoint when
one is present locally (or named by SOVITS_V2PRO_CKPT).

Runs from the repository root:

    python3 -m unittest discover -s python/tests -v
"""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

HAS_GGUF = importlib.util.find_spec("gguf") is not None
if HAS_GGUF:  # the converter imports gguf at module scope
    import convert_sovits_to_gguf as conv

DEFAULT_V2PRO_CKPT = (REPO_ROOT / "tests/t2s/models/pretrained_models/v2Pro/s2Gv2Pro.pth")


# ---------------------------------------------------------------------------
# A complete, correctly-shaped synthetic checkpoint (PyTorch layout).
# ---------------------------------------------------------------------------

def synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True):
    """Every tensor the converter reads, with the real architecture shapes."""
    def arr(*shape):
        return np.zeros(shape, np.float32)

    w = {}
    # ref_enc (MelStyleEncoder)
    w["ref_enc.spectral.0.fc.weight"] = arr(128, 704)
    w["ref_enc.spectral.0.fc.bias"] = arr(128)
    w["ref_enc.spectral.3.fc.weight"] = arr(128, 128)
    w["ref_enc.spectral.3.fc.bias"] = arr(128)
    for i in range(2):
        w[f"ref_enc.temporal.{i}.conv1.conv.weight"] = arr(256, 128, 5)
        w[f"ref_enc.temporal.{i}.conv1.conv.bias"] = arr(256)
    for n in ("q", "k", "v"):
        w[f"ref_enc.slf_attn.w_{n}s.weight"] = arr(128, 128)
        w[f"ref_enc.slf_attn.w_{n}s.bias"] = arr(128)
    w["ref_enc.slf_attn.fc.weight"] = arr(128, 128)
    w["ref_enc.slf_attn.fc.bias"] = arr(128)
    w["ref_enc.fc.fc.weight"] = arr(gin, 128)
    w["ref_enc.fc.fc.bias"] = arr(gin)

    # quantizer
    w["quantizer.vq.layers.0._codebook.embed"] = arr(1024, 768)

    # text encoder
    w["enc_p.text_embedding.weight"] = arr(732, 192)
    w["enc_p.ssl_proj.weight"] = arr(192, 768)
    w["enc_p.ssl_proj.bias"] = arr(192)
    for branch, n_layers in (("encoder_ssl", 3), ("encoder_text", 6), ("encoder2", 3)):
        for i in range(n_layers):
            p = f"enc_p.{branch}.attn_layers.{i}"
            for x in ("q", "k", "v", "o"):
                w[f"{p}.conv_{x}.weight"] = arr(192, 192, 1)
                w[f"{p}.conv_{x}.bias"] = arr(192)
            w[f"{p}.emb_rel_k"] = arr(1, 9, 96)
            w[f"{p}.emb_rel_v"] = arr(1, 9, 96)
            for nl in ("1", "2"):
                w[f"enc_p.{branch}.norm_layers_{nl}.{i}.gamma"] = arr(192)
                w[f"enc_p.{branch}.norm_layers_{nl}.{i}.beta"] = arr(192)
            w[f"enc_p.{branch}.ffn_layers.{i}.conv_1.weight"] = arr(768, 192, 3)
            w[f"enc_p.{branch}.ffn_layers.{i}.conv_1.bias"] = arr(768)
            w[f"enc_p.{branch}.ffn_layers.{i}.conv_2.weight"] = arr(192, 768, 3)
            w[f"enc_p.{branch}.ffn_layers.{i}.conv_2.bias"] = arr(192)
    w["enc_p.mrte.c_pre.weight"] = arr(512, 192, 1)
    w["enc_p.mrte.c_pre.bias"] = arr(512)
    w["enc_p.mrte.text_pre.weight"] = arr(512, 192, 1)
    w["enc_p.mrte.text_pre.bias"] = arr(512)
    for x in ("q", "k", "v", "o"):
        w[f"enc_p.mrte.cross_attention.conv_{x}.weight"] = arr(512, 512, 1)
        w[f"enc_p.mrte.cross_attention.conv_{x}.bias"] = arr(512)
    w["enc_p.mrte.c_post.weight"] = arr(192, 512, 1)
    w["enc_p.mrte.c_post.bias"] = arr(192)
    w["enc_p.proj.weight"] = arr(384, 192, 1)
    w["enc_p.proj.bias"] = arr(384)

    # flow coupling layers at checkpoint indices 0,2,4,6
    for idx in conv.FLOW_CKPT_INDICES:
        p = f"flow.flows.{idx}"
        w[f"{p}.pre.weight"] = arr(192, 96, 1)
        w[f"{p}.pre.bias"] = arr(192)
        w[f"{p}.post.weight"] = arr(96, 192, 1)
        w[f"{p}.post.bias"] = arr(96)
        w[f"{p}.enc.cond_layer.weight_g"] = arr(1536, 1, 1)
        w[f"{p}.enc.cond_layer.weight_v"] = arr(1536, gin, 1)
        w[f"{p}.enc.cond_layer.bias"] = arr(1536)
        for j in range(4):
            out = 384 if j < 3 else 192
            w[f"{p}.enc.in_layers.{j}.weight_g"] = arr(384, 1, 1)
            w[f"{p}.enc.in_layers.{j}.weight_v"] = arr(384, 192, 5)
            w[f"{p}.enc.in_layers.{j}.bias"] = arr(384)
            w[f"{p}.enc.res_skip_layers.{j}.weight_g"] = arr(out, 1, 1)
            w[f"{p}.enc.res_skip_layers.{j}.weight_v"] = arr(out, 192, 1)
            w[f"{p}.enc.res_skip_layers.{j}.bias"] = arr(out)

    # generator (HiFi-GAN)
    w["dec.conv_pre.weight"] = arr(gen_init, 192, 7)
    w["dec.conv_pre.bias"] = arr(gen_init)
    w["dec.cond.weight"] = arr(gen_init, gin, 1)
    w["dec.cond.bias"] = arr(gen_init)
    w["dec.conv_post.weight"] = arr(1, gen_init // 32, 7)
    upk = conv.GENERATOR_UP_K768 if gen_init == 768 else conv.GENERATOR_UP_K512
    for s in range(5):
        in_ch, out_ch = gen_init >> s, gen_init >> (s + 1)
        w[f"dec.ups.{s}.weight_g"] = arr(in_ch, 1, 1)
        w[f"dec.ups.{s}.weight_v"] = arr(in_ch, out_ch, upk[s])
        w[f"dec.ups.{s}.bias"] = arr(out_ch)
        for br, rk in enumerate(conv.GENERATOR_RESBLOCK_KERNELS):
            for l in range(3):
                for c in ("convs1", "convs2"):
                    b = f"dec.resblocks.{s * 3 + br}.{c}.{l}"
                    w[b + ".weight_g"] = arr(out_ch, 1, 1)
                    w[b + ".weight_v"] = arr(out_ch, out_ch, rk)
                    w[b + ".bias"] = arr(out_ch)

    if v2pro:
        w["sv_emb.weight"] = arr(gin, 20480)
        w["sv_emb.bias"] = arr(gin)
        w["ge_to512.weight"] = arr(512, gin)
        w["ge_to512.bias"] = arr(512)
        w["prelu.weight"] = arr(gin)
    return w


def synth_cfg(gin=1024, gen_init=512, **extra):
    cfg = {"gin_channels": gin, "upsample_initial_channel": gen_init,
           "semantic_frame_rate": "25hz"}
    cfg.update(extra)
    return cfg


def _quiet_convert(weights, cfg, out_path, dtype="f16"):
    with contextlib.redirect_stdout(io.StringIO()):
        conv.convert_from_weights(weights, cfg, str(out_path), dtype)


def _read_fields(path):
    import gguf
    reader = gguf.GGUFReader(path)
    fields = {}
    for f in reader.fields.values():
        if f.types[0] == gguf.GGUFValueType.STRING:
            fields[f.name] = bytes(f.parts[-1]).decode()
        else:
            fields[f.name] = f.parts[-1][0]
    return reader, fields


# ---------------------------------------------------------------------------
# Helper-level derivation / structure tests.
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_GGUF, "converter tests require gguf")
class DerivationTests(unittest.TestCase):
    def test_derive_gin_all_flows_agree(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        self.assertEqual(conv.derive_gin_channels(w, {"gin_channels": 1024}), 1024)
        self.assertTrue(conv.v2pro_present(w))

    def test_derive_gin_detects_non_first_flow(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        # Only the last coupling layer disagrees -> must still be rejected.
        w["flow.flows.6.enc.cond_layer.weight_v"] = np.zeros((1536, 512, 1), np.float32)
        with self.assertRaises(ValueError):
            conv.derive_gin_channels(w, {})

    def test_derive_gin_missing_flow_layer(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        del w["flow.flows.4.enc.cond_layer.weight_v"]
        with self.assertRaises(KeyError):
            conv.derive_gin_channels(w, {})

    def test_generator_structure_ok(self):
        for gen_init in (512, 768):
            conv.validate_generator_structure(
                synthetic_checkpoint(gin=1024, gen_init=gen_init, v2pro=True),
                1024, gen_init)

    def test_version_labels(self):
        self.assertEqual(conv._derive_version({}, False, 512), "v2")
        self.assertEqual(conv._derive_version({}, True, 512), "v2Pro")
        self.assertEqual(conv._derive_version({}, True, 768), "v2ProPlus")


# ---------------------------------------------------------------------------
# Single-fault rejections (helper level).
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_GGUF, "converter tests require gguf")
class SingleFaultHelperTests(unittest.TestCase):
    def test_config_gin_contradicts_weights(self):
        w = synthetic_checkpoint(gin=1024, v2pro=True)
        with self.assertRaises(ValueError):
            conv.derive_gin_channels(w, {"gin_channels": 512})

    def test_weights_disagree_on_gin(self):
        w = synthetic_checkpoint(gin=1024, v2pro=True)
        w["flow.flows.0.enc.cond_layer.weight_v"] = np.zeros((1536, 512, 1), np.float32)
        with self.assertRaises(ValueError):
            conv.derive_gin_channels(w, {})

    def test_partial_v2pro_module_set(self):
        w = synthetic_checkpoint(gin=1024, v2pro=True)
        del w["ge_to512.weight"]
        with self.assertRaises(ValueError):
            conv.v2pro_present(w)

    def test_generator_kernel_rejections(self):
        cases = [
            ("dec.conv_pre.weight", (512, 192, 5)),
            ("dec.conv_post.weight", (1, 16, 5)),
            ("dec.ups.0.weight_v", (512, 256, 12)),
            ("dec.resblocks.0.convs1.0.weight_v", (256, 256, 5)),
            ("dec.resblocks.0.convs2.2.weight_v", (256, 256, 5)),
        ]
        for name, shape in cases:
            with self.subTest(tensor=name):
                w = synthetic_checkpoint(gin=512, gen_init=512, v2pro=False)
                w[name] = np.zeros(shape, np.float32)
                with self.assertRaises(ValueError):
                    conv.validate_generator_structure(w, 512, 512)

    def test_generator_channel_rejections(self):
        w = synthetic_checkpoint(gin=512, gen_init=512, v2pro=False)
        w["dec.ups.2.bias"] = np.zeros((48,), np.float32)  # must be 64
        with self.assertRaises(ValueError):
            conv.validate_generator_structure(w, 512, 512)

    def test_bad_initial_channels(self):
        w = synthetic_checkpoint(gin=512, gen_init=384, v2pro=False)
        with self.assertRaises(ValueError):
            conv.validate_generator_structure(w, 512, 384)

    def test_version_conflicts(self):
        with self.assertRaises(ValueError):
            conv._derive_version({"version": "v2Pro"}, True, 768)
        with self.assertRaises(ValueError):
            conv._derive_version({"version": "v2ProPlus"}, True, 512)
        with self.assertRaises(ValueError):
            conv._derive_version({"version": "v2Pro"}, False, 512)


# ---------------------------------------------------------------------------
# Full synthetic conversion: positive control + single-fault conversions.
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_GGUF, "converter tests require gguf")
class SyntheticConvertTests(unittest.TestCase):
    def _convert_ok(self, weights, cfg, dtype="f16"):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "synth.gguf"
            _quiet_convert(weights, cfg, out, dtype)
            self.assertTrue(out.is_file())
            return _read_fields(out)

    def test_synthetic_v2pro_positive(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        reader, f = self._convert_ok(w, synth_cfg(1024, 512))
        self.assertEqual(f["general.architecture"], "sovits")
        self.assertTrue(f["sovits.v2pro"])
        self.assertEqual(f["sovits.version"], "v2Pro")
        self.assertEqual(f["sovits.ref_enc.out_dim"], 1024)
        self.assertEqual(f["sovits.flow.gin"], 1024)
        self.assertEqual(f["sovits.generator.gin_dim"], 1024)
        self.assertEqual(f["sovits.text_encoder.ge_dim"], 512)
        names = {t.name for t in reader.tensors}
        for n in ("v2pro.sv_emb_w", "v2pro.sv_emb_b", "v2pro.ge_to512_w",
                  "v2pro.ge_to512_b", "v2pro.prelu_w"):
            self.assertIn(n, names)
        shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}
        self.assertEqual(shapes["v2pro.sv_emb_w"], (20480, 1024))
        self.assertEqual(shapes["v2pro.ge_to512_w"], (1024, 512))
        self.assertEqual(shapes["v2pro.prelu_w"], (1024,))
        self.assertEqual(len(names), 484)

    def test_synthetic_v2proplus_positive(self):
        w = synthetic_checkpoint(gin=1024, gen_init=768, v2pro=True)
        _, f = self._convert_ok(w, synth_cfg(1024, 768))
        self.assertTrue(f["sovits.v2pro"])
        self.assertEqual(f["sovits.version"], "v2ProPlus")
        self.assertEqual(f["sovits.generator.gin_dim"], 1024)

    def test_synthetic_v2_positive(self):
        w = synthetic_checkpoint(gin=512, gen_init=512, v2pro=False)
        reader, f = self._convert_ok(w, synth_cfg(512, 512))
        self.assertFalse(f["sovits.v2pro"])
        self.assertEqual(f["sovits.version"], "v2")
        self.assertEqual(f["sovits.ref_enc.out_dim"], 512)
        names = {t.name for t in reader.tensors}
        self.assertNotIn("v2pro.sv_emb_w", names)
        self.assertEqual(len(names), 479)

    def _expect_reject(self, weights, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ValueError, KeyError)):
                _quiet_convert(weights, cfg, Path(tmp) / "bad.gguf")

    def test_convert_rejects_flow_gin_conflict(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["flow.flows.2.enc.cond_layer.weight_v"] = np.zeros((1536, 512, 1), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_flow_cond_bias_only(self):
        # Only the cond bias is wrong; the rest of the model is untouched.
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["flow.flows.4.enc.cond_layer.bias"] = np.zeros((1024,), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_flow_cond_weight_g_only(self):
        # Only the cond weight_g (norm magnitude) is wrong.
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["flow.flows.0.enc.cond_layer.weight_g"] = np.zeros((1024, 1, 1), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_flow_cond_kernel_only(self):
        # Only the cond Conv1d kernel (trailing dim) is wrong.
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["flow.flows.6.enc.cond_layer.weight_v"] = np.zeros((1536, 1024, 2), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_generator_kernel(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["dec.ups.0.weight_v"] = np.zeros((512, 256, 12), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_cfg_gin_conflict(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        self._expect_reject(w, synth_cfg(512, 512))  # cfg says 512, weights 1024

    def test_convert_rejects_version_conflict(self):
        w = synthetic_checkpoint(gin=1024, gen_init=768, v2pro=True)
        self._expect_reject(w, synth_cfg(1024, 768, version="v2Pro"))  # Plus weights

    def test_convert_rejects_partial_v2pro(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        del w["ge_to512.weight"]
        self._expect_reject(w, synth_cfg(1024, 512))

    def test_convert_rejects_v2pro_shape(self):
        w = synthetic_checkpoint(gin=1024, gen_init=512, v2pro=True)
        w["ge_to512.weight"] = np.zeros((256, 1024), np.float32)
        self._expect_reject(w, synth_cfg(1024, 512))


# ---------------------------------------------------------------------------
# Optional real-checkpoint end-to-end control.
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_GGUF, "converter tests require gguf")
class RealCheckpointConversionTests(unittest.TestCase):
    def setUp(self):
        self.ckpt = Path(os.environ.get("SOVITS_V2PRO_CKPT", DEFAULT_V2PRO_CKPT))
        if not self.ckpt.is_file():
            self.skipTest(f"v2Pro checkpoint not present: {self.ckpt}")

    def test_v2pro_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "v2pro.gguf"
            with contextlib.redirect_stdout(io.StringIO()):
                conv.convert(str(self.ckpt), str(out), "f16")
            reader, f = _read_fields(out)
            self.assertTrue(f["sovits.v2pro"])
            self.assertEqual(f["sovits.ref_enc.out_dim"], 1024)
            self.assertEqual(f["sovits.flow.gin"], 1024)
            self.assertEqual(f["sovits.generator.gin_dim"], 1024)
            self.assertEqual(f["sovits.text_encoder.ge_dim"], 512)
            self.assertIn(f["sovits.version"], ("v2Pro", "v2ProPlus"))
            shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}
            self.assertEqual(shapes["v2pro.sv_emb_w"], (20480, 1024))
            self.assertEqual(shapes["v2pro.ge_to512_w"], (1024, 512))


if __name__ == "__main__":
    unittest.main()
