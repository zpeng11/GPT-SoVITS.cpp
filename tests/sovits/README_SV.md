# Standalone SV (speaker-verification) tool

This directory holds the model artifacts, reference fixtures and notes for the
**standalone** ERes2NetV2 speaker-verification (SV) module. It is self-contained:
a GGUF weight container plus three C++ entry points (`include/gpt_sovits/sv.h`):

- `sv_model_load` — load the weights onto a backend and pre-fold BatchNorm;
- `sv_fbank` — Kaldi-style 80-mel frontend (16 kHz audio → features);
- `sv_forward3` — features → 20480-d reference embedding.

## Components

| Path | Role |
| --- | --- |
| `python/convert_sv_to_gguf.py` | Checkpoint → GGUF converter (torch-free) |
| `python/dump_sv_ref.py` | Developer fixture generator (runs the real model; needs torch) |
| `include/gpt_sovits/sv.h`, `src/sv/sv.cpp` | C++ runtime (`sv_model_load`, `sv_fbank`, `sv_forward3`) |
| `tests/sovits/test_sv_parity.cpp` | `sv_fbank` / `sv_forward3` parity and contract tests |
| `tests/sovits/test_sv_loader.cpp` | `sv_model_load` loader tests |
| `tests/sovits/models/sv-eres2netv2-f32.gguf` | Converted model (gitignored; generate locally) |
| `tests/sovits/ref/sv_*.npy` | Reference arrays (see "Fixture strategy") |

## Supported architecture

Fixed: `ERes2NetV2(baseWidth=24, scale=4, expansion=4)`, `feat_dim=80`,
`out_dim=20480` — i.e. the upstream `pretrained_eres2netv2w24s4ep4` checkpoint.
Only the **forward3** path is implemented (`conv1+bn1`, `layer1..4`,
`layer3_ds`, `fuse34`); the `seg_*` embedding head and `num_batches_tracked`
buffers are dropped. This is the only SV architecture implemented by the
converter and the C++ runtime, so neither script offers architecture options.

## Dependencies

- **Converter** (`convert_sv_to_gguf.py`): `gguf` + NumPy and the stdlib only
  (via `python/torch_ckpt_utils.py`). Deliberately torch-free.
- **Fixture generator** (`dump_sv_ref.py`): requires `torch` (it runs the real
  model). `torchaudio` is not used — the Kaldi frontend is inlined in pure
  torch, replicating the `kaldi.fbank` defaults (`dither=0`, 80 mel, 25 ms /
  10 ms, povey window, preemphasis 0.97, round-to-pow2 → 512 FFT, power
  spectrum, log, snip_edges) exactly as `SV.compute_embedding3` calls it.

## Upstream sources (parameterized, not hard-coded)

- ERes2Net package dir: `<GPT_SoVITS>/eres2net` (`ERes2NetV2.py`, `fusion.py`,
  `pooling_layers.py`, `kaldi.py`).
- Checkpoint: `<GPT_SoVITS>/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`.
  A working copy used by the tests lives at
  `tests/t2s/models/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`.

## Commands

### 1. Convert the model to GGUF (torch-free)

The model GGUF is not tracked (`.gitignore` ignores `*.gguf`), so generate it
locally before running the tests:

```sh
python python/convert_sv_to_gguf.py \
    tests/t2s/models/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
    -o tests/sovits/models/sv-eres2netv2-f32.gguf
```

Produces 842 F32 tensors (forward3 path).

### 2. Generate the reference fixtures (needs torch)

```sh
python python/dump_sv_ref.py \
    --eres2net-dir /path/to/GPT_SoVITS/eres2net \
    --ckpt /path/to/pretrained_eres2netv2w24s4ep4.ckpt \
    --out-dir tests/sovits/ref --stress
```

`--eres2net-dir` and `--ckpt` may instead come from the environment:
`GPT_SOVITS_ERES2NET_DIR` (or `GPT_SOVITS_ROOT`, which implies
`<root>/GPT_SoVITS/eres2net`) and `GPT_SOVITS_SV_CKPT`. `--out-dir` defaults to
`tests/sovits/ref`. Regeneration is deterministic in a fixed environment, so it
can be re-run to confirm the fixtures.

Options:

- default (no flags): `sv_audio16k.npy`, `sv_fbank.npy`, `sv_emb.npy`.
- `--save-intermediates`: additionally `sv_fuse34.npy`, `sv_out1..4.npy`.
- `--stress`: additionally `sv_stress_fbank.npy`, `sv_stress_emb.npy`.

### 3. Build and run the tests

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_sv_parity test_sv_loader -j 4
ctest --test-dir build -R '^(SVParity|SVLoader)\.' --output-on-failure
```

Or run a binary directly (GTest):

```sh
./build/bin/test_sv_parity
./build/bin/test_sv_loader
SV_TEST_CPU=1 ./build/bin/test_sv_parity  # force CPU instead of preferring Metal
```

The gtest_discover_tests names are `<gtest suite>.<test>`. Test binaries are
named after their source files (`test_sv_parity.cpp` → `test_sv_parity`,
`test_sv_loader.cpp` → `test_sv_loader`), and suite names are whatever the test
sources register — the test sources are the source of truth if a name differs.
The fbank-contract tests currently live in the `SVParity` suite.

## Fixture strategy

The fixtures split by origin:

- **Generated directly by `dump_sv_ref.py`** (deterministic synthesis, not a
  model run): `sv_audio16k.npy` (16 kHz mono waveform) and `sv_fbank.npy`
  (inlined Kaldi fbank of that waveform).
- **Outputs of the real upstream `ERes2NetV2.forward3`**: `sv_emb.npy`,
  `sv_stress_emb.npy`. The optional intermediates (`sv_fuse34.npy`,
  `sv_out1..4.npy`) come from the same loaded model's forward3 sub-steps.

Keep the five consumed fixtures (audio, fbank, embedding, and the stress pair)
with the tests. The large intermediate arrays are optional debug outputs and
are not needed for regression tests. The model GGUF is ignored by `.gitignore`.
Without a local GGUF the model parity tests explicitly skip; fbank and synthetic
loader tests still run. Synthetic loader tests use temporary full-shape GGUFs
(about 174 MiB each), remove them on exit, and need no pretrained checkpoint.

| Fixture | Shape | Purpose |
| --- | --- | --- |
| `sv_audio16k.npy` | `{48000}` | Deterministic 16 kHz mono waveform (seed 0, 3 s, three tones + seeded noise). Fed to the C++ `sv_fbank` so the frontend check uses identical samples. |
| `sv_fbank.npy` | `{298, 80}` | Inlined `kaldi.fbank` of `sv_audio16k`. Input to the forward3 check. |
| `sv_emb.npy` | `{1, 20480}` | `forward3(sv_fbank)` reference. |
| `sv_fuse34.npy` | `{1, 2048, 10, 38}` | `fuse34` output before flatten+mean (debug; `--save-intermediates`). |
| `sv_out1..4.npy` | `{1,256,80,298}` / `{1,512,40,149}` / `{1,1024,20,75}` / `{1,2048,10,38}` | Per-layer outputs for block-level checks (debug; `--save-intermediates`). |
| `sv_stress_fbank.npy` | `{98, 80}` | Stress input (see below). |
| `sv_stress_emb.npy` | `{1, 20480}` | `forward3(sv_stress_fbank)` reference. |

Consumers: `SVParity.Fbank` (`sv_fbank(sv_audio16k)` vs `sv_fbank.npy`;
`max_abs < 1e-3`, `rmse < 2e-4`), `SVParity.Forward3` (`sv_forward3(sv_fbank)`
vs `sv_emb.npy`; `max_abs < 1e-2`, `rmse < 3e-3`), and
`SVParity.Forward3Stress` (same tolerances as `Forward3`).

## Stress fixture and the `Hardtanh(0, 20)` boundary

`forward3`'s first op is `F.relu(bn1(conv1(x)))` — a plain ReLU. The residual
blocks use `ReLU = Hardtanh(0, 20)`, which clamps at 20. The C++ graph mirrors
this (`ggml_relu` for the first layer, clamped ReLU inside the blocks), so a
mis-applied upper clamp on the first layer would only show up on an input whose
first-layer activation exceeds 20. The primary `sv_fbank` fixture stays below
that bound (first-layer activation max ≈ 12.6), which is why this dedicated
stress case exists.

- Construction: `sv_stress_fbank.npy` is `fbank(seed=7, 1.0 s)` scaled by a
  fixed `gain = 4.0`; `sv_stress_emb.npy` is `forward3` of it. To keep the
  trigger reliable the script scales the fbank directly (the fbank is the model
  input, so this deterministically raises the first-layer response).
- Measured: first-layer activation max ≈ 30.6, with 57 / 501760 elements above
  20; the reference embedding is finite.
- If the first layer were clamped at 20 (the older bug), the embedding would
  differ from the reference by `max ≈ 0.16` (≈ 0.0055 mean) — ~15× the `Forward3`
  `max_abs < 1e-2` threshold, so the check discriminates the bug while leaving
  headroom for backend rounding on a correct implementation.
- Sizes: `sv_stress_fbank.npy` 31 KB, `sv_stress_emb.npy` 80 KB (the 20480-d
  embedding dominates and is fixed by the architecture).
