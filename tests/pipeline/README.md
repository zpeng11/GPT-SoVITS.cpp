# CPU reference conditioning and T2S → SoVITS bridge

## Reference conditioning

`src/pipeline/reference_internal.h` provides an internal, two-stage CPU v2Pro
reference path. `prepare_reference` owns its output buffers and composes mono PCM
preparation, frontend BERT assembly and HuBERT extraction. `encode_reference`
borrows those features plus loaded T2S/SoVITS/SV models and fresh initialized
sessions, then computes T2S reference embeddings and SoVITS `ge`/`ge_enc`.
It does not load models, generate target codes, decode a waveform or write files.

The separation lets a later request runner learn `T_ref` and `T_prompt` before
allocating its T2S slot budget. The encoder checks that the reference alone fits
that slot; a later runner must also budget target text and generated codes.
Models and the common CPU backend must remain at the same addresses and outlive
both sessions. Release sessions before models and backend. One session pair is
used for one reference; cached sessions are rejected. After a compute failure,
dispose of both sessions rather than retrying on partially prepared state.

### HuBERT input policy

The verified Python v2Pro path passes **unnormalized** 16 kHz PCM directly to
HuBERT and appends **9600 zero samples**. That is 0.6 seconds at 16 kHz: the
upstream expression uses `32000 * 0.3`, despite the waveform being 16 kHz.
The pipeline reproduces that sample count, with no second normalization.
SV receives unnormalized, unpadded 16 kHz PCM; SoVITS receives the first 704 bins
of the unnormalized 32 kHz reference spectrum.

`audio_prepare_reference` remains unchanged, including its normalized output and
no-padding behavior. The pipeline does not consume `normalized_pcm16k`.
`hubert_extract_features_from_pcm` consumes caller-provided samples verbatim;
`hubert_extract_features_from_normalized_pcm` remains a compatibility entry point
for existing normalized-input callers.

C++ still uses its existing direct-source resampler. Python's HuBERT branch uses
librosa, while its SV branch resamples through 32 kHz with torchaudio. Therefore
raw-recording integration and exact-dump-input component comparison are separate
checks. Neither claims bitwise resampling parity. Existing audio preparation also
does not apply Python's optional reference gain adjustment for peaks above one.

### Local matched case

Keep recordings, transcriptions, model identities and dumps under
`tests/pipeline/out/`, which is already ignored by Git. Do not copy private inputs
into tracked fixtures. A capture contains `manifest.json`, original `ref.wav`,
reference/target frontend JSON, BERT/phones, actual HuBERT/SV inputs and outputs,
conditioning tensors, generated codes, noise and waveforms from one run.
The latter target/decoder artifacts are retained for the next synthesis milestone,
but are not consumed by this reference-only delivery.

Configure a case explicitly (leave the setting empty to skip optional local
integration tests):

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON \
  -DGPTSOVITS_REFERENCE_CASE=/absolute/path/to/matched-case
cmake --build build --target gpt_sovits_cpp_static gpt_sovits_cpp_shared \
  test_reference test_reference_integration test_hubert_extract -j 4
ctest --test-dir build -R '^(Reference|HubertExtract|HubertOutputFrames)' \
  --output-on-failure
```

When a case is configured, missing/invalid files or models are failures, not
skips. Tests force the CPU backend even in a Metal-enabled build. Component
comparisons repack PyTorch `[1,C,T]` / `[C,T]` arrays into ggml `{C,T}` with
`out[t*C+c]`, and report numerical errors separately from raw-audio integration.
The recording path checks finite caches and dimensions and agrees with explicit
calls using the same prepared features. It never invokes target generation or
waveform forward.

For the local CPU FP32 capture used during development, all five source model
groups and the tokenizer matched their local counterparts by SHA-256. GGUF
weights were also checked against the converters' mappings. The distributed
checkpoints themselves store F16 weights; Python loads them into F32 execution.
Thus F16 GGUF storage does not imply a new weight quantization relative to these
checkpoints; compute/accumulation paths can still differ. Numerical comparisons
must use component-specific tolerances, not a blanket waveform tolerance.

The matched recording also exposed a HuBERT activation mismatch: `ggml_gelu`
uses an approximation, while the checkpoint's `gelu` activation is the exact
error-function form. The HuBERT graph now uses `ggml_gelu_erf` in its convolution,
positional and feed-forward paths. With the exact dumped 273-frame input, this
reduced peak error from about `1.94e-2` to `1.16e-5`, without relaxing a tolerance.
The matched-case F32 check requires peak error below `1e-4` and RMSE below `1e-5`.
It also compares the cached prompt embedding with an independent construction
from the dumped prompt codes and T2S embedding/position weights, so a changed
nearest-code assignment cannot be hidden by a small aggregate feature error.

Observed errors for the matched local CPU case (exact component inputs):

| Output | Maximum absolute error | RMSE |
| --- | ---: | ---: |
| Reference BERT | 1.55e-2 | 1.74e-3 |
| HuBERT | 1.16e-5 | 3.63e-7 |
| SV embedding | 6.38e-5 | 6.27e-6 |
| ge | 5.92e-2 | 1.08e-2 |
| ge_enc | 4.66e-2 | 1.52e-2 |
| Prompt embedding from recorded codes | 1.91e-6 | 1.13e-7 |

These are component numerical checks and reference-cache integration, not a
listening test or a claim of full synthesized-waveform parity.

## Prepared-feature bridge test

`T2SSoVITSBridge.CpuF16ProducesWaveform` composes the existing public APIs:

```text
prepared reference/target features
  → t2s_session_compute_ref_emb
  → t2s_session_generate
  → sovits_session_compute_ge
  → sovits_session_forward
```

It passes the complete generated codes and the same target phonemes to SoVITS,
then checks output length, finite samples, non-degeneracy and a trailing buffer
guard. It runs on CPU even in a Metal-enabled build: the pinned Metal
`CONV_TRANSPOSE_1D` kernel can hang during full decoder inference.

## Assets

Run the conversion commands from the repository root, using local checkpoints:

```sh
mkdir -p tests/t2s/models tests/sovits/models
python3 python/convert_t2s_to_gguf.py /path/to/s1v3.ckpt /path/to/s2Gv2Pro.pth \
  --output tests/t2s/models/s1v3-s2Gv2Pro-f16.gguf --type f16
python3 python/convert_sovits_to_gguf.py /path/to/s2Gv2Pro.pth \
  --output tests/sovits/models/v2pro-sovits-f16.gguf --type f16
```

Use the **same `s2Gv2Pro.pth` checkpoint** for both conversions. The test checks
that the T2S extract-latent codebook, rounded to F16, exactly matches the SoVITS
F16 codebook. GGUF files are local assets ignored by Git; no models are downloaded
by the test.

The test reuses two tracked arrays, described in
[the SoVITS reference provenance notes](../sovits/README_V2PRO.md#reference-provenance-and-limits):

- `tests/sovits/ref/v2pro_refer.npy`: F16 `[1,1025,243]` BCT spectrum. Its first
  704 channels are repacked as channel-fast `spec[t * 704 + c]`.
- `tests/sovits/ref/v2pro_sv_emb.npy`: F16 `[1,20480]` speaker embedding,
  converted to F32 when loaded.

The T2S reference/target phones, BERT features and HuBERT features are short
synthetic inputs matching the existing T2S generation integration baseline.
Sampling uses `top_k=1`, seed 42 and at most five new codes. SoVITS uses noise
scale 0.5 and seed 7. The test does not truncate the generated codes; at 25 Hz,
its waveform contains `codes.size() * 2 * 640` samples (at most 6400).

## Run

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_t2s_sovits_bridge -j 4
ctest --test-dir build -R '^T2SSoVITSBridge\.' --output-on-failure
```

Related CPU regressions:

```sh
cmake --build build --target test_t2s_generate test_t2s_generate_integration \
  test_sovits_v2pro_parity -j 4
ctest --test-dir build \
  -R '^(T2SGenerateSynthetic\.|T2SGenerateIntegration\.Cpu)|SessionForwardShortCodes' \
  --output-on-failure
```

Missing local GGUF models cause an explicit skip with the missing paths. Missing
tracked reference arrays, invalid layouts, failed loads, incompatible models,
empty generated codes and invalid waveform output are failures. A skip is not
successful bridge validation; with the assets prepared, the case must actually
pass. Asset paths are supplied by CMake and do not depend on the working directory.

## Limits

This is a prepared-feature integration smoke/contract test, not raw-text/audio
end-to-end synthesis. The synthetic T2S features and the SoVITS reference arrays
do not describe the same recording, so the output is not an audio-quality test.
It does not run HuBERT, RoBERTa or SV inference, invoke the Python frontend, write
WAV files, change sampling semantics, or claim PyTorch waveform parity. It only
runs one SoVITS forward; the session RNG advances between calls, and matching a
seed does not establish cross-platform or cross-language waveform identity.
