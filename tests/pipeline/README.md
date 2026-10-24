# CPU T2S → SoVITS bridge test

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
