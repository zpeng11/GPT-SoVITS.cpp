# CLI: single-request offline CPU synthesis

`gpt_sovits_cli` turns the public `synthesize_cpu` API
(`include/gpt_sovits/pipeline.h`) into a command-line program: it loads the
five v2Pro GGUF models onto one CPU backend, reads the reference audio and two
text-frontend JSON files, and writes the raw 32 kHz mono F32 result as an
IEEE-float WAV. The full synthesis contract (single reference, no streaming,
sampler semantics that deliberately differ from the Python `infer_panel`, no
output normalization) is the library's contract; see
[tests/pipeline/README.md](../tests/pipeline/README.md) for its verified
boundaries.

Build (independent, without tests):

```sh
cmake -S . -B build-cli -DGPTSOVITS_BUILD_CLI=ON -DGPTSOVITS_BUILD_TESTS=OFF
cmake --build build-cli --target gpt_sovits_cli -j
# -> build-cli/bin/gpt_sovits_cli
```

`GPTSOVITS_BUILD_CLI` defaults to ON and only needs `gpt_sovits_cpp_static`.

## Usage

```sh
build-cli/bin/gpt_sovits_cli \
    --hubert  models/chinese-hubert-base-f32.gguf \
    --roberta models/chinese-roberta-wwm-ext-large-f16.gguf \
    --t2s     models/s1v3-s2Gv2Pro-f16.gguf \
    --sovits  models/v2pro-sovits-f16.gguf \
    --sv      models/sv-eres2netv2-f32.gguf \
    --ref-audio ref.wav \
    --ref-frontend-json ref.json \
    --target-frontend-json target.json \
    --output out.wav
```

- `--roberta` is optional: it is only needed when either frontend JSON
  contains Chinese segments (pure-English requests work without it).
- Optional synthesis knobs, with the library defaults:
  `--max-new-tokens` (200), `--seed` (42), `--noise-seed` (7),
  `--noise-scale` (0.5), `--top-k` (15), `--top-p` (1.0),
  `--temperature` (1.0), `--repetition-penalty` (1.35).
- `--overwrite` is required to replace an existing `--output` file; the CLI
  never writes partial output on failure.

Exit codes: 0 success, 1 runtime failure (bad file, load or synthesis error),
2 usage error. On success a summary line reports the generated code count, the
stop reason (`eos` = natural end, `max_new_tokens` = possibly truncated audio),
the sample count and the duration.

## Producing the frontend JSON files

The CLI does not process raw text. Both JSON files come from the torch-free
Python frontend (`python/text_frontend`, schema `gpt-sovits-text-frontend/v1`):

```sh
PYTHONPATH=python python3 -m text_frontend \
    --text "请大家相信我说的话，都是真的。" --language zh \
    --tokenizer /path/to/chinese-roberta-wwm-ext-large/tokenizer.json \
    --json target.json --quiet
```

- Chinese needs `pypinyin` plus the RoBERTa `tokenizer.json`
  (`python/text_frontend/requirements.txt`); pass `--tokenizer` so the C++ side
  can compute BERT features.
- English works without a tokenizer via the dependency-free basic backend:
  add `--language en --english-backend basic`.
- Convert checkpoints to GGUF with `python/convert_*_to_gguf.py`.

## Reproducibility

With identical inputs, options, build and backend, repeated runs produce
identical output (`--seed` drives T2S sampling, `--noise-seed` the SoVITS
noise). Output is not expected to match Python-generated audio sample-for-
sample; the known numeric relationship is documented in
[tests/pipeline/README.md](../tests/pipeline/README.md).

## Tests

`tests/cli/check_cli.cmake` plus the `cli_*` ctest entries cover the no-model
paths: `--help`, missing required options, unreadable inputs, and a loader
failure that must not create the output file. They run with
`-DGPTSOVITS_BUILD_TESTS=ON` and need no model assets.
