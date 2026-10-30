# CLI: single-request offline CPU synthesis

`gpt_sovits_cli` turns the public `synthesize_cpu` API
(`include/gpt_sovits/pipeline.h`) into a command-line program: it loads the
five v2Pro GGUF models onto one CPU backend, reads the reference audio and two
text-frontend JSON files, and writes the raw 32 kHz mono F32 result as an
IEEE-float WAV. The full synthesis contract (single reference, no streaming,
no output normalization) is the library's contract. Repetition history includes
reference prompt codes plus accepted generated codes; repeated IDs are penalized
once. The EOS window (`--min-new-tokens`) approximates upstream early-EOS
suppression; duration-based stopping and RNG remain different, so this is not
full upstream inference parity. See
[tests/pipeline/README.md](../tests/pipeline/README.md) for the
verified boundaries.

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
  `--max-new-tokens` (200), `--min-new-tokens` (11), `--seed` (42),
  `--noise-seed` (7), `--noise-scale` (0.5), `--top-k` (15), `--top-p` (1.0),
  `--temperature` (1.0), `--repetition-penalty` (1.35).
- `--min-new-tokens N` (`N` in `[0, 2147483647]`) excludes EOS for the first
  `N` generation steps; step 0 is the prediction from prefill. `0` disables the
  window. It may exceed `--max-new-tokens`, which stays a hard cap: the length
  limit then wins and EOS is never emitted.
- `--overwrite` is required to replace an existing `--output` file; the CLI
  never writes partial output on failure.

Exit codes: 0 success, 1 runtime failure (bad file, load or synthesis error),
2 usage error. On success a summary line reports the generated code count, the
stop reason (`eos` = natural end, `max_new_tokens` = possibly truncated audio),
the sample count and the duration. With `--min-new-tokens` above the length cap,
`max_new_tokens` is the only possible stop reason.

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

### No-model smoke tests

`tests/cli/check_cli.cmake` plus the `cli_help`, `cli_missing_required_args`,
`cli_bad_input_path` and `cli_loader_failure_keeps_output_absent` ctest entries
cover `--help`, missing required options, unreadable inputs, and a loader
failure that must not create the output file. The `cli_min_new_tokens_*` entries
check the EOS-window option without models: `0` and a window larger than the
length cap (`--max-new-tokens 1 --min-new-tokens 11`) parse and reach the early
input-path check (exit 1), while negative, non-integer and overflowing values are
usage errors (exit 2). None may create the output file. They run with
`-DGPTSOVITS_BUILD_TESTS=ON` and need no model assets.

### Local CPU synthesis integration test

`cli_synthesis_e2e` starts the real CLI with a v2Pro model pair and a local
reference case. It checks successful exit, IEEE-float32/mono/32 kHz WAV format,
complete finite non-degenerate samples, and agreement between the WAV and the
CLI's code count, stop reason and sample count (`samples == codes * 1280`). It
then repeats the command without `--overwrite`, requiring exit code 1, the
existing-output diagnostic and byte-for-byte preservation of the first WAV.
Only the first command performs synthesis.

Place the same local models used by the pipeline integration tests at:

- `tests/hubert/models/chinese-hubert-base-f32.gguf`
- `tests/roberta/chinese-roberta-wwm-ext-large-f16.gguf`
- `tests/t2s/models/s1v3-s2Gv2Pro-f16.gguf`
- `tests/sovits/models/v2pro-sovits-f16.gguf`
- `tests/sovits/models/sv-eres2netv2-f32.gguf`

The case directory needs only `ref.wav`, `ref_frontend.json` and
`target_frontend.json`; no Python process or `.npy` dumps are needed. Both JSONs
must satisfy the CLI's v2 frontend contract, and the request must fit the
1024-position budget with 100 new tokens. See the
[pipeline local-case instructions](../tests/pipeline/README.md#local-matched-case)
for preparing inputs. Keep private recordings, models and generated files out
of Git.

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON -DGPTSOVITS_BUILD_CLI=ON \
    -DGPTSOVITS_REFERENCE_CASE=/absolute/path/to/local-case
# Also builds the CLI through an explicit target dependency.
cmake --build build --target cli_synthesis_check -j 4
ctest --test-dir build -R '^cli_synthesis_e2e$' --output-on-failure
```

The test uses greedy top-k 1, a 100-code limit, semantic seed 42, noise seed 7
and noise scale 0.5. Either EOS or a correctly reported length limit is accepted;
it does not require a fixed code count or Python waveform equality. These checks
verify the user entry point and basic output validity, not listening quality,
intelligibility or numerical parity with Python.

An empty `GPTSOVITS_REFERENCE_CASE` reports **Skipped**. Once configured, missing
or invalid assets fail the test rather than skipping. With CLI building disabled,
the test and its checker target are not registered. The test has the `cpu` and
`integration` labels and a 900-second timeout (600 seconds for synthesis).
Each run uses its own directory under `build/tests/cli-synthesis/`, including an
output filename with spaces. Successful runs clean up their directory; failed
runs retain stdout/stderr logs and any output there, and report the directory
in the failure message.
