# T2S teacher-forced CPU decode reference

## No-model sampler contract tests

`test_t2s_sampler` exercises the public `t2s_sampler_block_forward` on a CPU
backend with an identity LM-head projection, explicit logits and fixed positive
noise. It needs no GGUF, private reference files, PyTorch, or Python process.

The top-p rule follows `AR.models.utils.logits_to_probs`, used by `sample()`:
after repetition penalty, sort logits and compute cumulative softmax probability;
remove ranks whose cumulative probability is **strictly greater** than `top_p`,
but always retain rank 0. The first token crossing the threshold is not retained
unless it is rank 0. A cumulative probability exactly equal to the threshold
remains eligible. Temperature and top-k follow this filtering; `top_p >= 1`
disables it. For probabilities `[0.5, 0.3, 0.2]` and `top_p=0.7`, only the first
token remains. This is not the shifted-mask rule of upstream's separate
`top_k_top_p_filtering` helper.

Top-k uses the k-th largest logit as its pivot and removes only logits **strictly
below** it. Candidates tied with the pivot remain eligible, so more than k
candidates may survive, even when k is 1. `top_k <= 0` disables this filtering;
`top_k >= vocab` does not truncate the vocabulary. Top-k follows top-p filtering
and temperature scaling, matching the corresponding `logits_to_probs` cutoff.

Tests cover the crossing token, exact and zero thresholds, at least one candidate,
a single-token vocabulary, disabled filtering, the default top-k 15, top-k cutoff
inclusion and exclusion, pivot ties, k at or above vocabulary size, temperature
and top-k ordering, repetition penalty, token-ID remapping, EOS prefix suppression
and independent batch columns. Handwritten cases supply fixed sorted-rank noise;
the upstream fixture cases below adapt vocabulary-indexed noise only in the test.
Neither changes production noise ordering or claims cross-language RNG parity or
full stochastic-sequence parity.
The prior implementation already used this top-p rule; its right-shift comment
and variable name were misleading. No production sampling behavior is changed.

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_t2s_sampler -j 4
ctest --test-dir build -R '^T2SSampler' --output-on-failure
```

## Reproducible upstream sampler fixtures

`generate_sampler_reference.py` exports one self-contained JSON fixture from the
real upstream `AR.models.utils.logits_to_probs`, loaded directly from source under
`--upstream-root` (no package import, and existing bytecode caches are ignored).
Exporting needs PyTorch and NumPy; `--validate` and `--help` need only NumPy and
the standard library, and neither mode imports the upstream, PyTorch, the network,
nor installs anything. The fixture is synthetic public data, unlike private
model-backed exports under `tests/*/out/`.

```sh
python3 -I tests/t2s/generate_sampler_reference.py \
  --upstream-root /path/to/trusted/GPT-SoVITS \
  --output tests/t2s/out/sampler-a.json
python3 -I tests/t2s/generate_sampler_reference.py \
  --upstream-root /path/to/trusted/GPT-SoVITS \
  --output tests/t2s/out/sampler-b.json
cmp tests/t2s/out/sampler-a.json tests/t2s/out/sampler-b.json
python3 -I tests/t2s/generate_sampler_reference.py --validate tests/t2s/out/sampler-a.json
```

The output file must not already exist. One file with schema
`gpt-sovits-t2s-sampler-reference/v1` holds a fixed synthetic catalog: `logits`,
`previous_tokens`, `parameters` (including `suppress_eos`) and vocabulary-indexed
fixed positive `noise`, so cases do not depend on RNG draws. `expected` records the
upstream `adjusted_logits` (only over the valid vocabulary prefix, or `V-1` when
EOS is suppressed), full-vocabulary `probabilities` (EOS set to 0 when
suppressed), the `greedy_candidates` set, `sampled`, and `relative_margin`. The
clone inputs reproduce upstream's in-place repetition penalty; `top_k <= 0` maps
to `None`.

EOS suppression is an outer prefix adapter applied before the upstream call, not
an upstream function parameter or generation window. Execution is CPU/F32,
single-threaded and deterministic; numeric probabilities come only from upstream.
`sampled` is replayed from the upstream formula using `probabilities` and the
fixed noise, not a fresh random draw.

Upstream returns probabilities and consumes noise in vocabulary order, while the
C++ sampler consumes noise in sorted-rank order. Its test-only adapter therefore
remaps through CPU ggml sorting, as described below. The fixture does not make
same-seed noise equal across languages and is not a cross-language RNG oracle. Provenance records exporter/source hashes and environment versions,
with no paths, timestamps or private content. A canonical SHA-256 guards content
integrity; it is not a signature or numerical proof. Byte-identical exports are
promised only for the same source, tool and environment, not across versions or
machines. C++ scales positive-score repetition with an F32 reciprocal where
upstream divides, so bitwise agreement is not claimed.

The catalog covers the default top-k 15 with repetition penalty 1.35 on a
20-token vocabulary, top-p ordering and temperature, signed repetition penalties
with repeated history, disabled top-k and pivot ties, and EOS pairs. Fixture
tests use only NumPy and the standard library:

```sh
python3 -I -m unittest discover -s python/tests -p test_t2s_sampler_reference.py -v
```

They check the tracked `fixtures/sampler_reference.json` catalog, provenance and
content digests, probability normalization, greedy candidate sets and fixed-noise
replay, plus selected analytic probabilities and policy relationships. The
validator does not reimplement the full sampler or establish numerical truth for
all outputs. Only actual upstream exports author the fixture. Optional PyTorch
tests use an explicit sampler test double for parameter adaptation and CLI
plumbing, not as a real oracle; computation tests skip when PyTorch is absent.

### C++ fixture decision comparison

`T2SSamplerUpstream` in `test_t2s_sampler` runs all 19 tracked cases on the CPU
through the public sampler with the original logits and binary history. Every
sampled token must exactly match the fixture; greedy must belong to its recorded
maximum-score candidate set. The fixture is required, not optional: missing,
malformed or incompatible data fails the test rather than skipping. It is read
at test runtime, not during GoogleTest discovery. The existing build and CTest
commands above include these cases and the reader's invalid-data contract test.
No model, Python process, upstream checkout, network or extra configuration is
needed to run them.

A test-only mapping graph reproduces the C++ F32 repetition-penalty arithmetic
and CPU argsort on the active vocabulary, then reorders the fixed noise by those
token IDs. Matching the same input, shape and local argsort handles ties without
assuming a portable tie order. The adapter does not implement top-p, top-k or
softmax and never computes the expected token. Suppressed-EOS cases sort only the
non-EOS prefix for noise mapping but still exercise the full public sampler with
its suppression flag set. Production noise ordering and APIs are unchanged.

This is a token-decision comparison for these synthetic cases, not a direct
comparison of the full probability arrays: the public sampler returns only
sampled and greedy IDs. It does not establish stochastic-sequence parity,
bitwise probability agreement or cross-language RNG parity. Full fixture digest
and provenance validation remains in the separate Python validator above; the
C++ reader checks the schema, semantics, complete case names and consumed data.

The following sections describe the separate model-backed decode and generation
references.

`generate_reference.py` exports an independent upstream PyTorch reference for
multi-step C++ decode tests. It runs **one unpadded request on CPU in F32**, using
precomputed text features and semantic tokens. It does not run the text frontend,
HuBERT, a sampler, SoVITS, or audio IO. This tool does not change production
inference or add a Python dependency to the library, CLI, CMake, or CTest.

## Export

Use Python 3.10+ with NumPy, PyTorch, and the dependencies required to import the
local upstream `AR.models.t2s_model.Text2SemanticDecoder` (including
`torchmetrics` and `tqdm` in the current upstream). Nothing is installed or
downloaded automatically. `--upstream-root` imports and executes Python code:
point it only at a **trusted** GPT-SoVITS checkout, not an untrusted artifact.
Either the checkout root or its `GPT_SoVITS/` directory is accepted.

From the repository root:

```sh
python3 tests/t2s/generate_reference.py \
  --checkpoint /path/to/s1v3.ckpt \
  --upstream-root /path/to/GPT-SoVITS \
  --inputs-dir /path/to/precomputed-case \
  --output-dir tests/t2s/out/decode-reference-1 \
  --steps 8

# A second new directory, with identical source files and execution environment:
python3 tests/t2s/generate_reference.py \
  --checkpoint /path/to/s1v3.ckpt \
  --upstream-root /path/to/GPT-SoVITS \
  --inputs-dir /path/to/precomputed-case \
  --output-dir tests/t2s/out/decode-reference-2 \
  --steps 8
diff -r tests/t2s/out/decode-reference-1 tests/t2s/out/decode-reference-2
```

Use the **same source checkpoint** as the C++ T2S GGUF. The checkpoint must use
this repository's supported ZIP serialization, `model.*` weight names, and a
`config.model` matching the upstream architecture. It is read with the existing
restricted `python/torch_ckpt_utils.py` loader, then strictly loaded into the
upstream model. No checkpoint pickle is executed by `torch.load`.

The source directory must contain these `.npy` files, as in an inference dump:

| File | Accepted shape | Dtype |
| --- | --- | --- |
| `t2s_all_phones.npy` | `[T]` or `[1,T]` | int32 or int64 |
| `t2s_all_bert.npy` | `[1024,T]` or `[1,1024,T]` | float32 |
| `t2s_prompt.npy` | `[P]` or `[1,P]` | int32 or int64 |
| `generated_codes.npy` | `[D]`, `[1,D]`, or `[1,1,D]` | int32 or int64 |

`T` includes reference and target phones in their original concatenated order;
`P` is the reference audio prompt. Both must be nonempty. Phones must fit the
model vocabulary; prompt and generated codes must be nonnegative semantic IDs
**below EOS**. BERT must be finite. Object arrays and symlink inputs are rejected.

`--steps S` (default 8, positive) explicitly selects the **first S** generated
codes as teacher-forced inputs; `D < S` fails rather than reducing S. The entire
source token array is range-checked, even beyond this prefix. The final context
`T + P + S` must fit 1024 positions. This is a decode-reference limit, not a
change to the public generation budget `N + max_new_tokens - 1`.

For a separate local export environment that reuses an existing PyTorch install
without modifying it, create a venv in the ignored output directory:

```sh
/path/to/python-with-torch -m venv --system-site-packages tests/t2s/out/reference-env
tests/t2s/out/reference-env/bin/python -m pip install 'torchmetrics==1.5.0'
# Use this venv's Python for the export command above.
```

This is an explicit developer setup step, not something the exporter or CTest
runs automatically. It may need network access to install packages. The local
validation used Python 3.11.15, PyTorch 2.11.0, NumPy 1.26.4, torchmetrics 1.5.0
and tqdm 4.67.3. Torchmetrics' NumPy constraint was resolved inside this venv;
the existing Python environment was not changed.

The output path must **not exist**, even as an empty directory, file, or broken
symlink. Arrays are staged and validated before publishing to a newly reserved
directory; `manifest.json` is published last. Recoverable publication failures
clean up this run's files. A process kill can leave an incomplete directory;
without the completion manifest it is not a valid export. Existing data is never
overwritten or removed. Publication uses exclusive hard links within the output
filesystem; a filesystem without hard-link support fails explicitly. Private
inputs and generated data from these model-backed exports belong under the
ignored `tests/*/out/` directories, not in tracked fixtures. The synthetic,
upstream-authored sampler fixture described above is public data and an explicit
exception to this rule.

## Forward semantics and determinism

The exporter calls the upstream embedding, positional encoding,
`t2s_transformer.process_prompt`, `decode_next_token`, and `ar_predict_layer`
directly. It does not copy the transformer implementation. Text positions start
at zero over all T phones; audio positions start independently at zero over the
prompt and continue through the forced tokens.

Prefill order is `[all text | prompt audio]`. Text attends bidirectionally to
text and never to audio; audio attends to all text and causally to audio. Decode
uses the growing KV cache with one forced input at each audio position `P+i`.

- Row 0 of `logits` predicts the next token **after the complete prompt**.
- Row `i+1` predicts the next token **after consuming `forced_tokens[i]`**.
- `argmax` is diagnostic only; it is never fed back into the model.
- Logits include every vocabulary entry, including EOS. There is no repetition
  penalty, EOS suppression, top-k/top-p, temperature, stochastic sampling or
  predicted-EOS early stop. Therefore there are always **S+1** output rows.

Execution is CPU/F32, `eval()`, `torch.inference_mode()`, one intra-op and one
inter-op thread, deterministic algorithms, and initialization seed 0. The
upstream's explicit manual attention path (`torch_sdpa=False`) avoids an implicit
SDPA implementation choice. The seed only controls initialization before all
checkpoint weights are loaded, not sampling. Model storage dtype may be F16;
execution and exported floating-point arrays are F32.

Identical files, tool version, dependencies, machine, and execution settings
should produce byte-identical exports. Equality across PyTorch/NumPy versions,
architectures, or numeric backends is **not promised**. No timestamp or absolute
input/source path is written to the manifest. Arrays still encode private model
inputs; hashes and removal of paths do not anonymize their content.

## Output schema and layout

`manifest.json` uses `gpt-sovits-t2s-decode-reference/v1`, status `complete`.
It records:

- Model dimensions, head/layer counts, vocabularies and EOS.
- Text, prompt, decode-step, prefill (`N=T+P`), and final (`N+S`) lengths.
- Checkpoint, original input files, exporter and checkpoint-loader SHA-256.
- Relative names and SHA-256 for the imported upstream `AR/*.py` source files,
  including local modifications; a Git commit alone would not identify them.
  These modules and the checkpoint loader execute directly from source, ignoring
  existing bytecode caches so a stale `.pyc` cannot invalidate provenance.
- Execution settings plus Python/NumPy/PyTorch versions and system/architecture.
- Each array's dtype, C-order shape, reversed `ggml_ne`, and file SHA-256.

Arrays are little-endian int32 or float32 `.npy` files, without object data.
`H=hidden_dim`, `L=n_layer`, `V=vocab_size`:

| Array | NumPy C-order shape | Meaning |
| --- | --- | --- |
| `phones.npy` | `[T]` | Complete text IDs |
| `bert.npy` | `[T,1024]` | Phone-major BERT, ggml `{1024,T}` |
| `prompt.npy` | `[P]` | Reference semantic IDs |
| `forced_tokens.npy` | `[S]` | Actual teacher-forced inputs |
| `prefill_input.npy` | `[N,H]` | Concatenated positioned embeddings |
| `prefill_hidden.npy` | `[N,H]` | Full prefill transformer output |
| `logits.npy` | `[S+1,V]` | Raw next-token logits for each step |
| `argmax.npy` | `[S+1]` | Raw-logit argmax, including possible EOS |
| `context_lengths.npy` | `[S+1]` | `N, N+1, ..., N+S` |
| `audio_positions.npy` | `[S]` | `P, P+1, ..., P+S-1` |
| `prefill_k.npy`, `prefill_v.npy` | `[L,N,H]` | Per-layer prefill caches |
| `final_k.npy`, `final_v.npy` | `[L,N+S,H]` | Caches after every forced input |

Hidden dimensions are contiguous: `hidden[t*H+c]` and
`cache[(layer*context+t)*H+c]`. No transpose is needed to upload an individual
layer to ggml `{H,context}`. Cache values are the upstream's un-split projected
K/V before splitting heads. The prefill prefix must remain exactly unchanged in
the final cache.

## Validate without PyTorch

```sh
python3 tests/t2s/generate_reference.py --validate tests/t2s/out/decode-reference-1
python3 tests/t2s/generate_reference.py --help
```

These modes import only NumPy and the standard library. Validation checks the
schema and exact file list, source/output digest syntax, output file hashes,
shape/dtype/C-order, finite values, token ranges, argmax, positions, context
lengths and KV prefix preservation. It rejects unexpected files and symlink
artifacts. It cannot establish the identity of source files that are no longer
present, nor does it rerun the model: SHA-256 is an integrity/provenance aid, not
a signature or proof of numerical correctness.

## Tool tests and scope

```sh
python3 -m unittest discover -s python/tests -p test_t2s_reference_export.py -v
```

The NumPy/standard-library tests need no upstream or models. PyTorch-enabled
runs additionally exercise tiny deterministic test models: incremental outputs
are compared to a full causal forward, and the actual CLI loads a small
checkpoint through a temporary upstream test double. These tests write only to
temporary directories and do not download resources. PyTorch tests explicitly
skip if PyTorch is unavailable.

The exporter and its own tests do not establish C++ numerical parity. The
optional C++ comparison below supplies that separate check. Neither suite claims
sampled-sequence equality, waveform parity, intelligibility, or listening quality.
Existing tracked `ref/*.npy` files are not overwritten. The separate matched-policy
generation regression at the end of this document checks closed-loop semantic
codes under an explicitly restricted policy, not the upstream UI's defaults.

## Optional C++ CPU decode comparison

Use the export above and an F32 GGUF converted from the **same T2S checkpoint**:

```sh
python3 python/convert_t2s_to_gguf.py /path/to/s1v3.ckpt /path/to/s2Gv2Pro.pth \
  --type f32 --output tests/t2s/out/t2s-f32.gguf
python3 tests/t2s/generate_reference.py --validate tests/t2s/out/decode-reference-1

cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON \
  -DGPTSOVITS_T2S_DECODE_REFERENCE="$PWD/tests/t2s/out/decode-reference-1" \
  -DGPTSOVITS_T2S_DECODE_MODEL="$PWD/tests/t2s/out/t2s-f32.gguf"
cmake --build build --target test_t2s_decode_reference_contract test_t2s_decode_reference -j 4
ctest --test-dir build -R '^T2SDecodeReference' --output-on-failure
```

Both cache variables default to empty. Without a reference directory the two
numerical tests report **Skipped**. Once a reference is configured, missing or
invalid reference/model files fail, as do incompatible model dimensions or
non-F32 weights. Contract tests do not need a reference directory or model and
always run. Numerical tests force CPU, batch 1 and F32 KV, have the `cpu` and
`integration` labels, and a 300-second timeout.

`GPTSOVITS_T2S_DECODE_MODEL` must name a model converted from the checkpoint
whose SHA-256 is in the manifest. Matching dimensions alone do not prove model
identity: the GGUF format currently does not retain the source checkpoint hash.
The converter also requires a SoVITS checkpoint to package extract-latent weights;
these tests do not execute HuBERT or extract-latent inference.

The C++ reader checks the consumed manifest/NPY layout, finite values, token
ranges, positions, lengths and cache-prefix consistency. It does not recompute
SHA-256 or import Python. Run the independent `--validate` command first to check
file digests. Neither reader executes arbitrary NumPy pickle data. The C++ reader
accepts only NPY v1/v2 `<f4`/`<i4` on little-endian hosts, with up to 1024 context
positions, hidden/embedding dimensions 4096, 128 layers, 512 heads, vocabularies
65536, 512 MiB per array and 1 GiB total; larger inputs fail before allocation.

The numerical checks isolate the transformer from reference/text/audio
preparation by uploading the exported `prefill_input` directly. The flex path
checks full prefill hidden states, prefill/final layer caches and raw next-token
logits at all S+1 steps. Logits are independently projected in the test from the
returned hidden states and F32 LM-head weights, not inferred from sampled IDs.
The persistent path uses a fresh session and reuses one persistent decode graph;
it checks every input position, context length, raw greedy choice (repetition
penalty disabled), and layer caches. Both paths feed the same recorded tokens,
not their own predictions, and never stop on a predicted EOS. This does not test
sampler parity, automatic generation stopping or the upstream audio resamplers.

### Observed CPU F32 baseline

The local matched case used 66 text phones, 136 prompt codes, eight forced
inputs, 24 layers, hidden dimension 512 and vocabulary 1025. Both numerical tests
actually ran (not skipped). Two independent upstream exports were byte-identical,
and all 295 T2S tensors in the F32 GGUF exactly matched the source checkpoint's
F32 values through the converter mapping. This establishes the local model/input
pairing; it does not make the private case a distributed fixture.

| Compared output | Observed maximum absolute error | Worst RMSE | Asserted max / RMSE |
| --- | --- | --- | --- |
| Full prefill hidden | `9.25e-5` | `1.31e-6` | `2e-4` / `1e-5` |
| Flex raw logits, all 9 rows | `2.77e-5` | `5.79e-6` | `1e-4` / `2e-5` |
| Per-layer prefill/final K | `2.39e-5` | `1.32e-6` | `1e-4` / `1e-5` |
| Per-layer prefill/final V | `1.60e-5` | `7.22e-7` | `1e-4` / `1e-5` |
| New decode K/V rows, both paths | `6.14e-6` | `1.45e-6` | `1e-4` / `1e-5` |

Greedy IDs agreed at every compared step. The smallest reference top-1/top-2
logit gap was about `0.114`, well above the observed numerical error; the tests
assert exact argmax agreement and print the gap on failure. Thresholds are
component-specific CPU F32 budgets, not tolerances for F16/quantized weights,
quantized KV, other backends or arbitrary model architectures.

## Matched-policy closed-loop generation

`--generate` exports a **separate** `gpt-sovits-t2s-generation-reference/v2`
reference using the same trusted upstream model and CPU/F32 execution. The
forward is upstream code; the small outer loop explicitly implements the C++
policy rather than calling `infer_panel` or replaying a UI-generated sequence.
It does not read `generated_codes.npy`.

The restricted policy is top-k 1, top-p 1, temperature 1, with a finite positive
F32 repetition penalty. Seen tokens are a binary set of **reference prompt codes
plus accepted generated codes**, initialized from the prompt before the first
prediction and reset for each request. Repeated IDs are penalized once, not once
per occurrence. A seen negative logit is multiplied by the penalty; a nonnegative
logit is scaled by its F32 reciprocal. The maximum
adjusted logit supplies the candidate. For the first `min_new_tokens` predictions
(default 11, including prefill as step 0), EOS is excluded before both greedy and
sampled selection. Step 11 is the first possible EOS decision with the default.
`--min-new-tokens 0` restores unmasked EOS. The value must be a nonnegative int32;
it may exceed `max_new_tokens`, which remains the hard cap. An allowed EOS is
tested before acceptance; a length-limit stop retains the last accepted code. Top-k 1 removes
cross-language random-number differences; exact ties are not a portable sampling
oracle and should be avoided when preparing cases.

Repetition history and the default first-11-prediction EOS suppression match
upstream's corresponding policy. Duration-based stopping and cross-language RNG
remain different; nonnegative logits still use C++'s precomputed F32 reciprocal.
The EOS check remains on the adjusted greedy token or sampled token before
acceptance. This is not full `infer_panel` or stochastic sampling parity. The new
default can change sequences that previously stopped early; set the minimum to
0 for the previous EOS policy. A repetition penalty of 1.0 disables the history
effect, not EOS suppression.

### Export and run

Use the same checkpoint/upstream/precomputed inputs as above. `--reference-phones`
is the number of reference phones at the start of all-phones; the rest must be
nonempty target text. There is no automatic split inferred from text or filenames.
Both modes require `N + max_new_tokens - 1 <= 1024`.

```sh
python3 tests/t2s/generate_reference.py --generate \
  --checkpoint /path/to/s1v3.ckpt --upstream-root /path/to/GPT-SoVITS \
  --inputs-dir /path/to/precomputed-case --reference-phones 44 \
  --max-new-tokens 200 --min-new-tokens 11 --repetition-penalty 1.0 \
  --output-dir tests/t2s/out/generation-reference/rp1
python3 tests/t2s/generate_reference.py --generate \
  --checkpoint /path/to/s1v3.ckpt --upstream-root /path/to/GPT-SoVITS \
  --inputs-dir /path/to/precomputed-case --reference-phones 44 \
  --max-new-tokens 200 --min-new-tokens 11 --repetition-penalty 1.35 \
  --output-dir tests/t2s/out/generation-reference/rp135
python3 tests/t2s/generate_reference.py --validate tests/t2s/out/generation-reference/rp1
python3 tests/t2s/generate_reference.py --validate tests/t2s/out/generation-reference/rp135

cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON \
  -DGPTSOVITS_T2S_GENERATION_REFERENCE="$PWD/tests/t2s/out/generation-reference" \
  -DGPTSOVITS_T2S_DECODE_MODEL="$PWD/tests/t2s/out/t2s-f32.gguf"
cmake --build build --target test_t2s_generation_reference_contract test_t2s_generation_reference -j 4
ctest --test-dir build -R '^T2SGenerationReference' --output-on-failure
python3 -m unittest discover -s python/tests -p test_t2s_generation_reference.py -v
```

Use the isolated environment described above for exporting if needed; `--validate`
and NumPy-only tool tests need neither torch nor upstream imports. `--generate`
cannot be combined with `--steps`; generation policy flags are rejected without
`--generate`. Existing teacher-forced commands and schema remain unchanged.
Generation v2 manifests require `policy.min_new_tokens`,
`policy.eos == "suppressed-before-min-new-tokens"` and
`policy.history == "prompt+generated-binary"`. Old v1/unmasked or generated-only
exports are rejected and must be re-exported to a new directory. Changing schema
or policy strings alone is not a valid migration. Teacher-forced v1 exports are
unaffected.

`GPTSOVITS_T2S_GENERATION_REFERENCE` defaults to empty (two numerical tests skip).
Once configured, missing/invalid `rp1` or `rp135` cases, incompatible/F16 model
weights or a missing model fail. CTest never runs Python, installs dependencies or
downloads models. Numerical tests are CPU/F32, labeled `cpu;integration`, with a
300-second timeout. The same-source-model requirement and separate Python hash
validation apply as for the teacher-forced comparison.

### Generation artifact contract

The manifest adds `policy` and `stop_reason` instead of the teacher-forced
`semantics` field. Dimensions record total/reference/target text, prompt,
prefill N, accepted codes C, and decisions D. Source hashes identify only the
three actual input files (all-phones, all-BERT and prompt), in addition to the
checkpoint, exporter, loader and upstream sources. No private text, absolute
paths or timestamps are recorded in the manifest.

| Array | C-order shape | Dtype |
| --- | --- | --- |
| `phones.npy`, `bert.npy`, `prompt.npy` | `[T]`, `[T,1024]`, `[P]` | int32 / float32 / int32 |
| `prefill_input.npy` | `[N,H]` | float32 |
| `codes.npy` | `[C]` (possibly empty) | int32 |
| `logits.npy` | `[D,V]` before penalties/filtering | float32 |
| `selected.npy`, `context_lengths.npy` | `[D]`, `[D]` | int32 |

EOS stops have `D=C+1`, with one final unaccepted EOS selection and
`C >= min_new_tokens`. Length-limit stops have `C=D=max_new_tokens`, with no EOS
decision, including when the minimum exceeds the maximum. Context lengths are
`N..N+D-1`. Logits remain raw, finite and full-vocabulary, including EOS. The
validator recomputes every selection from the binary prompt-plus-generated
history, skipping EOS during the window; it also checks codes/decision/stop
consistency, layout, IDs, finite values and file hashes. Empty codes are valid
only for immediate EOS with minimum 0; the C++ NPY reader permits zero length
explicitly for that array, not for ordinary decode inputs.

The C++ integration test installs exported reference-text and prompt embeddings
plus prompt token IDs in a session-owned reference cache. It sends **target**
phones/BERT to public `t2s_session_generate`, covering real target embedding, host concatenation,
prefill, persistent decode, sampler state and stopping. It asserts exact codes
and stop reason, then verifies a short-budget prefix retains every accepted code
and a second seed reproduces the full result using the same decode graph. Slot
release and byte-identical reference caches are checked after every call.
Reference audio encoding, the text frontend, HuBERT, SV, SoVITS and WAV output are
not run here; this is not an audio-quality or raw-text-to-speech test.

### Observed generation baseline

On the same local CPU/F32 case (44 reference phones, 22 target phones, 136 prompt
codes, maximum 200, minimum 11), independent v2 exports were byte-identical for
both repetition penalties:

| Repetition penalty | Accepted codes | Decisions | Stop reason | Smallest adjusted top-2 gap |
| --- | ---: | ---: | --- | ---: |
| 1.0 | 200 | 200 | `max_new_tokens` | `0.0403` |
| 1.35 | 77 | 78 | `eos` | `0.00730` |

All eight arrays for both policies remain byte-identical to the preceding
prompt-history/unmasked baseline: this particular case did not choose early EOS.
The default-window behavior is separately tested with synthetic models whose
highest logit is always EOS, including stochastic sampling, the exact step-11
boundary, a one-token non-EOS vocabulary, independent batch flags and graph reuse.
The public API comparison also checks the full sequence, stop reason, short-budget
prefix and same-session repeats. The no-penalty limit is reported
as a limit, not as successful natural termination. These observations do not
imply other inputs will stop at the same lengths or that the UI's different
policy will generate the same sequence.
