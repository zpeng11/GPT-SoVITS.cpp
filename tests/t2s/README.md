# T2S teacher-forced CPU decode reference

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

The output path must **not exist**, even as an empty directory, file, or broken
symlink. Arrays are staged and validated before publishing to a newly reserved
directory; `manifest.json` is published last. Recoverable publication failures
clean up this run's files. A process kill can leave an incomplete directory;
without the completion manifest it is not a valid export. Existing data is never
overwritten or removed. Publication uses exclusive hard links within the output
filesystem; a filesystem without hard-link support fails explicitly. Private
inputs and generated data belong under the ignored `tests/*/out/` directories,
not in tracked fixtures.

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

This commit supplies the oracle exporter and its own tests. It does **not** yet
assert C++ decode numerical parity, sampled-sequence equality, waveform parity,
intelligibility, or listening quality. Existing tracked `ref/*.npy` files are not
overwritten. Independent C++ multi-step comparison is a separate change.
