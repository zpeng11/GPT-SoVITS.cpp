# Python regression tests

Run from the repository root:

```sh
python3 -m unittest discover -s python/tests -v
```

The core checkpoint tests require only NumPy and Python's standard library.
Converter tests additionally require the existing `gguf` dependency; they are
explicitly skipped if it is unavailable. No pytest, model downloads or CMake
build are needed. Tests write temporary files outside the source tree and clean
them up automatically.

## Coverage

- Bare `state_dict` (including a parameter literally named `weight`), wrapped
  checkpoints and metadata, and empty state dictionaries.
- Shared storage, transposed tensors, nonzero-offset stepped slices, zero-stride
  broadcast views, scalar and empty tensors. Checks include exact shape, dtype,
  values, writable C-contiguous output, and independent output storage.
- Float16/32/64, int16/32/64, uint8 and bool storage.
- Invalid top-level layouts, missing/ambiguous `data.pkl`, missing tensor storage,
  non-ZIP files, and unsupported pickle globals.
- HuBERT and RoBERTa conversion via their CLI entry points, using tiny synthetic
  checkpoints. F32/F16/Q8 GGUF output is read back to check tensor names, shapes,
  types, values and selected metadata. These checks cover HuBERT weight_g
  flattening/Conv1d handling, RoBERTa integer-buffer filtering, and skipped heads.
- Loader and converter subprocesses actively reject imports of `torch`, including
  when the parent suite runs in a PyTorch-enabled environment.

These are tooling tests, not model inference tests. Q4/Q5, bfloat16 storage,
legacy non-ZIP loading and arbitrary unsupported PyTorch serialization classes
are outside this suite's coverage. Non-ZIP input is tested for rejection only.

## Fixtures

`fixtures/*.pt` are small synthetic checkpoints produced by real `torch.save`
(PyTorch 2.11.0, default ZIP serialization). They contain no pretrained weights.
The checked-in fixtures make normal test runs independent of PyTorch. Expected
arrays are defined independently in NumPy in the tests.

To regenerate them in an environment with PyTorch installed:

```sh
python python/tests/generate_fixtures.py
```

Regeneration overwrites only those fixture files. The tensor values are
deterministic; byte-for-byte archive identity across PyTorch versions is not
promised.

## Optional PyTorch and real-model comparisons

If PyTorch is importable, the suite also tests fresh `torch.save` round trips
against `torch.load(..., weights_only=True)`. No PyTorch dependency is installed
automatically.

To compare every tensor from local HuBERT and RoBERTa checkpoints with PyTorch,
set these environment variables to directories containing `pytorch_model.bin`:

```sh
HUBERT_MODEL_DIR=/path/to/chinese-hubert-base \
ROBERTA_MODEL_DIR=/path/to/chinese-roberta-wwm-ext-large \
python -m unittest discover -s python/tests -v
```

The real-model tests are explicitly skipped without the variables or PyTorch.
An explicitly configured but missing/invalid checkpoint fails the test rather
than skipping. These checks compare loader outputs, not full real-model GGUF
conversion; only the tiny synthetic models are converted in the default suite.

## T2S decode reference exporter

```sh
python3 -m unittest discover -s python/tests -p test_t2s_reference_export.py -v
```

The exporter tests cover precomputed-input validation, the versioned manifest,
file integrity, output protection and failure cleanup using NumPy and the
standard library. `--help` and `--validate` are tested with PyTorch imports and
network access blocked. With PyTorch installed, tiny deterministic models also
compare incremental teacher-forced logits against a full causal forward and
exercise checkpoint loading through the actual export CLI. Without PyTorch,
only these numerical/CLI-export tests skip. No pretrained model or upstream
checkout is needed by this suite.

See [the T2S reference documentation](../../tests/t2s/README.md) for the explicit
local-upstream export command and output layout. This is an offline reference
tool; it does not add PyTorch to production inference or claim C++ parity.

The matched-policy closed-loop mode has a separate test suite:

```sh
python3 -m unittest discover -s python/tests -p test_t2s_generation_reference.py -v
```

It reuses the tiny model fixtures to check self-feedback, prompt-plus-generated
binary repetition history, the configurable initial EOS suppression window,
length-limit behavior, independent full-forward logits, v2 generation manifest
validation (including rejection of old unmasked exports) and cross-process export
identity. NumPy-only validation and CLI rejection tests still run without PyTorch;
model computation tests skip when
it is absent. This is not a comparison with the upstream UI's sampling defaults.
