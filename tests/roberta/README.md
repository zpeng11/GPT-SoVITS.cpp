# RoBERTa feature extraction

The public API has two distinct token-level outputs for
`chinese-roberta-wwm-ext-large`:

- `roberta_extract_features`: embeddings plus all 24 encoder layers, equivalent
  to Hugging Face `last_hidden_state`. Its existing behavior is unchanged.
- `roberta_extract_tts_features`: embeddings plus the first 22 encoder layers,
  equivalent to `hidden_states[-3]` (`hidden_states[22]`, after encoder layer
  index 21). This is the layer selected by upstream GPT-SoVITS for TTS. The final
  two encoder layers are not executed.

Both accept `1 <= T <= 512` token IDs and produce contiguous float32 ggml
`{1024, T}` data, indexed as `out[t * 1024 + channel]`. Any caller-supplied
CLS/SEP tokens count toward T and retain their output rows. Neither function
runs tokenization, removes special tokens or expands features to phonemes.
Use `roberta_expand_bert_features` with token spans for phoneme expansion;
special-token spans `-1/-1` are never selected. Mixed-language assembly and
English zero-fill remain outside these low-level APIs; use
`text_frontend_build_bert_features` from `gpt_sovits/text_frontend.h` to consume
parsed frontend results and assemble features for T2S. See the
[frontend contract](../frontend/README.md#phoneme-level-bert-feature-builder).

## Local reference assets

The tests do not download models. Prepare
`tests/roberta/chinese-roberta-wwm-ext-large-f16.gguf` with
`python/convert_roberta_to_gguf.py`, and generate references using a local Hugging
Face checkpoint containing `config.json` and model weights:

```sh
python3 tests/roberta/generate_reference.py /path/to/chinese-roberta-wwm-ext-large \
  --output-dir tests/roberta
```

The generator needs NumPy, PyTorch and Transformers. It runs CPU float32 eager
attention in evaluation mode, using 32 deterministic token IDs (seed 42) by
default. `ref_model_output.bin` is the final-layer reference;
`ref_tts_output.bin` is the TTS reference. Both retain all token rows, including
special tokens when supplied. The generated binary files and GGUF models are
ignored by Git.

If only the local F16/F32 GGUF is available, `--gguf` loads its tensors into a
Hugging Face `BertModel` and still executes the independent PyTorch forward:

```sh
python3 tests/roberta/generate_reference.py /path/to/config-directory \
  --gguf tests/roberta/chinese-roberta-wwm-ext-large-f16.gguf \
  --output-dir tests/roberta
```

This mode additionally needs the Python `gguf` package. It requires the 24-layer,
1024-channel model configuration and rejects quantized GGUF tensors. It tests
computation against the stored GGUF weights, not against the original checkpoint
before F16 rounding. It does not establish original-F32 or synthesis parity.
Generating references in an existing output directory overwrites its `ref_*.bin`
files; use a separate directory to preserve an earlier baseline.

## Build and test

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_roberta_extract test_roberta_parity -j 4
ctest --test-dir build \
  -R '^(RobertaExtract\.|RobertaExtractCPU\.|RobertaExtractValidation\.|RobertaExpand\.|RobertaReference\.|RobertaParity/)' \
  --output-on-failure
```

TTS parity runs on the normal test backend (Metal when available, otherwise
CPU), and explicitly on CPU. The test compares all 1024 channels of every token,
checks finite values and an output-buffer guard, and leaves the final two
layers absent in a borrowed model view to ensure they are not used. The TTS
reference must differ materially from the final-layer reference so accidentally
selecting `last_hidden_state` cannot pass. TTS checks each element with
`abs(actual - reference) <= 0.05 + 0.005 * abs(reference)` and RMSE < 0.01.
The amplitude-aware bound accounts for large intermediate activations while
being stricter near zero than the existing final-layer absolute bound of 0.1.
The unchanged graph blocks use ggml's tanh GELU approximation, whereas the Hugging
Face `gelu` reference uses erf; bitwise identity is not expected. Final-layer
regression retains its original maximum absolute error < 0.1 and RMSE < 0.01.
Additional CPU regressions cover T=1 and T=512, explicit zero/nonzero token-type
IDs, and repeated TTS/final-layer/TTS calls on the same model. Malformed reference
sizes are rejected before allocation/read, with model-free regression coverage.

Missing local models cause explicit skips, not successful numerical validation.
When the model exists, load failures and missing/invalid reference files fail
the extraction tests. Unloaded-model validation and host-side phoneme-expansion
tests require no model files. This suite does not test JSON feature assembly,
T2S generation, audio quality or end-to-end synthesis.
