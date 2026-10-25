# Text frontend JSON contract

This layer consumes the structured JSON produced by `python -m text_frontend`
and can build phoneme-level BERT features using a caller-loaded RoBERTa model.
It does not run Python, audio processing or synthesis. Only
`gpt-sovits-text-frontend/v1` with the `v2` (732-phoneme) vocabulary is supported.
The producer contract lives in `python/text_frontend/frontend.py` and
`python/text_frontend/tokenizer.py`.

## API

```cpp
#include "gpt_sovits/text_frontend.h"

gpt_sovits::text_frontend_result text;
std::string error;
if (!gpt_sovits::text_frontend_read_json("target.json", text, &error)) {
    // Report error to the caller.
}
```

`text_frontend_parse_json` accepts an in-memory UTF-8 JSON string. Both functions
return `bool`, own all result buffers, replace the result only on success, and
clear the supplied error on success. On failure, the old result is unchanged and
a diagnostic is returned through `error`, or printed to stderr if it is null.
Do not alias `error` with the input or a string in the output object. Recoverable
host allocation exceptions are caught. Resource exhaustion inside the JSON
library's `noexcept` DOM teardown can terminate the process and is not a supported
recovery point; this API does not promise recovery from all memory exhaustion.
Raw NUL bytes in JSON input (including after a complete document) and file paths
are rejected. Legal JSON `\u0000` escapes are not raw NUL bytes.

## Validation and interpretation

- Schema and vocabulary versions must match exactly. The request must contain
  phonemes and at least one `zh` or `en` segment. Required fields have strict
  types: booleans and floating-point values are not accepted as integers.
- Top-level and segment phoneme IDs, counts, indices and offsets must agree.
  Segments cover the entire phoneme array in order; zero-phoneme segments may
  have repeated offsets. Phoneme IDs are in `[0,732)`. Segment texts concatenate
  to the top-level normalized text.
- `normalized_start`, `normalized_end`, `char_count` (if present), and token
  spans count **Unicode code points, not UTF-8 bytes**. Token spans refer to the
  segment's normalized text, not the tokenizer's internally normalized text.
  Invalid UTF-8 and malformed Unicode escapes are rejected.
- Chinese `word2ph` has one non-negative integer per normalized character and
  sums to the segment's phoneme count. English `word2ph` is omitted or null.
- Segment tokenizers may be omitted or null: the parser retains this state
  instead of treating it as BERT-ready input. The feature builder rejects
  missing tokenizers for Chinese segments that require BERT. The parser
  **never substitutes the top-level tokenizer**, which is a separate encoding.
- A configured segment tokenizer has status `ok`, non-negative int32 token IDs,
  matching ID/span lengths, and exactly one leading CLS and trailing SEP outside
  its body. Their IDs match `cls_id`/`sep_id`; their null spans become `-1/-1`.
  Body spans are nonempty, within the segment, and use half-open intervals.
  Every Chinese character with `word2ph > 0` must have exactly one covering body
  token. Characters with zero phonemes need not have a unique covering token.
- Unknown metadata is ignored, including the top-level tokenizer and word2ph.
  Duplicate segment phoneme arrays are checked but not stored twice. The owned
  output keeps the top-level IDs and each segment's range.
- Tokenizer IDs are **not** phoneme IDs. Their actual embedding-table bounds and
  RoBERTa's per-segment 512-token limit are model-specific checks for the feature
  extraction layer, not this parser. No truncation or BERT feature computation
  happens here.

The JSON dependency is a private, vendored nlohmann/json header; it is not exposed
by `text_frontend.h` and adds no configure/build-time downloads. NPZ ingestion is
not implemented: the current NPZ output does not contain the full per-segment
WordPiece tokenization needed for mixed-text BERT inference.

## Phoneme-level BERT feature builder

```cpp
#include "gpt_sovits/roberta.h"
#include "gpt_sovits/text_frontend.h"

// text is parsed above; roberta is a caller-loaded roberta_model.
std::vector<float> bert;
if (!gpt_sovits::text_frontend_build_bert_features(text, &roberta, bert, &error)) {
    // Report error; bert retains its previous contents on failure.
}
// bert[p * 1024 + channel] is directly usable by T2S reference/generate APIs.
```

The builder runs `roberta_extract_tts_features` (`hidden_states[-3]`) per
nonzero-phoneme Chinese segment, then uses `roberta_expand_bert_features` with
that segment's Unicode spans and word2ph. It never uses a top-level tokenizer,
transposes features, silently truncates or chunks long inputs. Segment token
counts include CLS/SEP and must not exceed 512; IDs must fit the loaded model's
embedding vocabulary. Tokenizer/model compatibility is the caller's responsibility:
IDs being in range does not establish that two vocabularies have the same meaning.

English phoneme ranges are strictly zero-filled without accessing a tokenizer
or model. Pass `nullptr` instead of `&roberta` for English-only requests (also
allowed when every Chinese segment has zero phonemes). Zero-phoneme segments
skip inference and need no tokenizer, but must retain valid offsets and word2ph.
Nonzero-phoneme Chinese segments require both a tokenizer and a loaded model;
missing Chinese features are errors, never substituted with zeros.

Results are contiguous float32 ggml `{1024, T_phonemes}`, equivalent to NumPy
`(T_phonemes, 1024)` C-order. Success replaces `bert` and clears `error`. Failure
preserves the previous vector and reports diagnostics through `error`, or stderr
when null. The builder checks the owned struct's language, phoneme ranges and
IDs, concatenated UTF-8 texts/Unicode offsets, Chinese word2ph, and the tokenizer
structure used for inference, even for manually constructed/modified results.
It rejects non-finite extracted features and recoverable allocation/compute
failures. It does not recover from ggml aborts or invalid backend resources.
Do not alias `error` with a string in `text`.

The model/backend are borrowed for the call and must not be freed, reloaded or
used concurrently during it. There is no model loading, reference cache, Python
invocation, audio processing or T2S/SoVITS generation inside this function.

## Build and test

From the repository root:

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_text_frontend_json -j 4
ctest --test-dir build -R '^TextFrontendJson\.' --output-on-failure
```

Parser and host-assembly tests use tracked fixtures and require no Python,
model files or network access:

```sh
cmake --build build --target test_text_frontend_bert -j 4
ctest --test-dir build -R '^TextFrontendBert\.' --output-on-failure
```

Host tests supply deterministic token features through a private internal
provider, not a public mock API. They verify every output channel and segment
range, English zeros, Unicode/word2ph mapping, zero-phoneme segments, malformed
structs/tokenizers, the 512-token boundary, failure atomicity, retry, simulated
allocation/compute failures and rejection of NaN/Inf.

The separate CPU integration suite needs the local F16 RoBERTa model documented
in [the RoBERTa tests](../roberta/README.md#local-reference-assets):

```sh
cmake --build build --target test_text_frontend_bert_integration test_roberta_extract -j 4
ctest --test-dir build \
  -R '^(TextFrontendBertIntegration\.|RobertaExtractCPU\.)' --output-on-failure
```

`TextFrontendBertIntegration` is labeled `cpu;integration` with a 300-second
per-test timeout. It explicitly uses CPU even in Metal builds. Only a missing
model (ENOENT) skips; an unreadable/invalid model or missing/invalid tracked
fixture fails. A skip is not successful model validation. It compares the
builder against direct per-segment TTS extraction and expansion, checks real
token IDs, strict English zeros and finite/nondegenerate Chinese features, and
rejects model-vocabulary violations without changing output. This establishes
assembly consistency, not independent numerical parity. The existing RoBERTa
CPU TTS reference test supplies independent PyTorch parity for extraction.
Neither suite validates T2S generation, synthesis or audio quality.

## Fixtures

The explicit `generate_fixtures.py` script uses the existing `TextFrontend`
producer with a small local tokenizer vocabulary and the explicitly selected
`basic` English backend. It does not download resources or use model weights.
Chinese regeneration requires `pypinyin==0.55.0`; the generator disables optional
jieba/POS behavior so installation of jieba does not change the fixtures. Normal
CTest runs do not regenerate them.

```sh
python3 tests/frontend/generate_fixtures.py --check
# Deliberately regenerate the tracked producer outputs:
python3 tests/frontend/generate_fixtures.py
```

The mini vocabulary is for parser contracts only. Its IDs are not a substitute
for the real RoBERTa tokenizer, and its fixtures must not be presented as a
faithful synthesis or audio-quality case. Tests also mutate these producer
outputs to exercise invalid types, counts, offsets, spans and Unicode input.

### Real-tokenizer integration fixture

`fixtures/bert_mixed.json` is separate from the mini-vocabulary parser fixtures.
It contains `你好，世界！Hello world!再见。`, produced with the real
`chinese-roberta-wwm-ext-large` tokenizer, `pypinyin==0.55.0`, the `basic` English
backend and jieba disabled. Its unknown `fixture_provenance` metadata records
the tokenizer source identity, SHA-256 and generation settings. The fixture
contains no model weights or extracted feature arrays.

The generator requires an explicitly supplied local tokenizer; the model asset
directory is ignored by Git. Using an interpreter with the pinned dependency:

```sh
python3 tests/frontend/generate_bert_fixture.py \
  --tokenizer /path/to/chinese-roberta-wwm-ext-large/tokenizer.json --check
# Deliberately regenerate only this fixture (never the mini fixtures):
python3 tests/frontend/generate_bert_fixture.py \
  --tokenizer /path/to/chinese-roberta-wwm-ext-large/tokenizer.json
```

These commands do not install packages or download resources. Normal CTest
requires neither this tokenizer file nor Python. The mixed assembly test uses
this fixture with a compatible local RoBERTa model; it is not an end-to-end
synthesis or audio-quality fixture.
