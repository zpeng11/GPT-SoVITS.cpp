# Text frontend JSON contract

This layer consumes the structured JSON produced by `python -m text_frontend`.
It does not run Python, RoBERTa, audio processing or synthesis. Only
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
  instead of treating it as BERT-ready input. A later feature builder must
  reject missing tokenizers for Chinese segments that require BERT. The parser
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

## Build and test

From the repository root:

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_text_frontend_json -j 4
ctest --test-dir build -R '^TextFrontendJson\.' --output-on-failure
```

These C++ tests use tracked fixtures and require no Python, model files or
network access. Model-dependent inference limits and audio quality are not
validated by this suite.

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
