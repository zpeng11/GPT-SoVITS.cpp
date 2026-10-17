# SoVITS v2Pro / v2ProPlus integration

The unified SoVITS model supports the v2, v2Pro and v2ProPlus inference
architectures. The integration tests target CPU, batch size 1 and speed 1.
Full session tests are smoke/contract tests, not a claim of bitwise PyTorch
waveform parity or full Metal session support.

## Conditioning and model layout

| Variant | Style / flow / decoder `ge` | Encoder `ge_enc` | Generator initial channels |
| --- | --- | --- | --- |
| v2 | 512 | aliases `ge` | 512 |
| v2Pro | 1024 | `ge_to512(ge)`, 512 | 512 |
| v2ProPlus | 1024 | `ge_to512(ge)`, 512 | 768 |

Pro variants fuse each reference's 20480-element SV embedding into the style
encoder result, then apply PReLU. Multiple references are averaged **after**
this per-reference fusion. The encoder projection is applied to the resulting
mean; flow and decoder receive the unprojected 1024-element conditioning.

Both Pro variants use upsample strides `[10,8,2,2,2]`. ProPlus uses a first
upsample kernel of 20 rather than 16. This is support for these architectures,
not arbitrary generator configurations.

## Conversion

Run from the repository root, using local upstream checkpoints:

```sh
python3 python/convert_sovits_to_gguf.py /path/to/s2Gv2Pro.pth \
    --output tests/sovits/models/v2pro-sovits-f16.gguf --type f16
python3 python/convert_sovits_to_gguf.py /path/to/s2Gv2ProPlus.pth \
    --output tests/sovits/models/v2proplus-sovits-f16.gguf --type f16
```

GGUF files are ignored by Git. The converter derives the conditioning size and
variant from weights and checks configuration consistency. Existing older Pro
GGUFs may carry `sovits.version=v2` and stale informational flow/generator gin
metadata. The runtime identifies the Pro path through its tensors and the
`sovits.v2pro` flag, not that version string. Authoritative conditioning shapes
and `sovits.ref_enc.out_dim`, when present, must agree.

The audio-reference convenience API also needs the independent SV model; see
[README_SV.md](README_SV.md) for its conversion and reference generation.

## API migration and lifetime

`compute_ge` accepts `std::vector<sovits_refer_view>`:

```cpp
std::vector<gpt_sovits::sovits_refer_view> refs = {
    {spec.data(), frames, sv_embedding.data()},
};
bool ok = gpt_sovits::sovits_session_compute_ge(session, model, refs);
```

`spec` contains 704 channels with channel-fast layout `spec[t * 704 + c]`.
For a PyTorch `[1,1025,T]` reference spectrum, select the first 704 channels
and transpose/repack. A Pro reference requires its own SV embedding. Input
pointers are borrowed only for the call.

For v2 callers holding the previous `std::vector<std::pair<const float *,
int64_t>>`, change the function name to `sovits_session_compute_ge_specs`.
Existing two-field braced calls `compute_ge(session, model, {{spec, frames}})`
remain valid for v2. The named adapter avoids ambiguous braced overloads; it
is **not** preservation of the old exported ABI. Recompile downstream users.

Successful reference replacement invalidates borrowed `ge` / `ge_enc` tensor
pointers. A recoverable failed update preserves the previous pair. This is
transactional replacement, not thread-safe concurrent mutation. Backend
lifetime must exceed model/session lifetime; release sessions and models before
their borrowed backend.

## Tests

```sh
cmake -S . -B build -DGPTSOVITS_BUILD_TESTS=ON
cmake --build build --target test_sovits_loader test_sovits_v2pro_parity \
    test_sovits_session test_sovits_generator -j 4
./build/bin/test_sovits_loader
./build/bin/test_sovits_v2pro_parity
./build/bin/test_sovits_session
./build/bin/test_sovits_generator
python3 -m unittest discover -s python/tests -v
```

The Pro suite includes independent block parity, style conditioning, actual
short-code session forward, output capacity guards, independent encoder
projection comparison, multi-reference updates, failure/retry semantics, and
comparison of audio-driven SV with explicitly supplied SV embeddings. Tests
require matching local GGUF models and the reference arrays under `ref/`.

## Reference provenance and limits

The local Pro/Plus downstream arrays form one set of runs. The NAS export at
`/path/to/GPT-SoViTS` contains another text sequence: its Plus run has 180
frames whereas the local Plus run has 192. Do not selectively replace text,
codes, encoder, flow or decoder arrays between these runs.

The added `v2pro_refer.npy` was recovered from the NAS export and checked
byte-for-byte against the existing `v2proplus_refer.npy`; both contain the
same `[1,1025,243]` reference spectrum. Associated local SV embeddings and
conditioning inputs agree with that export. No downstream arrays were replaced.

`*_enc_p_input_ge.npy` contains **1024-element pre-projection conditioning**,
not the 512-element output of `ge_to512`.

The original Pro dump generator and a complete provenance manifest are not
part of this repository. Complete PyTorch session numerical parity would need
one matched run containing reference/SV inputs, codes/text, `ge`, `ge_to512`
output, encoder mean/log-scale, the actual scaled sampling noise, `z_p`, flow
output and waveform, plus checkpoint identity, upstream revision and dtype.
Matching random seeds alone is insufficient across PyTorch and C++ RNGs.
Existing decoder waveform references do not remove that sampling gap.
