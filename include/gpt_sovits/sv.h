#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// ERes2NetV2 speaker-verification model (forward3 path only).
//
// GPT-SoVITS computes the 20480-d reference embedding via
// SV.compute_embedding3 -> ERes2NetV2.forward3, which skips the embedding
// head (seg_1/pool) and returns fuse_out34.flatten(1,2).mean(-1).
//
// forward3 runs as a ggml compute graph on a backend (Metal on macOS, CPU
// elsewhere) -- exactly like the rest of the pipeline. Weights and pre-folded
// BatchNorm scale/shift tensors are uploaded once into a backend buffer at
// load time; sv_forward3 builds an ephemeral graph per call and runs it on the
// same backend. The GGUF from convert_sv_to_gguf.py is the weight source.
//
// Architecture is fixed (ERes2NetV2 baseWidth=24, scale=4, expansion=4):
//   num_blocks=[3,4,6,3], m_channels=64, feat_dim=80, out_dim=20480.
// Layer 1/2 use BasicBlockERes2NetV2 (residual split-add); layer 3/4 use the
// AFF variant (residual split-fuse). Metadata and weights must match these dims.
// ---------------------------------------------------------------------------

struct sv_hparams {
    int base_width = 24;
    int scale      = 4;
    int expansion  = 4;
    int feat_dim   = 80;
    int out_dim    = 20480;
};

struct sv_model {
    // Backend the model runs on (borrowed; not freed by sv_model_free).
    ggml_backend_t backend = nullptr;

    // Weight context: tensors allocated onto `buf_w` (backend-resident).
    struct ggml_context    * ctx_w   = nullptr;
    ggml_backend_buffer_t    buf_w   = nullptr;

    // Derived-constant context: per-BatchNorm scale/shift tensors, folded at
    // load time from bn.{weight,bias,running_mean,running_var}, allocated onto
    // `buf_bn`. Stored as "<bn-prefix>.scale" / "<bn-prefix>.shift" [C].
    struct ggml_context    * ctx_bn  = nullptr;
    ggml_backend_buffer_t    buf_bn  = nullptr;

    sv_hparams hparams = {};
};

// Load an SV GGUF (from convert_sv_to_gguf.py) onto `backend`: allocates every
// weight tensor into a backend buffer, streams the data once, and pre-folds all
// BatchNorm layers into scale/shift tensors. Supports F32/F16 convolution
// weights and F32 BatchNorm/bias tensors. Returns true on success, replacing any
// previously loaded model; failure leaves model unchanged. Backend must be non-null.
bool sv_model_load(const std::string & fname, sv_model & model,
                   ggml_backend_t backend);

// Free resources owned by the model (does not free the borrowed backend).
void sv_model_free(sv_model & model);

// Look up a weight tensor by GGUF name (e.g. "sv.layer1.0.convs.0.weight").
// Returns nullptr if absent. The tensor is backend-resident.
const struct ggml_tensor * sv_model_tensor(const sv_model & model,
                                           const std::string & name);

// ---------------------------------------------------------------------------
// Kaldi-style fbank frontend (matches SV.compute_embedding3 -> Kaldi.fbank).
//
// Fixed parameters (Kaldi.fbank defaults): 16 kHz, 80 mel bins, 25 ms / 10 ms
// frame length / shift, povey window, preemphasis 0.97, remove_dc_offset,
// round-to-power-of-two (400 -> 512 FFT), power spectrum, log mel, snip_edges.
//
// Parameters:
//   audio      - 16 kHz mono waveform samples (float32 in [-1, 1]-ish)
//   n_samples  - number of samples
//   out_feats  - appended with m*num_mel_bins row-major values (frame-major:
//                out[m*80 + b], i.e. {m, 80} row-major). Grown, not cleared.
//
// Returns the number of frames m (0 on failure).
int sv_fbank(const float * audio, int64_t n_samples,
             std::vector<float> & out_feats);

// ---------------------------------------------------------------------------
// ERes2NetV2 forward3: fbank features -> 20480-d embedding.
//
// Mirrors ERes2NetV2.forward3 exactly:
//   fbank {m, 80} -> {1,1,80,m} -> conv1+bn1+relu -> layer1..4
//   -> layer3_ds -> fuse34 -> flatten(C*F) -> time-mean -> {20480}
//
// Parameters:
//   model    - loaded SV model
//   fbank    - fbank features {m, feat_dim} row-major (frame-major)
//   m_frames - number of frames
//   out      - caller-allocated buffer of model.hparams.out_dim floats
//
// Returns true on success.
bool sv_forward3(const sv_model & model,
                 const float * fbank, int m_frames,
                 float * out);

} // namespace gpt_sovits
