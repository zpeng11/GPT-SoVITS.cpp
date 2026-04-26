#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <string>

namespace gpt_sovits {

// SoVITS v2 reference encoder (`ref_enc`) inference block.
//
// This block implements the `MelStyleEncoder` used by the SoVITS v2
// synthesizer path:
//   refer {704, T} -> spectral MLP -> 2x Conv1dGLU -> self-attention
//   -> projection -> temporal average pool -> style embedding {512, 1}
//
// Scope:
//   - single-sample inference only
//   - public activations use {channels, time}
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v2 inference path in
//     `module/models_onnx.py`: all frames are valid, attention is unmasked,
//     and temporal pooling averages across the full sequence.

struct sovits_mel_style_encoder_conv_glu_block_weights {
    // Conv1d(128, 256, kernel=5, padding=2)
    // PyTorch layout [out_channels, in_channels, kernel]
    // ggml layout {kernel, in_channels, out_channels}
    struct ggml_tensor * conv_w;   // {5, 128, 256}
    struct ggml_tensor * conv_b;   // {256}
};

struct sovits_mel_style_encoder_attention_block_weights {
    // Linear(128, 128) projections.
    struct ggml_tensor * q_w;      // {128, 128}
    struct ggml_tensor * q_b;      // {128}
    struct ggml_tensor * k_w;      // {128, 128}
    struct ggml_tensor * k_b;      // {128}
    struct ggml_tensor * v_w;      // {128, 128}
    struct ggml_tensor * v_b;      // {128}
    struct ggml_tensor * out_w;    // {128, 128}
    struct ggml_tensor * out_b;    // {128}
};

struct sovits_mel_style_encoder_block_weights {
    // spectral = Linear(704, 128) -> Mish -> Linear(128, 128) -> Mish
    struct ggml_tensor * spectral_1_w;   // {704, 128}
    struct ggml_tensor * spectral_1_b;   // {128}
    struct ggml_tensor * spectral_2_w;   // {128, 128}
    struct ggml_tensor * spectral_2_b;   // {128}

    // temporal = 2 x Conv1dGLU(128, 128, kernel=5)
    std::array<sovits_mel_style_encoder_conv_glu_block_weights, 2> temporal;

    // self attention over the temporal sequence.
    sovits_mel_style_encoder_attention_block_weights attention;

    // fc = Linear(128, 512)
    struct ggml_tensor * fc_w;           // {128, 512}
    struct ggml_tensor * fc_b;           // {512}
};

// Weights for the SoVITS RVQ decode path used by SynthesizerTrn.forward:
//   codes {T} -> codebook lookup -> quantized SSL features {768, T}
//
// Scope:
//   - single-sample inference only
//   - single RVQ layer only (matches v1/v2 export path: n_q = 1)
//   - input codes must be a 1D ggml vector of token ids
struct sovits_rvq_decode_block_weights {
    // EuclideanCodebook.embed stored in ggml layout.
    struct ggml_tensor * codebook;       // {768, 1024}
};

// Build the SoVITS v2 MelStyleEncoder graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   refer    - reference spectrogram features {704, T}
//   weights  - MelStyleEncoder weights
//
// Returns:
//   style embedding {512, 1}
struct ggml_tensor * sovits_mel_style_encoder_block_forward(
    struct ggml_context                              * ctx,
    struct ggml_tensor                               * refer,
    const sovits_mel_style_encoder_block_weights     & weights);

// Build the SoVITS single-layer RVQ decode graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   codes    - semantic token ids {T} (i32)
//   weights  - RVQ decode weights
//
// Returns:
//   quantized SSL features {768, T}
struct ggml_tensor * sovits_rvq_decode_block_forward(
    struct ggml_context                       * ctx,
    struct ggml_tensor                        * codes,
    const sovits_rvq_decode_block_weights     & weights);

// ---------------------------------------------------------------------------
// SoVITS ref_enc model: owns the loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
// ---------------------------------------------------------------------------

struct sovits_ref_enc_model {
    sovits_mel_style_encoder_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// SoVITS quantizer model: owns the loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
struct sovits_quantizer_model {
    sovits_rvq_decode_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// Load a SoVITS ref_enc model from a GGUF file produced by
// `convert_sovits_ref_enc_to_gguf.py`.
//
// Parameters:
//   fname   - path to the .gguf file
//   model   - output model struct (will be populated)
//   backend - ggml backend for tensor allocation (caller-owned; not freed by
//             sovits_ref_enc_model_free)
//
// Returns:
//   true on success, false on failure.
bool sovits_ref_enc_model_load(
    const std::string & fname,
    sovits_ref_enc_model & model,
    ggml_backend_t backend);

// Load a SoVITS quantizer model from a GGUF file produced by
// `convert_sovits_quantizer_to_gguf.py`.
bool sovits_quantizer_model_load(
    const std::string & fname,
    sovits_quantizer_model & model,
    ggml_backend_t backend);

// Free all resources owned by a SoVITS ref_enc model.
void sovits_ref_enc_model_free(sovits_ref_enc_model & model);

// Free all resources owned by a SoVITS quantizer model.
void sovits_quantizer_model_free(sovits_quantizer_model & model);

} // namespace gpt_sovits
