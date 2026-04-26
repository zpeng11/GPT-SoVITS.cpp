#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <string>

namespace gpt_sovits {

static constexpr int kSovitsTextEncoderSslLayers = 3;

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

// Weights for the SoVITS v2 `enc_p.ssl_proj + enc_p.encoder_ssl` path:
//   ssl {768, T} -> Conv1d(768, 192, k=1) -> 3 x [rel-pos self-attn + FFN]
//   -> {192, T}
//
// Scope:
//   - single-sample inference only
//   - fixed v2 hyperparameters from the shipped checkpoint:
//       hidden=192, ffn=768, n_heads=2, n_layers=3, kernel=3, window=4
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v2 inference path in
//     `module/models_onnx.py`: all frames are valid
struct sovits_text_encoder_ssl_layer_weights {
    // 1x1 Conv1d projections for self-attention.
    struct ggml_tensor * q_w;      // {1, 192, 192}
    struct ggml_tensor * q_b;      // {192}
    struct ggml_tensor * k_w;      // {1, 192, 192}
    struct ggml_tensor * k_b;      // {192}
    struct ggml_tensor * v_w;      // {1, 192, 192}
    struct ggml_tensor * v_b;      // {192}
    struct ggml_tensor * out_w;    // {1, 192, 192}
    struct ggml_tensor * out_b;    // {192}

    // Relative-position parameters. Stored in ggml layout after GGUF import.
    struct ggml_tensor * rel_k;    // {96, 9, 1}
    struct ggml_tensor * rel_v;    // {96, 9, 1}

    // LayerNorm(hidden)
    struct ggml_tensor * ln1_w;    // {192}
    struct ggml_tensor * ln1_b;    // {192}

    // FFN Conv1d(hidden, ffn, k=3) -> ReLU -> Conv1d(ffn, hidden, k=3)
    struct ggml_tensor * ffn_up_w;     // {3, 192, 768}
    struct ggml_tensor * ffn_up_b;     // {768}
    struct ggml_tensor * ffn_down_w;   // {3, 768, 192}
    struct ggml_tensor * ffn_down_b;   // {192}

    // LayerNorm(hidden)
    struct ggml_tensor * ln2_w;    // {192}
    struct ggml_tensor * ln2_b;    // {192}
};

struct sovits_text_encoder_ssl_block_weights {
    // Conv1d(768, 192, k=1)
    struct ggml_tensor * ssl_proj_w;    // {1, 768, 192}
    struct ggml_tensor * ssl_proj_b;    // {192}

    std::array<sovits_text_encoder_ssl_layer_weights, kSovitsTextEncoderSslLayers> layers;
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

// Build the SoVITS v2 `enc_p.ssl_proj + enc_p.encoder_ssl` graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   ssl      - quantized SSL features {768, T}
//   weights  - text-encoder SSL weights
//
// Returns:
//   encoded SSL features {192, T}
struct ggml_tensor * sovits_text_encoder_ssl_block_forward(
    struct ggml_context                               * ctx,
    struct ggml_tensor                                * ssl,
    const sovits_text_encoder_ssl_block_weights       & weights);

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

// SoVITS text-encoder SSL model: owns the loaded GGUF weights and ggml
// resources (except backend, which is borrowed from the caller).
struct sovits_text_encoder_ssl_model {
    sovits_text_encoder_ssl_block_weights weights = {};

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

// Load a SoVITS text_encoder_ssl model from a GGUF file produced by
// `convert_sovits_text_encoder_ssl_to_gguf.py`.
bool sovits_text_encoder_ssl_model_load(
    const std::string & fname,
    sovits_text_encoder_ssl_model & model,
    ggml_backend_t backend);

// Free all resources owned by a SoVITS ref_enc model.
void sovits_ref_enc_model_free(sovits_ref_enc_model & model);

// Free all resources owned by a SoVITS quantizer model.
void sovits_quantizer_model_free(sovits_quantizer_model & model);

// Free all resources owned by a SoVITS text_encoder_ssl model.
void sovits_text_encoder_ssl_model_free(sovits_text_encoder_ssl_model & model);

} // namespace gpt_sovits
