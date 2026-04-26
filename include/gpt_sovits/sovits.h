#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <string>

namespace gpt_sovits {

static constexpr int kSovitsTextEncoderSslLayers = 3;
static constexpr int kSovitsTextEncoderTextLayers = 6;
static constexpr int kSovitsTextEncoderPostLayers = 3;

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
struct sovits_relpos_encoder_layer_weights {
    // Fused 1x1 Conv1d projections for self-attention.
    // Output channels are laid out as [Q, K, V].
    struct ggml_tensor * qkv_w;    // {1, 192, 576}
    struct ggml_tensor * qkv_b;    // {576}
    struct ggml_tensor * out_w;    // {1, 192, 192}
    struct ggml_tensor * out_b;    // {192}

    // Relative-position parameters, prepacked for inference.
    struct ggml_tensor * rel_k;      // {96, 9}
    struct ggml_tensor * rel_v_t;    // {9, 96}

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

    std::array<sovits_relpos_encoder_layer_weights, kSovitsTextEncoderSslLayers> layers;
};

// Weights for the SoVITS v2 `enc_p.text_embedding + enc_p.encoder_text` path:
//   text ids {T} -> Embedding(732, 192) -> 6 x [rel-pos self-attn + FFN]
//   -> {192, T}
//
// Scope:
//   - single-sample inference only
//   - fixed v2 hyperparameters from the shipped checkpoint:
//       vocab=732, hidden=192, ffn=768, n_heads=2, n_layers=6, kernel=3, window=4
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v2 inference path in
//     `module/models_onnx.py`: all text tokens are treated as valid
struct sovits_text_encoder_text_block_weights {
    // Embedding(732, 192)
    struct ggml_tensor * text_embedding; // {192, vocab}

    std::array<sovits_relpos_encoder_layer_weights, kSovitsTextEncoderTextLayers> layers;
};

// Weights for an inference-only fused SoVITS v2 `enc_p.mrte` path:
//   ssl {192, T_ssl} -> fused [q, skip] projection {704, T_ssl}
//   text {192, T_text} -> fused [k, v] projection {1024, T_text}
//   ge {512, 1} -> fused ge projection {192, 1}
//   -> cross-attention(q, k, v) -> fused output projection {192, T_ssl}
//   -> attn_out + skip + ge_out -> {192, T_ssl}
//
// Scope:
//   - single-sample inference only
//   - exact offline fusion of the shipped v2 MRTE weights
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v1/v2 inference path in
//     `module/models_onnx.py`: all frames and text tokens are treated as valid
struct sovits_text_encoder_mrte_block_weights {
    struct ggml_tensor * ssl_fused_w;   // {1, 192, 704}
    struct ggml_tensor * ssl_fused_b;   // {704}
    struct ggml_tensor * text_kv_w;     // {1, 192, 1024}
    struct ggml_tensor * text_kv_b;     // {1024}
    struct ggml_tensor * attn_out_w;    // {1, 512, 192}
    struct ggml_tensor * attn_out_b;    // {192}
    struct ggml_tensor * ge_out_w;      // {1, 512, 192}
    struct ggml_tensor * ge_out_b;      // {192}
};

// Weights for the SoVITS v2 `enc_p.encoder2` path:
//   features {192, T} -> 3 x [rel-pos self-attn + FFN] -> {192, T}
//
// Scope:
//   - single-sample inference only
//   - fixed v2 hyperparameters from the shipped checkpoint:
//       hidden=192, ffn=768, n_heads=2, n_layers=3, kernel=3, window=4
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v2 inference path in
//     `module/models_onnx.py`: all frames are treated as valid
struct sovits_text_encoder_post_block_weights {
    std::array<sovits_relpos_encoder_layer_weights, kSovitsTextEncoderPostLayers> layers;
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

// Build the SoVITS v2 `enc_p.text_embedding + enc_p.encoder_text` graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   text     - phoneme token ids {T} (i32)
//   weights  - text-encoder text weights
//
// Returns:
//   encoded text features {192, T}
struct ggml_tensor * sovits_text_encoder_text_block_forward(
    struct ggml_context                                * ctx,
    struct ggml_tensor                                 * text,
    const sovits_text_encoder_text_block_weights       & weights);

// Build the fused SoVITS v2 `enc_p.mrte` graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   ssl      - encoded SSL features {192, T_ssl}
//   text     - encoded text features {192, T_text}
//   ge       - reference style embedding {512, 1}
//   weights  - MRTE weights
//
// Returns:
//   post-MRTE features {192, T_ssl}
struct ggml_tensor * sovits_text_encoder_mrte_block_forward(
    struct ggml_context                                * ctx,
    struct ggml_tensor                                 * ssl,
    struct ggml_tensor                                 * text,
    struct ggml_tensor                                 * ge,
    const sovits_text_encoder_mrte_block_weights       & weights);

// Build the SoVITS v2 `enc_p.encoder2` graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   x        - post-MRTE features {192, T}
//   weights  - encoder2 weights
//
// Returns:
//   encoded post-MRTE features {192, T}
struct ggml_tensor * sovits_text_encoder_post_block_forward(
    struct ggml_context                                * ctx,
    struct ggml_tensor                                 * x,
    const sovits_text_encoder_post_block_weights       & weights);

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

// SoVITS text-encoder text model: owns the loaded GGUF weights and ggml
// resources (except backend, which is borrowed from the caller).
struct sovits_text_encoder_text_model {
    sovits_text_encoder_text_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// SoVITS text-encoder MRTE model: owns the loaded GGUF weights and ggml
// resources (except backend, which is borrowed from the caller).
struct sovits_text_encoder_mrte_model {
    sovits_text_encoder_mrte_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// SoVITS text-encoder post model: owns the loaded GGUF weights and ggml
// resources (except backend, which is borrowed from the caller).
struct sovits_text_encoder_post_model {
    sovits_text_encoder_post_block_weights weights = {};

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

// Load a SoVITS text_encoder_text model from a GGUF file produced by
// `convert_sovits_text_encoder_text_to_gguf.py`.
bool sovits_text_encoder_text_model_load(
    const std::string & fname,
    sovits_text_encoder_text_model & model,
    ggml_backend_t backend);

// Load a SoVITS text_encoder_mrte model from a GGUF file produced by
// `convert_sovits_text_encoder_mrte_to_gguf.py`.
bool sovits_text_encoder_mrte_model_load(
    const std::string & fname,
    sovits_text_encoder_mrte_model & model,
    ggml_backend_t backend);

// Load a SoVITS text_encoder_post model from a GGUF file produced by
// `convert_sovits_text_encoder_post_to_gguf.py`.
bool sovits_text_encoder_post_model_load(
    const std::string & fname,
    sovits_text_encoder_post_model & model,
    ggml_backend_t backend);

// Free all resources owned by a SoVITS ref_enc model.
void sovits_ref_enc_model_free(sovits_ref_enc_model & model);

// Free all resources owned by a SoVITS quantizer model.
void sovits_quantizer_model_free(sovits_quantizer_model & model);

// Free all resources owned by a SoVITS text_encoder_ssl model.
void sovits_text_encoder_ssl_model_free(sovits_text_encoder_ssl_model & model);

// Free all resources owned by a SoVITS text_encoder_text model.
void sovits_text_encoder_text_model_free(sovits_text_encoder_text_model & model);

// Free all resources owned by a SoVITS text_encoder_mrte model.
void sovits_text_encoder_mrte_model_free(sovits_text_encoder_mrte_model & model);

// Free all resources owned by a SoVITS text_encoder_post model.
void sovits_text_encoder_post_model_free(sovits_text_encoder_post_model & model);

} // namespace gpt_sovits
