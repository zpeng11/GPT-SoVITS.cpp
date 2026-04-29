#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <string>

namespace gpt_sovits {

static constexpr int kSovitsTextEncoderSslLayers = 3;
static constexpr int kSovitsTextEncoderTextLayers = 6;
static constexpr int kSovitsTextEncoderPostLayers = 3;

static constexpr int kSovitsGeneratorIn = 192;
static constexpr int kSovitsGeneratorGin = 512;
static constexpr int kSovitsGeneratorStages = 5;
static constexpr int kSovitsGeneratorBranches = 3;
static constexpr int kSovitsGeneratorResLayers = 3;

static constexpr int kSovitsFlowChannels = 192;
static constexpr int kSovitsFlowHidden = 192;
static constexpr int kSovitsFlowHalfChannels = 96;
static constexpr int kSovitsFlowKernel = 5;
static constexpr int kSovitsFlowWNLayers = 4;
static constexpr int kSovitsFlowNFlows = 4;
static constexpr int kSovitsFlowGin = 512;

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

// Weights for one SoVITS v2 relative-position encoder layer:
//   hidden {192, T} -> relative-position self-attention -> residual + LayerNorm
//   -> Conv1d(192, 768, k=3) -> ReLU -> Conv1d(768, 192, k=3)
//   -> residual + LayerNorm -> {192, T}
//
// Shared by `enc_p.encoder_ssl`, `enc_p.encoder_text`, and `enc_p.encoder2`
// in `module/models_onnx.py`.
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

// Weights for the SoVITS v2 `enc_p.ssl_proj + enc_p.encoder_ssl` branch:
//   ssl {768, T} -> Conv1d(768, 192, k=1)
//   -> 3 x [relative-position encoder layer] -> {192, T}
//
// Scope:
//   - single-sample inference only
//   - fixed v2 hyperparameters from the shipped checkpoint:
//       hidden=192, ffn=768, n_heads=2, n_layers=3, kernel=3, window=4
//   - dropout is skipped (eval-mode behavior)
//   - mask handling is fixed to the exported v2 inference path in
//     `module/models_onnx.py`: all frames are valid
struct sovits_text_encoder_ssl_block_weights {
    // Conv1d(768, 192, k=1)
    struct ggml_tensor * ssl_proj_w;    // {1, 768, 192}
    struct ggml_tensor * ssl_proj_b;    // {192}

    std::array<sovits_relpos_encoder_layer_weights, kSovitsTextEncoderSslLayers> layers;
};

// Weights for the SoVITS v2 `enc_p.text_embedding + enc_p.encoder_text` branch:
//   text ids {T} -> Embedding(732, 192)
//   -> 6 x [relative-position encoder layer] -> {192, T}
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

// Weights for the inference-only fused SoVITS v2 `enc_p.mrte` branch:
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

// Weights for the SoVITS v2 `enc_p.encoder2` branch:
//   features {192, T} -> 3 x [relative-position encoder layer] -> {192, T}
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

    // Conv1d(192, 384, k=1), used by full `enc_p` to produce [m, logs].
    struct ggml_tensor * proj_w;    // {1, 192, 384}
    struct ggml_tensor * proj_b;    // {384}
};

// Weights for the full SoVITS v2 `enc_p` graph at fixed speed=1:
//   ssl {768, T_ssl} -> SSL branch
//   text ids {T_text} -> text branch
//   ssl/text/ge -> MRTE branch -> post branch -> proj -> split(m, logs)
//
// Returns the same activations as `TextEncoder.forward(..., speed=1)` in
// `module/models_onnx.py`, except the all-ones `y_mask` is omitted.
struct sovits_text_encoder_block_weights {
    sovits_text_encoder_ssl_block_weights  ssl;
    sovits_text_encoder_text_block_weights text;
    sovits_text_encoder_mrte_block_weights mrte;
    sovits_text_encoder_post_block_weights post;
};

// ---------------------------------------------------------------------------
// SoVITS v2 flow (ResidualCouplingBlock) inference block.
//
// This block implements the inverse flow used by SynthesizerTrn.forward
// with reverse=True.  The flow turns a Gaussian sample z_p {192, T} into
// the decoder input z {192, T}, conditioned on the style embedding
// g = ge {512, 1}.
//
// Architecture (v2 hyperparameters from the shipped checkpoint):
//   ResidualCouplingBlock(channels=192, hidden=192, kernel=5,
//                         dilation=1, n_layers=4, n_flows=4, gin=512,
//                         mean_only=True)
//
// Each of the 4 coupling layers contains a WaveNet (WN) with 4 dilated-
// Conv1d layers and gated tanh·sigmoid activations.  Channel flipping
// (torch.flip along the channel axis) alternates which half is
// transformed.
//
// Scope:
//   - single-sample inference (reverse mode) only
//   - x_mask is all-ones (fixed-length input, no padding)
//   - dropout is skipped (eval-mode behaviour)
//   - weight_norm is fused at GGUF-conversion time
// ---------------------------------------------------------------------------

// One WN layer: dilated Conv1d(k=5, dil=1) + gated activation + 1x1 projection.
struct sovits_wn_layer_weights {
    // in_layer: weight_norm Conv1d(H, 2H, K=5, dil=1, pad=2)
    // ggml layout: {kernel, in_channels, out_channels}
    struct ggml_tensor * in_w;     // {5, 192, 384}
    struct ggml_tensor * in_b;     // {384}

    // res_skip_layer: weight_norm Conv1d(H, out, K=1)
    // out = 2H for layers 0..2, out = H for the last layer
    struct ggml_tensor * rs_w;     // {1, 192, 384} or {1, 192, 192}
    struct ggml_tensor * rs_b;     // {384} or {192}
};

// WaveNet inside one coupling layer (4 dilated layers + global condition).
struct sovits_wn_weights {
    // cond_layer: Conv1d(gin, 2*H*n_layers, K=1)  -- feeds g into each layer
    struct ggml_tensor * cond_w;   // {1, 512, 1536}
    struct ggml_tensor * cond_b;   // {1536}

    std::array<sovits_wn_layer_weights, kSovitsFlowWNLayers> layers;
};

// One coupling (affine-coupling) layer.
// input {192, T} is split into x0 {96, T} and x1 {96, T}.
// x0 passes through pre → WN → post to predict a mean correction m {96, T}.
// With mean_only=True the reverse step is simply  x1 = x1 - m.
struct sovits_flow_layer_weights {
    // pre: Conv1d(half, H, K=1)
    struct ggml_tensor * pre_w;    // {1, 96, 192}
    struct ggml_tensor * pre_b;    // {192}

    sovits_wn_weights enc;

    // post: Conv1d(H, half, K=1)  -- mean_only → 96 output channels
    struct ggml_tensor * post_w;   // {1, 192, 96}
    struct ggml_tensor * post_b;   // {96}
};

// Full flow block: 4 coupling layers interspersed with channel flips.
struct sovits_flow_block_weights {
    std::array<sovits_flow_layer_weights, kSovitsFlowNFlows> layers;
};

// ---------------------------------------------------------------------------
// SoVITS v2 Generator inference block.
//
// This block implements `SynthesizerTrn.dec` in the shipped SoVITS v2
// checkpoint, with all architecture choices fixed to the exported inference
// path in `module/models_onnx.py`:
//   Conv1d(192, 512, k=7) + global condition Conv1d(512, 512, k=1)
//   -> 5 x [LeakyReLU -> ConvTranspose1d -> 3-way ResBlock1 average]
//   -> LeakyReLU -> Conv1d(16, 1, k=7, bias=False) -> tanh
//
// Each stage uses the checkpoint-fixed HiFi-GAN style layout:
//   stages[0]: upsample 512 -> 256, kernel=16, stride=10, padding=3
//   stages[1]: upsample 256 -> 128, kernel=16, stride=8,  padding=4
//   stages[2]: upsample 128 -> 64, kernel=8,  stride=2,  padding=3
//   stages[3]: upsample 64  -> 32, kernel=2,  stride=2,  padding=0
//   stages[4]: upsample 32  -> 16, kernel=2,  stride=2,  padding=0
//
// Each stage then averages 3 parallel ResBlock1 branches with kernel sizes
// {3, 7, 11}; every ResBlock1 contains 3 residual sublayers with dilations
// {1, 3, 5} followed by a dilation-1 Conv1d of the same kernel size.
//
// Scope:
//   - single-sample inference only
//   - fixed SoVITS v2 architecture and channel counts
//   - weight_norm is fused at GGUF-conversion time for ConvTranspose1d and
//     ResBlock1 Conv1d weights
// ---------------------------------------------------------------------------

struct sovits_generator_conv_weights {
    struct ggml_tensor * w = nullptr;
    struct ggml_tensor * b = nullptr;
};

struct sovits_generator_resblock1_weights {
    std::array<sovits_generator_conv_weights, kSovitsGeneratorResLayers> convs1;
    std::array<sovits_generator_conv_weights, kSovitsGeneratorResLayers> convs2;
};

struct sovits_generator_stage_weights {
    sovits_generator_conv_weights up;
    std::array<sovits_generator_resblock1_weights, kSovitsGeneratorBranches> resblocks;
};

struct sovits_generator_block_weights {
    sovits_generator_conv_weights conv_pre;
    sovits_generator_conv_weights cond;
    std::array<sovits_generator_stage_weights, kSovitsGeneratorStages> stages;
    struct ggml_tensor * conv_post_w = nullptr;
};

struct sovits_text_encoder_result {
    struct ggml_tensor * x;     // {192, T_ssl}
    struct ggml_tensor * m;     // {192, T_ssl}
    struct ggml_tensor * logs;  // {192, T_ssl}
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

// Build the full SoVITS v2 `enc_p` graph at fixed speed=1.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   ssl      - quantized SSL features {768, T_ssl}
//   text     - phoneme token ids {T_text} (i32)
//   ge       - reference style embedding {512, 1}
//   weights  - full text-encoder weights
//
// Returns:
//   x, m, logs where each tensor uses ggml layout and `x`/`m`/`logs`
//   match `TextEncoder.forward(..., speed=1)`.
sovits_text_encoder_result sovits_text_encoder_block_forward(
    struct ggml_context                           * ctx,
    struct ggml_tensor                            * ssl,
    struct ggml_tensor                            * text,
    struct ggml_tensor                            * ge,
    const sovits_text_encoder_block_weights       & weights);

// Build the SoVITS v2 flow (ResidualCouplingBlock) inverse graph.
//
// This implements the inference path of SynthesizerTrn.forward where
//   z = self.flow(z_p, y_mask, g=ge, reverse=True)
// The flow turns a Gaussian sample z_p {192, T} into the decoder input
// z {192, T}, conditioned on the style embedding g = ge {512, 1}.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   x        - Gaussian sample z_p {192, T}
//   g        - style embedding ge {512, 1}
//   weights  - flow block weights
//
// Returns:
//   decoder input z {192, T}
struct ggml_tensor * sovits_flow_block_inverse_forward(
    struct ggml_context                        * ctx,
    struct ggml_tensor                         * x,
    struct ggml_tensor                         * g,
    const sovits_flow_block_weights            & weights);

// Build the SoVITS v2 Generator graph.
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   z        - decoder input latent {192, T}
//   g        - style embedding ge {512, 1}
//   weights  - Generator weights for the fixed v2 architecture
//
// Returns:
//   waveform {1, T * 640}
struct ggml_tensor * sovits_generator_block_forward(
    struct ggml_context                         * ctx,
    struct ggml_tensor                          * z,
    struct ggml_tensor                          * g,
    const sovits_generator_block_weights        & weights);

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

// SoVITS full text-encoder model: owns the loaded GGUF weights and ggml
// resources (except backend, which is borrowed from the caller).
struct sovits_text_encoder_model {
    sovits_text_encoder_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// SoVITS flow model: owns the loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
struct sovits_flow_model {
    sovits_flow_block_weights weights = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// SoVITS generator model: owns the loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
struct sovits_generator_model {
    sovits_generator_block_weights weights = {};

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

// Load a SoVITS full text_encoder model from a GGUF file produced by
// `convert_sovits_text_encoder_to_gguf.py`.
bool sovits_text_encoder_model_load(
    const std::string & fname,
    sovits_text_encoder_model & model,
    ggml_backend_t backend);

// Load a SoVITS flow model from a GGUF file produced by
// `convert_sovits_flow_to_gguf.py`.
bool sovits_flow_model_load(
    const std::string & fname,
    sovits_flow_model & model,
    ggml_backend_t backend);

// Load a SoVITS generator model from a GGUF file produced by
// `convert_sovits_generator_to_gguf.py`.
bool sovits_generator_model_load(
    const std::string & fname,
    sovits_generator_model & model,
    ggml_backend_t backend);

// Free all resources owned by a SoVITS ref_enc model.
void sovits_ref_enc_model_free(sovits_ref_enc_model & model);

// Free all resources owned by a SoVITS quantizer model.
void sovits_quantizer_model_free(sovits_quantizer_model & model);

// Free all resources owned by a SoVITS text_encoder model.
void sovits_text_encoder_model_free(sovits_text_encoder_model & model);

// Free all resources owned by a SoVITS flow model.
void sovits_flow_model_free(sovits_flow_model & model);

// Free all resources owned by a SoVITS generator model.
void sovits_generator_model_free(sovits_generator_model & model);

} // namespace gpt_sovits
