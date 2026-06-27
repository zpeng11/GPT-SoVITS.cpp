#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// SoVITS v2 model hyperparameters
//
// Read from GGUF KV metadata at load time. Currently only carries the
// semantic_frame_rate flag; the rest of v2 architecture is fixed in block.cpp.
// ---------------------------------------------------------------------------

struct sovits_hparams {
    // True when the source checkpoint's semantic_frame_rate is "25hz".
    // Requires time-doubling of quantized SSL features before enc_p, to
    // match the 50hz expectation of the text encoder / flow / decoder.
    // Read from quantizer GGUF KV "sovits.semantic_frame_rate"; defaults
    // to true (matches the shipped v2 pretrained checkpoint).
    bool semantic_frame_rate_25hz = true;
};

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
// The self-attention's Q/K/V projections are exported from the GGUF as a
// single fused linear weight (ref_enc.attention.qkv_w / qkv_b), produced by
// concatenating the PyTorch w_qs/w_ks/w_vs weights along the output axis at
// conversion time. This matches the layout used by the text-encoder relpos
// layers and lets inference issue one matmul instead of three.
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
    // Fused Q/K/V projection exported as a single linear weight, matching
    // the layout used by the text-encoder relpos layers. The 3*hidden output
    // channels are laid out as [Q, K, V] so attention_block_forward can split
    // the linear output into q/k/v with three O(1) channel views.
    struct ggml_tensor * qkv_w;    // {128, 384}
    struct ggml_tensor * qkv_b;    // {384}
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
    // Fused 1x1 projections for self-attention exported as linear weights.
    // Output channels are laid out as [Q, K, V].
    struct ggml_tensor * qkv_w;    // {192, 576}
    struct ggml_tensor * qkv_b;    // {576}
    struct ggml_tensor * out_w;    // {192, 192}
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
    // Conv1d(768, 192, k=1) exported as a linear projection.
    struct ggml_tensor * ssl_proj_w;    // {768, 192}
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
    struct ggml_tensor * ssl_fused_w;   // {192, 704}
    struct ggml_tensor * ssl_fused_b;   // {704}
    struct ggml_tensor * text_kv_w;     // {192, 1024}
    struct ggml_tensor * text_kv_b;     // {1024}
    struct ggml_tensor * attn_out_w;    // {512, 192}
    struct ggml_tensor * attn_out_b;    // {192}
    struct ggml_tensor * ge_out_w;      // {512, 192}
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

    // Conv1d(192, 384, k=1) exported as a linear projection, used by full
    // `enc_p` to produce [m, logs].
    struct ggml_tensor * proj_w;    // {192, 384}
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
// SoVITS v2 pipeline helpers (stateless graph builders)
// ---------------------------------------------------------------------------

// Double the time dimension of quantized SSL features (25hz -> 50hz upsample).
//
// Mirrors the python path in SynthesizerTrn.forward when
// semantic_frame_rate == "25hz":
//   dquantized = torch.cat([quantized, quantized]).permute(1, 2, 0)
//   quantized  = dquantized.contiguous().view(1, self.ssl_dim, -1)
// which is equivalent to torch.repeat_interleave(quantized, 2, dim=time).
//
// Parameters:
//   ctx       - ggml context for tensor/op allocation
//   quantized - input features {768, T}
//   indices   - caller-built index tensor {2T} (I32) with values
//               [0, 0, 1, 1, ..., T-1, T-1]; the caller is responsible for
//               filling it before graph compute.
//
// Returns:
//   upsampled features {768, 2T} where output[:, 2t] == output[:, 2t+1] == input[:, t].
struct ggml_tensor * sovits_quantized_double_25hz_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * quantized,
    struct ggml_tensor  * indices);

// Sample z_p from text-encoder outputs (m, logs) and caller-provided noise.
//
// Mirrors the python:
//   z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
//
// noise_scale is folded into the caller-provided randn input (host-side RNG
// uses std=noise_scale), so the graph op is just z_p = m + randn * exp(logs).
// This keeps the graph topology independent of noise_scale.
//
// Parameters:
//   ctx   - ggml context
//   m     - mean {192, T} from text encoder
//   logs  - log-std {192, T} from text encoder
//   randn - caller-filled N(0, noise_scale²) samples {192, T} (graph input)
//
// Returns:
//   z_p {192, T}
struct ggml_tensor * sovits_sample_z_p_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * m,
    struct ggml_tensor  * logs,
    struct ggml_tensor  * randn);

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

// ---------------------------------------------------------------------------
// SoVITS v2 aggregate: bundles all 5 sub-models + hparams
//
// Each sub-model retains independent ownership of its GGUF weights and
// ggml resources; the aggregate only groups references so callers don't
// have to manage 5 separate structs.
// ---------------------------------------------------------------------------

struct sovits_models {
    sovits_ref_enc_model      ref_enc;
    sovits_quantizer_model    quantizer;
    sovits_text_encoder_model text_encoder;
    sovits_flow_model         flow;
    sovits_generator_model    generator;

    sovits_hparams hparams = {};
};

// Load all 5 SoVITS sub-models from their respective GGUF files.
//
// Reads sovits.semantic_frame_rate from the quantizer GGUF KV to populate
// models.hparams. If the key is missing, defaults to "25hz" (the shipped
// v2 pretrained checkpoint value).
//
// On failure, any sub-models loaded before the failure are freed before
// returning; the caller does not need to clean up.
//
// Parameters:
//   ref_enc_path      - path to ref_enc GGUF (MelStyleEncoder weights)
//   quantizer_path    - path to quantizer GGUF (RVQ codebook + hparams KV)
//   text_encoder_path - path to text_encoder GGUF (enc_p weights)
//   flow_path         - path to flow GGUF (ResidualCouplingBlock weights)
//   generator_path    - path to generator GGUF (HiFi-GAN Generator weights)
//   models            - output aggregate struct (will be populated)
//   backend           - ggml backend (borrowed; not freed by sovits_models_free)
//
// Returns true on success, false on failure (errors printed to stderr).
bool sovits_models_load(
    const std::string & ref_enc_path,
    const std::string & quantizer_path,
    const std::string & text_encoder_path,
    const std::string & flow_path,
    const std::string & generator_path,
    sovits_models & models,
    ggml_backend_t backend);

// Free all sub-models owned by sovits_models.
void sovits_models_free(sovits_models & models);

// ---------------------------------------------------------------------------
// SoVITS v2 session: cached style embedding + host-side RNG
//
// Manages the cross-call state needed by SynthesizerTrn.forward:
//   - cached ge {512, 1} style embedding (one reference per session)
//   - host-side RNG for z_p sampling noise (bundled so the caller does not
//     need to manage noise generation externally)
//   - scratch buffers reused across forward calls (grown on demand)
//
// Each forward call builds a fresh graph (T varies per call); there is no
// persistent decode graph like t2s_session has, since sovits forward is a
// single-shot pipeline rather than an iterative AR decode.
// ---------------------------------------------------------------------------

struct sovits_session {
    // Configuration (set at init, immutable across forward calls)
    float    noise_scale = 0.5f;   // folded into host-side RNG std
    uint64_t rng_seed    = 0;      // stored for diagnostics

    // Borrowed backend (not freed by sovits_session_free)
    ggml_backend_t backend = nullptr;

    // Host-side RNG state (bundled so caller doesn't manage noise)
    std::mt19937 rng;

    // Cached style embedding ge {512, 1} F32, computed once per reference.
    // Owned: ctx_ge holds the tensor metadata, buf_ge holds its data.
    struct ggml_context  * ctx_ge = nullptr;
    ggml_backend_buffer_t  buf_ge = nullptr;
    struct ggml_tensor   * ge     = nullptr;

    // Scratch buffers reused across forward calls (grown on demand,
    // never shrunk; steady-state forward is allocation-free).
    std::vector<int32_t> scratch_double_idx;  // 25hz index, [0,0,1,1,...]
    std::vector<float>   scratch_randn;       // N(0, noise_scale²) samples
};

// Initialize a SoVITS session.
//
// The session does not own a backend; the caller must keep the backend alive
// for the lifetime of the session.
//
// Parameters:
//   session     - output session struct (will be populated)
//   backend     - ggml backend (borrowed)
//   noise_scale - noise scale for z_p sampling, folded into host-side RNG
//                 (default 0.5 matches python SynthesizerTrn.forward default)
//   rng_seed    - RNG seed for reproducible noise (use std::random_device{}
//                 externally for true randomness)
//
// Returns true on success, false on failure.
bool sovits_session_init(
    sovits_session & session,
    ggml_backend_t   backend,
    float            noise_scale = 0.5f,
    uint64_t         rng_seed    = 0xdeadbeefULL);

// Free all resources owned by a SoVITS session (ge cache + scratch buffers).
// Does not free the borrowed backend.
void sovits_session_free(sovits_session & session);

// Compute and cache the style embedding ge from reference audio.
//
// For each refer slice, builds a temporary graph that runs MelStyleEncoder
// on the reference features; the per-slice results are averaged element-wise
// (mirrors SynthesizerTrn.decode's `torch.stack(ges, 0).mean(0)` for list
// refer input), and the mean is copied into a session-owned persistent
// tensor (session.ge). Subsequent sovits_session_forward calls reuse it.
//
// Switching references requires re-calling this; the previously cached ge
// is freed first. Each refer slice's `data` must already be sliced to the
// v2 704-channel slice (i.e. caller does refer[:, :704]).
//
// Parameters:
//   session - initialized SoVITS session
//   models  - loaded SoVITS models (ref_enc weights are read)
//   refers  - list of {ptr to reference spectrogram features {704, T},
//             T frames}. Must be non-empty. Each entry's data is F32,
//             row-major C*T layout matching ggml {704, T}. Different
//             entries may have different T.
//
// Returns true on success, false on failure.
bool sovits_session_compute_ge(
    sovits_session       & session,
    const sovits_models  & models,
    const std::vector<std::pair<const float *, int64_t>> & refers);

// Get the cached style embedding tensor, or nullptr if not computed.
struct ggml_tensor * sovits_session_get_ge(const sovits_session & session);

// Run the full SoVITS v2 forward pipeline end-to-end.
//
// Mirrors SynthesizerTrn.forward (v2 path, speed=1):
//   1. codes -> quantizer.decode         -> quantized {768, T_codes}
//   2. (if 25hz) double time dim         -> quantized {768, 2*T_codes}
//   3. enc_p(quantized, text, ge)        -> m, logs   {192, T_ssl} each
//   4. host: randn ~ N(0, noise_scale²)  -> randn     {192, T_ssl}
//   5. z_p = m + randn * exp(logs)       -> z_p       {192, T_ssl}
//   6. z = flow(z_p, ge, reverse=True)   -> z         {192, T_ssl}
//   7. wav = dec(z, ge)                  -> wav       {1, T_ssl*640}
//   8. copy wav to wav_out
// where T_ssl = (semantic_frame_rate_25hz ? 2 : 1) * T_codes.
//
// Parameters:
//   session - initialized session with ge cached
//   models  - loaded SoVITS models
//   codes   - semantic token ids {T_codes} (i32), host pointer
//   T_codes - number of semantic tokens
//   text    - phoneme token ids {T_text} (i32), host pointer
//   T_text  - number of phoneme tokens
//   wav_out - caller-allocated output buffer (F32)
//   wav_cap - capacity of wav_out in floats; must be >= T_ssl * 640
//
// Returns true on success. On success, exactly T_ssl * 640 samples are
// written to wav_out.
bool sovits_session_forward(
    sovits_session       & session,
    const sovits_models  & models,
    const int32_t        * codes,
    int64_t                T_codes,
    const int32_t        * text,
    int64_t                T_text,
    float                * wav_out,
    int64_t                wav_cap);

} // namespace gpt_sovits
