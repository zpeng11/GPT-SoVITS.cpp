#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gpt_sovits/sv.h"

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

    // v2Pro / v2ProPlus path (SynthesizerTrn with version in v2pro_set).
    // When true, MelStyleEncoder emits a gin_channels-wide ge (1024), an
    // external speaker-verification embedding is fused via sv_emb + PReLU,
    // and ge_to512 projects it down to 512 for enc_p. flow/dec still take
    // the 1024-wide ge. Read from GGUF KV "sovits.v2pro"; defaults to false.
    bool is_v2pro = false;

    // ge (style embedding) channel count: 512 for v2, 1024 for v2Pro/Plus.
    // Read from GGUF KV "sovits.ref_enc.out_dim"; defaults to 512.
    int64_t gin_channels = 512;
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

    // fc = Linear(128, gin_channels). 512 for v2, 1024 for v2Pro/Plus.
    struct ggml_tensor * fc_w;           // {128, gin_channels}
    struct ggml_tensor * fc_b;           // {gin_channels}
};

// ---------------------------------------------------------------------------
// SoVITS v2Pro / v2ProPlus extra weights (absent in v2).
//
// SynthesizerTrn adds three modules when `version in v2pro_set`:
//   sv_emb   = Linear(20480, gin_channels)   speaker-verif embedding fuse
//   ge_to512 = Linear(gin_channels, 512)     projects ge down for enc_p
//   prelu    = PReLU(gin_channels)           per-channel slope
//
// Inference path (per refer slice):
//   ge   = PReLU(MelStyleEncoder(spec) + sv_emb(sv))   -> {gin_channels, 1}
//   ge512= ge_to512(ge)                                -> {512, 1}  (enc_p)
// flow / dec still consume the 1024-wide `ge`.
//
// These tensors are only populated when hparams.is_v2pro; otherwise the
// pointers stay nullptr and the v2 path is used unchanged.
struct sovits_v2pro_block_weights {
    // Linear weights stored as PyTorch [out, in]; ggml reverses to {in, out},
    // matching the linear_2d convention (ne[0]=in, ne[1]=out).
    struct ggml_tensor * sv_emb_w;    // {20480, gin_channels}
    struct ggml_tensor * sv_emb_b;    // {gin_channels}
    struct ggml_tensor * ge_to512_w;  // {gin_channels, 512}
    struct ggml_tensor * ge_to512_b;  // {512}
    // PReLU per-channel negative slope.
    struct ggml_tensor * prelu_w;     // {gin_channels}
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

// Build the v2Pro/v2ProPlus ge post-processing graph.
//
// Mirrors SynthesizerTrn.decode's get_ge() is_v2pro branch:
//   sv_proj = sv_emb(sv)                 -> {gin_channels, 1}
//   ge     = PReLU(ge_raw + sv_proj)     -> {gin_channels, 1}
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   ge_raw  - MelStyleEncoder output {gin_channels, 1}
//   sv      - speaker-verification embedding {20480} (graph input)
//   weights - v2Pro extra weights (sv_emb + prelu)
//
// Returns:
//   flow/dec conditioning ge {gin_channels, 1}
struct ggml_tensor * sovits_v2pro_ge_forward(
    struct ggml_context                          * ctx,
    struct ggml_tensor                           * ge_raw,
    struct ggml_tensor                           * sv,
    const sovits_v2pro_block_weights             & weights);

// Project the v2Pro ge down to 512 channels for enc_p.
//
// Mirrors:  ge512 = ge_to512(ge)   -> {512, 1}
struct ggml_tensor * sovits_ge_to512_forward(
    struct ggml_context                          * ctx,
    struct ggml_tensor                           * ge,
    const sovits_v2pro_block_weights             & weights);

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
// SoVITS v2 unified model: owns the loaded GGUF weights and ggml resources
// for all 5 inference blocks (except backend, which is borrowed from the
// caller).
//
// A single unified GGUF (produced by `convert_sovits_to_gguf.py`) holds every
// block under disjoint tensor-name prefixes (ref_enc.*, quantizer.*,
// text_encoder_*.*, flow.*, generator.*), so one file + one load replaces the
// previous 5-file pipeline. Each member below is the weight sub-struct used
// directly by the matching *_block_forward graph builder in block.cpp.
// ---------------------------------------------------------------------------

struct sovits_model {
    sovits_mel_style_encoder_block_weights  ref_enc;       // MelStyleEncoder
    sovits_rvq_decode_block_weights        quantizer;      // RVQ codebook
    sovits_text_encoder_block_weights      text_encoder;   // enc_p (ssl/text/mrte/post)
    sovits_flow_block_weights              flow;           // ResidualCouplingBlock
    sovits_generator_block_weights         generator;      // HiFi-GAN
    sovits_v2pro_block_weights             v2pro;          // v2Pro/Plus extras (else nullptr)

    sovits_hparams hparams = {};

    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context     * ctx_w   = nullptr;
};

// Load a unified SoVITS model from a single GGUF file produced by
// `convert_sovits_to_gguf.py`.
//
// Reads sovits.semantic_frame_rate from the GGUF KV to populate
// model.hparams. If the key is missing, defaults to "25hz" (the shipped
// v2 pretrained checkpoint value).
//
// Parameters:
//   fname   - path to the unified .gguf file
//   model   - output model struct (will be populated)
//   backend - ggml backend for tensor allocation (caller-owned; not freed by
//             sovits_model_free)
//
// Returns:
//   true on success, false on failure.
bool sovits_model_load(
    const std::string & fname,
    sovits_model & model,
    ggml_backend_t backend);

// Free all resources owned by a unified SoVITS model.
void sovits_model_free(sovits_model & model);

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

    // Cached style embedding ge {gin_channels, 1} F32, computed once per
    // reference. Consumed by flow + dec. Owned: ctx_ge holds the tensor
    // metadata, buf_ge holds its data. Replaced atomically by compute_ge only
    // after the new pair is fully computed (a failed compute_ge leaves the
    // previous cache in place).
    struct ggml_context  * ctx_ge = nullptr;
    ggml_backend_buffer_t  buf_ge = nullptr;
    struct ggml_tensor   * ge     = nullptr;

    // Cached enc_p conditioning ge_enc {512, 1} F32. For v2 this aliases ge
    // (ge is already 512-wide; ctx_ge_enc/buf_ge_enc stay null). For
    // v2Pro/v2ProPlus it is a separately allocated tensor holding ge_to512(ge).
    // nullptr until compute_ge runs. When aliased, freeing the ge buffer also
    // releases ge_enc.
    struct ggml_context  * ctx_ge_enc = nullptr;
    ggml_backend_buffer_t  buf_ge_enc = nullptr;
    struct ggml_tensor   * ge_enc     = nullptr;

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

// One reference slice for style-embedding computation.
//
// spec is the v2 704-channel MelStyleEncoder input: caller supplies refer[:, :704]
// for one reference clip. Memory layout is ggml {704, T} row-major, i.e. the
// element for channel c (0..703) at frame t (0..T-1) is spec[t * 704 + c].
// (PyTorch stores refer as [1, 1025, T] BCT; slice channels to 704 and repack
// so that channel is the fast axis.)
//
// Pointer lifetime: spec and sv are BORROWED for the duration of the
// sovits_session_compute_ge[_audio] call only. The session copies the resulting
// ge/ge_enc into session-owned buffers and never retains these pointers, so the
// caller may free/reuse spec and sv as soon as the call returns.
struct sovits_refer_view {
    const float * spec = nullptr;   // {704, T} row-major, spec[t*704 + c]
    int64_t       T    = 0;

    // Speaker-verification embedding {20480} for v2Pro/v2ProPlus (one per
    // slice, produced externally by ERes2NetV2). Must be non-null for every
    // slice when model.hparams.is_v2pro; ignored (may be null) for v2.
    const float * sv = nullptr;
};

// Compute and cache the style embedding ge from reference audio.
//
// For each refer slice, builds a temporary graph that runs MelStyleEncoder
// on the reference features; for v2Pro/v2ProPlus it additionally fuses the
// per-slice sv embedding (sv_emb + PReLU). The per-slice ge results are
// averaged element-wise (mirrors SynthesizerTrn.decode's
// `torch.stack(ges, 0).mean(0)` for list refer input) and stored in the
// session. Note the mean is taken over the POST-fuse per-slice ge, so for
// v2Pro the average is of PReLU(MelStyleEncoder(spec_i) + sv_emb(sv_i)), which
// is not the same as fusing the averaged encoder output.
//
// Two conditionings are produced:
//   ge     {gin_channels, 1}  gin_channels = 512 (v2) or 1024 (v2Pro/Plus).
//                             Consumed by flow + decoder.
//   ge_enc {512, 1}           enc_p conditioning. For v2Pro/Plus it is
//                             ge_to512(ge); for v2 it ALIASES ge (ge is
//                             already 512-wide, so get_ge_enc() == get_ge()).
//
// Replacement is atomic: the new ge/ge_enc are fully computed and allocated in
// temporaries first, and the previously cached pair is only freed once both are
// valid. On failure the function returns false and the PREVIOUS cache is left
// unchanged, so a caller can safely retry or keep using the old reference.
//
// Recovery scope: host/C++ std::bad_alloc raised while building the ge cache
// (host vectors, ggml host allocations, C++ trampolines) is caught and returns
// false with the old cache preserved and nothing leaked. This does NOT apply to
// backend buffer-type / allocator callbacks: they follow the C ABI and must not
// throw (a throw from inside an alloc callback unwinds through C frames and is
// not a supported recovery point). Failures inside ggml itself (asserts /
// aborts) or a corrupt backend are likewise not recovery points.
//
// Input validation returns false rather than asserting.
//
// Parameters:
//   session - initialized SoVITS session
//   model   - loaded unified SoVITS model (ref_enc / v2pro weights read)
//   refers  - non-empty list of refer slices (spec + optional sv)
//
// Returns true on success, false on failure.
bool sovits_session_compute_ge(
    sovits_session       & session,
    const sovits_model   & model,
    const std::vector<sovits_refer_view> & refers);

// v2 legacy compatibility entry: same as sovits_session_compute_ge but takes
// the pre-v2Pro API's (spec, T) pair list.
//
// This is a separate named function, NOT an overload, on purpose. Adding an
// overload taking std::vector<std::pair<const float *, int64_t>> would make the
// common braced call
//     compute_ge(session, model, {{spec, T}})
// ambiguous: the braced list can equally initialize a sovits_refer_view
// (aggregate, sv defaults to null) or a std::pair. The main entry above still
// accepts that braced form via aggregate initialization; this named entry
// exists for callers that already hold a std::vector<std::pair<...>>.
//
// It passes sv = nullptr for every slice, so it only works on v2 models; on a
// v2Pro/v2ProPlus model sovits_session_compute_ge returns false (missing sv).
bool sovits_session_compute_ge_specs(
    sovits_session       & session,
    const sovits_model   & model,
    const std::vector<std::pair<const float *, int64_t>> & refers);

// Get the cached flow/dec style embedding tensor (ge), or nullptr if not
// computed. Returned ge/ge_enc pointers are borrowed: successful cache replacement
// or session_free invalidates them; a failed cache update preserves them.
struct ggml_tensor * sovits_session_get_ge(const sovits_session & session);

// Get the cached enc_p conditioning tensor (ge_enc {512, 1}), or nullptr.
// For v2 this is the same tensor as ge.
struct ggml_tensor * sovits_session_get_ge_enc(const sovits_session & session);

// One reference slice for the audio-driven ge path (v2Pro/v2ProPlus): in
// addition to the 704-channel spec, carries the 16 kHz mono waveform used to
// compute the SV embedding in-process via sv_fbank + sv_forward3.
//
// spec uses the same layout/lifetime as sovits_refer_view (spec[t*704 + c],
// borrowed only for the call). audio16k / n_samples are likewise borrowed for
// the duration of the call; the samples are consumed by sv_fbank and not
// retained.
struct sovits_audio_refer_view {
    const float * spec     = nullptr;   // {704, T} row-major, spec[t*704 + c]
    int64_t       T        = 0;
    const float * audio16k = nullptr;   // 16 kHz mono waveform (float32)
    int64_t       n_samples = 0;
};

// Compute and cache ge, deriving the per-refer SV embedding in-process from
// the 16 kHz audio (no external sv dump needed). v2Pro/v2ProPlus only.
//
// Thinly composes sv_fbank + sv_forward3 (per refer) with the existing
// sovits_session_compute_ge, so correctness rests on the two parity-verified
// halves. Requires model.hparams.is_v2pro and a loaded sv_model.
//
// Input validation returns false rather than asserting. The cache is updated
// with the same atomic semantics as sovits_session_compute_ge: any failure
// (bad input, fbank/SV error, or ge compute error) leaves the previous
// ge/ge_enc unchanged. Produces both conditionings (ge for flow/dec, ge_enc
// for enc_p) exactly like sovits_session_compute_ge.
//
// Parameters:
//   session  - initialized SoVITS session
//   model    - loaded unified SoVITS model (v2pro weights used)
//   sv_model - loaded ERes2NetV2 SV model
//   refers   - non-empty list of audio refer slices (spec + audio16k)
//
// Returns true on success, false on failure.
bool sovits_session_compute_ge_audio(
    sovits_session       & session,
    const sovits_model   & model,
    const sv_model       & sv_model,
    const std::vector<sovits_audio_refer_view> & refers);

// Run the full SoVITS v2/v2Pro/v2ProPlus pipeline end-to-end.
//
// Mirrors SynthesizerTrn.decode (speed=1, batch=1):
//   1. codes -> quantizer.decode         -> quantized {768, T_codes}
//   2. (if 25hz) double time dim         -> quantized {768, 2*T_codes}
//   3. enc_p(quantized, text, ge_enc)    -> m, logs   {192, T_ssl} each
//   4. host: randn ~ N(0, noise_scale²)  -> randn     {192, T_ssl}
//   5. z_p = m + randn * exp(logs)       -> z_p       {192, T_ssl}
//   6. z = flow(z_p, ge, reverse=True)   -> z         {192, T_ssl}
//   7. wav = dec(z, ge)                  -> wav       {1, T_ssl*640}
//   8. copy wav to wav_out
// where T_ssl = (semantic_frame_rate_25hz ? 2 : 1) * T_codes.
//
// Parameters:
//   session - initialized session with ge cached
//   model   - loaded unified SoVITS model
//   codes   - semantic token ids {T_codes} (i32), host pointer
//   T_codes - number of semantic tokens
//   text    - phoneme token ids {T_text} (i32), host pointer
//   T_text  - number of phoneme tokens
//   wav_out - caller-allocated output buffer (F32)
//   wav_cap - capacity of wav_out in floats; must be >= T_ssl * 640
//
// Input validation returns false rather than asserting. That includes the
// length arithmetic (25hz doubling and the *640 / *192 products are
// overflow-checked) and every token id in `codes` / `text` being within
// [0, codebook_size) / [0, text_vocab). Host/allocator std::bad_alloc is caught
// and reported as false; ggml asserts / a corrupt backend are not recovery
// points.
//
// Returns true on success. On success, exactly T_ssl * 640 samples are
// written to wav_out.
bool sovits_session_forward(
    sovits_session       & session,
    const sovits_model   & model,
    const int32_t        * codes,
    int64_t                T_codes,
    const int32_t        * text,
    int64_t                T_text,
    float                * wav_out,
    int64_t                wav_cap);

} // namespace gpt_sovits
