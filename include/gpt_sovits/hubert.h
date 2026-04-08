#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <string>

namespace gpt_sovits {

// CN-HuBERT base inference blocks.
//
// These blocks match the Hugging Face `HubertModel` eager-attention inference
// path used by GPT-SoVITS:
//   raw waveform -> 7-layer Conv1d feature encoder -> feature projection
//   -> positional Conv1d embedding -> 12-layer Transformer encoder.
//
// Scope of this component:
//   - single-sample inference only
//   - no attention-mask support
//   - no SpecAugment
//   - no layerdrop
//   - no stable-layer-norm variant
//
// Public activations use the project's usual {channels, time} convention.

struct hubert_feature_encoder_block_weights {
    // Feature extractor Conv1d kernels in ggml layout {kernel, in_channels, out_channels}.
    // Layer 0: {10, 1, 512}
    // Layers 1-4: {3, 512, 512}
    // Layers 5-6: {2, 512, 512}
    std::array<struct ggml_tensor *, 7> conv_w;

    // GroupNorm(512, 512) affine parameters for feature extractor layer 0.
    struct ggml_tensor * conv0_norm_w;   // {512}
    struct ggml_tensor * conv0_norm_b;   // {512}
};

// Build the 7-layer HuBERT feature extractor graph.
//
// Parameters:
//   ctx          - ggml context for tensor/op allocation
//   input_values - mono waveform samples {T} or {T, 1}
//   weights      - feature extractor weights
//
// Returns:
//   Extracted features {512, T'}.
struct ggml_tensor * hubert_feature_encoder_block_forward(
    struct ggml_context                       * ctx,
    struct ggml_tensor                        * input_values,
    const hubert_feature_encoder_block_weights & weights);

struct hubert_feature_projection_block_weights {
    // LayerNorm(512)
    struct ggml_tensor * layer_norm_w;   // {512}
    struct ggml_tensor * layer_norm_b;   // {512}

    // Linear(512, 768)
    struct ggml_tensor * projection_w;   // {512, 768}
    struct ggml_tensor * projection_b;   // {768}
};

// Build the HuBERT feature projection graph:
//   LayerNorm(512) -> Linear(512, 768)
//
// Parameters:
//   ctx      - ggml context for tensor/op allocation
//   features - extracted features {512, T}
//   weights  - projection weights
//
// Returns:
//   Projected features {768, T}.
struct ggml_tensor * hubert_feature_projection_block_forward(
    struct ggml_context                          * ctx,
    struct ggml_tensor                           * features,
    const hubert_feature_projection_block_weights & weights);

struct hubert_positional_conv_block_weights {
    // Hugging Face weight_norm parameters for Conv1d(768, 768, kernel=128,
    // groups=16), stored in ggml tensor layout so the graph can reconstruct the
    // effective grouped-conv kernel exactly.
    //   PyTorch weight_v [out_channels, in_channels_per_group, kernel_size]
    //   maps to ggml {kernel_size, in_channels_per_group, out_channels}.
    //   weight_g may be stored as {kernel_size}, {kernel_size, 1, 1}, or any
    //   equivalent shape with 128 total elements.
    struct ggml_tensor * weight_v;       // {128, 48, 768}
    struct ggml_tensor * weight_g;       // 128 total elements
    struct ggml_tensor * bias;           // {768}
};

// Build the HuBERT positional convolution graph.
//
// Parameters:
//   ctx           - ggml context for tensor/op allocation
//   hidden_states - projected features {768, T}
//   weights       - positional-conv weights
//
// Returns:
//   Positional convolution embedding {768, T}.
struct ggml_tensor * hubert_positional_conv_block_forward(
    struct ggml_context                      * ctx,
    struct ggml_tensor                       * hidden_states,
    const hubert_positional_conv_block_weights & weights);

struct hubert_attention_block_weights {
    // Self-attention projections Linear(768, 768)
    struct ggml_tensor * q_proj_w;       // {768, 768}
    struct ggml_tensor * q_proj_b;       // {768}
    struct ggml_tensor * k_proj_w;       // {768, 768}
    struct ggml_tensor * k_proj_b;       // {768}
    struct ggml_tensor * v_proj_w;       // {768, 768}
    struct ggml_tensor * v_proj_b;       // {768}
    struct ggml_tensor * out_proj_w;     // {768, 768}
    struct ggml_tensor * out_proj_b;     // {768}
};

// Build eager self-attention for a full HuBERT encoder sequence.
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - input activations {768, T}
//   weights - attention weights
//
// Returns:
//   Attention output after output projection {768, T}.
struct ggml_tensor * hubert_attention_block_forward(
    struct ggml_context                   * ctx,
    struct ggml_tensor                    * x,
    const hubert_attention_block_weights  & weights);

struct hubert_encoder_layer_block_weights {
    hubert_attention_block_weights attention;

    // Post-attention LayerNorm(768)
    struct ggml_tensor * ln1_w;          // {768}
    struct ggml_tensor * ln1_b;          // {768}

    // Feed-forward Linear(768, 3072) -> GELU -> Linear(3072, 768)
    struct ggml_tensor * ffn_up_w;       // {768, 3072}
    struct ggml_tensor * ffn_up_b;       // {3072}
    struct ggml_tensor * ffn_down_w;     // {3072, 768}
    struct ggml_tensor * ffn_down_b;     // {768}

    // Final LayerNorm(768)
    struct ggml_tensor * ln2_w;          // {768}
    struct ggml_tensor * ln2_b;          // {768}
};

// Build a single HuBERT encoder layer (post-norm).
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - input activations {768, T}
//   weights - encoder-layer weights
//
// Returns:
//   Encoder-layer output {768, T}.
struct ggml_tensor * hubert_encoder_layer_block_forward(
    struct ggml_context                        * ctx,
    struct ggml_tensor                         * x,
    const hubert_encoder_layer_block_weights   & weights);

struct hubert_encoder_block_weights {
    hubert_positional_conv_block_weights pos_conv;

    // LayerNorm applied after adding positional convolution.
    struct ggml_tensor * layer_norm_w;   // {768}
    struct ggml_tensor * layer_norm_b;   // {768}

    std::array<hubert_encoder_layer_block_weights, 12> layers;
};

// Build the HuBERT encoder graph:
//   x + positional_conv(x) -> LayerNorm -> 12 encoder layers.
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - projected features {768, T}
//   weights - encoder weights
//
// Returns:
//   Final encoder hidden states {768, T}.
struct ggml_tensor * hubert_encoder_block_forward(
    struct ggml_context                  * ctx,
    struct ggml_tensor                   * x,
    const hubert_encoder_block_weights   & weights);

struct hubert_model_block_weights {
    hubert_feature_encoder_block_weights    feature_encoder;
    hubert_feature_projection_block_weights feature_projection;
    hubert_encoder_block_weights            encoder;
};

// Build the full CN-HuBERT base inference graph.
//
// Parameters:
//   ctx          - ggml context for tensor/op allocation
//   input_values - mono waveform samples {T} or {T, 1}
//   weights      - model weights
//
// Returns:
//   Final HuBERT hidden states {768, T'}.
struct ggml_tensor * hubert_model_block_forward(
    struct ggml_context                * ctx,
    struct ggml_tensor                 * input_values,
    const hubert_model_block_weights   & weights);

// ---------------------------------------------------------------------------
// HuBERT model: owns the loaded GGUF weights and ggml resources.
// ---------------------------------------------------------------------------

struct hubert_model {
    // Populated weight struct (all tensor pointers are owned by ctx_w).
    hubert_model_block_weights weights = {};

    // ggml resources -- managed by hubert_model_free().
    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buf_w   = nullptr;
    struct ggml_context      * ctx_w   = nullptr;
};

// Load a HuBERT model from a GGUF file.
//
// Parameters:
//   fname  - path to the .gguf file produced by convert_hubert_to_gguf.py
//   model  - output model struct (will be populated)
//
// Returns:
//   true on success, false on failure (with errors printed to stderr).
bool hubert_model_load(const std::string & fname, hubert_model & model);

// Free all resources owned by a HuBERT model.
void hubert_model_free(hubert_model & model);

} // namespace gpt_sovits
