#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <string>

namespace gpt_sovits {

// chinese-roberta-wwm-ext-large (BERT-large) forward blocks.
//
// These blocks match the Hugging Face BertModel eager-attention inference path
// used by GPT-SoVITS.  Despite the "roberta" name in the model checkpoint, the
// architecture is standard BERT (model_type = "bert") with post-norm Transformer
// encoder layers.
//
// Configuration:
//   hidden_size = 1024, num_heads = 16, head_dim = 64,
//   intermediate_size = 4096, num_layers = 24,
//   max_position_embeddings = 512, vocab_size = 21128,
//   hidden_act = gelu, layer_norm_eps = 1e-12
//
// Scope:
//   - single-sample inference only (no batching)
//   - no attention mask (full bidirectional)
//   - no token-type-id mask (always zeros in GPT-SoVITS)
//   - no MLM head or pooler
//
// Tensor layout convention: {channels, time} (ne[0] = innermost dim).

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

struct roberta_embeddings_block_weights {
    // nn.Embedding(vocab_size, hidden_size)
    struct ggml_tensor * word_embeddings;        // {21128, 1024}

    // nn.Embedding(max_position_embeddings, hidden_size)
    struct ggml_tensor * position_embeddings;    // {512, 1024}

    // nn.Embedding(type_vocab_size, hidden_size)
    struct ggml_tensor * token_type_embeddings;  // {2, 1024}

    // LayerNorm(hidden_size)
    struct ggml_tensor * layer_norm_w;           // {1024}
    struct ggml_tensor * layer_norm_b;           // {1024}
};

// Build the BERT embeddings graph:
//   word_embed(input_ids) + pos_embed[0:T] + type_embed(token_type_ids) + LayerNorm
//
// Parameters:
//   ctx            - ggml context for tensor/op allocation
//   input_ids      - token IDs                     {T} (i32)
//   token_type_ids - segment IDs or nullptr        {T} (i32)
//                    When nullptr, token-type contribution is skipped.
//   weights        - embedding weights
//
// Returns:
//   Embedded representation {1024, T}.
struct ggml_tensor * roberta_embeddings_block_forward(
    struct ggml_context                       * ctx,
    struct ggml_tensor                        * input_ids,
    struct ggml_tensor                        * token_type_ids,
    const roberta_embeddings_block_weights    & weights);

// ---------------------------------------------------------------------------
// Self-attention (per-layer)
// ---------------------------------------------------------------------------

struct roberta_self_attention_block_weights {
    // Separate Q/K/V projections  Linear(1024, 1024)
    struct ggml_tensor * q_w;       // {1024, 1024}
    struct ggml_tensor * q_b;       // {1024}
    struct ggml_tensor * k_w;       // {1024, 1024}
    struct ggml_tensor * k_b;       // {1024}
    struct ggml_tensor * v_w;       // {1024, 1024}
    struct ggml_tensor * v_b;       // {1024}

    // Output projection  Linear(1024, 1024)
    struct ggml_tensor * out_w;     // {1024, 1024}
    struct ggml_tensor * out_b;     // {1024}
};

// Build the BERT self-attention graph:
//   Q/K/V projections -> multi-head attention (flash_attn_ext) -> out_proj
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - input activations                     {1024, T}
//   weights - self-attention weights
//
// Returns:
//   Attention output after output projection {1024, T}.
struct ggml_tensor * roberta_self_attention_block_forward(
    struct ggml_context                          * ctx,
    struct ggml_tensor                           * x,
    const roberta_self_attention_block_weights   & weights);

// ---------------------------------------------------------------------------
// Encoder layer (per-layer)
// ---------------------------------------------------------------------------

struct roberta_encoder_layer_block_weights {
    roberta_self_attention_block_weights attention;

    // Post-attention LayerNorm
    struct ggml_tensor * attn_ln_w;   // {1024}
    struct ggml_tensor * attn_ln_b;   // {1024}

    // FFN: up (1024 -> 4096) + GELU + down (4096 -> 1024)
    struct ggml_tensor * ffn_up_w;    // {4096, 1024}
    struct ggml_tensor * ffn_up_b;    // {4096}
    struct ggml_tensor * ffn_down_w;  // {1024, 4096}
    struct ggml_tensor * ffn_down_b;  // {1024}

    // Post-FFN LayerNorm
    struct ggml_tensor * ffn_ln_w;    // {1024}
    struct ggml_tensor * ffn_ln_b;    // {1024}
};

// Build a single BERT encoder layer graph (post-norm):
//   x -> self-attn -> residual + LN -> FFN -> residual + LN
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - input activations                     {1024, T}
//   weights - encoder-layer weights
//
// Returns:
//   Encoder-layer output {1024, T}.
struct ggml_tensor * roberta_encoder_layer_block_forward(
    struct ggml_context                             * ctx,
    struct ggml_tensor                              * x,
    const roberta_encoder_layer_block_weights       & weights);

// ---------------------------------------------------------------------------
// Encoder (24 layers)
// ---------------------------------------------------------------------------

struct roberta_encoder_block_weights {
    std::array<roberta_encoder_layer_block_weights, 24> layers;
};

// Build the BERT encoder graph: 24 encoder layers applied sequentially.
//
// Parameters:
//   ctx     - ggml context for tensor/op allocation
//   x       - embedded input                        {1024, T}
//   weights - encoder weights
//
// Returns:
//   Final encoder hidden states {1024, T}.
struct ggml_tensor * roberta_encoder_block_forward(
    struct ggml_context                    * ctx,
    struct ggml_tensor                     * x,
    const roberta_encoder_block_weights    & weights);

// ---------------------------------------------------------------------------
// Full model (embeddings + encoder)
// ---------------------------------------------------------------------------

struct roberta_model_block_weights {
    roberta_embeddings_block_weights embeddings;
    roberta_encoder_block_weights    encoder;
};

// Build the full BERT-large inference graph.
//
// Parameters:
//   ctx            - ggml context for tensor/op allocation
//   input_ids      - token IDs                          {T} (i32)
//   token_type_ids - segment IDs or nullptr             {T} (i32)
//   weights        - model weights
//
// Returns:
//   Last hidden states {1024, T} (equivalent to BertModel last_hidden_state).
struct ggml_tensor * roberta_model_block_forward(
    struct ggml_context                  * ctx,
    struct ggml_tensor                   * input_ids,
    struct ggml_tensor                   * token_type_ids,
    const roberta_model_block_weights    & weights);

// ---------------------------------------------------------------------------
// RoBERTa model: owns loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
// ---------------------------------------------------------------------------

struct roberta_model {
    // Populated weight struct (all tensor pointers are owned by ctx_w).
    roberta_model_block_weights weights = {};

    // ggml resources -- managed by roberta_model_free().
    ggml_backend_t            backend = nullptr;  // borrowed
    ggml_backend_buffer_t     buf_w   = nullptr;  // owned
    struct ggml_context     * ctx_w   = nullptr;  // owned
};

// Load a RoBERTa model from a GGUF file.
//
// Parameters:
//   fname   - path to the .gguf file produced by convert_roberta_to_gguf.py
//   model   - output model struct (will be populated)
//   backend - ggml backend for tensor allocation (caller-owned; not freed by roberta_model_free)
//
// Returns:
//   true on success, false on failure (with errors printed to stderr).
bool roberta_model_load(const std::string & fname, roberta_model & model, ggml_backend_t backend);

// Free all resources owned by a RoBERTa model.
void roberta_model_free(roberta_model & model);

} // namespace gpt_sovits
