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

// ---------------------------------------------------------------------------
// End-to-end feature extraction and phoneme alignment
// ---------------------------------------------------------------------------

// Run full RoBERTa (chinese-roberta-wwm-ext-large) feature extraction
// end-to-end (embeddings + 24 encoder layers), encapsulating the standard
// ephemeral-graph compute sequence.
//
// Parameters:
//   model          - loaded RoBERTa model (weights read, not modified)
//   token_ids      - token IDs {T} (I32), host pointer; T must be <= 512
//   T              - sequence length (> 0)
//   out            - caller-allocated destination (F32), ggml {1024, T} layout:
//                    out[t * 1024 + c]
//   out_cap        - capacity of out in floats; must be >= 1024 * T
//   token_type_ids - optional segment IDs {T} (I32); nullptr = all zeros
//
// Input validation returns false rather than asserting.
// Returns true on success.
bool roberta_extract_features(const roberta_model & model,
                              const int32_t       * token_ids,
                              int64_t               T,
                              float               * out,
                              int64_t               out_cap,
                              const int32_t       * token_type_ids = nullptr);

// Expand per-WordPiece-token BERT features into per-phoneme BERT features,
// replicating python/text_frontend/layout.py:expand_bert_features.
//
// For each normalized character i (0 <= i < n_chars), word2ph[i] phonemes are
// emitted; they all receive a copy of the 1024-dim feature vector of the body
// token whose char span covers i (char_starts[t] <= i < char_ends[t]).
// Special tokens ([CLS]/[SEP]) have span -1/-1 and are never selected.
// sum(word2ph) must equal n_phones.
//
// Span-conflict rule (same as the Python reference): for every character with
// word2ph[i] > 0 the Python code requires EXACTLY ONE covering body token and
// raises LayoutError when there are zero or several matches.  There is no
// "first token wins" / "last token wins" precedence: a character covered by
// more than one token span is invalid, and this function returns false.
// Characters with word2ph[i] == 0 are not looked up at all.
//
// Parameters:
//   tok_feat    - per-token features {1024, n_tok} F32, layout tok_feat[t*1024 + c]
//   n_tok       - number of WordPiece tokens (incl. specials)
//   char_starts / char_ends - token char spans, length n_tok; -1 for specials
//   word2ph     - phoneme count per normalized char, length n_chars
//   n_phones    - total phoneme count (== sum(word2ph))
//   out         - caller-allocated {1024, n_phones} F32, out[p*1024 + c]
//
// English-segment zero-fill is NOT this function's job: callers handle
// multi-segment text by calling this per Chinese segment and zero-filling
// English segments' phoneme ranges themselves.
// Returns false on invalid input (nulls, span out of range, no covering token,
// several covering tokens, sum(word2ph) != n_phones).
bool roberta_expand_bert_features(const float   * tok_feat,
                                  int64_t         n_tok,
                                  const int32_t * char_starts,
                                  const int32_t * char_ends,
                                  const int32_t * word2ph,
                                  int64_t         n_chars,
                                  int64_t         n_phones,
                                  float         * out);

} // namespace gpt_sovits
