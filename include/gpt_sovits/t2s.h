#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

// Weights for the SoVITS semantic extractor path used by
// SynthesizerTrn.extract_latent(...):
//   hubert_feature -> ssl_proj Conv1d -> RVQ nearest-code lookup -> codes
//
// This block matches the inference-only path that produces prompt semantic
// tokens from HuBERT features. The current GPT-SoVITS models use a single RVQ
// layer (n_q = 1), so only one codebook is required here.
struct sovits_extract_latent_block_weights {
    // Conv1d(768, 768, kernel=2, stride=2) in ggml kernel layout.
    // PyTorch weight [out_channels, in_channels, kernel_size]
    // maps to ggml {kernel_size, in_channels, out_channels}.
    struct ggml_tensor * ssl_proj_w;     // {2, 768, 768}
    struct ggml_tensor * ssl_proj_b;     // {768}

    // RVQ codebook for the first and only quantizer layer.
    // PyTorch embed shape [1024, 768] maps to ggml {768, 1024} so
    // ggml_mul_mat(codebook, ssl) yields per-token scores {1024, T'}.
    struct ggml_tensor * codebook;       // {768, 1024}
};

// Build the computation graph for the inference path used by
// SynthesizerTrn.extract_latent(...).
//
// Parameters:
//   ctx            - ggml context for tensor/op allocation
//   hubert_feature - CN-HuBERT features               {768, T}
//   weights        - block weights (see above)
//
// Returns:
//   codes {T'} (i32), where T' is the post-conv sequence length.
//
// Notes:
//   - This implements only the inference path. Training-only EMA updates,
//     commitment loss, and straight-through quantized outputs are omitted.
//   - Current GPT-SoVITS checkpoints force semantic_frame_rate = "25hz" in
//     TTS inference, so ssl_proj is the stride-2 Conv1d path.
struct ggml_tensor * sovits_extract_latent_block_forward(
    struct ggml_context                       * ctx,
    struct ggml_tensor                        * hubert_feature,
    const sovits_extract_latent_block_weights & weights);

// Build the computation graph for the T2S sampler path used by
// AR.models.utils.sample(...).
//
// Parameters:
//   ctx                - ggml context for tensor/op allocation
//   y                  - hidden state from last attention layer  {d_model} or {d_model, 1}
//   lm_head_w          - projection weight (no bias)            {d_model, vocab}
//   seen_mask          - optional seen-token mask               {vocab} or {vocab, 1}
//                        Values are treated as binary via step(mask):
//                        > 0 means token has appeared before.
//   top_k              - <= 0 disables top-k filtering
//   top_p              - >= 1 disables top-p filtering
//   temperature        - logits are divided by max(temperature, 1e-5)
//   repetition_penalty - 1.0 disables repetition penalty
//   exp_noise      - optional Exp(1) noise samples      {vocab} or {vocab, 1}
//                    When null, this falls back to greedy argmax.
//
// Returns:
//   next token id {1} (i32)
//
// Notes:
//   - This block targets single-sample inference.
//   - Top-p is applied in sorted-logit space exactly as in AR.models.utils.
//   - Masked logits use a large negative finite sentinel so softmax underflows
//     them to zero without requiring a dedicated masked_fill op.
//   - ggml does not provide a graph RNG op, so randomized sampling requires
//     `exp_noise` to be supplied by the caller.
struct ggml_tensor * t2s_sampler_block_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * y,
    struct ggml_tensor  * lm_head_w,
    struct ggml_tensor  * seen_mask,
    int                   top_k,
    float                 top_p,
    float                 temperature,
    float                 repetition_penalty,
    struct ggml_tensor  * exp_noise);

// Build the computation graph for audio token embedding with positional
// encoding. Reusable for both prompt prefill (start_position=0) and
// autoregressive decode steps (start_position=y_len+idx).
//
// Parameters:
//   ctx             - ggml context for tensor/op allocation
//   token_ids       - audio/prompt token ids            {T} or {1} (i32)
//   audio_embedding - embedding weight                  {d_model, semantic_vocab}
//   audio_pos_alpha - positional embedding scale        scalar or {1}
//   start_position  - position offset for PE generation (0 for prefill)
//
// Returns:
//   Embedded tokens with positional encoding {d_model, T}.
struct ggml_tensor * t2s_audio_embed_block_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * token_ids,
    struct ggml_tensor  * audio_embedding,
    struct ggml_tensor  * audio_pos_alpha,
    int64_t               start_position);

// Weights for building the T2S prefill input sequence up to xy_pos.
//
// This matches the Python path used before process_prompt(...):
//   x_tokens -> text embedding
//           + bert_proj(bert_feature)
//           + SinePositionalEmbedding.forward(x)
//   prompt_tokens -> audio embedding + SinePositionalEmbedding.forward(y)
//   xy_pos = concat(text, prompt)
//
// All tensors use ggml's shape convention (ne[0] = innermost dim).
// This block currently targets single-sample inference.
struct t2s_embed_block_weights {
    // Text path
    struct ggml_tensor * text_embedding;  // {d_model, phoneme_vocab}
    struct ggml_tensor * bert_proj_w;     // {1024, d_model}
    struct ggml_tensor * bert_proj_b;     // {d_model}
    struct ggml_tensor * text_pos_alpha;  // scalar or {1}

    // Prompt/audio path
    struct ggml_tensor * audio_embedding; // {d_model, semantic_vocab}
    struct ggml_tensor * audio_pos_alpha; // scalar or {1}
};

// Precompute the reference portion of the T2S input sequence.
//
// Processes reference text (phonemes + BERT + positional) and reference
// audio (HuBERT → semantic tokens → audio embed + positional), producing
// a single concatenated tensor.  The result is deterministic for a given
// reference and can be cached across inference calls.
//
// Parameters:
//   ctx                - ggml context for tensor/op allocation
//   ref_token          - reference phoneme token ids       {T_ref}  (i32)
//   ref_bert_feature   - reference BERT features           {1024, T_ref}
//   hubert_feature     - HuBERT features from ref audio    {768, T_hub}
//   extract_latent_weights - weights for the SoVITS extract-latent block
//   embed_weights           - weights for the T2S embedding block
//
// Returns:
//   ref_emb {d_model, T_ref + T_prompt}
//   Layout: [ref_text_emb | prompt_audio_emb]
struct ggml_tensor * t2s_embed_ref_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * ref_token,
    struct ggml_tensor  * ref_bert_feature,
    struct ggml_tensor  * hubert_feature,
    const sovits_extract_latent_block_weights & extract_latent_weights,
    const t2s_embed_block_weights    & embed_weights);

// Compute the input text embedding portion of the T2S sequence.
//
// Embeds input text tokens with BERT projection and positional encoding
// starting at position T_ref, so the positions continue seamlessly
// after the reference text portion.
//
// Parameters:
//   ctx                - ggml context for tensor/op allocation
//   input_token        - input phoneme token ids           {T_in}   (i32)
//   input_bert_feature - input BERT features               {1024, T_in}
//   T_ref              - number of reference text tokens (position offset)
//   embed_weights      - weights for the T2S embedding block
//
// Returns:
//   input_text_emb {d_model, T_in}
struct ggml_tensor * t2s_embed_input_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * input_token,
    struct ggml_tensor  * input_bert_feature,
    int64_t               T_ref,
    const t2s_embed_block_weights & embed_weights);

// Per-layer weights for a T2S (Text-to-Semantic) attention block.
//
// Implements a post-norm Transformer encoder layer:
//   x -> QKV proj -> flash attention -> out proj -> residual + LN
//     -> FFN up (ReLU) -> FFN down -> residual + LN
//
// Weight tensor shapes use ggml convention (ne[0] = innermost dim):
//   Linear(in, out)  ->  weight.ne = {in, out}
//   LayerNorm(dim)   ->  weight.ne = {dim}, bias.ne = {dim}
struct t2s_attention_block_weights {
    // Self-attention: fused QKV projection  Linear(d_model, 3*d_model)
    struct ggml_tensor * qkv_w;       // {d_model, 3*d_model}
    struct ggml_tensor * qkv_b;       // {3*d_model}

    // Self-attention: output projection  Linear(d_model, d_model)
    struct ggml_tensor * out_proj_w;   // {d_model, d_model}
    struct ggml_tensor * out_proj_b;   // {d_model}

    // Post-attention layer norm
    struct ggml_tensor * ln1_w;        // {d_model}
    struct ggml_tensor * ln1_b;        // {d_model}

    // Feed-forward network: up projection  Linear(d_model, d_ff)
    struct ggml_tensor * ffn_up_w;     // {d_model, d_ff}
    struct ggml_tensor * ffn_up_b;     // {d_ff}

    // Feed-forward network: down projection  Linear(d_ff, d_model)
    struct ggml_tensor * ffn_down_w;   // {d_ff, d_model}
    struct ggml_tensor * ffn_down_b;   // {d_model}

    // Post-FFN layer norm
    struct ggml_tensor * ln2_w;        // {d_model}
    struct ggml_tensor * ln2_b;        // {d_model}
};

// Build the computation graph for a single T2S attention block.
//
// Writes new K/V entries into the caches (scatter-write) and performs
// self-attention over [0, n_kv) followed by a feed-forward network,
// both with post-norm residuals.
//
// Parameters:
//   ctx       - ggml context for tensor/op allocation (typically no_alloc)
//   gf        - computation graph; KV cache write ops are added via
//               ggml_build_forward_expand so they execute before attention
//   x         - input activations               {d_model, N}
//   mask      - attention mask (f16, contiguous) {n_kv, N}
//               0 = attend, -inf = masked; caller is responsible for
//               masking out any invalid positions in [0, n_kv)
//   k_cache   - key cache buffer                 {d_model, max_ctx}
//   v_cache   - value cache buffer               {d_model, max_ctx}
//   kv_pos    - scatter-write positions           {N} I32
//               kv_pos[i] = cache slot (column index) for token i;
//               positions may be non-contiguous and in any order;
//               duplicate positions produce undefined behaviour
//   weights   - layer weights (see t2s_attention_block_weights)
//   n_kv      - total KV entries for attention readback; K and V are
//               read from cache positions [0, n_kv)
//   n_head    - number of attention heads (head_dim = d_model / n_head)
//   eps       - layer-norm epsilon (e.g. 1e-5)
//
// Returns:
//   Output activations {d_model, N}.
struct ggml_tensor * t2s_attention_block_forward(
    struct ggml_context       * ctx,
    struct ggml_cgraph        * gf,
    struct ggml_tensor        * x,
    struct ggml_tensor        * mask,
    struct ggml_tensor        * k_cache,
    struct ggml_tensor        * v_cache,
    struct ggml_tensor        * kv_pos,
    const t2s_attention_block_weights   & weights,
    int                         n_kv,
    int                         n_head,
    float                       eps);

// ---------------------------------------------------------------------------
// T2S model hyperparameters (read from GGUF KV metadata)
// ---------------------------------------------------------------------------

struct t2s_hparams {
    uint32_t embedding_dim      = 512;
    uint32_t hidden_dim         = 512;
    uint32_t n_head             = 16;
    uint32_t linear_units       = 2048;
    uint32_t n_layer            = 24;
    uint32_t vocab_size         = 1025;
    uint32_t phoneme_vocab_size = 732;
    uint32_t eos                = 1024;
    uint32_t inter_channels     = 192;   // SoVITS extract-latent
};

// ---------------------------------------------------------------------------
// Aggregate weight struct for all T2S + extract-latent blocks
// ---------------------------------------------------------------------------

struct t2s_model_weights {
    // SoVITS extract-latent path
    sovits_extract_latent_block_weights extract_latent;

    // T2S encoder embedding path
    t2s_embed_block_weights embed;

    // T2S transformer attention layers
    std::vector<t2s_attention_block_weights> attention;

    // T2S sampler (lm_head projection)
    struct ggml_tensor * lm_head_w;      // {d_model, vocab}
};

// ---------------------------------------------------------------------------
// T2S model: owns loaded GGUF weights and ggml resources
// (except backend, which is borrowed from the caller).
// ---------------------------------------------------------------------------

struct t2s_model {
    t2s_hparams         hparams = {};
    t2s_model_weights   weights = {};

    // ggml resources -- managed by t2s_model_free().
    ggml_backend_t         backend = nullptr;  // borrowed
    ggml_backend_buffer_t  buf_w   = nullptr;  // owned
    struct ggml_context  * ctx_w   = nullptr;  // owned
};

// Load a T2S model from a GGUF file.
//
// Parameters:
//   fname   - path to the .gguf file produced by convert_t2s_to_gguf.py
//   model   - output model struct (will be populated)
//   backend - ggml backend for tensor allocation (caller-owned; not freed)
//
// Returns:
//   true on success, false on failure (with errors printed to stderr).
bool t2s_model_load(const std::string & fname, t2s_model & model, ggml_backend_t backend);

// Free all resources owned by a T2S model.
void t2s_model_free(t2s_model & model);

// ---------------------------------------------------------------------------
// T2S session: pre-allocated KV caches and slot management
// ---------------------------------------------------------------------------

// Manages pre-allocated KV caches for all transformer layers, designed for
// CUDA Graph compatibility (fixed tensor shapes across all steps).
//
// Layout:
//   K/V cache per layer:  {d_model, n_batch * slot_size}
//   Slot i occupies:      columns [i * slot_size, (i+1) * slot_size)
//   kv_pos:               {slot_size} I32
//   mask:                 {n_batch * slot_size, n_batch} F16
struct t2s_session {
    // Configuration (set at init, immutable)
    uint32_t n_batch   = 0;
    uint32_t slot_size = 0;

    // Per-slot state
    struct slot_state {
        bool in_use = false;
        int  n_pos  = 0;
    };
    std::vector<slot_state> slots;

    // Per-layer KV caches
    std::vector<struct ggml_tensor *> k_caches;  // [n_layer]
    std::vector<struct ggml_tensor *> v_caches;  // [n_layer]

    // Shared input tensors (caller fills before graph build)
    struct ggml_tensor * kv_pos = nullptr;  // {n_batch} I32
    // Decode-only mask. Each column corresponds to one query token and masks
    // out all KV positions outside that query's own slot, providing per-slot
    // isolation during batch decode. NOT suitable for prefill.
    struct ggml_tensor * mask   = nullptr;  // {n_batch * slot_size, n_batch} F16

    // CPU-side mask mirror for incremental updates.
    // Linear indexing: mask_host[col * max_ctx + row] corresponds to mask(row, col).
    std::vector<ggml_fp16_t> mask_host;

    // KV cache element type (default F32; can be set to Q8_0 etc. for memory savings)
    enum ggml_type kv_cache_type = GGML_TYPE_F32;

    // ggml resources -- managed by t2s_session_free()
    ggml_backend_t        backend = nullptr;  // borrowed
    ggml_backend_buffer_t buf_kv  = nullptr;  // owned
    struct ggml_context * ctx_kv  = nullptr;  // owned

    // Decode computation graph (built by t2s_session_build_decode_graph)
    struct ggml_context * ctx_graph = nullptr;  // owned
    struct ggml_cgraph  * gf_dec    = nullptr;  // owned
    struct ggml_tensor  * x_dec     = nullptr;  // input  {d_model, n_batch} F32
    struct ggml_tensor  * y_dec     = nullptr;  // output {d_model, n_batch} F32
    ggml_gallocr_t        alloc_dec = nullptr;  // owned — pre-allocates intermediate storage

    // Reference embedding (owned, computed once per session)
    struct ggml_context  * ctx_ref    = nullptr;  // owned - holds ref_emb tensor
    ggml_backend_buffer_t  buf_ref    = nullptr;  // owned - data buffer for ref_emb
    struct ggml_tensor   * ref_emb    = nullptr;  // cached {d_model, T_ref + T_prompt}
    int64_t                ref_T_ref  = 0;        // ref text token count (input pos offset)
};

// Initialize a T2S inference session with pre-allocated KV caches.
//
// Parameters:
//   session   - output session struct (will be populated)
//   hparams   - model hyperparameters (d_model, n_layer, etc.)
//   backend   - ggml backend for tensor allocation (borrowed, not freed)
//   n_batch   - number of concurrent request slots
//   slot_size - maximum tokens per request slot
//
// Returns true on success, false on failure (with errors printed to stderr).
bool t2s_session_init(t2s_session      & session,
                      const t2s_hparams & hparams,
                      ggml_backend_t     backend,
                      uint32_t           n_batch,
                      uint32_t           slot_size,
                      enum ggml_type     kv_cache_type = GGML_TYPE_F32);

// Free all resources owned by a T2S session.
void t2s_session_free(t2s_session & session);

// Allocate an available slot with n_pos prefilled tokens.
// Updates mask to allow attending to the prefilled positions.
// Returns slot ID (0-based) or -1 if full.
int t2s_session_slot_alloc(t2s_session & session, int n_pos);

// Release a slot, making it available for reuse.
// Masks out the slot's entire column in the decode mask.
void t2s_session_slot_release(t2s_session & session, int slot_id);

// Advance a slot by one decode step.
// Increments n_pos and reveals the newly written KV position in the mask.
void t2s_session_slot_decode_step(t2s_session & session, int slot_id);

// Get the current number of valid tokens in a slot.
int t2s_session_slot_n_pos(const t2s_session & session, int slot_id);

// Accessor helpers for graph building.
struct t2s_layer_caches {
    struct ggml_tensor * k;
    struct ggml_tensor * v;
};

t2s_layer_caches  t2s_session_get_layer_caches(const t2s_session & session, int layer);
struct ggml_tensor * t2s_session_get_kv_pos(const t2s_session & session);
struct ggml_tensor * t2s_session_get_mask(const t2s_session & session);

// Total KV entries for attention readback (always n_batch * slot_size).
int t2s_session_get_n_kv(const t2s_session & session);

// Build a *persistent* decode graph for this session.  Call exactly once
// after t2s_session_init; the graph is stored in the session and reused for
// every subsequent decode step.
//
// IMPORTANT — this graph assumes **all** `n_batch` slots are occupied and
// every slot is in the single-token decode phase (one new token per slot
// per invocation).  It is NOT suitable for prefill where a slot may consume
// many tokens at once.
bool t2s_session_build_decode_graph(t2s_session & session, const t2s_model & model);

// Accessors for the decode graph.
struct ggml_tensor * t2s_session_get_x_dec(const t2s_session & session);
struct ggml_tensor * t2s_session_get_y_dec(const t2s_session & session);
struct ggml_cgraph * t2s_session_get_decode_graph(const t2s_session & session);

// Compute and cache the reference embedding in the session.
//
// Accepts raw host data pointers for all inputs. Internally builds a temporary
// graph, executes it, and copies the result to a session-owned persistent tensor.
// Call once after session init; one session = one ref.
//
// Parameters:
//   session     - initialized T2S session
//   model       - loaded T2S model (weights are read, not modified)
//   ref_token   - reference phoneme token ids       {T_ref}  (i32), host pointer
//   T_ref       - number of reference text tokens
//   ref_bert    - reference BERT features            {1024, T_ref}, host pointer (F32)
//   hubert_data - HuBERT features from ref audio     {768, T_hub}, host pointer (F32)
//   T_hub       - number of HuBERT time frames
//
// Returns true on success, false on failure.
bool t2s_session_compute_ref_emb(t2s_session       & session,
                                  const t2s_model   & model,
                                  const int32_t     * ref_token,
                                  int64_t             T_ref,
                                  const float       * ref_bert,
                                  const float       * hubert_data,
                                  int64_t             T_hub);

// Get the cached reference embedding tensor, or nullptr if not computed.
struct ggml_tensor * t2s_session_get_ref_emb(const t2s_session & session);

// Get the cached T_ref (number of reference text tokens), or 0 if not computed.
int64_t t2s_session_get_ref_T_ref(const t2s_session & session);

} // namespace gpt_sovits
