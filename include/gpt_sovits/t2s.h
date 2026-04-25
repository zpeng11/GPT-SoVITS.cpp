#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpt_sovits {

// Build the computation graph for the T2S sampler path used by
// AR.models.utils.sample(...).
//
// Parameters:
//   ctx                - ggml context for tensor/op allocation
//   y                  - hidden states from last attention layer {d_model, n_batch}
//   lm_head_w          - projection weight (no bias)            {d_model, vocab}
//   seen_mask          - optional seen-token mask               {vocab, n_batch} or nullptr
//                        Values are treated as binary via step(mask):
//                        > 0 means token has appeared before.
//   top_k              - <= 0 disables top-k filtering
//   top_p              - >= 1 disables top-p filtering
//   temperature        - logits are divided by max(temperature, 1e-5)
//   repetition_penalty - 1.0 disables repetition penalty
//   exp_noise          - optional Exp(1) noise samples          {vocab, n_batch} or nullptr
//                        When null, this falls back to greedy argmax.
//
// Returns:
//   t2s_sampler_result with per-slot token ids, each {n_batch} (i32).
//
// Notes:
//   - Batched: each column of y is sampled independently.
//   - No in-graph RNG; caller must supply exp_noise for random sampling.
struct t2s_sampler_result {
    struct ggml_tensor * sampled;   // randomly sampled (or greedy) tokens {n_batch} (i32)
    struct ggml_tensor * greedy;    // argmax tokens (before noise)        {n_batch} (i32)
};

struct t2s_sampler_result t2s_sampler_block_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * y,
    struct ggml_tensor  * lm_head_w,
    struct ggml_tensor  * seen_mask,
    int                   top_k,
    float                 top_p,
    float                 temperature,
    float                 repetition_penalty,
    struct ggml_tensor  * exp_noise);

// Tensor shapes use ggml convention throughout: ne[0] is the innermost dim.
// PyTorch Linear(in, out).weight has shape [out, in] → ggml {in, out}.
// All shapes in comments use ggml notation.

// Weights for the SoVITS extract-latent path:
//   hubert_feature -> ssl_proj Conv1d -> RVQ nearest-code lookup -> codes
struct sovits_extract_latent_block_weights {
    struct ggml_tensor * ssl_proj_w;     // Conv1d(768, 768, k=2, s=2)  {2, 768, 768}
    struct ggml_tensor * ssl_proj_b;     // {768}

    // RVQ codebook (single quantizer layer, n_q = 1).
    struct ggml_tensor * codebook;       // {768, 1024}
};

// Build the computation graph for audio token embedding with positional
// encoding. Reusable for both prompt prefill (start_position=0) and
// autoregressive decode steps (start_position=y_len+idx).
//
// Parameters:
// Weights for building the T2S prefill input sequence up to xy_pos.
//
// This matches the Python path used before process_prompt(...):
//   x_tokens -> text embedding
//           + bert_proj(bert_feature)
//           + SinePositionalEmbedding.forward(x)
//   prompt_tokens -> audio embedding + SinePositionalEmbedding.forward(y)
//   xy_pos = concat(text, prompt)
//
// Embedding weights for the T2S input sequence (single-sample inference).
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
//   Combined embedding {d_model, T_ref + T_prompt}.
//   Layout: [ref_text_emb | prompt_audio_emb].
//   The caller may split this into separate tensors for storage.
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

// Per-layer weights for a T2S attention block (post-norm Transformer):
//   x -> QKV -> attention -> out proj -> residual + LN
//     -> FFN up (ReLU) -> FFN down -> residual + LN
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
//   ctx       - ggml context
//   gf        - computation graph
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
// Sampler configuration (baked into decode graph at build time)
// ---------------------------------------------------------------------------

// Sampling parameters — bound to the session at init time (see t2s_session_init).
struct t2s_sampler_config {
    int   top_k              = 15;      // <= 0 disables top-k filtering
    float top_p              = 1.0f;   // >= 1.0 disables top-p filtering
    float temperature        = 1.0f;   // logits are divided by max(temperature, 1e-5)
    float repetition_penalty = 1.35f;  // 1.0 disables repetition penalty
};

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
        int n_pos        = 0;  // total tokens written to this slot's KV cache
        int audio_offset = 0;  // first audio token index (T_ref + T_in); decode PE = n_pos - audio_offset
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
    ggml_gallocr_t        alloc_dec = nullptr;  // owned — pre-allocates intermediate storage

    // Decode embedding inputs (filled per decode step)
    struct ggml_tensor  * token_id  = nullptr;  // {n_batch} I32 — audio token to embed
    struct ggml_tensor  * position  = nullptr;  // {n_batch} I32 — PE position index

    // Precomputed sine PE table: {d_model, slot_size} F32
    // Row p = [sin0, cos0, sin1, cos1, ...] for position p.
    struct ggml_tensor  * pe_table  = nullptr;

    // Sampler configuration (baked into decode graph topology at build time)
    t2s_sampler_config   sampler_cfg;

    // Sampler graph inputs (caller fills per decode step)
    struct ggml_tensor  * seen_mask  = nullptr;  // {vocab, n_batch} F32
    struct ggml_tensor  * exp_noise  = nullptr;  // {vocab, n_batch} F32

    // Sampler graph outputs
    struct ggml_tensor  * sampled    = nullptr;  // {n_batch} I32 — sampled token ids
    struct ggml_tensor  * greedy     = nullptr;  // {n_batch} I32 — argmax token ids

    // Reference embedding (owned, computed once per session)
    struct ggml_context  * ctx_ref        = nullptr;  // owned - holds ref text/audio tensors
    ggml_backend_buffer_t  buf_ref        = nullptr;  // owned - data buffer for ref tensors
    struct ggml_tensor   * ref_text_emb   = nullptr;  // cached {d_model, T_ref}
    struct ggml_tensor   * ref_audio_emb  = nullptr;  // cached {d_model, T_prompt}
    int64_t                ref_T_ref      = 0;        // ref text token count (input pos offset)
    int64_t                ref_T_prompt   = 0;        // ref prompt audio token count

    // Shared allocator for flexible computation graphs (owned).
    // Reused across t2s_session_build_flex_graph calls to avoid repeated
    // buffer allocation/free when consecutive plans have similar sizes.
    ggml_gallocr_t alloc_flex = nullptr;
};

// Initialize a T2S inference session with pre-allocated KV caches.
//
// The sampler configuration is bound at init time: all subsequent graph builds
// (decode graph, flex graph) use this config. One session = one sampler config.
//
// Parameters:
//   session     - output session struct (will be populated)
//   hparams     - model hyperparameters
//   backend     - ggml backend (borrowed, not freed)
//   n_batch     - number of concurrent request slots
//   slot_size   - maximum tokens per request slot
//   sampler_cfg - sampler config (baked into all graphs)
//
// Returns true on success, false on failure.
bool t2s_session_init(t2s_session             & session,
                      const t2s_hparams       & hparams,
                      ggml_backend_t            backend,
                      uint32_t                  n_batch,
                      uint32_t                  slot_size,
                      const t2s_sampler_config & sampler_cfg   = {},
                      enum ggml_type            kv_cache_type  = GGML_TYPE_F32);

// Free all resources owned by a T2S session.
void t2s_session_free(t2s_session & session);

// Find the first free slot without modifying session state.
// Returns slot ID (0-based) or -1 if all slots are in use.
// The caller activates the slot via t2s_session_flex_advance by including
// it in the batch plan; no separate activation step is needed.
int t2s_session_find_free_slot(const t2s_session & session);

// Release a slot, making it available for reuse.
// Masks out the slot's entire column in the decode mask.
void t2s_session_slot_release(t2s_session & session, int slot_id);

// Advance all slots by one decode step.
// Increments n_pos for each slot and reveals the newly written KV
// positions in the session decode mask.  Updates kv_pos so that the
// persistent decode graph scatter-writes at the correct columns.
//
// IMPORTANT — asserts that ALL n_batch slots are active (in_use) with
// n_pos > 0.  The persistent decode graph processes every slot; partial
// occupancy must use the flexible graph path (t2s_session_build_flex_graph)
// instead.
void t2s_session_decode_advance(t2s_session & session);

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

// Get the cached reference text embedding tensor, or nullptr if not computed.
struct ggml_tensor * t2s_session_get_ref_text_emb(const t2s_session & session);

// Get the cached reference audio/prompt embedding tensor, or nullptr if not computed.
struct ggml_tensor * t2s_session_get_ref_audio_emb(const t2s_session & session);

// Get the cached T_ref (number of reference text tokens), or 0 if not computed.
int64_t t2s_session_get_ref_T_ref(const t2s_session & session);

// Get the cached T_prompt (number of reference prompt audio tokens), or 0 if not computed.
int64_t t2s_session_get_ref_T_prompt(const t2s_session & session);

// Build a persistent decode graph for this session. Call exactly once after
// t2s_session_init; the graph is reused for every decode step.
//
// Includes the sampler (using session's sampler_cfg). Outputs are `sampled`
// and `greedy` token id tensors.
//
// IMPORTANT — assumes ALL n_batch slots are in single-token decode phase.
// NOT suitable for prefill; use flex graph for mixed batches.
bool t2s_session_build_decode_graph(
    t2s_session             & session,
    const t2s_model         & model);

// Accessors for the decode graph.
struct ggml_tensor * t2s_session_get_token_id(const t2s_session & session);
struct ggml_tensor * t2s_session_get_position(const t2s_session & session);
struct ggml_cgraph * t2s_session_get_decode_graph(const t2s_session & session);
struct ggml_tensor * t2s_session_get_sampled(const t2s_session & session);
struct ggml_tensor * t2s_session_get_greedy(const t2s_session & session);
struct ggml_tensor * t2s_session_get_seen_mask(const t2s_session & session);
struct ggml_tensor * t2s_session_get_exp_noise(const t2s_session & session);

// ---------------------------------------------------------------------------
// Flexible computation graph for mixed prefill/decode batch steps
// ---------------------------------------------------------------------------

// Describes the batch configuration for one graph invocation.
// Each entry corresponds to one slot; n_query[i] == 0 means slot i is idle.
struct t2s_batch_plan {
    // Per-slot token count.  Size must equal session.n_batch.
    // n_query[i] > 0: slot i processes n_query[i] tokens.
    // n_query[i] == 0: slot i is idle.
    std::vector<int> n_query;

    // Total query tokens across all active slots.
    int total() const;
};

// A built computation graph with variable shapes.
// Owns the graph context and tensor metadata (freed by t2s_flex_graph_free).
// Borrows KV caches from the session and weight tensors from the model.
// The allocator is owned by the session and reused across graph builds.
// The caller must ensure the session and model outlive this object.
struct t2s_flex_graph {
    // ---- Input tensors ----

    // Prefill embeddings in slot order (null if no prefill slots).  {d_model, N_prefill} F32
    struct ggml_tensor * x_prefill = nullptr;

    // Decode token IDs in slot order (null if no decode slots).  {n_decode} I32
    struct ggml_tensor * decode_token_ids = nullptr;

    // ---- Runtime tensors (filled by t2s_session_flex_advance) ----

    // Decode PE positions — auto-filled by flex_advance (null if no decode slots).  {n_decode} I32
    struct ggml_tensor * decode_positions = nullptr;

    struct ggml_tensor * kv_pos = nullptr;   // {N} I32
    struct ggml_tensor * mask   = nullptr;   // {n_kv, N} F16

    // ---- Output tensors ----

    struct ggml_tensor * y      = nullptr;   // {d_model, N} F32 (full attention output)

    // Sampler inputs/outputs (caller fills seen_mask/exp_noise, reads sampled/greedy)
    struct ggml_tensor * seen_mask = nullptr;  // {vocab, n_active} F32 input (optional)
    struct ggml_tensor * exp_noise = nullptr;  // {vocab, n_active} F32 input (optional)
    struct ggml_tensor * sampled   = nullptr;  // {n_active} I32 output
    struct ggml_tensor * greedy    = nullptr;  // {n_active} I32 output

    // ggml resources (ctx/gf owned, backend borrowed from session)
    struct ggml_context * ctx     = nullptr;
    struct ggml_cgraph  * gf      = nullptr;
    ggml_backend_t        backend = nullptr;  // borrowed from session

    int N        = 0;  // total query tokens (sum of active slot n_query)
    int n_active = 0;  // number of active slots (sampler batch dim)
    int n_decode = 0;  // number of decode slots (decode_token_ids batch dim)
};

// Build a flexible computation graph for the given batch plan.
//
// Input tensors (caller fills unless noted):
//   - x_prefill:        {d_model, N_prefill} F32 — prefill embeddings
//   - decode_token_ids: {n_decode} I32           — decode token IDs
//   - decode_positions: {n_decode} I32            — auto-filled by flex_advance
//   - kv_pos, mask:                               — filled by flex_advance
//
// Token ordering: slot 0's tokens first, then slot 1's, etc.
// Idle slots (n_query == 0) contribute zero tokens.
//
// The sampler extracts the last token per active slot and applies the session's
// sampler_cfg. Full attention output is available as graph.y.
//
// Returns a t2s_flex_graph with ctx != nullptr on success, or an empty graph
// on failure. Call t2s_flex_graph_free when done (frees ctx only, not the allocator).
t2s_flex_graph t2s_session_build_flex_graph(
    t2s_session             & session,
    const t2s_model         & model,
    const t2s_batch_plan    & plan);

// Free all resources owned by a t2s_flex_graph.
void t2s_flex_graph_free(t2s_flex_graph & graph);

// Advance session state and fill graph runtime inputs for the given plan.
// Call after t2s_session_build_flex_graph and before ggml_backend_graph_compute.
//
// For each active slot (n_query[i] > 0):
//   - Validates slot state (decode: n_pos > 0; prefill: n_pos == 0)
//   - Fills kv_pos scatter-write positions: slot_col[i] * slot_size + n_pos + j
//   - Builds attention mask (decode: attend all valid; prefill: text bidirectional + audio causal)
//   - Decode slots: auto-fills decode_positions from (n_pos - audio_offset)
//   - Prefill slots: sets audio_offset = n_query - ref_T_prompt
//   - Reveals new KV positions in session decode mask; increments n_pos
void t2s_session_flex_advance(t2s_session & session, const t2s_batch_plan & plan, t2s_flex_graph & graph);

} // namespace gpt_sovits
