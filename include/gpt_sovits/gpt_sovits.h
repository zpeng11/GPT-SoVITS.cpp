#pragma once

#include "ggml.h"

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
struct t2s_encoder_block_weights {
    // Text path
    struct ggml_tensor * text_embedding;  // {d_model, phoneme_vocab}
    struct ggml_tensor * bert_proj_w;     // {1024, d_model}
    struct ggml_tensor * bert_proj_b;     // {d_model}
    struct ggml_tensor * text_pos_alpha;  // scalar or {1}

    // Prompt/audio path
    struct ggml_tensor * audio_embedding; // {d_model, semantic_vocab}
    struct ggml_tensor * audio_pos_alpha; // scalar or {1}
};

// Build the computation graph for the prefill embedding block up to xy_pos.
//
// Parameters:
//   ctx          - ggml context for tensor/op allocation
//   x_tokens     - phoneme token ids                 {T_x} (i32)
//   bert_feature - BERT features                     {1024, T_x}
//   prompt_tokens- semantic prompt token ids         {T_y} (i32), or nullptr
//   weights      - encoder weights (see t2s_encoder_block_weights)
//
// Returns:
//   xy_pos {d_model, T_x + T_y} when prompt_tokens is present,
//          {d_model, T_x} otherwise.
//
// Positional embeddings are generated inside the block using the same
// frequency schedule as SinePositionalEmbedding.extend_pe() in GPT-SoVITS,
// with x_scale = 1.0 (the current model configuration uses scale=False).
struct ggml_tensor * t2s_encoder_block_forward(
    struct ggml_context              * ctx,
    struct ggml_tensor               * x_tokens,
    struct ggml_tensor               * bert_feature,
    struct ggml_tensor               * prompt_tokens,
    const t2s_encoder_block_weights  & weights);

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
// Appends new K/V entries to the caches and performs self-attention
// followed by a feed-forward network, both with post-norm residuals.
//
// Parameters:
//   ctx       - ggml context for tensor/op allocation (typically no_alloc)
//   gf        - computation graph; KV cache copy ops are added via
//               ggml_build_forward_expand so they execute before attention
//   x         - input activations               {d_model, N}
//   mask      - attention mask (f16, contiguous) {n_kv, N}
//               0 = attend, -inf = masked
//   k_cache   - key cache buffer                 {d_model, max_ctx}
//   v_cache   - value cache buffer               {d_model, max_ctx}
//   weights   - layer weights (see t2s_attention_block_weights)
//   n_past    - number of tokens already in the KV cache
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
    const t2s_attention_block_weights   & weights,
    int                         n_past,
    int                         n_head,
    float                       eps);

} // namespace gpt_sovits
