#pragma once

#include "ggml.h"

namespace gpt_sovits {

// Per-layer weights for a T2S (Text-to-Semantic) Transformer block.
//
// Implements a post-norm Transformer encoder layer:
//   x -> QKV proj -> flash attention -> out proj -> residual + LN
//     -> FFN up (ReLU) -> FFN down -> residual + LN
//
// Weight tensor shapes use ggml convention (ne[0] = innermost dim):
//   Linear(in, out)  ->  weight.ne = {in, out}
//   LayerNorm(dim)   ->  weight.ne = {dim}, bias.ne = {dim}
struct t2s_block_weights {
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

// Build the computation graph for a single T2S Transformer block.
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
//   weights   - layer weights (see t2s_block_weights)
//   n_past    - number of tokens already in the KV cache
//   n_head    - number of attention heads (head_dim = d_model / n_head)
//   eps       - layer-norm epsilon (e.g. 1e-5)
//
// Returns:
//   Output activations {d_model, N}.
struct ggml_tensor * t2s_block_forward(
    struct ggml_context       * ctx,
    struct ggml_cgraph        * gf,
    struct ggml_tensor        * x,
    struct ggml_tensor        * mask,
    struct ggml_tensor        * k_cache,
    struct ggml_tensor        * v_cache,
    const t2s_block_weights   & weights,
    int                         n_past,
    int                         n_head,
    float                       eps);

} // namespace gpt_sovits
