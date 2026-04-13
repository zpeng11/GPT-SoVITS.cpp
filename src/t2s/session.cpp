#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Mask helpers (internal)
// ---------------------------------------------------------------------------

// mask_host linear indexing: mask_host[col * max_ctx + row] corresponds to
// tensor element mask(row, col).  Column col is contiguous in memory.

static void mask_upload_range(t2s_session & session, int slot_id,
                              int64_t start_row, int64_t count) {
    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const size_t  offset  = (size_t)(slot_id * max_ctx + start_row) * sizeof(ggml_fp16_t);
    const size_t  size    = (size_t) count * sizeof(ggml_fp16_t);
    ggml_backend_tensor_set(session.mask,
                            &session.mask_host[slot_id * max_ctx + start_row],
                            offset, size);
}

bool t2s_session_init(t2s_session      & session,
                      const t2s_hparams & hparams,
                      ggml_backend_t     backend,
                      uint32_t           n_batch,
                      uint32_t           slot_size)
{
    GGML_ASSERT(backend != nullptr);
    if (n_batch == 0 || slot_size == 0) {
        fprintf(stderr, "%s: n_batch and slot_size must be > 0\n", __func__);
        return false;
    }

    const uint32_t n_layer  = hparams.n_layer;
    const int64_t  d_model  = (int64_t) hparams.hidden_dim;
    const int64_t  max_ctx  = (int64_t) n_batch * slot_size;

    // Total tensors: n_layer * 2 (K/V) + 1 (kv_pos) + 1 (mask) + 1 (x_dec)
    const size_t n_tensors = (size_t) n_layer * 2 + 3;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 1),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    session.ctx_kv = ggml_init(params);
    if (!session.ctx_kv) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    // Allocate tensors.
    session.k_caches.resize(n_layer);
    session.v_caches.resize(n_layer);

    for (uint32_t i = 0; i < n_layer; i++) {
        session.k_caches[i] = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, d_model, max_ctx);
        session.v_caches[i] = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, d_model, max_ctx);
    }

    session.kv_pos = ggml_new_tensor_1d(session.ctx_kv, GGML_TYPE_I32, slot_size);
    session.mask   = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F16, max_ctx, n_batch);

    // Decode input tensor: {d_model, n_batch}
    session.x_dec = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, d_model, n_batch);

    // Allocate backend buffer.
    session.backend = backend;
    session.buf_kv  = ggml_backend_alloc_ctx_tensors(session.ctx_kv, backend);
    if (!session.buf_kv) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        t2s_session_free(session);
        return false;
    }

    // Zero-initialize kv_pos.
    {
        std::vector<int32_t> zeros(slot_size, 0);
        ggml_backend_tensor_set(session.kv_pos, zeros.data(), 0, slot_size * sizeof(int32_t));
    }

    // Initialize mask_host to all -inf and upload.
    {
        const size_t mask_n = (size_t) max_ctx * n_batch;
        session.mask_host.resize(mask_n);
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::fill(session.mask_host.begin(), session.mask_host.end(), neg_inf);
        ggml_backend_tensor_set(session.mask, session.mask_host.data(),
                                0, mask_n * sizeof(ggml_fp16_t));
    }

    session.n_batch   = n_batch;
    session.slot_size = slot_size;
    session.slots.resize(n_batch);

    fprintf(stderr, "%s: n_batch=%u, slot_size=%u, max_ctx=%lld, n_layer=%u\n",
            __func__, n_batch, slot_size, (long long) max_ctx, n_layer);
    return true;
}

void t2s_session_free(t2s_session & session) {
    if (session.alloc_dec) {
        ggml_gallocr_free(session.alloc_dec);
        session.alloc_dec = nullptr;
    }
    if (session.ctx_graph) {
        ggml_free(session.ctx_graph);
        session.ctx_graph = nullptr;
    }
    session.gf_dec = nullptr;
    session.y_dec  = nullptr;

    if (session.buf_kv) {
        ggml_backend_buffer_free(session.buf_kv);
        session.buf_kv = nullptr;
    }
    if (session.ctx_kv) {
        ggml_free(session.ctx_kv);
        session.ctx_kv = nullptr;
    }
    session.backend   = nullptr;
    session.n_batch   = 0;
    session.slot_size = 0;
    session.slots.clear();
    session.k_caches.clear();
    session.v_caches.clear();
    session.kv_pos = nullptr;
    session.mask   = nullptr;
    session.x_dec  = nullptr;
    session.mask_host.clear();
}

int t2s_session_slot_alloc(t2s_session & session, int n_pos) {
    for (size_t i = 0; i < session.slots.size(); i++) {
        if (!session.slots[i].in_use) {
            const int slot_id = (int) i;
            session.slots[slot_id].in_use = true;
            session.slots[slot_id].n_pos  = n_pos;

            // Update mask_host: reveal [slot_id*slot_size, slot_id*slot_size+n_pos)
            // in column slot_id.
            if (n_pos > 0) {
                const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
                const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
                for (int k = 0; k < n_pos; k++) {
                    session.mask_host[slot_id * max_ctx + slot_id * session.slot_size + k] = zero;
                }
                mask_upload_range(session, slot_id,
                                  slot_id * session.slot_size, n_pos);
            }

            return slot_id;
        }
    }
    return -1;
}

void t2s_session_slot_release(t2s_session & session, int slot_id) {
    GGML_ASSERT(slot_id >= 0 && (size_t) slot_id < session.slots.size());

    // Mask out the slot's entire region in column slot_id.
    {
        const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (uint32_t k = 0; k < session.slot_size; k++) {
            session.mask_host[slot_id * max_ctx + slot_id * session.slot_size + k] = neg_inf;
        }
        mask_upload_range(session, slot_id,
                          slot_id * session.slot_size, session.slot_size);
    }

    session.slots[slot_id].in_use = false;
    session.slots[slot_id].n_pos  = 0;
}

int t2s_session_slot_n_pos(const t2s_session & session, int slot_id) {
    GGML_ASSERT(slot_id >= 0 && (size_t) slot_id < session.slots.size());
    return session.slots[slot_id].n_pos;
}

void t2s_session_slot_decode_step(t2s_session & session, int slot_id) {
    GGML_ASSERT(slot_id >= 0 && (size_t) slot_id < session.slots.size());
    GGML_ASSERT(session.slots[slot_id].in_use);
    GGML_ASSERT(session.slots[slot_id].n_pos < (int) session.slot_size);

    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const int     row     = slot_id * session.slot_size + session.slots[slot_id].n_pos;

    session.mask_host[slot_id * max_ctx + row] = ggml_fp32_to_fp16(0.0f);
    mask_upload_range(session, slot_id, row, 1);

    session.slots[slot_id].n_pos++;
}

t2s_layer_caches t2s_session_get_layer_caches(const t2s_session & session, int layer) {
    GGML_ASSERT(layer >= 0 && (size_t) layer < session.k_caches.size());
    return { session.k_caches[layer], session.v_caches[layer] };
}

struct ggml_tensor * t2s_session_get_kv_pos(const t2s_session & session) {
    return session.kv_pos;
}

struct ggml_tensor * t2s_session_get_mask(const t2s_session & session) {
    return session.mask;
}

int t2s_session_get_n_kv(const t2s_session & session) {
    return (int) session.n_batch * session.slot_size;
}

bool t2s_session_build_decode_graph(t2s_session & session, const t2s_model & model) {
    const auto & hparams = model.hparams;
    const int64_t d_model = hparams.hidden_dim;
    const int     n_layer = (int) hparams.n_layer;
    const int     n_head  = (int) hparams.n_head;
    const int     n_kv    = t2s_session_get_n_kv(session);

    // Graph context: holds intermediate tensors and the cgraph structure.
    const size_t n_intermediates = (size_t) n_layer * 32;
    const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 1) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    session.ctx_graph = ggml_init(params);
    if (!session.ctx_graph) {
        fprintf(stderr, "%s: ggml_init() for graph context failed\n", __func__);
        return false;
    }

    session.gf_dec = ggml_new_graph_custom(session.ctx_graph, graph_size, false);

    // kv_pos view: {n_batch} from the full {slot_size}
    struct ggml_tensor * kv_pos_view = ggml_view_1d(
        session.ctx_graph, session.kv_pos, session.n_batch, 0);

    // Build 24-layer decode graph.
    struct ggml_tensor * x = session.x_dec;
    for (int i = 0; i < n_layer; i++) {
        x = t2s_attention_block_forward(
            session.ctx_graph, session.gf_dec, x,
            session.mask,
            session.k_caches[i], session.v_caches[i],
            kv_pos_view,
            model.weights.attention[i],
            n_kv, n_head, 1e-5f);
    }
    ggml_build_forward_expand(session.gf_dec, x);
    session.y_dec = x;

    // Pre-allocate intermediate tensor storage for stable data pointers.
    // This ensures CUDA Graph can reuse the captured graph even when other
    // graphs execute on the same backend between decode steps.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
    session.alloc_dec = ggml_gallocr_new(buft);
    if (!session.alloc_dec) {
        fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(session.alloc_dec, session.gf_dec)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }

    return true;
}

struct ggml_tensor * t2s_session_get_x_dec(const t2s_session & session) {
    return session.x_dec;
}

struct ggml_tensor * t2s_session_get_y_dec(const t2s_session & session) {
    return session.y_dec;
}

struct ggml_cgraph * t2s_session_get_decode_graph(const t2s_session & session) {
    return session.gf_dec;
}

} // namespace gpt_sovits
