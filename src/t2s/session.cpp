#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

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
                      uint32_t           slot_size,
                      enum ggml_type     kv_cache_type)
{
    GGML_ASSERT(backend != nullptr);
    if (n_batch == 0 || slot_size == 0) {
        fprintf(stderr, "%s: n_batch and slot_size must be > 0\n", __func__);
        return false;
    }

    session.kv_cache_type = kv_cache_type;

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
        session.k_caches[i] = ggml_new_tensor_2d(session.ctx_kv, session.kv_cache_type, d_model, max_ctx);
        session.v_caches[i] = ggml_new_tensor_2d(session.ctx_kv, session.kv_cache_type, d_model, max_ctx);
    }

    session.kv_pos = ggml_new_tensor_1d(session.ctx_kv, GGML_TYPE_I32, n_batch);
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
        std::vector<int32_t> zeros(n_batch, 0);
        ggml_backend_tensor_set(session.kv_pos, zeros.data(), 0, n_batch * sizeof(int32_t));
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
    // Free ref_emb cache.
    if (session.buf_ref) {
        ggml_backend_buffer_free(session.buf_ref);
        session.buf_ref = nullptr;
    }
    if (session.ctx_ref) {
        ggml_free(session.ctx_ref);
        session.ctx_ref = nullptr;
    }
    session.ref_emb   = nullptr;
    session.ref_T_ref = 0;

    if (session.alloc_dec) {
        ggml_gallocr_free(session.alloc_dec);
        session.alloc_dec = nullptr;
    }
    if (session.alloc_flex) {
        ggml_gallocr_free(session.alloc_flex);
        session.alloc_flex = nullptr;
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

int t2s_session_find_free_slot(const t2s_session & session) {
    for (size_t i = 0; i < session.slots.size(); i++) {
        if (session.slots[i].n_pos == 0) {
            return (int) i;
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

    session.slots[slot_id].n_pos = 0;
}

int t2s_session_slot_n_pos(const t2s_session & session, int slot_id) {
    GGML_ASSERT(slot_id >= 0 && (size_t) slot_id < session.slots.size());
    return session.slots[slot_id].n_pos;
}

void t2s_session_decode_advance(t2s_session & session) {
    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);

    // The persistent decode graph processes all n_batch slots.
    // Every slot must be in the decode phase (n_pos > 0).
    for (uint32_t i = 0; i < session.n_batch; i++) {
        GGML_ASSERT(session.slots[i].n_pos > 0);
    }

    for (uint32_t i = 0; i < session.n_batch; i++) {
        GGML_ASSERT(session.slots[i].n_pos < (int) session.slot_size);

        const int row = (int)(i * session.slot_size) + session.slots[i].n_pos;

        // Reveal the new KV position in the decode mask.
        session.mask_host[i * max_ctx + row] = zero;
        mask_upload_range(session, (int)i, row, 1);

        // Set kv_pos for this slot so the graph scatter-writes at the correct column.
        const int32_t write_pos = row;
        ggml_backend_tensor_set(session.kv_pos, &write_pos,
                                i * sizeof(int32_t), sizeof(int32_t));

        session.slots[i].n_pos++;
    }
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

    // Build 24-layer decode graph.
    struct ggml_tensor * x = session.x_dec;
    for (int i = 0; i < n_layer; i++) {
        x = t2s_attention_block_forward(
            session.ctx_graph, session.gf_dec, x,
            session.mask,
            session.k_caches[i], session.v_caches[i],
            session.kv_pos,
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

// ---------------------------------------------------------------------------
// Flexible computation graph
// ---------------------------------------------------------------------------

int t2s_batch_plan::total() const {
    int sum = 0;
    for (int v : n_query) { sum += v; }
    return sum;
}

void t2s_flex_graph_free(t2s_flex_graph & graph) {
    // Note: allocator is owned by the session and reused across builds.
    // ggml_free(graph.ctx) releases tensor metadata; the underlying
    // data buffer stays alive inside session.alloc_flex.
    if (graph.ctx) {
        ggml_free(graph.ctx);
        graph.ctx = nullptr;
    }
    graph.gf      = nullptr;
    graph.x       = nullptr;
    graph.y       = nullptr;
    graph.kv_pos  = nullptr;
    graph.mask    = nullptr;
    graph.backend = nullptr;
    graph.N       = 0;
}

t2s_flex_graph t2s_session_build_flex_graph(
    t2s_session          & session,
    const t2s_model      & model,
    const t2s_batch_plan & plan)
{
    t2s_flex_graph graph;

    const int N = plan.total();
    if (N <= 0) {
        fprintf(stderr, "%s: plan has no tokens (total=%d)\n", __func__, N);
        return graph;
    }
    if (plan.n_query.size() != session.n_batch) {
        fprintf(stderr, "%s: plan size (%zu) != n_batch (%u)\n",
                __func__, plan.n_query.size(), session.n_batch);
        return graph;
    }

    const auto & hparams  = model.hparams;
    const int64_t d_model = hparams.hidden_dim;
    const int     n_layer = (int) hparams.n_layer;
    const int     n_head  = (int) hparams.n_head;
    const int     n_kv    = t2s_session_get_n_kv(session);

    // --- 1. Create graph context and tensors ---
    const size_t n_intermediates = (size_t) n_layer * 32;
    const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    graph.ctx = ggml_init(params);
    if (!graph.ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return graph;
    }

    graph.x = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, d_model, N);
    ggml_set_name(graph.x, "x_in");
    ggml_set_input(graph.x);

    graph.kv_pos = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, N);
    ggml_set_name(graph.kv_pos, "kv_pos");
    ggml_set_input(graph.kv_pos);

    graph.mask = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, n_kv, N);
    ggml_set_name(graph.mask, "mask");
    ggml_set_input(graph.mask);

    // --- 2. Build computation graph ---
    graph.gf = ggml_new_graph_custom(graph.ctx, graph_size, false);

    struct ggml_tensor * cur = graph.x;
    for (int i = 0; i < n_layer; i++) {
        cur = t2s_attention_block_forward(
            graph.ctx, graph.gf, cur,
            graph.mask,
            session.k_caches[i], session.v_caches[i],
            graph.kv_pos,
            model.weights.attention[i],
            n_kv, n_head, 1e-5f);
    }
    ggml_build_forward_expand(graph.gf, cur);
    graph.y = cur;
    ggml_set_name(graph.y, "y_out");
    ggml_set_output(graph.y);

    // --- 3. Allocate intermediate storage (reuse session allocator) ---
    if (!session.alloc_flex) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
        session.alloc_flex = ggml_gallocr_new(buft);
        if (!session.alloc_flex) {
            fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
            t2s_flex_graph_free(graph);
            return graph;
        }
    }
    if (!ggml_gallocr_alloc_graph(session.alloc_flex, graph.gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        t2s_flex_graph_free(graph);
        return graph;
    }

    graph.backend = session.backend;
    graph.N = N;

    return graph;
}

void t2s_session_flex_advance(t2s_session       & session,
                          const t2s_batch_plan & plan,
                          t2s_flex_graph            & graph)
{
    GGML_ASSERT(graph.ctx != nullptr);
    GGML_ASSERT(graph.N == plan.total());
    GGML_ASSERT(plan.n_query.size() == session.n_batch);

    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const int     n_kv    = t2s_session_get_n_kv(session);
    const int     N       = graph.N;

    const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    // --- 0. Validate plan ---
    for (uint32_t i = 0; i < session.n_batch; i++) {
        const int nq = plan.n_query[i];
        if (nq <= 0) continue;
        const int n_pos = session.slots[i].n_pos;

        if (nq == 1) {
            // Decode: slot must have existing context.
            GGML_ASSERT(n_pos > 0);
        } else {
            // Prefill: slot must be clean.
            GGML_ASSERT(n_pos == 0);
        }
    }

    // --- 1. Fill graph.kv_pos ---
    {
        std::vector<int32_t> kv_pos_host(N);
        int offset = 0;
        for (uint32_t i = 0; i < session.n_batch; i++) {
            const int nq = plan.n_query[i];
            if (nq <= 0) continue;
            const int n_pos      = session.slots[i].n_pos;
            const int slot_start = (int)(i * session.slot_size);
            GGML_ASSERT(n_pos + nq <= (int)session.slot_size);
            const int base_pos = slot_start + n_pos;
            for (int j = 0; j < nq; j++) {
                kv_pos_host[offset + j] = base_pos + j;
            }
            offset += nq;
        }
        ggml_backend_tensor_set(graph.kv_pos, kv_pos_host.data(),
                                0, (size_t) N * sizeof(int32_t));
    }

    // --- 2. Fill graph.mask ---
    {
        std::vector<ggml_fp16_t> mask_buf((size_t) n_kv * N, neg_inf);

        int col = 0;
        for (uint32_t i = 0; i < session.n_batch; i++) {
            const int nq = plan.n_query[i];
            if (nq <= 0) continue;
            const int slot_start = (int)(i * session.slot_size);
            const int n_pos      = session.slots[i].n_pos;

            for (int j = 0; j < nq; j++) {
                const int attend_up_to = n_pos + j + 1;
                for (int r = 0; r < attend_up_to && r < (int)session.slot_size; r++) {
                    mask_buf[(size_t)(col + j) * n_kv + slot_start + r] = zero;
                }
            }
            col += nq;
        }

        ggml_backend_tensor_set(graph.mask, mask_buf.data(),
                                0, (size_t) n_kv * N * sizeof(ggml_fp16_t));
    }

    // --- 3. Update persistent decode mask ---
    for (uint32_t i = 0; i < session.n_batch; i++) {
        const int nq = plan.n_query[i];
        if (nq <= 0) continue;
        const int old_n_pos  = session.slots[i].n_pos;
        const int slot_start = (int)(i * session.slot_size);
        for (int k = 0; k < nq; k++) {
            session.mask_host[i * max_ctx + slot_start + old_n_pos + k] = zero;
        }
        mask_upload_range(session, (int)i, slot_start + old_n_pos, nq);
    }

    // --- 4. Advance n_pos ---
    for (uint32_t i = 0; i < session.n_batch; i++) {
        if (plan.n_query[i] > 0) {
            session.slots[i].n_pos += plan.n_query[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Reference embedding
// ---------------------------------------------------------------------------

struct ggml_tensor * t2s_session_get_ref_emb(const t2s_session & session) {
    return session.ref_emb;
}

int64_t t2s_session_get_ref_T_ref(const t2s_session & session) {
    return session.ref_T_ref;
}

bool t2s_session_compute_ref_emb(t2s_session       & session,
                                  const t2s_model   & model,
                                  const int32_t     * ref_token_data,
                                  int64_t             T_ref,
                                  const float       * ref_bert_data,
                                  const float       * hubert_data,
                                  int64_t             T_hub)
{
    GGML_ASSERT(session.backend != nullptr);
    GGML_ASSERT(ref_token_data != nullptr);
    GGML_ASSERT(ref_bert_data  != nullptr);
    GGML_ASSERT(hubert_data    != nullptr);
    GGML_ASSERT(T_ref > 0);
    GGML_ASSERT(T_hub > 0);

    const int64_t d_model    = (int64_t) model.hparams.hidden_dim;
    const int64_t bert_dim   = 1024;
    const int64_t hubert_dim = 768;

    // --- Temporary graph context ---
    const size_t n_intermediates = 64;
    const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx_tmp = ggml_init(params);
    if (!ctx_tmp) {
        fprintf(stderr, "%s: ggml_init() for temp context failed\n", __func__);
        return false;
    }

    // --- Input tensors ---
    struct ggml_tensor * ref_token_tensor = ggml_new_tensor_1d(ctx_tmp, GGML_TYPE_I32, T_ref);
    ggml_set_name(ref_token_tensor, "ref_token");
    ggml_set_input(ref_token_tensor);

    struct ggml_tensor * ref_bert_tensor = ggml_new_tensor_2d(ctx_tmp, GGML_TYPE_F32, bert_dim, T_ref);
    ggml_set_name(ref_bert_tensor, "ref_bert");
    ggml_set_input(ref_bert_tensor);

    struct ggml_tensor * hubert_tensor = ggml_new_tensor_2d(ctx_tmp, GGML_TYPE_F32, hubert_dim, T_hub);
    ggml_set_name(hubert_tensor, "hubert");
    ggml_set_input(hubert_tensor);

    // --- Build graph ---
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_tmp, graph_size, false);

    struct ggml_tensor * result = t2s_embed_ref_forward(
        ctx_tmp,
        ref_token_tensor,
        ref_bert_tensor,
        hubert_tensor,
        model.weights.extract_latent,
        model.weights.embed);

    ggml_set_name(result, "ref_emb_out");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // --- Allocate and execute ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!alloc) {
        fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
        ggml_free(ctx_tmp);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx_tmp);
        return false;
    }

    ggml_backend_tensor_set(ref_token_tensor, ref_token_data,
                            0, (size_t) T_ref * sizeof(int32_t));
    ggml_backend_tensor_set(ref_bert_tensor, ref_bert_data,
                            0, (size_t) bert_dim * T_ref * sizeof(float));
    ggml_backend_tensor_set(hubert_tensor, hubert_data,
                            0, (size_t) hubert_dim * T_hub * sizeof(float));

    if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx_tmp);
        return false;
    }

    // --- Create persistent tensor and copy result ---
    const int64_t out_d = result->ne[0];
    const int64_t out_T = result->ne[1];

    struct ggml_init_params ref_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    session.ctx_ref = ggml_init(ref_params);
    if (!session.ctx_ref) {
        fprintf(stderr, "%s: ggml_init() for ref context failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx_tmp);
        return false;
    }

    session.ref_emb = ggml_new_tensor_2d(session.ctx_ref, GGML_TYPE_F32, out_d, out_T);
    ggml_set_name(session.ref_emb, "ref_emb");

    session.buf_ref = ggml_backend_alloc_ctx_tensors(session.ctx_ref, session.backend);
    if (!session.buf_ref) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() for ref failed\n", __func__);
        session.ctx_ref = nullptr;
        session.ref_emb = nullptr;
        ggml_gallocr_free(alloc);
        ggml_free(ctx_tmp);
        return false;
    }

    // Copy: result (temp alloc) -> session.ref_emb (persistent alloc).
    {
        const size_t nbytes = ggml_nbytes(result);
        std::vector<uint8_t> tmp_buf(nbytes);
        ggml_backend_tensor_get(result, tmp_buf.data(), 0, nbytes);
        ggml_backend_tensor_set(session.ref_emb, tmp_buf.data(), 0, nbytes);
    }

    session.ref_T_ref = T_ref;

    // --- Free temporaries ---
    ggml_gallocr_free(alloc);
    ggml_free(ctx_tmp);

    fprintf(stderr, "%s: cached ref_emb {%lld, %lld} (T_ref=%lld)\n",
            __func__, (long long) out_d, (long long) out_T, (long long) T_ref);
    return true;
}

} // namespace gpt_sovits
