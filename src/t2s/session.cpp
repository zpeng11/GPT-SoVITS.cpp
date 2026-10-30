#include "gpt_sovits/t2s.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <utility>
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

// Upload the entire persistent decode mask from mask_host to the device tensor
// in a single transfer. Used after batching multiple scattered mask_host
// updates to avoid many small per-slot H2D uploads.
static void mask_upload_full(t2s_session & session) {
    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const size_t  nbytes  = (size_t) max_ctx * session.n_batch * sizeof(ggml_fp16_t);
    ggml_backend_tensor_set(session.mask, session.mask_host.data(), 0, nbytes);
}

// Precompute sine positional embedding table on CPU.
// Output layout per position p: [sin0, cos0, sin1, cos1, ...] with d_model elements.
// This matches the reorder performed in build_sine_positional_embedding (block.cpp).
static std::vector<float> build_pe_table_cpu(int64_t d_model, int64_t max_pos,
                                              int max_period = 10000) {
    const int64_t half = d_model / 2;
    std::vector<float> table((size_t)(d_model * max_pos), 0.0f);

    // Precompute frequency values: freq[j] = exp(-log(max_period) * j / half)
    std::vector<float> freq(half);
    for (int64_t j = 0; j < half; j++) {
        freq[j] = std::exp(-std::log((float)max_period) * (float)j / (float)half);
    }

    for (int64_t p = 0; p < max_pos; p++) {
        float * row = table.data() + p * d_model;
        for (int64_t j = 0; j < half; j++) {
            float arg = (float)p * freq[j];
            row[2 * j]     = std::sin(arg);  // sin_j
            row[2 * j + 1] = std::cos(arg);  // cos_j
        }
    }
    return table;
}

bool t2s_session_init(t2s_session             & session,
                      const t2s_hparams       & hparams,
                      ggml_backend_t            backend,
                      uint32_t                  n_batch,
                      uint32_t                  slot_size,
                      const t2s_sampler_config & sampler_cfg,
                      enum ggml_type            kv_cache_type)
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
    const int64_t  vocab    = (int64_t) hparams.vocab_size;

    // Total tensors: n_layer * 2 (K/V) + 1 (kv_pos) + 1 (mask)
    //              + 1 (token_id) + 1 (position) + 1 (pe_table)
    //              + 3 (seen_mask, exp_noise, suppress_eos)
    const size_t n_tensors = (size_t) n_layer * 2 + 8;

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

    // Decode embedding inputs: {n_batch} I32 each
    session.token_id = ggml_new_tensor_1d(session.ctx_kv, GGML_TYPE_I32, n_batch);
    session.position = ggml_new_tensor_1d(session.ctx_kv, GGML_TYPE_I32, n_batch);

    // Precomputed PE table: {d_model, slot_size} F32
    session.pe_table = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, d_model, slot_size);

    // Sampler graph inputs: {vocab, n_batch}
    session.seen_mask = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, vocab, n_batch);
    session.exp_noise = ggml_new_tensor_2d(session.ctx_kv, GGML_TYPE_F32, vocab, n_batch);
    session.suppress_eos = ggml_new_tensor_1d(session.ctx_kv, GGML_TYPE_I32, n_batch);

    // Allocate backend buffer.
    session.backend = backend;
    session.buf_kv  = ggml_backend_alloc_ctx_tensors(session.ctx_kv, backend);
    if (!session.buf_kv) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        t2s_session_free(session);
        return false;
    }

    // Low-level session callers retain unmasked sampling unless explicitly enabled.
    {
        std::vector<int32_t> zeros(n_batch, 0);
        ggml_backend_tensor_set(session.kv_pos, zeros.data(), 0, n_batch * sizeof(int32_t));
        ggml_backend_tensor_set(session.suppress_eos, zeros.data(), 0, n_batch * sizeof(int32_t));
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

    // Compute and upload PE table.
    {
        auto pe_data = build_pe_table_cpu(d_model, slot_size);
        ggml_backend_tensor_set(session.pe_table, pe_data.data(),
                                0, (size_t)(d_model * slot_size) * sizeof(float));
    }

    session.n_batch      = n_batch;
    session.slot_size    = slot_size;
    session.sampler_cfg  = sampler_cfg;
    session.slots.resize(n_batch);

    fprintf(stderr, "%s: n_batch=%u, slot_size=%u, max_ctx=%lld, n_layer=%u\n",
            __func__, n_batch, slot_size, (long long) max_ctx, n_layer);
    return true;
}

void t2s_session_free(t2s_session & session) {
    // Free ref embedding cache.
    if (session.buf_ref) {
        ggml_backend_buffer_free(session.buf_ref);
        session.buf_ref = nullptr;
    }
    if (session.ctx_ref) {
        ggml_free(session.ctx_ref);
        session.ctx_ref = nullptr;
    }
    session.ref_text_emb  = nullptr;
    session.ref_audio_emb = nullptr;
    session.ref_T_ref     = 0;
    session.ref_T_prompt  = 0;
    session.ref_prompt_tokens.clear();
    session.ref_model     = nullptr;

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
    session.decode_model = nullptr;
    session.gf_dec   = nullptr;
    session.sampled  = nullptr;
    session.greedy   = nullptr;
    session.seen_mask = nullptr;
    session.exp_noise = nullptr;
    session.suppress_eos = nullptr;

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
    session.token_id  = nullptr;
    session.position  = nullptr;
    session.pe_table  = nullptr;
    session.seen_mask = nullptr;
    session.exp_noise = nullptr;
    session.suppress_eos = nullptr;
    session.mask_host.clear();
    session.scratch_mask.clear();
    session.scratch_kv_pos.clear();
    session.scratch_decode_pos.clear();
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

    session.slots[slot_id].n_pos        = 0;
    session.slots[slot_id].audio_offset = 0;
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

    // Stage kv_pos / position into scratch buffers so we can bulk-upload once
    // after the loop instead of issuing 2 * n_batch small H2D transfers.
    // Buffers grow on demand; size never shrinks, so steady-state decode is
    // allocation-free.
    if (session.scratch_kv_pos.size() < session.n_batch) {
        session.scratch_kv_pos.resize(session.n_batch);
    }
    if (session.scratch_decode_pos.size() < session.n_batch) {
        session.scratch_decode_pos.resize(session.n_batch);
    }
    int32_t * kv_pos_host = session.scratch_kv_pos.data();
    int32_t * pos_host    = session.scratch_decode_pos.data();

    for (uint32_t i = 0; i < session.n_batch; i++) {
        GGML_ASSERT(session.slots[i].n_pos < (int) session.slot_size);

        const int row = (int)(i * session.slot_size) + session.slots[i].n_pos;

        // Reveal the new KV position in the decode mask (host mirror only —
        // uploaded once after the loop to avoid n_batch small H2D transfers).
        session.mask_host[i * max_ctx + row] = zero;

        // kv_pos: graph scatter-writes the new K/V at this column.
        kv_pos_host[i] = row;

        // Auto-fill PE position: pe_pos = n_pos - audio_offset.
        // This mirrors the logic in t2s_session_flex_advance for decode slots.
        GGML_ASSERT(session.slots[i].audio_offset >= 0 &&
                    session.slots[i].audio_offset < (int) session.slot_size &&
                    "decode on slot with invalid audio_offset");
        const int32_t pe_pos = session.slots[i].n_pos - session.slots[i].audio_offset;
        GGML_ASSERT(pe_pos >= 0 && pe_pos < (int) session.slot_size &&
                    "decode PE position out of pe_table bounds");
        pos_host[i] = pe_pos;

        session.slots[i].n_pos++;
    }

    // Single bulk upload for each of the three small input tensors.
    mask_upload_full(session);
    ggml_backend_tensor_set(session.kv_pos, kv_pos_host,
                            0, (size_t) session.n_batch * sizeof(int32_t));
    ggml_backend_tensor_set(session.position, pos_host,
                            0, (size_t) session.n_batch * sizeof(int32_t));
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

bool t2s_session_build_decode_graph(
    t2s_session             & session,
    const t2s_model         & model)
{
    if (session.ctx_graph || session.gf_dec || session.alloc_dec || session.decode_model) {
        fprintf(stderr, "%s: decode graph already built\n", __func__);
        return false;
    }
    if (!session.backend || !session.ctx_kv || !session.buf_kv) {
        fprintf(stderr, "%s: session is not initialized\n", __func__);
        return false;
    }

    struct graph_resources {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ~graph_resources() {
            if (alloc) ggml_gallocr_free(alloc);
            if (ctx) ggml_free(ctx);
        }
    } resources;

    try {
        const auto & hparams = model.hparams;
        const int     n_layer = (int) hparams.n_layer;
        const int     n_head  = (int) hparams.n_head;
        const int     n_kv    = t2s_session_get_n_kv(session);
        const int     n_batch = (int) session.n_batch;
        const t2s_sampler_config & sampler_cfg = session.sampler_cfg;

        // Reserve both full-vocabulary and EOS-excluded sampler branches.
        const size_t n_intermediates = (size_t) n_layer * 32 + (size_t) n_batch * 200 + 12;
        const size_t graph_size      = std::max((size_t) GGML_DEFAULT_GRAPH_SIZE, n_intermediates);
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 1) +
                             ggml_graph_overhead_custom(graph_size, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        resources.ctx = ggml_init(params);
        if (!resources.ctx) {
            fprintf(stderr, "%s: ggml_init() for graph context failed\n", __func__);
            return false;
        }
        auto * ctx = resources.ctx;
        auto * gf = ggml_new_graph_custom(ctx, graph_size, false);

        struct ggml_tensor * emb = ggml_get_rows(ctx,
            model.weights.embed.audio_embedding, session.token_id);
        struct ggml_tensor * pe = ggml_get_rows(ctx, session.pe_table, session.position);
        struct ggml_tensor * scaled_pe = ggml_mul(ctx, pe, model.weights.embed.audio_pos_alpha);
        struct ggml_tensor * x = ggml_add(ctx, emb, scaled_pe);

        for (int i = 0; i < n_layer; i++) {
            x = t2s_attention_block_forward(
                ctx, gf, x, session.mask,
                session.k_caches[i], session.v_caches[i], session.kv_pos,
                model.weights.attention[i], n_kv, n_head, 1e-5f);
        }

        t2s_sampler_result sampler_result = t2s_sampler_block_forward(
            ctx, x, model.weights.lm_head_w, session.seen_mask,
            sampler_cfg.top_k, sampler_cfg.top_p, sampler_cfg.temperature,
            sampler_cfg.repetition_penalty, session.exp_noise,
            hparams.vocab_size > 1 ? session.suppress_eos : nullptr);

        ggml_set_name(sampler_result.sampled, "sampled");
        ggml_set_output(sampler_result.sampled);
        ggml_build_forward_expand(gf, sampler_result.sampled);
        ggml_set_name(sampler_result.greedy, "greedy");
        ggml_set_output(sampler_result.greedy);
        ggml_build_forward_expand(gf, sampler_result.greedy);

        // Persistent allocation keeps data pointers stable for CUDA graph reuse.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
        resources.alloc = ggml_gallocr_new(buft);
        if (!resources.alloc) {
            fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(resources.alloc, gf)) {
            fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
            return false;
        }

        session.ctx_graph = resources.ctx;
        session.alloc_dec = resources.alloc;
        session.gf_dec = gf;
        session.sampled = sampler_result.sampled;
        session.greedy = sampler_result.greedy;
        session.decode_model = &model;
        resources.ctx = nullptr;
        resources.alloc = nullptr;
        return true;
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        return false;
    }
}

struct ggml_tensor * t2s_session_get_token_id(const t2s_session & session) {
    return session.token_id;
}

struct ggml_tensor * t2s_session_get_position(const t2s_session & session) {
    return session.position;
}

struct ggml_cgraph * t2s_session_get_decode_graph(const t2s_session & session) {
    return session.gf_dec;
}

struct ggml_tensor * t2s_session_get_sampled(const t2s_session & session) {
    return session.sampled;
}

struct ggml_tensor * t2s_session_get_greedy(const t2s_session & session) {
    return session.greedy;
}

struct ggml_tensor * t2s_session_get_seen_mask(const t2s_session & session) {
    return session.seen_mask;
}

struct ggml_tensor * t2s_session_get_exp_noise(const t2s_session & session) {
    return session.exp_noise;
}

struct ggml_tensor * t2s_session_get_suppress_eos(const t2s_session & session) {
    return session.suppress_eos;
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
    graph.gf              = nullptr;
    graph.x_prefill       = nullptr;
    graph.decode_token_ids = nullptr;
    graph.decode_positions = nullptr;
    graph.y               = nullptr;
    graph.kv_pos          = nullptr;
    graph.mask            = nullptr;
    graph.seen_mask       = nullptr;
    graph.exp_noise       = nullptr;
    graph.suppress_eos    = nullptr;
    graph.sampled         = nullptr;
    graph.greedy          = nullptr;
    graph.backend         = nullptr;
    graph.N               = 0;
    graph.n_active        = 0;
    graph.n_decode        = 0;
}

t2s_flex_graph t2s_session_build_flex_graph(
    t2s_session             & session,
    const t2s_model         & model,
    const t2s_batch_plan    & plan)
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
    const t2s_sampler_config & sampler_cfg = session.sampler_cfg;
    const int64_t d_model = hparams.hidden_dim;
    const int     n_layer = (int) hparams.n_layer;
    const int     n_head  = (int) hparams.n_head;
    const int     n_kv    = t2s_session_get_n_kv(session);
    const int64_t vocab   = hparams.vocab_size;

    // Classify slots: prefill (n_query > 1), decode (n_query == 1), idle (0).
    int n_active  = 0;
    int n_decode  = 0;
    int N_prefill = 0;
    for (int i = 0; i < (int) plan.n_query.size(); i++) {
        const int nq = plan.n_query[i];
        if (nq <= 0) continue;
        n_active++;
        if (nq == 1) {
            n_decode++;
        } else {
            N_prefill += nq;
        }
    }
    const int N_decode = n_decode;  // each decode slot contributes exactly 1 token

    // --- 1. Create graph context ---
    // Reserve both sampler branches and ~4 decode embedding intermediates per slot.
    const size_t n_intermediates = (size_t) n_layer * 32
                                 + (size_t) n_active * 200
                                 + (size_t) n_decode * 4
                                 + 8;
    const size_t graph_size      = std::max((size_t) GGML_DEFAULT_GRAPH_SIZE, n_intermediates);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 8) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    graph.ctx = ggml_init(params);
    if (!graph.ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return graph;
    }

    struct graph_cleanup {
        t2s_flex_graph * graph;
        t2s_session & session;
        ~graph_cleanup() {
            if (!graph) return;
            t2s_flex_graph_free(*graph);
            // Failed gallocr reservations can retain a plan without its buffer.
            if (session.alloc_flex) ggml_gallocr_free(session.alloc_flex);
            session.alloc_flex = nullptr;
        }
    } cleanup{&graph, session};

    // --- 2. Create input tensors ---
    if (N_prefill > 0) {
        graph.x_prefill = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, d_model, N_prefill);
        ggml_set_name(graph.x_prefill, "x_prefill");
        ggml_set_input(graph.x_prefill);
    }

    if (n_decode > 0) {
        graph.decode_token_ids = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, n_decode);
        ggml_set_name(graph.decode_token_ids, "decode_token_ids");
        ggml_set_input(graph.decode_token_ids);

        graph.decode_positions = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, n_decode);
        ggml_set_name(graph.decode_positions, "decode_positions");
        ggml_set_input(graph.decode_positions);
    }

    graph.kv_pos = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, N);
    ggml_set_name(graph.kv_pos, "kv_pos");
    ggml_set_input(graph.kv_pos);

    graph.mask = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, n_kv, N);
    ggml_set_name(graph.mask, "mask");
    ggml_set_input(graph.mask);

    // --- 3. Build full input sequence by interleaving prefill/decode in slot order ---
    graph.gf = ggml_new_graph_custom(graph.ctx, graph_size, false);

    struct ggml_tensor * full_x = nullptr;

    {
        // Batch all decode token embeddings at once: get_rows yields {d_model, n_decode}
        struct ggml_tensor * decode_embs = nullptr;
        if (n_decode > 0) {
            struct ggml_tensor * emb = ggml_get_rows(graph.ctx,
                model.weights.embed.audio_embedding, graph.decode_token_ids);
            struct ggml_tensor * pe  = ggml_get_rows(graph.ctx,
                session.pe_table, graph.decode_positions);
            struct ggml_tensor * scaled_pe = ggml_mul(graph.ctx, pe,
                model.weights.embed.audio_pos_alpha);
            decode_embs = ggml_add(graph.ctx, emb, scaled_pe);
        }

        // Walk slots in order, concatenating chunks to build the full input sequence.
        int prefill_offset = 0;  // column offset into x_prefill
        int decode_idx     = 0;  // index into decode_embs columns

        for (int i = 0; i < (int) plan.n_query.size(); i++) {
            const int nq = plan.n_query[i];
            if (nq <= 0) continue;

            struct ggml_tensor * chunk = nullptr;

            if (nq == 1) {
                // Decode slot: extract one column from decode_embs.
                chunk = ggml_view_2d(graph.ctx, decode_embs,
                    d_model, 1, decode_embs->nb[1],
                    (size_t) decode_idx * decode_embs->nb[1]);
                decode_idx++;
            } else {
                // Prefill slot: extract nq columns from x_prefill.
                chunk = ggml_view_2d(graph.ctx, graph.x_prefill,
                    d_model, nq, graph.x_prefill->nb[1],
                    (size_t) prefill_offset * graph.x_prefill->nb[1]);
                prefill_offset += nq;
            }

            if (!full_x) {
                full_x = chunk;
            } else {
                full_x = ggml_concat(graph.ctx, full_x, chunk, 1);
            }
        }
    }

    // --- 4. Build transformer layers ---
    struct ggml_tensor * cur = full_x;
    for (int i = 0; i < n_layer; i++) {
        cur = t2s_attention_block_forward(
            graph.ctx, graph.gf, cur,
            graph.mask,
            session.k_caches[i], session.v_caches[i],
            graph.kv_pos,
            model.weights.attention[i],
            n_kv, n_head, 1e-5f);
    }

    // Mark full attention output as graph output (for verification / parity tests).
    graph.y = cur;
    ggml_set_name(graph.y, "y_out");
    ggml_set_output(graph.y);
    ggml_build_forward_expand(graph.gf, graph.y);

    // --- 5. Attach sampler ---
    // For each active slot, extract the last token's hidden state:
    //   - decode (n_query == 1): the single column
    //   - prefill (n_query > 1): the last column in the slot's range
    {
        struct ggml_tensor * y = cur;  // {d_model, N}

        // Collect per-slot column views from y.
        std::vector<struct ggml_tensor *> sample_cols;
        int offset = 0;
        for (int i = 0; i < (int) plan.n_query.size(); i++) {
            if (plan.n_query[i] > 0) {
                int col_idx = offset + plan.n_query[i] - 1;
                struct ggml_tensor * col = ggml_view_2d(
                    graph.ctx, y, d_model, 1,
                    y->nb[1], (size_t) col_idx * y->nb[1]);
                sample_cols.push_back(col);
            }
            offset += plan.n_query[i];
        }

        // Concat into y_sample = {d_model, n_active}.
        struct ggml_tensor * y_sample;
        if (sample_cols.size() == 1) {
            y_sample = sample_cols[0];
        } else {
            y_sample = sample_cols[0];
            for (size_t i = 1; i < sample_cols.size(); i++) {
                y_sample = ggml_concat(graph.ctx, y_sample, sample_cols[i], 1);
            }
        }

        // Sampler graph inputs.
        // seen_mask is only reachable in the graph when repetition_penalty != 1.0,
        // so only create it when it will actually be used by the sampler.
        struct ggml_tensor * seen_mask_arg = nullptr;
        if (sampler_cfg.repetition_penalty != 1.0f) {
            graph.seen_mask = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, vocab, n_active);
            ggml_set_name(graph.seen_mask, "seen_mask");
            ggml_set_input(graph.seen_mask);
            seen_mask_arg = graph.seen_mask;
        }

        graph.exp_noise = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, vocab, n_active);
        ggml_set_name(graph.exp_noise, "exp_noise");
        ggml_set_input(graph.exp_noise);
        if (vocab > 1) {
            graph.suppress_eos = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, n_active);
            ggml_set_name(graph.suppress_eos, "suppress_eos");
            ggml_set_input(graph.suppress_eos);
        }

        // Attach sampler: project hidden states to logits, apply filtering, sample tokens.
        t2s_sampler_result sampler_result = t2s_sampler_block_forward(
            graph.ctx,
            y_sample,
            model.weights.lm_head_w,
            seen_mask_arg,
            sampler_cfg.top_k,
            sampler_cfg.top_p,
            sampler_cfg.temperature,
            sampler_cfg.repetition_penalty,
            graph.exp_noise,
            graph.suppress_eos);

        graph.sampled = sampler_result.sampled;
        ggml_set_name(graph.sampled, "sampled");
        ggml_set_output(graph.sampled);
        ggml_build_forward_expand(graph.gf, graph.sampled);

        graph.greedy = sampler_result.greedy;
        ggml_set_name(graph.greedy, "greedy");
        ggml_set_output(graph.greedy);
        ggml_build_forward_expand(graph.gf, graph.greedy);
    }

    // --- 6. Allocate intermediate storage (reuse session allocator) ---
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

    if (graph.suppress_eos != nullptr) {
        const std::vector<int32_t> zeros((size_t) n_active, 0);
        ggml_backend_tensor_set(graph.suppress_eos, zeros.data(), 0,
                                zeros.size() * sizeof(int32_t));
    }

    graph.backend  = session.backend;
    graph.N        = N;
    graph.n_active = n_active;
    graph.n_decode = n_decode;

    cleanup.graph = nullptr;
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
        if ((int) session.scratch_kv_pos.size() < N) {
            session.scratch_kv_pos.resize(N);
        }
        int32_t * kv_pos_host = session.scratch_kv_pos.data();
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
        ggml_backend_tensor_set(graph.kv_pos, kv_pos_host,
                                0, (size_t) N * sizeof(int32_t));
    }

    // --- 2. Fill graph.mask ---
    {
        const size_t mask_n = (size_t) n_kv * (size_t) N;
        if (session.scratch_mask.size() < mask_n) {
            session.scratch_mask.resize(mask_n);
        }
        ggml_fp16_t * mask_buf = session.scratch_mask.data();
        // Only the active [0, mask_n) region needs to be reset each step;
        // the rest of the buffer (if any) is left untouched.
        std::fill(mask_buf, mask_buf + mask_n, neg_inf);

        int col = 0;
        for (uint32_t i = 0; i < session.n_batch; i++) {
            const int nq = plan.n_query[i];
            if (nq <= 0) continue;
            const int slot_start = (int)(i * session.slot_size);
            const int n_pos      = session.slots[i].n_pos;

            if (nq > 1) {
                // Prefill: three-part mask matching the Python model.
                // Sequence layout: [T_ref (ref text) | T_in (input text) | T_prompt (ref audio)]
                // Text tokens see all text bidirectionally but no audio.
                // Audio tokens see all text + causal (autoregressive) audio.
                const int T_text = nq - (int) session.ref_T_prompt;
                for (int j = 0; j < nq; j++) {
                    const int attend_up_to = (j < T_text) ? T_text : (j + 1);
                    for (int r = 0; r < attend_up_to && r < (int)session.slot_size; r++) {
                        mask_buf[(size_t)(col + j) * n_kv + slot_start + r] = zero;
                    }
                }
            } else {
                // Decode: single token attends to all valid KV positions in slot.
                const int attend_up_to = n_pos + 1;
                for (int r = 0; r < attend_up_to && r < (int)session.slot_size; r++) {
                    mask_buf[(size_t) col * n_kv + slot_start + r] = zero;
                }
            }
            col += nq;
        }

        ggml_backend_tensor_set(graph.mask, mask_buf,
                                0, (size_t) n_kv * N * sizeof(ggml_fp16_t));
    }

    // --- 3. Fill decode_positions and set audio_offset ---
    if (graph.n_decode > 0 && graph.decode_positions) {
        if ((int) session.scratch_decode_pos.size() < graph.n_decode) {
            session.scratch_decode_pos.resize(graph.n_decode);
        }
        int32_t * pos_host = session.scratch_decode_pos.data();
        int decode_idx = 0;
        for (uint32_t i = 0; i < session.n_batch; i++) {
            const int nq = plan.n_query[i];
            if (nq <= 0) continue;

            if (nq == 1) {
                // Decode: PE position = n_pos - audio_offset
                GGML_ASSERT(decode_idx < graph.n_decode);
                GGML_ASSERT(session.slots[i].audio_offset >= 0 &&
                            session.slots[i].audio_offset < (int) session.slot_size &&
                            "decode on slot with invalid audio_offset");
                const int pe_pos = session.slots[i].n_pos - session.slots[i].audio_offset;
                GGML_ASSERT(pe_pos >= 0 && pe_pos < (int) session.slot_size &&
                            "decode PE position out of pe_table bounds");
                pos_host[decode_idx] = pe_pos;
                decode_idx++;
            } else {
                // Prefill: set audio_offset = nq - ref_T_prompt (= T_ref + T_in)
                const int offset = nq - (int) session.ref_T_prompt;
                GGML_ASSERT(offset >= 0 &&
                            "prefill n_query < ref_T_prompt (audio_offset would be negative)");
                session.slots[i].audio_offset = offset;
            }
        }
        ggml_backend_tensor_set(graph.decode_positions, pos_host,
                                0, (size_t) graph.n_decode * sizeof(int32_t));
    } else {
        // No decode slots — only set audio_offset for prefill slots.
        for (uint32_t i = 0; i < session.n_batch; i++) {
            const int nq = plan.n_query[i];
            if (nq > 1) {
                const int offset = nq - (int) session.ref_T_prompt;
                GGML_ASSERT(offset >= 0 &&
                            "prefill n_query < ref_T_prompt (audio_offset would be negative)");
                session.slots[i].audio_offset = offset;
            }
        }
    }

    // --- 4. Update persistent decode mask ---
    // Write all revealed positions to mask_host first, then bulk-upload once
    // to avoid per-slot H2D transfers.
    for (uint32_t i = 0; i < session.n_batch; i++) {
        const int nq = plan.n_query[i];
        if (nq <= 0) continue;
        const int old_n_pos  = session.slots[i].n_pos;
        const int slot_start = (int)(i * session.slot_size);
        for (int k = 0; k < nq; k++) {
            session.mask_host[i * max_ctx + slot_start + old_n_pos + k] = zero;
        }
    }
    mask_upload_full(session);

    // --- 5. Advance n_pos ---
    for (uint32_t i = 0; i < session.n_batch; i++) {
        if (plan.n_query[i] > 0) {
            session.slots[i].n_pos += plan.n_query[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Reference embedding
// ---------------------------------------------------------------------------

namespace {

// RAII guards for the short-lived graph resources and the new persistent ref
// cache. Ownership is handed to the session only after every step succeeds;
// on any failure the destructors release exactly the resources created here.
struct ref_temp_resources {
    struct ggml_context * ctx   = nullptr;
    ggml_gallocr_t        alloc = nullptr;
    ~ref_temp_resources() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

struct ref_cache_resources {
    struct ggml_context  * ctx = nullptr;
    ggml_backend_buffer_t  buf = nullptr;
    ~ref_cache_resources() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// Overflow-checked multiply for host-side size arithmetic (operands >= 0).
static bool mul_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > INT64_MAX / a) return false;
    out = a * b;
    return true;
}

// Convert an element count to bytes, rejecting negative or overflowing sizes.
static bool bytes_ok(int64_t n, size_t elem_size, size_t & out) {
    if (n < 0) return false;
    const size_t ne = (size_t) n;
    if (elem_size != 0 && ne > SIZE_MAX / elem_size) return false;
    out = ne * elem_size;
    return true;
}

} // namespace

struct ggml_tensor * t2s_session_get_ref_text_emb(const t2s_session & session) {
    return session.ref_text_emb;
}

struct ggml_tensor * t2s_session_get_ref_audio_emb(const t2s_session & session) {
    return session.ref_audio_emb;
}

int64_t t2s_session_get_ref_T_ref(const t2s_session & session) {
    return session.ref_T_ref;
}

int64_t t2s_session_get_ref_T_prompt(const t2s_session & session) {
    return session.ref_T_prompt;
}

bool t2s_session_compute_ref_emb(t2s_session       & session,
                                  const t2s_model   & model,
                                  const int32_t     * ref_token_data,
                                  int64_t             T_ref,
                                  const float       * ref_bert_data,
                                  const float       * hubert_data,
                                  int64_t             T_hub)
{
    // One session = one reference. Reject duplicates before touching any state
    // so a second call cannot overwrite (and leak) the cached tensors.
    if (session.ctx_ref || session.buf_ref || session.ref_text_emb ||
        session.ref_audio_emb || session.ref_model || !session.ref_prompt_tokens.empty()) {
        fprintf(stderr, "%s: reference embedding already computed\n", __func__);
        return false;
    }
    if (session.backend == nullptr) {
        fprintf(stderr, "%s: session backend is null\n", __func__);
        return false;
    }
    if (ref_token_data == nullptr || ref_bert_data == nullptr || hubert_data == nullptr) {
        fprintf(stderr, "%s: input data must be non-null\n", __func__);
        return false;
    }
    if (T_ref <= 0) {
        fprintf(stderr, "%s: T_ref must be > 0 (got %lld)\n",
                __func__, (long long) T_ref);
        return false;
    }
    // extract-latent uses Conv1d(kernel=2, stride=2), which needs a complete
    // two-frame window; T_hub < 2 yields no prompt frame.
    if (T_hub < 2) {
        fprintf(stderr, "%s: T_hub must be >= 2 (got %lld)\n",
                __func__, (long long) T_hub);
        return false;
    }
    if (model.hparams.phoneme_vocab_size == 0) {
        fprintf(stderr, "%s: model has no phoneme vocabulary\n", __func__);
        return false;
    }

    const int64_t bert_dim   = 1024;
    const int64_t hubert_dim = 768;

    // Validate every host-side size before allocating so a bad input cannot
    // throw std::length_error / std::bad_alloc out of the public API.
    size_t token_bytes = 0;
    size_t bert_bytes  = 0;
    size_t hub_bytes   = 0;
    {
        int64_t bert_elems   = 0;
        int64_t hubert_elems = 0;
        if (!bytes_ok(T_ref, sizeof(int32_t), token_bytes) ||
            !mul_ok(bert_dim, T_ref, bert_elems) ||
            !bytes_ok(bert_elems, sizeof(float), bert_bytes) ||
            !mul_ok(hubert_dim, T_hub, hubert_elems) ||
            !bytes_ok(hubert_elems, sizeof(float), hub_bytes)) {
            fprintf(stderr, "%s: input size overflow\n", __func__);
            return false;
        }
    }

    // Only walk the token array once T_ref has passed the size checks above,
    // otherwise a bogus length could read past the caller's buffer.
    for (int64_t i = 0; i < T_ref; i++) {
        const int32_t tok = ref_token_data[i];
        if (tok < 0 || (uint32_t) tok >= model.hparams.phoneme_vocab_size) {
            fprintf(stderr, "%s: ref_token[%lld]=%d out of range [0, %u)\n",
                    __func__, (long long) i, tok, model.hparams.phoneme_vocab_size);
            return false;
        }
    }

    ref_temp_resources  temp;
    ref_cache_resources cache;

    try {
        // --- Temporary graph context ---
        const size_t n_intermediates = 64;
        const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                             ggml_graph_overhead_custom(graph_size, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        temp.ctx = ggml_init(params);
        if (!temp.ctx) {
            fprintf(stderr, "%s: ggml_init() for temp context failed\n", __func__);
            return false;
        }

        // --- Input tensors ---
        struct ggml_tensor * ref_token_tensor = ggml_new_tensor_1d(temp.ctx, GGML_TYPE_I32, T_ref);
        ggml_set_name(ref_token_tensor, "ref_token");
        ggml_set_input(ref_token_tensor);

        struct ggml_tensor * ref_bert_tensor = ggml_new_tensor_2d(temp.ctx, GGML_TYPE_F32, bert_dim, T_ref);
        ggml_set_name(ref_bert_tensor, "ref_bert");
        ggml_set_input(ref_bert_tensor);

        struct ggml_tensor * hubert_tensor = ggml_new_tensor_2d(temp.ctx, GGML_TYPE_F32, hubert_dim, T_hub);
        ggml_set_name(hubert_tensor, "hubert");
        ggml_set_input(hubert_tensor);

        // --- Build graph ---
        struct ggml_cgraph * gf = ggml_new_graph_custom(temp.ctx, graph_size, false);

        struct ggml_tensor * result = t2s_embed_ref_forward(
            temp.ctx,
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
        temp.alloc = ggml_gallocr_new(buft);
        if (!temp.alloc) {
            fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
            return false;
        }

        if (!ggml_gallocr_alloc_graph(temp.alloc, gf)) {
            fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
            return false;
        }

        ggml_backend_tensor_set(ref_token_tensor, ref_token_data, 0, token_bytes);
        ggml_backend_tensor_set(ref_bert_tensor, ref_bert_data, 0, bert_bytes);
        ggml_backend_tensor_set(hubert_tensor, hubert_data, 0, hub_bytes);

        if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
            return false;
        }

        // --- Validate output shape and derive the persistent sizes ---
        const int64_t out_d = result->ne[0];
        const int64_t out_T = result->ne[1];
        if (out_d <= 0 || out_T <= 0 || out_T < T_ref) {
            fprintf(stderr, "%s: invalid ref embedding shape {%lld, %lld}\n",
                    __func__, (long long) out_d, (long long) out_T);
            return false;
        }
        const int64_t T_prompt = out_T - T_ref;
        if (T_prompt <= 0) {
            fprintf(stderr, "%s: ref prompt token count must be > 0\n", __func__);
            return false;
        }

        int64_t text_elems  = 0;
        int64_t audio_elems = 0;
        size_t  text_bytes  = 0;
        size_t  audio_bytes = 0;
        size_t  prompt_bytes = 0;
        if (!mul_ok(out_d, T_ref, text_elems) ||
            !bytes_ok(text_elems, sizeof(float), text_bytes) ||
            !mul_ok(out_d, T_prompt, audio_elems) ||
            !bytes_ok(audio_elems, sizeof(float), audio_bytes) ||
            !bytes_ok(T_prompt, sizeof(int32_t), prompt_bytes)) {
            fprintf(stderr, "%s: output size overflow\n", __func__);
            return false;
        }

        struct ggml_tensor * prompt = ggml_graph_get_tensor(gf, "ref_prompt_tokens");
        if (!prompt || prompt->type != GGML_TYPE_I32 || prompt->ne[0] != T_prompt ||
            prompt->ne[1] != 1 || prompt->ne[2] != 1 || prompt->ne[3] != 1 ||
            !ggml_is_contiguous(prompt)) {
            fprintf(stderr, "%s: invalid reference prompt token layout\n", __func__);
            return false;
        }
        std::vector<int32_t> prompt_tokens((size_t) T_prompt);
        ggml_backend_tensor_get(prompt, prompt_tokens.data(), 0, prompt_bytes);
        for (int32_t token : prompt_tokens) {
            if (token < 0 || (uint32_t) token >= model.hparams.eos) {
                fprintf(stderr, "%s: reference prompt token %d outside [0, %u)\n",
                        __func__, token, model.hparams.eos);
                return false;
            }
        }

        // --- Create (but do not yet publish) the persistent tensors ---
        struct ggml_init_params ref_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 3,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        cache.ctx = ggml_init(ref_params);
        if (!cache.ctx) {
            fprintf(stderr, "%s: ggml_init() for ref context failed\n", __func__);
            return false;
        }

        struct ggml_tensor * new_text =
            ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, out_d, T_ref);
        ggml_set_name(new_text, "ref_text_emb");
        struct ggml_tensor * new_audio =
            ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, out_d, T_prompt);
        ggml_set_name(new_audio, "ref_audio_emb");

        cache.buf = ggml_backend_alloc_ctx_tensors(cache.ctx, session.backend);
        if (!cache.buf) {
            fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() for ref failed\n", __func__);
            return false;
        }

        // Copy: split result (temp alloc) into two persistent tensors.
        {
            std::vector<uint8_t> tmp_buf(text_bytes);
            ggml_backend_tensor_get(result, tmp_buf.data(), 0, text_bytes);
            ggml_backend_tensor_set(new_text, tmp_buf.data(), 0, text_bytes);
        }
        {
            std::vector<uint8_t> tmp_buf(audio_bytes);
            ggml_backend_tensor_get(result, tmp_buf.data(), text_bytes, audio_bytes);
            ggml_backend_tensor_set(new_audio, tmp_buf.data(), 0, audio_bytes);
        }

        // --- All steps succeeded: atomically publish the cache. Everything
        //     past this point is a non-throwing move. ---
        session.ctx_ref       = cache.ctx;
        session.buf_ref       = cache.buf;
        session.ref_text_emb  = new_text;
        session.ref_audio_emb = new_audio;
        session.ref_T_ref     = T_ref;
        session.ref_T_prompt  = T_prompt;
        session.ref_prompt_tokens = std::move(prompt_tokens);
        session.ref_model     = &model;
        cache.ctx = nullptr;
        cache.buf = nullptr;
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory (reference cache left unchanged)\n", __func__);
        return false;
    } catch (const std::length_error &) {
        fprintf(stderr, "%s: input size too large (reference cache left unchanged)\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: cached ref_text_emb {%lld, %lld}, ref_audio_emb {%lld, %lld}\n",
            __func__, (long long) session.ref_text_emb->ne[0],  (long long) session.ref_T_ref,
                    (long long) session.ref_audio_emb->ne[0], (long long) session.ref_T_prompt);
    return true;
}

} // namespace gpt_sovits
