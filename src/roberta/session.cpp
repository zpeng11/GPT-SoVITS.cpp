#include "gpt_sovits/roberta.h"
#include "block_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Constants (must match block.cpp / the BERT-large configuration)
// ---------------------------------------------------------------------------

static constexpr int64_t kHiddenSize   = 1024;
static constexpr int64_t kMaxPositions = 512;
static constexpr int kNumLayers = 24;
static constexpr int kTTSLayers = 22; // hidden_states includes embeddings at index 0.

// Same graph budget the RoBERTa parity test uses for the full model
// (embeddings + 24 encoder layers).
static constexpr size_t kMaxNodes = 8192;

namespace {

// ---------------------------------------------------------------------------
// RAII guards for the short-lived graph resources.
// ---------------------------------------------------------------------------

struct ctx_guard {
    ggml_context * p = nullptr;
    ctx_guard() = default;
    explicit ctx_guard(ggml_context * q) : p(q) {}
    ~ctx_guard() { if (p) ggml_free(p); }
    ctx_guard(const ctx_guard &) = delete;
    ctx_guard & operator=(const ctx_guard &) = delete;
};

struct gallocr_guard {
    ggml_gallocr_t p = nullptr;
    gallocr_guard() = default;
    explicit gallocr_guard(ggml_gallocr_t q) : p(q) {}
    ~gallocr_guard() { if (p) ggml_gallocr_free(p); }
    gallocr_guard(const gallocr_guard &) = delete;
    gallocr_guard & operator=(const gallocr_guard &) = delete;
};

// Overflow-checked multiply for host-side length arithmetic (operands >= 0).
static bool mul_ok(int64_t a, int64_t b, int64_t & out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > INT64_MAX / a) return false;
    out = a * b;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

static bool roberta_extract_features_impl(const roberta_model & model,
                                         const int32_t       * token_ids,
                                         int64_t               T,
                                         float               * out,
                                         int64_t               out_cap,
                                         const int32_t       * token_type_ids,
                                         int                   n_layers)
{
    if (model.backend == nullptr) {
        fprintf(stderr, "%s: model has no backend (load it first)\n", __func__);
        return false;
    }
    if (token_ids == nullptr) {
        fprintf(stderr, "%s: token_ids must be non-null\n", __func__);
        return false;
    }
    if (out == nullptr) {
        fprintf(stderr, "%s: out must be non-null\n", __func__);
        return false;
    }
    if (T <= 0) {
        fprintf(stderr, "%s: T=%lld must be > 0\n", __func__, (long long) T);
        return false;
    }
    if (T > kMaxPositions) {
        fprintf(stderr, "%s: T=%lld exceeds max positions (%lld)\n",
                __func__, (long long) T, (long long) kMaxPositions);
        return false;
    }

    int64_t n_floats = 0;
    if (!mul_ok(kHiddenSize, T, n_floats)) {
        fprintf(stderr, "%s: 1024 * T=%lld overflows\n", __func__, (long long) T);
        return false;
    }
    if (out_cap < n_floats) {
        fprintf(stderr, "%s: out_cap=%lld too small (need %lld for T=%lld)\n",
                __func__, (long long) out_cap, (long long) n_floats, (long long) T);
        return false;
    }

    // The block forward asserts on missing weights; validate externally instead.
    const roberta_embeddings_block_weights & emb = model.weights.embeddings;
    if (emb.word_embeddings == nullptr || emb.position_embeddings == nullptr ||
        emb.token_type_embeddings == nullptr ||
        emb.layer_norm_w == nullptr || emb.layer_norm_b == nullptr) {
        fprintf(stderr, "%s: model embeddings weights are not loaded\n", __func__);
        return false;
    }
    if (emb.position_embeddings->ne[1] < T) {
        fprintf(stderr, "%s: model has only %lld position embeddings (need %lld)\n",
                __func__, (long long) emb.position_embeddings->ne[1], (long long) T);
        return false;
    }

    // An out-of-range token id would index outside the embedding tables during
    // the graph build, so reject it up front.
    const int64_t n_vocab = emb.word_embeddings->ne[1];
    for (int64_t i = 0; i < T; ++i) {
        const int32_t id = token_ids[i];
        if (id < 0 || static_cast<int64_t>(id) >= n_vocab) {
            fprintf(stderr, "%s: token_ids[%lld]=%d out of range [0,%lld)\n",
                    __func__, (long long) i, id, (long long) n_vocab);
            return false;
        }
    }
    if (token_type_ids != nullptr) {
        const int64_t n_types = emb.token_type_embeddings->ne[1];
        for (int64_t i = 0; i < T; ++i) {
            const int32_t id = token_type_ids[i];
            if (id < 0 || static_cast<int64_t>(id) >= n_types) {
                fprintf(stderr, "%s: token_type_ids[%lld]=%d out of range [0,%lld)\n",
                        __func__, (long long) i, id, (long long) n_types);
                return false;
            }
        }
    }

    // Host/allocator OOM is caught below; ggml asserts / a corrupt backend are
    // not recovery points and are not caught.
    try {
        // --- Ephemeral graph context (no-alloc; gallocr owns activations) ---
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes +
                             ggml_graph_overhead_custom(kMaxNodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx_guard ctx(ggml_init(params));
        if (!ctx.p) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }

        // --- Inputs ---
        struct ggml_tensor * inp = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T);
        ggml_set_name(inp, "token_ids");
        ggml_set_input(inp);

        struct ggml_tensor * type_ids = nullptr;
        if (token_type_ids != nullptr) {
            type_ids = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T);
            ggml_set_name(type_ids, "token_type_ids");
            ggml_set_input(type_ids);
        }

        // --- Build ---
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx.p, kMaxNodes, false);

        struct ggml_tensor * result = detail::roberta_model_block_forward_n_layers(
            ctx.p, inp, type_ids, model.weights, n_layers);
        ggml_set_name(result, "features");
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);

        // --- Allocate ---
        ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(model.backend);
        gallocr_guard alloc(ggml_gallocr_new(buft));
        if (!alloc.p) {
            fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(alloc.p, gf)) {
            fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
            return false;
        }

        // --- Upload inputs ---
        ggml_backend_tensor_set(inp, token_ids, 0,
                                (size_t) T * sizeof(int32_t));
        if (type_ids != nullptr) {
            ggml_backend_tensor_set(type_ids, token_type_ids, 0,
                                    (size_t) T * sizeof(int32_t));
        }

        // --- Compute ---
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
            return false;
        }

        // --- Copy features out ({1024, T} with time as the outer index) ---
        ggml_backend_tensor_get(result, out, 0,
                                (size_t) n_floats * sizeof(float));
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        return false;
    }

    return true;
}

bool roberta_extract_features(const roberta_model & model,
                              const int32_t       * token_ids,
                              int64_t               T,
                              float               * out,
                              int64_t               out_cap,
                              const int32_t       * token_type_ids)
{
    return roberta_extract_features_impl(
        model, token_ids, T, out, out_cap, token_type_ids, kNumLayers);
}

bool roberta_extract_tts_features(const roberta_model & model,
                                  const int32_t       * token_ids,
                                  int64_t               T,
                                  float               * out,
                                  int64_t               out_cap,
                                  const int32_t       * token_type_ids)
{
    return roberta_extract_features_impl(
        model, token_ids, T, out, out_cap, token_type_ids, kTTSLayers);
}

// ---------------------------------------------------------------------------
// Token -> phoneme feature expansion
// ---------------------------------------------------------------------------

bool roberta_expand_bert_features(const float   * tok_feat,
                                  int64_t         n_tok,
                                  const int32_t * char_starts,
                                  const int32_t * char_ends,
                                  const int32_t * word2ph,
                                  int64_t         n_chars,
                                  int64_t         n_phones,
                                  float         * out)
{
    if (n_tok < 0 || n_chars < 0 || n_phones < 0) {
        fprintf(stderr, "%s: negative length (n_tok=%lld n_chars=%lld n_phones=%lld)\n",
                __func__, (long long) n_tok, (long long) n_chars,
                (long long) n_phones);
        return false;
    }
    if (n_tok > 0 &&
        (tok_feat == nullptr || char_starts == nullptr || char_ends == nullptr)) {
        fprintf(stderr, "%s: tok_feat/char_starts/char_ends must be non-null\n",
                __func__);
        return false;
    }
    if (n_chars > 0 && word2ph == nullptr) {
        fprintf(stderr, "%s: word2ph must be non-null\n", __func__);
        return false;
    }
    if (n_phones > 0 && out == nullptr) {
        fprintf(stderr, "%s: out must be non-null\n", __func__);
        return false;
    }

    // Validate the host-side length arithmetic once; every row offset below is
    // then bounded by these products.
    int64_t tok_floats   = 0;
    int64_t phone_floats = 0;
    if (!mul_ok(n_tok, kHiddenSize, tok_floats) ||
        !mul_ok(n_phones, kHiddenSize, phone_floats)) {
        fprintf(stderr, "%s: length overflow (n_tok=%lld n_phones=%lld)\n",
                __func__, (long long) n_tok, (long long) n_phones);
        return false;
    }
    (void) tok_floats;
    (void) phone_floats;

    // Token spans: either the special-token sentinel (-1, -1) or a valid
    // half-open [start, end) range inside the normalized text.
    for (int64_t t = 0; t < n_tok; ++t) {
        const int32_t s = char_starts[t];
        const int32_t e = char_ends[t];
        if (s == -1 && e == -1) continue;
        if (s < 0 || e < 0) {
            fprintf(stderr, "%s: token %lld has half-set span (%d, %d)\n",
                    __func__, (long long) t, s, e);
            return false;
        }
        if (s > e || static_cast<int64_t>(e) > n_chars) {
            fprintf(stderr, "%s: token %lld span [%d, %d) out of range [0, %lld]\n",
                    __func__, (long long) t, s, e, (long long) n_chars);
            return false;
        }
    }

    // word2ph must be non-negative and sum exactly to n_phones.
    int64_t total = 0;
    for (int64_t i = 0; i < n_chars; ++i) {
        const int32_t count = word2ph[i];
        if (count < 0) {
            fprintf(stderr, "%s: word2ph[%lld]=%d must be >= 0\n",
                    __func__, (long long) i, count);
            return false;
        }
        if (total > INT64_MAX - count) {
            fprintf(stderr, "%s: sum(word2ph) overflows at char %lld\n",
                    __func__, (long long) i);
            return false;
        }
        total += count;
    }
    if (total != n_phones) {
        fprintf(stderr, "%s: sum(word2ph)=%lld != n_phones=%lld\n",
                __func__, (long long) total, (long long) n_phones);
        return false;
    }

    int64_t p = 0;
    for (int64_t i = 0; i < n_chars; ++i) {
        const int32_t count = word2ph[i];
        if (count == 0) continue;

        // Exactly one body token must cover this character (see the header).
        int64_t match     = -1;
        int64_t n_matches = 0;
        for (int64_t t = 0; t < n_tok; ++t) {
            const int32_t s = char_starts[t];
            const int32_t e = char_ends[t];
            if (s == -1 && e == -1) continue;  // special token
            if (static_cast<int64_t>(s) <= i && i < static_cast<int64_t>(e)) {
                if (n_matches == 0) match = t;
                ++n_matches;
                if (n_matches > 1) break;
            }
        }
        if (n_matches == 0) {
            fprintf(stderr, "%s: char %lld has no covering BERT token\n",
                    __func__, (long long) i);
            return false;
        }
        if (n_matches > 1) {
            fprintf(stderr, "%s: char %lld is covered by multiple BERT tokens\n",
                    __func__, (long long) i);
            return false;
        }

        const float * src = tok_feat + match * kHiddenSize;
        for (int32_t k = 0; k < count; ++k) {
            std::memcpy(out + p * kHiddenSize, src, kHiddenSize * sizeof(float));
            ++p;
        }
    }

    // sum(word2ph) == n_phones guarantees this; it is an internal invariant.
    GGML_ASSERT(p == n_phones);
    return true;
}

} // namespace gpt_sovits
