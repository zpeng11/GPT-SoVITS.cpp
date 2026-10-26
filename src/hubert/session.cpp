// src/hubert/session.cpp
//
// Host-side runner for full HuBERT feature extraction.
//
// hubert_extract_features_from_pcm() consumes the caller's waveform verbatim:
// it owns only the graph construction and compute sequence and never
// normalizes, resamples, or pads the input.
//
// hubert_extract_features_from_normalized_pcm() is kept as a compatibility
// proxy for the historical "caller pre-normalized" contract; it delegates to
// hubert_extract_features_from_pcm() without touching the samples.
//
// This wraps the same ephemeral-graph compute sequence used by
// tests/hubert/test_hubert_parity.cpp: build a no-alloc context/graph, create
// the waveform input, expand the model graph, allocate via ggml_gallocr, upload
// the PCM, compute, and copy the hidden states back to the caller.
//
// Output layout follows ggml convention: the result is {768, T'}, where ne[0]
// (the hidden size) is contiguous, so out[t * 768 + c] is frame t, channel c.

#include "gpt_sovits/hubert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace gpt_sovits {

namespace {

// Full HuBERT base inference graph size, matching the parity test.
static constexpr size_t kMaxNodes = 8192;

// Feature encoder geometry (must match src/hubert/block.cpp):
//   strides      {5, 2, 2, 2, 2, 2, 2}
//   kernels      {10, 3, 3, 3, 3, 2, 2}
// The total stride is 5 * 2^6 = 320 and the receptive field is 400, so with no
// padding the output length is floor((n - 400) / 320) + 1.
static constexpr int64_t kHubertStrideProduct   = 320;
static constexpr int64_t kHubertReceptiveField  = 400;
static constexpr int64_t kHubertHiddenSize      = 768;

// RAII guards for the ephemeral ggml resources. They make sure a failure in a
// later step cannot leak the context / gallocr.
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

int64_t hubert_output_frames(int64_t n_samples) {
    if (n_samples < kHubertReceptiveField) {
        return 0;
    }
    return 1 + (n_samples - kHubertReceptiveField) / kHubertStrideProduct;
}

bool hubert_extract_features_from_pcm(const hubert_model & model,
                                      const float        * pcm,
                                      int64_t              n_samples,
                                      float              * out,
                                      int64_t              out_cap,
                                      int64_t            * out_frames)
{
    // --- Input validation: external errors return false, never assert. ---
    if (model.backend == nullptr) {
        fprintf(stderr, "%s: model has no backend\n", __func__);
        return false;
    }
    if (pcm == nullptr) {
        fprintf(stderr, "%s: pcm must be non-null\n", __func__);
        return false;
    }
    if (out == nullptr) {
        fprintf(stderr, "%s: out must be non-null\n", __func__);
        return false;
    }
    if (n_samples <= 0) {
        fprintf(stderr, "%s: n_samples=%lld must be > 0\n",
                __func__, (long long) n_samples);
        return false;
    }

    const int64_t T = hubert_output_frames(n_samples);

    // A waveform shorter than the receptive field has no output frames; the
    // call still succeeds and reports T' = 0.
    if (T == 0) {
        if (out_frames != nullptr) {
            *out_frames = 0;
        }
        return true;
    }

    // --- Overflow-checked size arithmetic (n_samples / T are caller values). ---
    int64_t in_bytes = 0;
    if (!mul_ok(n_samples, static_cast<int64_t>(sizeof(float)), in_bytes)) {
        fprintf(stderr, "%s: n_samples=%lld overflows byte count\n",
                __func__, (long long) n_samples);
        return false;
    }

    int64_t needed = 0;
    if (!mul_ok(kHubertHiddenSize, T, needed)) {
        fprintf(stderr, "%s: 768 * T'=%lld overflows\n",
                __func__, (long long) T);
        return false;
    }
    if (out_cap < needed) {
        fprintf(stderr, "%s: out_cap=%lld too small (need %lld floats for T'=%lld)\n",
                __func__, (long long) out_cap, (long long) needed, (long long) T);
        return false;
    }

    // Host/allocator OOM is caught below; ggml asserts / a corrupt backend are
    // not recovery points and are not caught.
    try {
        // --- Ephemeral no-alloc graph context, sized like the parity test. ---
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes +
                             ggml_graph_overhead_custom(kMaxNodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx_guard ctx(ggml_init(params));
        if (ctx.p == nullptr) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx.p, kMaxNodes, false);
        if (gf == nullptr) {
            fprintf(stderr, "%s: ggml_new_graph_custom() failed\n", __func__);
            return false;
        }

        // --- Input: raw waveform {n_samples}. ---
        struct ggml_tensor * inp =
            ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, n_samples);
        if (inp == nullptr) {
            fprintf(stderr, "%s: failed to create input tensor\n", __func__);
            return false;
        }
        ggml_set_name(inp, "input");
        ggml_set_input(inp);

        // --- Build the full model graph. ---
        struct ggml_tensor * out_t =
            hubert_model_block_forward(ctx.p, inp, model.weights);
        if (out_t == nullptr) {
            fprintf(stderr, "%s: hubert_model_block_forward() failed\n", __func__);
            return false;
        }
        ggml_set_name(out_t, "output");
        ggml_set_output(out_t);
        ggml_build_forward_expand(gf, out_t);

        // Internal invariant: the graph shape must agree with the host-side
        // frame formula.
        if (out_t->ne[0] != kHubertHiddenSize || out_t->ne[1] != T) {
            fprintf(stderr, "%s: graph output {%lld, %lld} disagrees with "
                    "expected {%lld, %lld}\n",
                    __func__,
                    (long long) out_t->ne[0], (long long) out_t->ne[1],
                    (long long) kHubertHiddenSize, (long long) T);
            return false;
        }

        // --- Allocate the graph. ---
        ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(model.backend);
        if (buft == nullptr) {
            fprintf(stderr, "%s: backend has no default buffer type\n", __func__);
            return false;
        }

        gallocr_guard alloc(ggml_gallocr_new(buft));
        if (alloc.p == nullptr) {
            fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(alloc.p, gf)) {
            fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
            return false;
        }

        // --- Upload input, compute, download output. ---
        ggml_backend_tensor_set(inp, pcm, 0, static_cast<size_t>(in_bytes));

        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
            return false;
        }

        const int64_t out_bytes = needed * static_cast<int64_t>(sizeof(float));
        ggml_backend_tensor_get(out_t, out, 0, static_cast<size_t>(out_bytes));
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        return false;
    }

    if (out_frames != nullptr) {
        *out_frames = T;
    }
    return true;
}

bool hubert_extract_features_from_normalized_pcm(const hubert_model & model,
                                                 const float        * pcm,
                                                 int64_t              n_samples,
                                                 float              * out,
                                                 int64_t              out_cap,
                                                 int64_t            * out_frames)
{
    // Compatibility proxy: the historical API only documents that normalization
    // is the caller's responsibility. The runner has no normalization stage, so
    // it simply forwards the samples unchanged.
    return hubert_extract_features_from_pcm(model, pcm, n_samples, out, out_cap,
                                            out_frames);
}

} // namespace gpt_sovits
