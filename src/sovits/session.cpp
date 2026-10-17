#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <new>
#include <random>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Constants (must match block.cpp v2 hardcodes)
// ---------------------------------------------------------------------------

static constexpr int64_t kMelChannels        = 704;   // refer sliced to [:, :704]
static constexpr int64_t kRVQDim             = 768;
static constexpr int64_t kEncPGeDim          = 512;   // enc_p ge input dim (always 512)
static constexpr int64_t kInterChannels      = 192;   // m / logs / z channels
static constexpr int64_t kGeneratorFrameMul = 640;    // dec upsamples T -> T*640

// ---------------------------------------------------------------------------
// RAII guards for ggml resources gathered on the stack.
//
// The session allocates short-lived graph contexts, gallocr instances, and
// persistent ge / ge_enc contexts+buffers. Holding these as raw pointers leaks
// them if any later step (another ggml_init, a std allocation, the graph build)
// throws std::bad_alloc. Wrapping them guarantees unwinding frees everything;
// release() is used only once ownership is handed to the session.
// ---------------------------------------------------------------------------

namespace {

struct ctx_guard {
    ggml_context * p = nullptr;
    ctx_guard() = default;
    explicit ctx_guard(ggml_context * q) : p(q) {}
    ~ctx_guard() { if (p) ggml_free(p); }
    ctx_guard(const ctx_guard &) = delete;
    ctx_guard & operator=(const ctx_guard &) = delete;
    ggml_context * release() { ggml_context * r = p; p = nullptr; return r; }
};

struct gallocr_guard {
    ggml_gallocr_t p = nullptr;
    gallocr_guard() = default;
    explicit gallocr_guard(ggml_gallocr_t q) : p(q) {}
    ~gallocr_guard() { if (p) ggml_gallocr_free(p); }
    gallocr_guard(const gallocr_guard &) = delete;
    gallocr_guard & operator=(const gallocr_guard &) = delete;
};

struct buf_guard {
    ggml_backend_buffer_t p = nullptr;
    buf_guard() = default;
    explicit buf_guard(ggml_backend_buffer_t q) : p(q) {}
    ~buf_guard() { if (p) ggml_backend_buffer_free(p); }
    buf_guard(const buf_guard &) = delete;
    buf_guard & operator=(const buf_guard &) = delete;
    ggml_backend_buffer_t release() { ggml_backend_buffer_t r = p; p = nullptr; return r; }
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
// sovits_session_compute_ge mirrors t2s_session_compute_ref_emb:
// builds a temp graph, runs it, copies the result into a session-owned
// persistent tensor. Caller must slice refer to [:, :704] before passing in.
// ---------------------------------------------------------------------------

bool sovits_session_init(
    sovits_session & session,
    ggml_backend_t   backend,
    float            noise_scale,
    uint64_t         rng_seed)
{
    if (backend == nullptr) {
        fprintf(stderr, "%s: backend must be non-null\n", __func__);
        return false;
    }
    if (noise_scale <= 0.0f) {
        fprintf(stderr, "%s: noise_scale must be > 0 (got %f)\n", __func__, noise_scale);
        return false;
    }

    session.backend     = backend;
    session.noise_scale = noise_scale;
    session.rng_seed    = rng_seed;
    session.rng.seed(rng_seed);

    return true;
}

// Free the cached ge / ge_enc tensors and their owning context+buffer, leaving
// every member null. For v2, ge_enc aliases ge (ge is already 512-wide) and
// only the ge context/buffer are populated, so the alias never double-frees.
static void release_ge_cache(sovits_session & session) {
    if (session.buf_ge_enc) {
        ggml_backend_buffer_free(session.buf_ge_enc);
        session.buf_ge_enc = nullptr;
    }
    if (session.ctx_ge_enc) {
        ggml_free(session.ctx_ge_enc);
        session.ctx_ge_enc = nullptr;
    }
    session.ge_enc = nullptr;

    if (session.buf_ge) {
        ggml_backend_buffer_free(session.buf_ge);
        session.buf_ge = nullptr;
    }
    if (session.ctx_ge) {
        ggml_free(session.ctx_ge);
        session.ctx_ge = nullptr;
    }
    session.ge = nullptr;
}

void sovits_session_free(sovits_session & session) {
    release_ge_cache(session);

    session.backend = nullptr;
    session.scratch_double_idx.clear();
    session.scratch_randn.clear();
}

// Run MelStyleEncoder on a single refer slice and return the ge vector in
// `out` (must have room for `gin_channels` floats). For v2Pro/v2ProPlus the
// per-slice sv embedding is fused (sv_emb + add + PReLU) inside the same
// graph. Graph is built, allocated, executed, and torn down per call so
// different slices can have different T.
static bool compute_ge_single(
    ggml_backend_t                                 backend,
    const sovits_mel_style_encoder_block_weights & ref_enc,
    const sovits_v2pro_block_weights            * v2pro,        // nullptr for v2
    const float                                  * refer_data,
    int64_t                                        T_refer,
    const float                                  * sv_data,      // {20480} or nullptr
    int64_t                                        gin_channels,
    std::vector<float>                           & out)
{
    const bool is_v2pro = (v2pro != nullptr);
    if (is_v2pro && sv_data == nullptr) {
        fprintf(stderr, "%s: v2Pro slice requires a non-null sv embedding\n", __func__);
        return false;
    }

    // --- Temp graph context for the MelStyleEncoder (+ v2pro) forward ---
    const size_t n_intermediates = 80;
    const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx_guard ctx_tmp(ggml_init(params));
    if (!ctx_tmp.p) {
        fprintf(stderr, "%s: ggml_init() for temp context failed\n", __func__);
        return false;
    }

    // --- Input tensor: refer {704, T_refer} ---
    struct ggml_tensor * refer = ggml_new_tensor_2d(ctx_tmp.p, GGML_TYPE_F32, kMelChannels, T_refer);
    ggml_set_name(refer, "refer");
    ggml_set_input(refer);

    // --- Optional sv input {20480} ---
    struct ggml_tensor * sv = nullptr;
    if (is_v2pro) {
        sv = ggml_new_tensor_1d(ctx_tmp.p, GGML_TYPE_F32, v2pro->sv_emb_w->ne[0]);
        ggml_set_name(sv, "sv");
        ggml_set_input(sv);
    }

    // --- Build graph: MelStyleEncoder(refer) -> [v2pro: sv_emb+add+PReLU] -> ge ---
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_tmp.p, graph_size, false);

    struct ggml_tensor * result = sovits_mel_style_encoder_block_forward(
        ctx_tmp.p, refer, ref_enc);
    if (is_v2pro) {
        result = sovits_v2pro_ge_forward(ctx_tmp.p, result, sv, *v2pro);
    }

    ggml_set_name(result, "ge_out");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // --- Allocate and execute ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    gallocr_guard alloc(ggml_gallocr_new(buft));
    if (!alloc.p) {
        fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(alloc.p, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }

    ggml_backend_tensor_set(refer, refer_data,
                            0, (size_t) kMelChannels * (size_t) T_refer * sizeof(float));
    if (is_v2pro) {
        const size_t sv_nbytes = (size_t) v2pro->sv_emb_w->ne[0] * sizeof(float);
        ggml_backend_tensor_set(sv, sv_data, 0, sv_nbytes);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }

    // --- Read back ge {gin_channels} into `out` ---
    const size_t nbytes = (size_t) gin_channels * sizeof(float);
    ggml_backend_tensor_get(result, out.data(), 0, nbytes);

    return true;
}

// Project the averaged ge {gin, 1} through ge_to512 into ge_enc {512, 1}.
// Runs a tiny standalone graph; result is read back into `out` (512 floats).
static bool compute_ge_enc_512(
    ggml_backend_t                       backend,
    const sovits_v2pro_block_weights   & v2pro,
    const float                        * ge_mean,     // {gin_channels}
    int64_t                              gin_channels,
    std::vector<float>                 & out)         // 512 floats
{
    const size_t n_intermediates = 16;
    const size_t graph_size      = GGML_DEFAULT_GRAPH_SIZE;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 4) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_guard ctx(ggml_init(params));
    if (!ctx.p) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    struct ggml_tensor * ge_in =
        ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, gin_channels, 1);
    ggml_set_name(ge_in, "ge_mean");
    ggml_set_input(ge_in);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx.p, graph_size, false);
    struct ggml_tensor * result = sovits_ge_to512_forward(ctx.p, ge_in, v2pro);
    ggml_set_name(result, "ge_enc");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    gallocr_guard alloc(ggml_gallocr_new(buft));
    if (!alloc.p || !ggml_gallocr_alloc_graph(alloc.p, gf)) {
        fprintf(stderr, "%s: gallocr alloc failed\n", __func__);
        return false;
    }

    ggml_backend_tensor_set(ge_in, ge_mean, 0,
                            (size_t) gin_channels * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph_compute failed\n", __func__);
        return false;
    }

    ggml_backend_tensor_get(result, out.data(), 0, kEncPGeDim * sizeof(float));
    return true;
}

bool sovits_session_compute_ge(
    sovits_session       & session,
    const sovits_model   & model,
    const std::vector<sovits_refer_view> & refers)
{
    // All input validation returns false (never asserts): a caller error must
    // leave the previously cached ge / ge_enc untouched.
    if (session.backend == nullptr) {
        fprintf(stderr, "%s: session has no backend (call sovits_session_init)\n",
                __func__);
        return false;
    }
    if (refers.empty()) {
        fprintf(stderr, "%s: refer list is empty\n", __func__);
        return false;
    }

    const bool     is_v2pro     = model.hparams.is_v2pro;
    const int64_t  gin_channels = model.hparams.gin_channels;
    const sovits_v2pro_block_weights * v2pro =
        is_v2pro ? &model.v2pro : nullptr;

    if (gin_channels <= 0) {
        fprintf(stderr, "%s: invalid gin_channels=%lld\n",
                __func__, (long long) gin_channels);
        return false;
    }
    if (is_v2pro &&
        (v2pro->sv_emb_w == nullptr || v2pro->sv_emb_b == nullptr ||
         v2pro->ge_to512_w == nullptr || v2pro->ge_to512_b == nullptr ||
         v2pro->prelu_w == nullptr)) {
        fprintf(stderr, "%s: v2Pro model is missing sv_emb/ge_to512/prelu weights\n",
                __func__);
        return false;
    }

    for (size_t i = 0; i < refers.size(); ++i) {
        if (refers[i].spec == nullptr) {
            fprintf(stderr, "%s: refer[%zu].spec is null\n", __func__, i);
            return false;
        }
        if (refers[i].T <= 0) {
            fprintf(stderr, "%s: refer[%zu].T=%lld must be > 0\n",
                    __func__, i, (long long) refers[i].T);
            return false;
        }
        if (is_v2pro && refers[i].sv == nullptr) {
            fprintf(stderr, "%s: refer[%zu].sv is null (required for v2Pro)\n",
                    __func__, i);
            return false;
        }
    }

    // --- Run MelStyleEncoder (+ v2pro fuse) per refer slice, accumulate mean.
    //     Everything up to the cache swap is computed into fresh host buffers;
    //     the session cache is only replaced once every step has succeeded.
    //
    // Scope of the recovery guarantee: this catches std::bad_alloc only, so a
    // host/allocator OOM leaves the previous cache intact and returns false.
    // Failures inside ggml itself (asserts / abort) or a corrupt backend are
    // NOT recovery points and are not promised to be catchable.
    try {
        std::vector<double> acc(gin_channels, 0.0);
        std::vector<float>  tmp(gin_channels);
        for (size_t i = 0; i < refers.size(); ++i) {
            if (!compute_ge_single(session.backend, model.ref_enc, v2pro,
                                   refers[i].spec, refers[i].T, refers[i].sv,
                                   gin_channels, tmp)) {
                fprintf(stderr, "%s: compute_ge_single failed on refer %zu/%zu\n",
                        __func__, i + 1, refers.size());
                return false;
            }
            for (int64_t j = 0; j < gin_channels; ++j) {
                acc[j] += static_cast<double>(tmp[j]);
            }
        }

        const double inv = 1.0 / static_cast<double>(refers.size());
        std::vector<float> mean(gin_channels);
        for (int64_t j = 0; j < gin_channels; ++j) {
            mean[j] = static_cast<float>(acc[j] * inv);
        }

        // Project the averaged ge through ge_to512 before touching the cache,
        // so a failed projection cannot leave a half-updated cache.
        std::vector<float> ge_enc_host;
        if (is_v2pro) {
            ge_enc_host.resize(kEncPGeDim);
            if (!compute_ge_enc_512(session.backend, *v2pro, mean.data(),
                                    gin_channels, ge_enc_host)) {
                fprintf(stderr, "%s: compute_ge_enc_512 failed\n", __func__);
                return false;
            }
        }

        // Allocate the new persistent ge {gin_channels, 1} in guarded locals
        // (not yet in the session) so any failure frees only the new resources.
        struct ggml_init_params ge_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 2,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx_guard new_ge_ctx(ggml_init(ge_params));
        if (!new_ge_ctx.p) {
            fprintf(stderr, "%s: ggml_init() for ge context failed\n", __func__);
            return false;
        }
        struct ggml_tensor * new_ge =
            ggml_new_tensor_2d(new_ge_ctx.p, GGML_TYPE_F32, gin_channels, 1);
        ggml_set_name(new_ge, "ge");
        buf_guard new_ge_buf(
            ggml_backend_alloc_ctx_tensors(new_ge_ctx.p, session.backend));
        if (!new_ge_buf.p) {
            fprintf(stderr, "%s: alloc_ctx_tensors() for ge failed\n", __func__);
            return false;
        }
        ggml_backend_tensor_set(new_ge, mean.data(), 0,
                                (size_t) gin_channels * sizeof(float));

        // ge_enc: enc_p conditioning (512). v2Pro/Plus project via ge_to512; v2
        // aliases ge (already 512-wide) and allocates no separate buffer.
        ctx_guard new_enc_ctx;
        buf_guard new_enc_buf;
        struct ggml_tensor * new_ge_enc = nullptr;
        if (is_v2pro) {
            new_enc_ctx.p = ggml_init(ge_params);
            if (!new_enc_ctx.p) {
                fprintf(stderr, "%s: ggml_init() for ge_enc context failed\n", __func__);
                return false;
            }
            new_ge_enc = ggml_new_tensor_2d(new_enc_ctx.p, GGML_TYPE_F32, kEncPGeDim, 1);
            ggml_set_name(new_ge_enc, "ge_enc");
            new_enc_buf.p = ggml_backend_alloc_ctx_tensors(new_enc_ctx.p, session.backend);
            if (!new_enc_buf.p) {
                fprintf(stderr, "%s: alloc_ctx_tensors() for ge_enc failed\n", __func__);
                return false;
            }
            ggml_backend_tensor_set(new_ge_enc, ge_enc_host.data(), 0,
                                    kEncPGeDim * sizeof(float));
        }

        // --- All new resources are live and valid: atomically replace the
        //     cache. Everything past this point is non-throwing pointer moves. ---
        release_ge_cache(session);
        session.ctx_ge = new_ge_ctx.release();
        session.buf_ge = new_ge_buf.release();
        session.ge     = new_ge;
        if (is_v2pro) {
            session.ctx_ge_enc = new_enc_ctx.release();
            session.buf_ge_enc = new_enc_buf.release();
            session.ge_enc     = new_ge_enc;
        } else {
            session.ge_enc = session.ge;  // v2 alias: ge is already 512-wide
        }
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory (cache left unchanged)\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: cached ge {%lld, 1}%s from %zu refer slice%s (mean)\n",
            __func__, (long long) gin_channels,
            is_v2pro ? " + ge_enc {512, 1}" : "",
            refers.size(), refers.size() == 1 ? "" : "s");
    return true;
}

struct ggml_tensor * sovits_session_get_ge(const sovits_session & session) {
    return session.ge;
}

struct ggml_tensor * sovits_session_get_ge_enc(const sovits_session & session) {
    return session.ge_enc;
}

bool sovits_session_compute_ge_audio(
    sovits_session       & session,
    const sovits_model   & model,
    const sv_model       & sv_model,
    const std::vector<sovits_audio_refer_view> & refers)
{
    if (!model.hparams.is_v2pro) {
        fprintf(stderr, "%s: requires a v2Pro/v2ProPlus model\n", __func__);
        return false;
    }
    if (refers.empty()) {
        fprintf(stderr, "%s: empty refer list\n", __func__);
        return false;
    }
    if (sv_model.backend == nullptr) {
        fprintf(stderr, "%s: sv_model is not loaded\n", __func__);
        return false;
    }
    if (sv_model.hparams.out_dim <= 0) {
        fprintf(stderr, "%s: invalid sv out_dim=%d\n", __func__, sv_model.hparams.out_dim);
        return false;
    }
    // Validate every slice up front (return false, never assert) so a bad
    // refer cannot leave the cache half-updated.
    for (size_t i = 0; i < refers.size(); ++i) {
        if (refers[i].spec == nullptr || refers[i].T <= 0) {
            fprintf(stderr, "%s: refer[%zu] has null spec or T<=0\n", __func__, i);
            return false;
        }
        if (refers[i].audio16k == nullptr || refers[i].n_samples <= 0) {
            fprintf(stderr, "%s: refer[%zu] has null audio16k or n_samples<=0\n",
                    __func__, i);
            return false;
        }
    }

    // Compute the per-refer SV embedding in-process (audio -> fbank -> forward3),
    // then delegate to the parity-verified compute_ge with those sv vectors.
    // std::bad_alloc is caught here (host/allocator OOM) for the same
    // cache-preserving contract as compute_ge; ggml asserts / corrupt backend
    // are not recovery points.
    try {
        std::vector<std::vector<float>> feat_owned(refers.size());
        std::vector<std::vector<float>> sv_owned(refers.size(),
            std::vector<float>(static_cast<size_t>(sv_model.hparams.out_dim), 0.0f));
        std::vector<sovits_refer_view> views;
        views.reserve(refers.size());

        for (size_t i = 0; i < refers.size(); ++i) {
            std::vector<float> feats;
            const int m = sv_fbank(refers[i].audio16k, refers[i].n_samples, feats);
            if (m <= 0) {
                fprintf(stderr, "%s: sv_fbank produced 0 frames for refer %zu\n", __func__, i);
                return false;
            }
            if (!sv_forward3(sv_model, feats.data(), m, sv_owned[i].data())) {
                fprintf(stderr, "%s: sv_forward3 failed for refer %zu\n", __func__, i);
                return false;
            }
            views.push_back({refers[i].spec, refers[i].T, sv_owned[i].data()});
        }

        return sovits_session_compute_ge(session, model, views);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory (cache left unchanged)\n", __func__);
        return false;
    }
}

// v2 legacy compatibility entry: the pre-v2Pro API took a list of
// (spec, T) pairs. It is kept as a distinct, clearly-named function rather
// than an overload of sovits_session_compute_ge, because adding an overload
// taking std::vector<std::pair<const float *, int64_t>> would make the common
// braced call `compute_ge(session, model, {{spec, T}})` ambiguous against the
// sovits_refer_view overload (both are viable list-initializations). The main
// entry still accepts that braced form as an aggregate sovits_refer_view with
// a default (null) sv; this named entry is for callers that already hold a
// std::vector<std::pair<const float *, int64_t>>.
bool sovits_session_compute_ge_specs(
    sovits_session       & session,
    const sovits_model   & model,
    const std::vector<std::pair<const float *, int64_t>> & refers)
{
    try {
        std::vector<sovits_refer_view> views;
        views.reserve(refers.size());
        for (const auto & r : refers) {
            views.push_back({r.first, r.second, nullptr});
        }
        return sovits_session_compute_ge(session, model, views);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory (cache left unchanged)\n", __func__);
        return false;
    }
}

bool sovits_session_forward(
    sovits_session       & session,
    const sovits_model   & model,
    const int32_t        * codes,
    int64_t                T_codes,
    const int32_t        * text,
    int64_t                T_text,
    float                * wav_out,
    int64_t                wav_cap)
{
    if (session.backend == nullptr) {
        fprintf(stderr, "%s: session has no backend (call sovits_session_init)\n", __func__);
        return false;
    }
    if (codes == nullptr || text == nullptr || wav_out == nullptr) {
        fprintf(stderr, "%s: codes/text/wav_out must be non-null\n", __func__);
        return false;
    }
    if (T_codes <= 0 || T_text <= 0) {
        fprintf(stderr, "%s: T_codes=%lld and T_text=%lld must both be > 0\n",
                __func__, (long long) T_codes, (long long) T_text);
        return false;
    }

    if (session.ge == nullptr || session.ge_enc == nullptr) {
        fprintf(stderr, "%s: ge not cached; call sovits_session_compute_ge first\n", __func__);
        return false;
    }

    // --- Overflow-checked length math (T_codes / T_text come from the caller,
    //     so a hostile/buggy value must not wrap int64 past the buffer checks). ---
    const bool is_25hz = model.hparams.semantic_frame_rate_25hz;
    int64_t T_ssl = T_codes;
    if (is_25hz) {
        if (T_codes > INT64_MAX / 2) {
            fprintf(stderr, "%s: T_codes=%lld overflows 25hz doubling\n",
                    __func__, (long long) T_codes);
            return false;
        }
        T_ssl = 2 * T_codes;
    }
    // The 25hz time-doubling runs through an I32 index tensor holding values
    // 0..T_codes-1, so the largest index must fit int32.
    if (is_25hz && (T_codes - 1) > INT32_MAX) {
        fprintf(stderr, "%s: T_codes=%lld exceeds the int32 25hz index range\n",
                __func__, (long long) T_codes);
        return false;
    }
    int64_t wav_len = 0;
    if (!mul_ok(T_ssl, kGeneratorFrameMul, wav_len)) {
        fprintf(stderr, "%s: T_ssl=%lld * 640 overflows\n",
                __func__, (long long) T_ssl);
        return false;
    }
    int64_t randn_count = 0;
    if (!mul_ok(kInterChannels, T_ssl, randn_count)) {
        fprintf(stderr, "%s: 192 * T_ssl=%lld overflows\n",
                __func__, (long long) T_ssl);
        return false;
    }
    if (wav_cap < wav_len) {
        fprintf(stderr, "%s: wav_cap=%lld too small (need %lld for T_ssl=%lld)\n",
                __func__, (long long) wav_cap, (long long) wav_len, (long long) T_ssl);
        return false;
    }

    // --- Validate token ids: an out-of-range id would index outside the RVQ
    //     codebook / text embedding during the graph build. ---
    if (model.quantizer.codebook == nullptr) {
        fprintf(stderr, "%s: model has no quantizer codebook\n", __func__);
        return false;
    }
    const int64_t n_code_vocab = model.quantizer.codebook->ne[1];
    for (int64_t i = 0; i < T_codes; ++i) {
        const int32_t c = codes[i];
        if (c < 0 || static_cast<int64_t>(c) >= n_code_vocab) {
            fprintf(stderr, "%s: codes[%lld]=%d out of range [0,%lld)\n",
                    __func__, (long long) i, c, (long long) n_code_vocab);
            return false;
        }
    }
    if (model.text_encoder.text.text_embedding == nullptr) {
        fprintf(stderr, "%s: model has no text embedding\n", __func__);
        return false;
    }
    const int64_t n_text_vocab = model.text_encoder.text.text_embedding->ne[1];
    for (int64_t i = 0; i < T_text; ++i) {
        const int32_t id = text[i];
        if (id < 0 || static_cast<int64_t>(id) >= n_text_vocab) {
            fprintf(stderr, "%s: text[%lld]=%d out of range [0,%lld)\n",
                    __func__, (long long) i, id, (long long) n_text_vocab);
            return false;
        }
    }

    // Host/allocator OOM is caught below; ggml asserts / a corrupt backend are
    // not recovery points and are not caught.
    try {
        // --- Build host-side scratch: 25hz index + randn ---
        struct ggml_tensor * idx_tensor = nullptr;
        if (is_25hz) {
            session.scratch_double_idx.resize(static_cast<size_t>(2 * T_codes));
            for (int64_t t = 0; t < T_codes; ++t) {
                session.scratch_double_idx[2 * t]     = static_cast<int32_t>(t);
                session.scratch_double_idx[2 * t + 1] = static_cast<int32_t>(t);
            }
        }

        session.scratch_randn.resize(static_cast<size_t>(randn_count));
        {
            std::normal_distribution<float> dist(0.0f, session.noise_scale);
            for (float & v : session.scratch_randn) {
                v = dist(session.rng);
            }
        }

    // --- Temp graph context ---
    // The combined pipeline includes the text encoder (whose relative-position
    // attention alone needs ~65k nodes — see tests/sovits/test_sovits_text_encoder.cpp
    // kMaxNodes), plus the flow inverse and the generator. Size the metadata
    // buffer to fit the union: ~131k nodes / ~131k tensor overheads.
    const size_t graph_size      = 131072;
    const size_t n_intermediates = graph_size;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_intermediates + 8) +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx_guard ctx(ggml_init(params));
    if (!ctx.p) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    // --- Input tensors ---
    struct ggml_tensor * codes_t = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T_codes);
    ggml_set_name(codes_t, "codes");
    ggml_set_input(codes_t);

    struct ggml_tensor * text_t = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, T_text);
    ggml_set_name(text_t, "text");
    ggml_set_input(text_t);

    if (is_25hz) {
        idx_tensor = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, 2 * T_codes);
        ggml_set_name(idx_tensor, "double_idx");
        ggml_set_input(idx_tensor);
    }

    struct ggml_tensor * randn_t =
        ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, kInterChannels, T_ssl);
    ggml_set_name(randn_t, "randn");
    ggml_set_input(randn_t);

    // --- Build the pipeline ---
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx.p, graph_size, false);

    struct ggml_tensor * quantized = sovits_rvq_decode_block_forward(
        ctx.p, codes_t, model.quantizer);   // {768, T_codes}

    if (is_25hz) {
        quantized = sovits_quantized_double_25hz_forward(
            ctx.p, quantized, idx_tensor);            // {768, 2*T_codes}
    }

    sovits_text_encoder_result enc = sovits_text_encoder_block_forward(
        ctx.p, quantized, text_t, session.ge_enc, model.text_encoder);

    struct ggml_tensor * z_p = sovits_sample_z_p_forward(
        ctx.p, enc.m, enc.logs, randn_t);            // {192, T_ssl}

    struct ggml_tensor * z = sovits_flow_block_inverse_forward(
        ctx.p, z_p, session.ge, model.flow);   // {192, T_ssl}

    struct ggml_tensor * wav = sovits_generator_block_forward(
        ctx.p, z, session.ge, model.generator);   // {1, T_ssl*640}

    ggml_set_name(wav, "wav");
    ggml_set_output(wav);
    ggml_build_forward_expand(gf, wav);

    // --- Allocate ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
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
    ggml_backend_tensor_set(codes_t, codes,
                            0, (size_t) T_codes * sizeof(int32_t));
    ggml_backend_tensor_set(text_t, text,
                            0, (size_t) T_text * sizeof(int32_t));
    if (is_25hz) {
        ggml_backend_tensor_set(idx_tensor, session.scratch_double_idx.data(),
                                0, (size_t) (2 * T_codes) * sizeof(int32_t));
    }
    ggml_backend_tensor_set(randn_t, session.scratch_randn.data(),
                            0, (size_t) randn_count * sizeof(float));

    // --- Compute ---
    if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }

    // --- Copy waveform out ---
    {
        const size_t nbytes = (size_t) wav_len * sizeof(float);
        ggml_backend_tensor_get(wav, wav_out, 0, nbytes);
    }
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        return false;
    }

    return true;
}

} // namespace gpt_sovits
