#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <random>
#include <vector>

namespace gpt_sovits {

// ---------------------------------------------------------------------------
// Constants (must match block.cpp v2 hardcodes)
// ---------------------------------------------------------------------------

static constexpr int64_t kMelChannels        = 704;   // refer sliced to [:, :704]
static constexpr int64_t kRVQDim             = 768;
static constexpr int64_t kStyleOut           = 512;   // ge channels
static constexpr int64_t kInterChannels      = 192;   // m / logs / z channels
static constexpr int64_t kGeneratorFrameMul = 640;    // dec upsamples T -> T*640

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
    GGML_ASSERT(backend != nullptr);
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

void sovits_session_free(sovits_session & session) {
    if (session.buf_ge) {
        ggml_backend_buffer_free(session.buf_ge);
        session.buf_ge = nullptr;
    }
    if (session.ctx_ge) {
        ggml_free(session.ctx_ge);
        session.ctx_ge = nullptr;
    }
    session.ge = nullptr;

    session.backend = nullptr;
    session.scratch_double_idx.clear();
    session.scratch_randn.clear();
}

// Run MelStyleEncoder on a single refer slice and return the ge {512} vector
// in `out` (must have room for kStyleOut floats). Graph is built, allocated,
// executed, and torn down per call so different slices can have different T.
static bool compute_ge_single(
    ggml_backend_t                       backend,
    const sovits_mel_style_encoder_block_weights & weights,
    const float                        * refer_data,
    int64_t                              T_refer,
    std::vector<float>                 & out)
{
    // --- Temp graph context for the MelStyleEncoder forward ---
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

    // --- Input tensor: refer {704, T_refer} ---
    struct ggml_tensor * refer = ggml_new_tensor_2d(ctx_tmp, GGML_TYPE_F32, kMelChannels, T_refer);
    ggml_set_name(refer, "refer");
    ggml_set_input(refer);

    // --- Build graph: MelStyleEncoder(refer) -> ge {512, 1} ---
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_tmp, graph_size, false);

    struct ggml_tensor * result = sovits_mel_style_encoder_block_forward(
        ctx_tmp, refer, weights);

    ggml_set_name(result, "ge_out");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // --- Allocate and execute ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
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

    ggml_backend_tensor_set(refer, refer_data,
                            0, (size_t) kMelChannels * (size_t) T_refer * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx_tmp);
        return false;
    }

    // --- Read back ge {512} into `out` ---
    const size_t nbytes = (size_t) kStyleOut * sizeof(float);
    ggml_backend_tensor_get(result, out.data(), 0, nbytes);

    ggml_gallocr_free(alloc);
    ggml_free(ctx_tmp);
    return true;
}

bool sovits_session_compute_ge(
    sovits_session       & session,
    const sovits_model   & model,
    const std::vector<std::pair<const float *, int64_t>> & refers)
{
    GGML_ASSERT(session.backend != nullptr);
    GGML_ASSERT(!refers.empty());

    // Release any previously cached ge first.
    if (session.buf_ge) {
        ggml_backend_buffer_free(session.buf_ge);
        session.buf_ge = nullptr;
    }
    if (session.ctx_ge) {
        ggml_free(session.ctx_ge);
        session.ctx_ge = nullptr;
    }
    session.ge = nullptr;

    // --- Run MelStyleEncoder per refer slice, accumulate element-wise mean ---
    std::vector<double>  acc(kStyleOut, 0.0);
    std::vector<float>   tmp(kStyleOut);
    for (size_t i = 0; i < refers.size(); ++i) {
        const float  * refer_data = refers[i].first;
        const int64_t  T_refer    = refers[i].second;
        GGML_ASSERT(refer_data != nullptr);
        GGML_ASSERT(T_refer > 0);

        if (!compute_ge_single(session.backend, model.ref_enc,
                               refer_data, T_refer, tmp)) {
            fprintf(stderr, "%s: compute_ge_single failed on refer %zu/%zu\n",
                    __func__, i + 1, refers.size());
            return false;
        }
        for (size_t j = 0; j < kStyleOut; ++j) {
            acc[j] += static_cast<double>(tmp[j]);
        }
    }

    const double inv = 1.0 / static_cast<double>(refers.size());
    std::vector<float> mean(kStyleOut);
    for (size_t j = 0; j < kStyleOut; ++j) {
        mean[j] = static_cast<float>(acc[j] * inv);
    }

    // --- Allocate persistent ge {512, 1} tensor owned by the session ---
    struct ggml_init_params ge_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    session.ctx_ge = ggml_init(ge_params);
    if (!session.ctx_ge) {
        fprintf(stderr, "%s: ggml_init() for ge context failed\n", __func__);
        return false;
    }

    session.ge = ggml_new_tensor_2d(session.ctx_ge, GGML_TYPE_F32, kStyleOut, 1);
    ggml_set_name(session.ge, "ge");

    session.buf_ge = ggml_backend_alloc_ctx_tensors(session.ctx_ge, session.backend);
    if (!session.buf_ge) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() for ge failed\n", __func__);
        session.ctx_ge = nullptr;
        session.ge     = nullptr;
        return false;
    }

    // --- Copy averaged ge into session.ge ---
    {
        const size_t nbytes = (size_t) kStyleOut * sizeof(float);
        ggml_backend_tensor_set(session.ge, mean.data(), 0, nbytes);
    }

    fprintf(stderr, "%s: cached ge {%lld, 1} from %zu refer slice%s (mean)\n",
            __func__, (long long) kStyleOut, refers.size(),
            refers.size() == 1 ? "" : "s");
    return true;
}

struct ggml_tensor * sovits_session_get_ge(const sovits_session & session) {
    return session.ge;
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
    GGML_ASSERT(session.backend != nullptr);
    GGML_ASSERT(codes != nullptr);
    GGML_ASSERT(text != nullptr);
    GGML_ASSERT(wav_out != nullptr);
    GGML_ASSERT(T_codes > 0);
    GGML_ASSERT(T_text > 0);

    if (session.ge == nullptr) {
        fprintf(stderr, "%s: ge not cached; call sovits_session_compute_ge first\n", __func__);
        return false;
    }

    const bool is_25hz = model.hparams.semantic_frame_rate_25hz;
    const int64_t T_ssl = is_25hz ? 2 * T_codes : T_codes;
    const int64_t wav_len = T_ssl * kGeneratorFrameMul;
    if (wav_cap < wav_len) {
        fprintf(stderr, "%s: wav_cap=%lld too small (need %lld for T_ssl=%lld)\n",
                __func__, (long long) wav_cap, (long long) wav_len, (long long) T_ssl);
        return false;
    }

    // --- Build host-side scratch: 25hz index + randn ---
    struct ggml_tensor * idx_tensor = nullptr;
    if (is_25hz) {
        session.scratch_double_idx.resize(static_cast<size_t>(2 * T_codes));
        for (int64_t t = 0; t < T_codes; ++t) {
            session.scratch_double_idx[2 * t]     = static_cast<int32_t>(t);
            session.scratch_double_idx[2 * t + 1] = static_cast<int32_t>(t);
        }
    }

    session.scratch_randn.resize(static_cast<size_t>(kInterChannels * T_ssl));
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

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    // --- Input tensors ---
    struct ggml_tensor * codes_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_codes);
    ggml_set_name(codes_t, "codes");
    ggml_set_input(codes_t);

    struct ggml_tensor * text_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_t, "text");
    ggml_set_input(text_t);

    if (is_25hz) {
        idx_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * T_codes);
        ggml_set_name(idx_tensor, "double_idx");
        ggml_set_input(idx_tensor);
    }

    struct ggml_tensor * randn_t =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInterChannels, T_ssl);
    ggml_set_name(randn_t, "randn");
    ggml_set_input(randn_t);

    // --- Build the pipeline ---
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    struct ggml_tensor * quantized = sovits_rvq_decode_block_forward(
        ctx, codes_t, model.quantizer);   // {768, T_codes}

    if (is_25hz) {
        quantized = sovits_quantized_double_25hz_forward(
            ctx, quantized, idx_tensor);            // {768, 2*T_codes}
    }

    sovits_text_encoder_result enc = sovits_text_encoder_block_forward(
        ctx, quantized, text_t, session.ge, model.text_encoder);

    struct ggml_tensor * z_p = sovits_sample_z_p_forward(
        ctx, enc.m, enc.logs, randn_t);            // {192, T_ssl}

    struct ggml_tensor * z = sovits_flow_block_inverse_forward(
        ctx, z_p, session.ge, model.flow);   // {192, T_ssl}

    struct ggml_tensor * wav = sovits_generator_block_forward(
        ctx, z, session.ge, model.generator);   // {1, T_ssl*640}

    ggml_set_name(wav, "wav");
    ggml_set_output(wav);
    ggml_build_forward_expand(gf, wav);

    // --- Allocate ---
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(session.backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!alloc) {
        fprintf(stderr, "%s: ggml_gallocr_new() failed\n", __func__);
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
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
                            0, (size_t) kInterChannels * (size_t) T_ssl * sizeof(float));

    // --- Compute ---
    if (ggml_backend_graph_compute(session.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // --- Copy waveform out ---
    {
        const size_t nbytes = (size_t) wav_len * sizeof(float);
        ggml_backend_tensor_get(wav, wav_out, 0, nbytes);
    }

    // --- Free temporaries ---
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    return true;
}

} // namespace gpt_sovits
