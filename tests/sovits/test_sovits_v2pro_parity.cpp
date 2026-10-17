// tests/sovits/test_sovits_v2pro_parity.cpp
//
// Block-level parity for SynthesizerTrn.decode on the v2Pro / v2ProPlus path.
// Each test feeds a PyTorch-recorded input and compares the C++ output to the
// recorded tensor. Variants share the is_v2pro code path and gin=1024; Plus
// additionally exercises the weight-driven generator (initial=768, k=20).
//
//   GetGeFromRefer          refer[:704] + sv  -> ge {1024}
//                           (skip when <prefix>_refer.npy is absent)
//   QuantizerDecode         codes             -> quantized {768, T}
//   TextEncoderWithGeTo512  ssl + text + ge   -> x, m, logs  (ge via ge_to512)
//   FlowInverse             z_p + ge          -> z
//   Generator               z + ge            -> wav
//
// GetGeFromRefer is the only test that goes through sovits_session_compute_ge
// (the real get_ge glue). The rest call *_block_forward so a failure names
// the exact block. Downstream tests consume the dumped post-PReLU ge; they
// do not depend on GetGeFromRefer having run.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"
#include "gpt_sovits/sv.h"

#include "cnpy.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "npy_loader.h"
#include "test_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Test-only: the failing-backend wrapper needs the ggml_backend struct/iface
// layout, which lives in third_party/ggml/src (not on the tests' public include
// path, which only exposes third_party/ggml/include).
#include "../../../third_party/ggml/src/ggml-backend-impl.h"

namespace {

static const std::string kTestDir = SOVITS_TEST_DIR;
static const std::string kRefDir  = kTestDir + "ref/";

static constexpr int64_t kInterChannels = 192;   // m / logs / z channels
static constexpr int64_t kGin           = 1024;  // v2Pro/v2ProPlus ge channels
static constexpr int64_t kEncPGeDim     = 512;   // enc_p ge input (ge_to512 out)
static constexpr int64_t kRVQDim        = 768;
static constexpr int64_t kMelChannels   = 704;   // ref_enc input slice
static constexpr int64_t kSvDim         = 20480; // ERes2NetV2.forward3
static constexpr int64_t kFrameMul      = 640;   // dec: T -> T*640

struct GraphContext {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = nullptr;

    explicit GraphContext(size_t max_nodes) {
        const size_t sz = ggml_tensor_overhead() * max_nodes
                        + ggml_graph_overhead_custom(max_nodes, false);
        buf.resize(sz);
        struct ggml_init_params params = {
            /*.mem_size   =*/ sz,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
    }
    ~GraphContext() { if (ctx) ggml_free(ctx); }
    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;
    operator struct ggml_context *() { return ctx; } // NOLINT
};

struct ErrorStats {
    double max_abs  = 0.0;
    double rmse     = 0.0;
    double mean_abs = 0.0;
    // A silent zero return on a size mismatch would let EXPECT_LT(0, tol) pass,
    // and std::max(a, NaN) can drop a non-finite sample from max_abs entirely.
    // Both are flagged here (and poison every metric) so parity checks fail.
    bool   size_mismatch = false;
    bool   nonfinite     = false;
};

struct NpyShapeInfo { std::vector<float> data; std::vector<size_t> shape; };

static ErrorStats compute_errors(const std::vector<float> & actual,
                                 const std::vector<float> & expected) {
    const double kInf = std::numeric_limits<double>::infinity();
    ErrorStats s{};
    if (actual.size() != expected.size() || actual.empty()) {
        s.size_mismatch = true;
        s.max_abs = s.rmse = s.mean_abs = kInf;
        return s;
    }
    double sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double x = static_cast<double>(actual[i]);
        const double y = static_cast<double>(expected[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            s.nonfinite = true;
            continue;
        }
        const double e = std::abs(x - y);
        s.max_abs = std::max(s.max_abs, e);
        sum_sq  += e * e;
        sum_abs += e;
    }
    s.rmse     = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    s.mean_abs = sum_abs / static_cast<double>(actual.size());
    if (s.nonfinite) {   // force a loud failure on NaN/Inf samples.
        s.max_abs = s.rmse = s.mean_abs = kInf;
    }
    return s;
}

static NpyShapeInfo load_npy_with_shape(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    return {load_npy_as_f32(path), arr.shape};
}

static std::vector<int32_t> load_npy_as_i32(const std::string & path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size == sizeof(int32_t)) {
        const int32_t * src = arr.data<int32_t>();
        return std::vector<int32_t>(src, src + arr.num_vals);
    }
    if (arr.word_size == sizeof(int64_t)) {
        const int64_t * src = arr.data<int64_t>();
        std::vector<int32_t> out(arr.num_vals);
        for (size_t i = 0; i < arr.num_vals; ++i) out[i] = static_cast<int32_t>(src[i]);
        return out;
    }
    ADD_FAILURE() << "load_npy_as_i32: unsupported word_size=" << arr.word_size
                  << " in '" << path << "'";
    return {};
}

// PyTorch [B, C, T] -> ggml {C, T} row-major (index (c,t) -> dst[t*C + c]).
static std::vector<float> pack_bct_to_ggml(
    const std::vector<float> & tensor, const std::vector<size_t> & shape,
    int64_t channels)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_EQ(shape[1], static_cast<size_t>(channels));
    if (shape.size() != 3 || shape[0] != 1u ||
        shape[1] != static_cast<size_t>(channels)) {
        return {};
    }
    const size_t time = shape[2];
    std::vector<float> packed(static_cast<size_t>(channels) * time);
    for (size_t t = 0; t < time; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(channels); ++c) {
            packed[t * static_cast<size_t>(channels) + c] = tensor[c * time + t];
        }
    }
    return packed;
}

// Same layout, but keep only the first `take` channels (refer[:, :704]).
static std::vector<float> pack_bct_prefix_to_ggml(
    const std::vector<float> & tensor, const std::vector<size_t> & shape,
    int64_t take)
{
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_GE(shape[1], static_cast<size_t>(take));
    if (shape.size() != 3 || shape[0] != 1u ||
        shape[1] < static_cast<size_t>(take)) {
        return {};
    }
    const size_t time = shape[2];
    std::vector<float> packed(static_cast<size_t>(take) * time);
    for (size_t t = 0; t < time; ++t) {
        for (size_t c = 0; c < static_cast<size_t>(take); ++c) {
            packed[t * static_cast<size_t>(take) + c] = tensor[c * time + t];
        }
    }
    return packed;
}

static std::vector<float> read_tensor_f32(struct ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n, 0.0f);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    return out;
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// Read a possibly-f16 weight tensor into f32 (graph outputs are f32; the f16
// models store GGUF weights as f16).
static std::vector<float> read_tensor_any_f32(struct ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n, 0.0f);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    }
    return out;
}

static bool vec_all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Session-glue support: one shared CPU backend, one cached GGUF per variant,
// and one cached SV model, so the two heavy v2Pro GGUFs are loaded once for the
// whole suite instead of once per test.
//
// All of it lives in one process-wide registry whose destructor frees the
// models BEFORE the backend they borrow (models -> sv models -> backend), so
// teardown ordering is explicit rather than relying on the reverse-order
// destruction of scattered function-local statics. A failed load is neither
// inserted into the cache nor handed back to callers.
// ---------------------------------------------------------------------------
struct ModelRegistry {
    ggml_backend_t                                  backend = nullptr;
    std::map<std::string, gpt_sovits::sovits_model> sovits;
    std::map<std::string, gpt_sovits::sv_model>     sv;

    ~ModelRegistry() {
        for (auto & kv : sv)     gpt_sovits::sv_model_free(kv.second);
        for (auto & kv : sovits) gpt_sovits::sovits_model_free(kv.second);
        if (backend) ggml_backend_free(backend);
    }
};

static ModelRegistry & model_registry() {
    static ModelRegistry reg;
    if (!reg.backend) reg.backend = ggml_backend_cpu_init();
    return reg;
}

static ggml_backend_t shared_cpu_backend() {
    return model_registry().backend;
}

// Returns nullptr if the load failed (nothing cached, nothing to use).
// The slot is inserted empty first (nothing loaded yet, so a throwing emplace
// cannot leak), then loaded in place; a failed load erases the slot.
static gpt_sovits::sovits_model * cached_sovits_model(const std::string & path) {
    ModelRegistry & reg = model_registry();
    auto it = reg.sovits.find(path);
    if (it != reg.sovits.end()) return &it->second;
    try {
        auto res = reg.sovits.emplace(path, gpt_sovits::sovits_model{});
        gpt_sovits::sovits_model & slot = res.first->second;
        if (!gpt_sovits::sovits_model_load(path, slot, reg.backend)) {
            ADD_FAILURE() << "failed to load sovits model " << path;
            reg.sovits.erase(res.first);
            return nullptr;
        }
        return &slot;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

static gpt_sovits::sv_model * cached_sv_model(const std::string & path) {
    ModelRegistry & reg = model_registry();
    auto it = reg.sv.find(path);
    if (it != reg.sv.end()) return &it->second;
    try {
        auto res = reg.sv.emplace(path, gpt_sovits::sv_model{});
        gpt_sovits::sv_model & slot = res.first->second;
        if (!gpt_sovits::sv_model_load(path, slot, reg.backend)) {
            ADD_FAILURE() << "failed to load sv model " << path;
            reg.sv.erase(res.first);
            return nullptr;
        }
        return &slot;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

// One reference slice: packed spec {704, T} + sv {20480}.
struct ReferSlice {
    std::vector<float> spec;   // {704, T}, spec[t*704 + c]
    int64_t            T = 0;
    std::vector<float> sv;     // {20480}
};

static ReferSlice load_refer_slice(const std::string & refer_path,
                                   const std::string & sv_path) {
    ReferSlice out;
    const auto refer = load_npy_with_shape(refer_path);
    const auto sv    = load_npy_with_shape(sv_path);
    if (refer.shape.size() != 3u || refer.shape[0] != 1u ||
        refer.shape[1] < static_cast<size_t>(kMelChannels)) {
        return out;
    }
    out.T    = static_cast<int64_t>(refer.shape[2]);
    out.spec = pack_bct_prefix_to_ggml(refer.data, refer.shape, kMelChannels);
    out.sv   = sv.data;
    return out;
}

// ---------------------------------------------------------------------------
// Test-only failing-backend wrapper. Delegates every ggml_backend_i entry to a
// real backend except graph_compute, which can be armed to force
// GGML_STATUS_FAILED on a chosen call. This lets us exercise the "compute
// failed -> cache unchanged" paths with no production test-only hook. Buffer
// allocation / tensor IO / buffer-type lookup pass through to the wrapped
// backend because the wrapper copies its device and per-buffer callbacks are
// reached through the tensors, not the backend.
struct FailInjection {
    ggml_backend_t real         = nullptr;   // borrowed, never freed here
    int            fail_at_call = -1;        // 1-based call index to fail; -1 = never
    int            calls        = 0;
};

static enum ggml_status fail_injection_graph_compute(ggml_backend_t backend,
                                                     struct ggml_cgraph * cgraph) {
    FailInjection * inj = static_cast<FailInjection *>(backend->context);
    inj->calls += 1;
    if (inj->fail_at_call > 0 && inj->calls >= inj->fail_at_call) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(inj->real, cgraph);
}

static const char * fail_injection_get_name(ggml_backend_t) {
    return "test-fail-injection";
}

static void fail_injection_free(ggml_backend_t backend) {
    delete static_cast<FailInjection *>(backend->context);
    delete backend;
}

// Wrap `real` (not freed by the wrapper). The injection state is written to
// *out so the test can arm/disarm the failure.
static ggml_backend_t wrap_failing_backend(ggml_backend_t real, FailInjection ** out) {
    FailInjection * inj = new FailInjection{real, -1, 0};
    ggml_backend_t w = new ggml_backend{
        /* .guid    = */ real->guid,
        /* .iface   = */ real->iface,
        /* .device  = */ real->device,
        /* .context = */ inj,
    };
    w->iface.graph_compute = fail_injection_graph_compute;
    w->iface.get_name      = fail_injection_get_name;
    w->iface.free          = fail_injection_free;
    *out = inj;
    return w;
}

// ---------------------------------------------------------------------------
// Test-only allocation-injecting backend.
//
// Wraps a real backend so a test can (a) force ggml_backend_graph_compute to
// return GGML_STATUS_FAILED, (b) force a chosen buffer allocation to return
// null or to throw std::bad_alloc, and (c) count live backend buffers to prove
// no leak. ggml_backend_get_default_buffer_type() resolves through
// backend->device->get_buffer_type(), so the wrapper installs a device wrapper
// that returns a custom buffer type; its alloc_buffer delegates to the real one
// and wraps the result so frees are counted. Everything but the handful of
// overridden entry points forwards to the real backend/device via the copied
// struct fields. No production test hook.
// ---------------------------------------------------------------------------
struct AllocInjector;

struct InjectBufWrap {
    ggml_backend_buffer_t real = nullptr;
    AllocInjector       * inj  = nullptr;
};

struct AllocInjector {
    ggml_backend_t             real        = nullptr;
    ggml_backend_buffer_type_t real_buft   = nullptr;
    ggml_backend_buffer_type_t inject_buft = nullptr;   // owned
    ggml_backend_dev_t         inject_dev  = nullptr;   // owned

    int fail_compute_at_call  = -1;
    int throw_compute_at_call = -1;   // throw std::bad_alloc from graph_compute
    int compute_calls         = 0;

    int fail_alloc_at_call  = -1;   // fail free for alloc calls >= this index
    int alloc_calls   = 0;
    int alloc_success = 0;
    int live_buffers  = 0;
    std::vector<size_t> sizes;      // requested size of each alloc_buffer call
};

static InjectBufWrap * injw(ggml_backend_buffer_t b) {
    return static_cast<InjectBufWrap *>(b->context);
}

// NOTE: ggml_backend_buffer_free() calls this then `delete buffer`, so the
// wrapper must free only the underlying real buffer + its own context, never
// the ggml_backend_buffer struct (that would double free).
static void inj_buf_free(ggml_backend_buffer_t buffer) {
    InjectBufWrap * w = injw(buffer);
    if (w->inj) w->inj->live_buffers -= 1;
    ggml_backend_buffer_free(w->real);
    delete w;
}
static void * inj_buf_base(ggml_backend_buffer_t b) {
    InjectBufWrap * w = injw(b);
    return w->real->iface.get_base(w->real);
}
static enum ggml_status inj_buf_init_tensor(ggml_backend_buffer_t b, struct ggml_tensor * t) {
    InjectBufWrap * w = injw(b);
    return w->real->iface.init_tensor ? w->real->iface.init_tensor(w->real, t)
                                      : GGML_STATUS_SUCCESS;
}
static void inj_buf_memset(ggml_backend_buffer_t b, struct ggml_tensor * t,
                           uint8_t v, size_t off, size_t sz) {
    InjectBufWrap * w = injw(b);
    if (w->real->iface.memset_tensor) w->real->iface.memset_tensor(w->real, t, v, off, sz);
}
static void inj_buf_set(ggml_backend_buffer_t b, struct ggml_tensor * t,
                        const void * d, size_t off, size_t sz) {
    InjectBufWrap * w = injw(b);
    w->real->iface.set_tensor(w->real, t, d, off, sz);
}
static void inj_buf_get(ggml_backend_buffer_t b, const struct ggml_tensor * t,
                        void * d, size_t off, size_t sz) {
    InjectBufWrap * w = injw(b);
    w->real->iface.get_tensor(w->real, t, d, off, sz);
}
static bool inj_buf_cpy(ggml_backend_buffer_t b, const struct ggml_tensor * s,
                        struct ggml_tensor * d) {
    InjectBufWrap * w = injw(b);
    return w->real->iface.cpy_tensor ? w->real->iface.cpy_tensor(w->real, s, d) : false;
}
static void inj_buf_clear(ggml_backend_buffer_t b, uint8_t v) {
    InjectBufWrap * w = injw(b);
    if (w->real->iface.clear) w->real->iface.clear(w->real, v);
}
static void inj_buf_reset(ggml_backend_buffer_t b) {
    InjectBufWrap * w = injw(b);
    if (w->real->iface.reset) w->real->iface.reset(w->real);
}

static const char * inj_buft_name(ggml_backend_buffer_type_t) { return "test-inject-buft"; }
static size_t inj_buft_align(ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_type_t r = static_cast<AllocInjector *>(buft->context)->real_buft;
    return r->iface.get_alignment(r);
}
static size_t inj_buft_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_type_t r = static_cast<AllocInjector *>(buft->context)->real_buft;
    return r->iface.get_max_size ? r->iface.get_max_size(r) : (size_t) -1;
}
static size_t inj_buft_alloc_size(ggml_backend_buffer_type_t buft,
                                  const struct ggml_tensor * t) {
    ggml_backend_buffer_type_t r = static_cast<AllocInjector *>(buft->context)->real_buft;
    return r->iface.get_alloc_size ? r->iface.get_alloc_size(r, t) : ggml_nbytes(t);
}
static bool inj_buft_is_host(ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_type_t r = static_cast<AllocInjector *>(buft->context)->real_buft;
    return r->iface.is_host ? r->iface.is_host(r) : true;
}
static ggml_backend_buffer_t inj_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    AllocInjector * inj = static_cast<AllocInjector *>(buft->context);
    // This runs inside ggml's C allocation path, so it must obey the C ABI and
    // never let an exception escape: an std::bad_alloc from our own bookkeeping
    // is turned into a null (allocation-failure) return.
    try {
        inj->alloc_calls += 1;
        inj->sizes.push_back(size);
        if (inj->fail_alloc_at_call > 0 && inj->alloc_calls >= inj->fail_alloc_at_call) {
            return nullptr;
        }
        ggml_backend_buffer_t real = ggml_backend_buft_alloc_buffer(inj->real_buft, size);
        if (!real) return nullptr;

        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
            real_owner(real, ggml_backend_buffer_free);
        auto w_owner = std::make_unique<InjectBufWrap>(InjectBufWrap{real, inj});
        InjectBufWrap * w = w_owner.get();
        ggml_backend_buffer_t wb = new ggml_backend_buffer{
            /* .iface   = */ real->iface,
            /* .buft    = */ buft,
            /* .context = */ w,
            /* .size    = */ real->size,
            /* .usage   = */ real->usage,
        };
        wb->iface.free_buffer   = inj_buf_free;
        wb->iface.get_base      = inj_buf_base;
        wb->iface.init_tensor   = inj_buf_init_tensor;
        wb->iface.memset_tensor = inj_buf_memset;
        wb->iface.set_tensor    = inj_buf_set;
        wb->iface.get_tensor    = inj_buf_get;
        wb->iface.cpy_tensor    = inj_buf_cpy;
        wb->iface.clear         = inj_buf_clear;
        wb->iface.reset         = inj_buf_reset;
        inj->alloc_success += 1;
        inj->live_buffers  += 1;
        w_owner.release();
        real_owner.release();
        return wb;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

static ggml_backend_buffer_type_t inj_dev_buffer_type(ggml_backend_dev_t dev) {
    return static_cast<AllocInjector *>(dev->context)->inject_buft;
}

static enum ggml_status inj_backend_graph_compute(ggml_backend_t backend,
                                                  struct ggml_cgraph * cgraph) {
    AllocInjector * inj = static_cast<AllocInjector *>(backend->context);
    inj->compute_calls += 1;
    // Throwing here is safe: graph_compute is invoked from C++ ggml-backend
    // (not from an allocator callback), the gallocr already owns its buffers
    // and frees them on unwind, and session.cpp catches std::bad_alloc.
    if (inj->throw_compute_at_call > 0 && inj->compute_calls >= inj->throw_compute_at_call) {
        throw std::bad_alloc();
    }
    if (inj->fail_compute_at_call > 0 && inj->compute_calls >= inj->fail_compute_at_call) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(inj->real, cgraph);
}
static const char * inj_backend_get_name(ggml_backend_t) { return "test-alloc-inject"; }
static void inj_backend_free(ggml_backend_t backend) {
    AllocInjector * inj = static_cast<AllocInjector *>(backend->context);
    delete inj->inject_buft;
    delete inj->inject_dev;
    delete inj;
    delete backend;
}

// Wrap `real` (not freed by the wrapper). The injection state is returned via
// *out so the test can arm/disarm failures and read the counters.
static ggml_backend_t wrap_inject_backend(ggml_backend_t real, AllocInjector ** out) {
    AllocInjector * inj = new AllocInjector{};
    inj->real      = real;
    inj->real_buft = ggml_backend_get_default_buffer_type(real);

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type{};
    buft->iface.get_name       = inj_buft_name;
    buft->iface.alloc_buffer   = inj_buft_alloc;
    buft->iface.get_alignment  = inj_buft_align;
    buft->iface.get_max_size   = inj_buft_max_size;
    buft->iface.get_alloc_size = inj_buft_alloc_size;
    buft->iface.is_host        = inj_buft_is_host;
    buft->device  = nullptr;
    buft->context = inj;
    inj->inject_buft = buft;

    ggml_backend_dev_t rdev = real->device;
    ggml_backend_dev_t dev = new ggml_backend_device{ rdev->iface, rdev->reg, inj };
    dev->iface.get_buffer_type = inj_dev_buffer_type;
    inj->inject_dev = dev;
    buft->device = dev;

    ggml_backend_t w = new ggml_backend{ real->guid, real->iface, dev, inj };
    w->iface.graph_compute = inj_backend_graph_compute;
    w->iface.get_name      = inj_backend_get_name;
    w->iface.free          = inj_backend_free;
    *out = inj;
    return w;
}

// Run a single-output graph on the CPU backend: allocates, uploads `inputs`
// (tensor pointer -> host data/nbytes), computes, and reads `out` back.
static bool run_graph(ggml_backend_t backend, struct ggml_cgraph * gf,
                      struct ggml_tensor * out,
                      const std::vector<std::pair<struct ggml_tensor *,
                                  std::pair<const void *, size_t>>> & inputs,
                      std::vector<float> & out_vals)
{
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        return false;
    }
    for (const auto & in : inputs) {
        ggml_backend_tensor_set(in.first, in.second.first, 0, in.second.second);
    }
    const bool ok = (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    if (ok) {
        const size_t n = ggml_nelements(out);
        out_vals.assign(n, 0.0f);
        ggml_backend_tensor_get(out, out_vals.data(), 0, n * sizeof(float));
    }
    ggml_gallocr_free(alloc);
    return ok;
}

// ---------------------------------------------------------------------------
// Value-parameterized over both v2Pro and v2ProPlus: each variant picks its
// own GGUF + `v2pro`/`v2proplus` reference-fixture prefix. Both share the
// is_v2pro model code path and gin=1024; v2ProPlus additionally exercises the
// weight-driven generator channel plan (initial=768, first kernel=20).

struct VariantSpec {
    const char * label;         // "v2pro" / "v2proplus"
    const char * model_rel;     // gguf path relative to SOVITS_TEST_DIR/models/
    const char * ref_prefix;    // "v2pro" / "v2proplus"
};

class SoVITSV2ProParity : public ::testing::TestWithParam<VariantSpec> {
protected:
    std::string model_path() const {
        return kTestDir + "models/" + GetParam().model_rel;
    }
    std::string ref(const char * suffix) const {
        return kRefDir + std::string(GetParam().ref_prefix) + "_" + suffix + ".npy";
    }
    const char * label() const { return GetParam().label; }
};

INSTANTIATE_TEST_SUITE_P(Variants, SoVITSV2ProParity, ::testing::Values(
    VariantSpec{"v2pro",     "v2pro-sovits-f16.gguf",     "v2pro"},
    VariantSpec{"v2proplus", "v2proplus-sovits-f16.gguf", "v2proplus"}
));

// SynthesizerTrn.decode get_ge: MelStyleEncoder(refer[:, :704]) + sv_emb + PReLU.
// Goes through sovits_session_compute_ge so the session glue is in the same
// check. Skips when the variant has no refer dump (v2Pro today).
TEST_P(SoVITSV2ProParity, GetGeFromRefer) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) {
        GTEST_SKIP() << label() << ": no " << GetParam().ref_prefix
                     << "_refer.npy (get_ge dump not available)";
    }

    const auto refer = load_npy_with_shape(refer_path);
    const auto sv    = load_npy_with_shape(ref("sv_emb"));
    const auto ge    = load_npy_with_shape(ref("flow_input_ge"));
    ASSERT_EQ(refer.shape.size(), 3u);
    ASSERT_EQ(refer.shape[0], 1u);
    ASSERT_GE(refer.shape[1], static_cast<size_t>(kMelChannels));
    ASSERT_EQ(sv.data.size(), static_cast<size_t>(kSvDim));

    const int64_t T = static_cast<int64_t>(refer.shape[2]);
    ASSERT_GT(T, 0);
    const auto spec     = pack_bct_prefix_to_ggml(refer.data, refer.shape, kMelChannels);
    const auto expected = pack_bct_to_ggml(ge.data, ge.shape, kGin);
    ASSERT_EQ(spec.size(), static_cast<size_t>(kMelChannels) * static_cast<size_t>(T));
    ASSERT_EQ(expected.size(), static_cast<size_t>(kGin));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, m, backend));
    ASSERT_TRUE(m.hparams.is_v2pro);
    ASSERT_EQ(m.hparams.gin_channels, kGin);

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, backend, 0.5f, 0u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        session, m, {{spec.data(), T, sv.data.data()}}));

    struct ggml_tensor * ge_t     = gpt_sovits::sovits_session_get_ge(session);
    struct ggml_tensor * ge_enc_t = gpt_sovits::sovits_session_get_ge_enc(session);
    ASSERT_NE(ge_t, nullptr);
    ASSERT_NE(ge_enc_t, nullptr);
    EXPECT_EQ(ge_t->ne[0], kGin);
    EXPECT_EQ(ge_t->ne[1], 1);
    EXPECT_EQ(ge_enc_t->ne[0], kEncPGeDim);
    EXPECT_NE(ge_enc_t, ge_t) << "v2Pro ge_enc must be a ge_to512 projection, not an alias";

    const auto actual = read_tensor_f32(ge_t);
    const auto err    = compute_errors(actual, expected);
    size_t worst = 0;
    double worst_e = -1.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double e = std::abs(static_cast<double>(actual[i]) -
                                  static_cast<double>(expected[i]));
        if (e > worst_e) { worst_e = e; worst = i; }
    }
    printf("[%s get_ge] refer={%lld, %lld} -> ge(max_abs=%.6f rmse=%.6f mean_abs=%.6f) "
           "worst[%zu] act=%.5f exp=%.5f\n",
           label(), (long long) refer.shape[1], (long long) T,
           err.max_abs, err.rmse, err.mean_abs,
           worst, actual[worst], expected[worst]);
    // Dump ge is float16 and sits in ~[-40, 50], where fp16 ulp is 0.03–0.06.
    // Plus f16 weights, so this is a few ulp — not an f32 bit-match.
    EXPECT_FALSE(err.size_mismatch) << "ge size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "ge has non-finite values";
    EXPECT_LT(err.max_abs, 1.0e-1);
    EXPECT_LT(err.rmse,    2.0e-2);

    gpt_sovits::sovits_session_free(session);
    gpt_sovits::sovits_model_free(m);
    ggml_backend_free(backend);
}

// End-to-end wiring: audio -> fbank -> sv (in-process) -> ge{1024} + ge_enc{512}.
// Confirms sovits_session_compute_ge_audio composes the two parity-verified SV
// halves without integration regressions (shapes + finiteness). Moved here from
// test_sv_parity.cpp: it needs the SoVITS session, so it does not belong in the
// standalone (sv-only) parity file. Semantics are preserved; it now runs for
// each v2Pro variant (both share gin=1024).
TEST_P(SoVITSV2ProParity, ComputeGeAudioWiring) {
    const std::string model     = model_path();
    const std::string sv_model  = kTestDir + "models/sv-eres2netv2-f32.gguf";
    ASSERT_TRUE(file_exists(model));
    ASSERT_TRUE(file_exists(sv_model));

    const auto audio = load_npy_with_shape(kRefDir + "sv_audio16k.npy");
    ASSERT_FALSE(audio.data.empty());

    // Use a short slice (~1s) to keep the SV forward fast; the spec is a
    // synthetic {704, T} stand-in (wiring test, not a ge-value parity).
    const int64_t n_use = std::min<int64_t>(16000, static_cast<int64_t>(audio.data.size()));
    const int T_spec = 100;
    std::vector<float> spec(static_cast<size_t>(kMelChannels) * T_spec, 0.0f);
    for (size_t i = 0; i < spec.size(); ++i) {
        spec[i] = 0.01f * static_cast<float>(i % 17) - 0.08f;  // deterministic, finite
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::sovits_model smodel{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, smodel, backend));
    ASSERT_TRUE(smodel.hparams.is_v2pro);
    ASSERT_EQ(smodel.hparams.gin_channels, kGin);

    gpt_sovits::sv_model svmodel{};
    ASSERT_TRUE(gpt_sovits::sv_model_load(sv_model, svmodel, backend));

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, backend, 0.5f, 42u));

    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge_audio(
        session, smodel, svmodel,
        {{spec.data(), T_spec, audio.data.data(), n_use}}));

    struct ggml_tensor * ge     = gpt_sovits::sovits_session_get_ge(session);
    struct ggml_tensor * ge_enc = gpt_sovits::sovits_session_get_ge_enc(session);
    ASSERT_NE(ge, nullptr);
    ASSERT_NE(ge_enc, nullptr);
    EXPECT_EQ(ge->ne[0], kGin);
    EXPECT_EQ(ge_enc->ne[0], kEncPGeDim);

    auto all_finite = [](struct ggml_tensor * t) {
        const float * p = static_cast<const float *>(t->data);
        for (int i = 0; i < ggml_nelements(t); ++i) {
            if (!std::isfinite(p[i])) return false;
        }
        return true;
    };
    EXPECT_TRUE(all_finite(ge))     << "ge has non-finite values";
    EXPECT_TRUE(all_finite(ge_enc)) << "ge_enc has non-finite values";
    printf("[%s sv wiring] ge{%lld} ge_enc{%lld} both finite\n",
           label(), (long long) ge->ne[0], (long long) ge_enc->ne[0]);

    gpt_sovits::sovits_session_free(session);
    gpt_sovits::sv_model_free(svmodel);
    gpt_sovits::sovits_model_free(smodel);
    ggml_backend_free(backend);
}

// flow inverse with gin=1024 conditioning.
TEST_P(SoVITSV2ProParity, FlowInverse) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    const auto ref_zp = load_npy_with_shape(ref("flow_input_z_p"));
    const auto ref_ge = load_npy_with_shape(ref("flow_input_ge"));
    const auto ref_z  = load_npy_with_shape(ref("flow_output_z"));
    ASSERT_EQ(ref_zp.shape.size(), 3u);
    const int64_t time = static_cast<int64_t>(ref_zp.shape[2]);

    auto packed_zp = pack_bct_to_ggml(ref_zp.data, ref_zp.shape, kInterChannels);
    auto packed_ge = pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kGin);
    auto expected  = pack_bct_to_ggml(ref_z.data,  ref_z.shape,  kInterChannels);
    ASSERT_FALSE(packed_zp.empty());
    ASSERT_FALSE(packed_ge.empty());

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, m, backend));
    EXPECT_TRUE(m.hparams.is_v2pro);
    EXPECT_EQ(m.hparams.gin_channels, kGin);

    GraphContext gctx(8192);
    struct ggml_tensor * z_p = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInterChannels, time);
    ggml_set_input(z_p);
    struct ggml_tensor * ge  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGin, 1);
    ggml_set_input(ge);

    struct ggml_tensor * out = sovits_flow_block_inverse_forward(gctx, z_p, ge, m.flow);
    ASSERT_NE(out, nullptr);
    out = ggml_cont(gctx, out);
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
    ggml_build_forward_expand(gf, out);

    std::vector<float> actual;
    ASSERT_TRUE(run_graph(backend, gf, out, {
        {z_p, {packed_zp.data(), packed_zp.size() * sizeof(float)}},
        {ge,  {packed_ge.data(), packed_ge.size() * sizeof(float)}},
    }, actual));

    const auto err = compute_errors(actual, expected);
    printf("[%s flow] T=%lld z(max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           label(), (long long) time, err.max_abs, err.rmse, err.mean_abs);
    EXPECT_FALSE(err.size_mismatch) << "flow output size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "flow output has non-finite values";
    EXPECT_LT(err.max_abs, 2.0e-2);
    EXPECT_LT(err.rmse,    5.0e-3);

    gpt_sovits::sovits_model_free(m);
    ggml_backend_free(backend);
}

// generator (dec) with gin=1024 conditioning. For v2ProPlus this also covers
// the weight-driven 768-channel plan and first-kernel=20 upsample.
TEST_P(SoVITSV2ProParity, Generator) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    const auto ref_z  = load_npy_with_shape(ref("dec_input_z"));
    const auto ref_ge = load_npy_with_shape(ref("dec_input_ge"));
    const auto ref_o  = load_npy_with_shape(ref("dec_output_o"));
    ASSERT_EQ(ref_z.shape.size(), 3u);
    const int64_t time = static_cast<int64_t>(ref_z.shape[2]);
    ASSERT_EQ(ref_o.shape.size(), 3u);
    ASSERT_EQ(ref_o.shape[1], 1u);

    auto packed_z  = pack_bct_to_ggml(ref_z.data,  ref_z.shape,  kInterChannels);
    auto packed_ge = pack_bct_to_ggml(ref_ge.data, ref_ge.shape, kGin);
    // dec output is {1, 1, T*640}; single-channel BCT needs no repack.
    const std::vector<float> & expected = ref_o.data;
    ASSERT_FALSE(packed_z.empty());
    ASSERT_FALSE(packed_ge.empty());

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, m, backend));

    GraphContext gctx(32768);
    struct ggml_tensor * z  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kInterChannels, time);
    ggml_set_input(z);
    struct ggml_tensor * ge = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGin, 1);
    ggml_set_input(ge);

    struct ggml_tensor * out = sovits_generator_block_forward(gctx, z, ge, m.generator);
    ASSERT_NE(out, nullptr);
    out = ggml_cont(gctx, out);
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 32768, false);
    ggml_build_forward_expand(gf, out);

    std::vector<float> actual;
    ASSERT_TRUE(run_graph(backend, gf, out, {
        {z,  {packed_z.data(),  packed_z.size()  * sizeof(float)}},
        {ge, {packed_ge.data(), packed_ge.size() * sizeof(float)}},
    }, actual));

    const auto err = compute_errors(actual, expected);
    printf("[%s dec] T=%lld o(max_abs=%.6f rmse=%.6f mean_abs=%.6f)\n",
           label(), (long long) time, err.max_abs, err.rmse, err.mean_abs);
    EXPECT_FALSE(err.size_mismatch) << "generator output size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "generator output has non-finite values";
    EXPECT_LT(err.max_abs, 3.0e-2);
    EXPECT_LT(err.rmse,    8.0e-3);

    gpt_sovits::sovits_model_free(m);
    ggml_backend_free(backend);
}

// enc_p, fed ge_enc = ge_to512(ge). Validates the ge_to512 projection plus the
// (v2-identical) text encoder internals at the v2Pro ge path.
TEST_P(SoVITSV2ProParity, TextEncoderWithGeTo512) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    const auto ref_ssl  = load_npy_with_shape(ref("enc_p_input_quantized"));
    const auto ref_text = load_npy_as_i32    (ref("enc_p_input_text"));
    const auto ref_ge   = load_npy_with_shape(ref("flow_input_ge"));  // raw ge (1024)
    const auto ref_x    = load_npy_with_shape(ref("enc_p_output_x"));
    const auto ref_mp   = load_npy_with_shape(ref("enc_p_output_m_p"));
    const auto ref_lp   = load_npy_with_shape(ref("enc_p_output_logs_p"));
    ASSERT_EQ(ref_ssl.shape.size(), 3u);
    const int64_t T_ssl  = static_cast<int64_t>(ref_ssl.shape[2]);
    const int64_t T_text = static_cast<int64_t>(ref_text.size());
    ASSERT_GT(T_text, 0);

    auto packed_ssl = pack_bct_to_ggml(ref_ssl.data, ref_ssl.shape, kRVQDim);
    auto packed_ge  = pack_bct_to_ggml(ref_ge.data,  ref_ge.shape,  kGin);
    auto expected_x = pack_bct_to_ggml(ref_x.data,   ref_x.shape,   kInterChannels);
    auto expected_m = pack_bct_to_ggml(ref_mp.data,  ref_mp.shape,  kInterChannels);
    auto expected_l = pack_bct_to_ggml(ref_lp.data,  ref_lp.shape,  kInterChannels);
    ASSERT_FALSE(packed_ssl.empty());

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, m, backend));

    // enc_p needs the relative-position attention graph (~65k nodes).
    constexpr size_t kMaxNodes = 131072;
    GraphContext gctx(kMaxNodes);

    struct ggml_tensor * ssl  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kRVQDim, T_ssl);
    ggml_set_input(ssl);
    struct ggml_tensor * text = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_text);
    ggml_set_input(text);
    // raw ge {1024,1} as graph input, projected to 512 in-graph via ge_to512.
    struct ggml_tensor * ge_raw = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGin, 1);
    ggml_set_input(ge_raw);
    struct ggml_tensor * ge_enc = sovits_ge_to512_forward(gctx, ge_raw, m.v2pro);
    ASSERT_NE(ge_enc, nullptr);
    EXPECT_EQ(ge_enc->ne[0], kEncPGeDim);

    gpt_sovits::sovits_text_encoder_result enc =
        sovits_text_encoder_block_forward(gctx, ssl, text, ge_enc, m.text_encoder);

    struct ggml_tensor * out_x = ggml_cont(gctx, enc.x);    ggml_set_output(out_x);
    struct ggml_tensor * out_m = ggml_cont(gctx, enc.m);    ggml_set_output(out_m);
    struct ggml_tensor * out_l = ggml_cont(gctx, enc.logs); ggml_set_output(out_l);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, kMaxNodes, false);
    ggml_build_forward_expand(gf, out_x);
    ggml_build_forward_expand(gf, out_m);
    ggml_build_forward_expand(gf, out_l);

    // Allocate + upload + compute. All three outputs must be read BEFORE the
    // allocator is freed (it owns the backing buffer for every tensor).
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(ssl,    packed_ssl.data(), 0,
                            packed_ssl.size() * sizeof(float));
    ggml_backend_tensor_set(text,   ref_text.data(), 0,
                            ref_text.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ge_raw, packed_ge.data(), 0,
                            packed_ge.size() * sizeof(float));
    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    std::vector<float> act_x, act_m, act_l;
    auto read_out = [&](struct ggml_tensor * t, std::vector<float> & dst) {
        const size_t n = ggml_nelements(t);
        dst.assign(n, 0.0f);
        ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
    };
    read_out(out_x, act_x);
    read_out(out_m, act_m);
    read_out(out_l, act_l);
    ggml_gallocr_free(alloc);

    const auto ex = compute_errors(act_x, expected_x);
    const auto em = compute_errors(act_m, expected_m);
    const auto el = compute_errors(act_l, expected_l);
    printf("[%s enc_p] T_ssl=%lld x(max_abs=%.6f rmse=%.6f) "
           "m_p(max_abs=%.6f rmse=%.6f) logs_p(max_abs=%.6f rmse=%.6f)\n",
           label(), (long long) T_ssl, ex.max_abs, ex.rmse, em.max_abs, em.rmse,
           el.max_abs, el.rmse);
    EXPECT_FALSE(ex.size_mismatch) << "enc_p x size mismatch";
    EXPECT_FALSE(em.size_mismatch) << "enc_p m_p size mismatch";
    EXPECT_FALSE(el.size_mismatch) << "enc_p logs_p size mismatch";
    EXPECT_FALSE(ex.nonfinite || em.nonfinite || el.nonfinite)
        << "enc_p output has non-finite values";
    EXPECT_LT(ex.max_abs, 3.0e-2);
    EXPECT_LT(ex.rmse,    8.0e-3);
    EXPECT_LT(em.max_abs, 3.0e-2);
    EXPECT_LT(el.max_abs, 3.0e-2);

    gpt_sovits::sovits_model_free(m);
    ggml_backend_free(backend);
}

// RVQ decode (gin-independent sanity check).
TEST_P(SoVITSV2ProParity, QuantizerDecode) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    const std::vector<int32_t> codes = load_npy_as_i32(ref("quantizer_codes"));
    const auto ref_dec = load_npy_with_shape(ref("quantizer_decoded"));
    ASSERT_FALSE(codes.empty());
    ASSERT_EQ(ref_dec.shape.size(), 3u);
    ASSERT_EQ(ref_dec.shape[1], static_cast<size_t>(kRVQDim));
    const int64_t T_codes = static_cast<int64_t>(codes.size());

    auto expected = pack_bct_to_ggml(ref_dec.data, ref_dec.shape, kRVQDim);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::sovits_model m{};
    ASSERT_TRUE(gpt_sovits::sovits_model_load(model, m, backend));

    GraphContext gctx(4096);
    struct ggml_tensor * codes_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_codes);
    ggml_set_input(codes_t);
    struct ggml_tensor * out = sovits_rvq_decode_block_forward(gctx, codes_t, m.quantizer);
    ASSERT_NE(out, nullptr);
    out = ggml_cont(gctx, out);
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 4096, false);
    ggml_build_forward_expand(gf, out);

    std::vector<float> actual;
    ASSERT_TRUE(run_graph(backend, gf, out, {
        {codes_t, {codes.data(), codes.size() * sizeof(int32_t)}},
    }, actual));

    const auto err = compute_errors(actual, expected);
    printf("[%s quant] T_codes=%lld decoded(max_abs=%.6f rmse=%.6f)\n",
           label(), (long long) T_codes, err.max_abs, err.rmse);
    EXPECT_FALSE(err.size_mismatch) << "quantizer output size mismatch";
    EXPECT_FALSE(err.nonfinite)     << "quantizer output has non-finite values";
    EXPECT_LT(err.max_abs, 1.0e-2);
    EXPECT_LT(err.rmse,    2.0e-3);

    gpt_sovits::sovits_model_free(m);
    ggml_backend_free(backend);
}

// ===========================================================================
// Session-glue tests (both variants, CPU, model cached across the suite)
// ===========================================================================

static constexpr int64_t kShortCodes = 4;

// Snapshot of the session ge cache identity + values, used to prove a failed
// compute_ge leaves the previous cache untouched (same pointers, same data).
// Both conditionings are recorded: ge and ge_enc (the latter is separate for
// v2Pro; for the aliased v2 case it is the same tensor).
struct GeCacheSnapshot {
    struct ggml_context  * ctx_ge  = nullptr;
    ggml_backend_buffer_t  buf_ge  = nullptr;
    struct ggml_tensor   * ge      = nullptr;
    struct ggml_context  * ctx_enc = nullptr;
    ggml_backend_buffer_t  buf_enc = nullptr;
    struct ggml_tensor   * enc     = nullptr;
    std::vector<float>     ge_values;
    std::vector<float>     enc_values;
};

static GeCacheSnapshot snapshot_ge_cache(const gpt_sovits::sovits_session & s) {
    GeCacheSnapshot snap;
    snap.ctx_ge  = s.ctx_ge;
    snap.buf_ge  = s.buf_ge;
    snap.ge      = s.ge;
    snap.ctx_enc = s.ctx_ge_enc;
    snap.buf_enc = s.buf_ge_enc;
    snap.enc     = s.ge_enc;
    if (s.ge)     snap.ge_values  = read_tensor_f32(s.ge);
    if (s.ge_enc) snap.enc_values = read_tensor_f32(s.ge_enc);
    return snap;
}

static void expect_ge_cache_unchanged(const gpt_sovits::sovits_session & s,
                                      const GeCacheSnapshot & snap,
                                      const char * what) {
    EXPECT_EQ(s.ctx_ge,     snap.ctx_ge)  << what;
    EXPECT_EQ(s.buf_ge,     snap.buf_ge)  << what;
    EXPECT_EQ(s.ge,         snap.ge)      << what;
    EXPECT_EQ(s.ctx_ge_enc, snap.ctx_enc) << what;
    EXPECT_EQ(s.buf_ge_enc, snap.buf_enc) << what;
    EXPECT_EQ(s.ge_enc,     snap.enc)     << what;
    const auto ge_now = read_tensor_f32(s.ge);
    EXPECT_EQ(compute_errors(ge_now, snap.ge_values).max_abs, 0.0)
        << what << ": cached ge values changed";
    const auto enc_now = read_tensor_f32(s.ge_enc);
    EXPECT_EQ(compute_errors(enc_now, snap.enc_values).max_abs, 0.0)
        << what << ": cached ge_enc values changed";
}

// Real end-to-end session forward for both v2Pro and v2ProPlus with short
// codes: compute ge from refer+sv, then run the full pipeline. Checks shape
// (T_ssl*640), finiteness and non-degeneracy.
TEST_P(SoVITSV2ProParity, SessionForwardShortCodes) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) {
        GTEST_SKIP() << label() << ": no refer dump";
    }
    ASSERT_TRUE(file_exists(ref("sv_emb")));
    ASSERT_TRUE(file_exists(ref("quantizer_codes")));
    ASSERT_TRUE(file_exists(ref("enc_p_input_text")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    ASSERT_EQ(rd.sv.size(), static_cast<size_t>(kSvDim));

    const auto codes_all = load_npy_as_i32(ref("quantizer_codes"));
    const auto text      = load_npy_as_i32(ref("enc_p_input_text"));
    ASSERT_GT(codes_all.size(), 0u);
    ASSERT_GT(text.size(), 0u);
    const int64_t T_codes = std::min<int64_t>(kShortCodes, (int64_t) codes_all.size());
    const std::vector<int32_t> codes(codes_all.begin(), codes_all.begin() + T_codes);

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    ASSERT_TRUE(m.hparams.is_v2pro);
    ASSERT_EQ(m.hparams.gin_channels, kGin);

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 7u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        session, m, {{rd.spec.data(), rd.T, rd.sv.data()}}));

    const int64_t T_ssl   = 2 * T_codes;   // 25hz doubling
    const int64_t wav_len = T_ssl * kFrameMul;
    std::vector<float> wav(static_cast<size_t>(wav_len), 0.0f);
    ASSERT_TRUE(gpt_sovits::sovits_session_forward(
        session, m, codes.data(), T_codes, text.data(), (int64_t) text.size(),
        wav.data(), wav_len));

    size_t n_finite = 0;
    float vmin = +INFINITY, vmax = -INFINITY;
    for (float v : wav) {
        if (std::isfinite(v)) {
            ++n_finite;
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
    EXPECT_EQ(n_finite, wav.size()) << "session forward produced non-finite samples";
    EXPECT_LT(vmin, 0.0f);
    EXPECT_GT(vmax, 0.0f);
    EXPECT_GT(vmax - vmin, 0.1f) << "output range suspiciously small";
    printf("[%s session fwd] T_codes=%lld wav=%lld range=[%g, %g]\n",
           label(), (long long) T_codes, (long long) wav_len,
           (double) vmin, (double) vmax);

    // --- Negative: token-id range and length-overflow validation must return
    //     false without touching the output buffer. ---
    const int64_t n_code_vocab = m.quantizer.codebook->ne[1];
    const int64_t n_text_vocab = m.text_encoder.text.text_embedding->ne[1];
    std::vector<float> guard(static_cast<size_t>(wav_len), 7.0f);
    {
        std::vector<int32_t> bad = codes;
        bad[0] = static_cast<int32_t>(n_code_vocab);   // one past the codebook
        EXPECT_FALSE(gpt_sovits::sovits_session_forward(
            session, m, bad.data(), T_codes, text.data(), (int64_t) text.size(),
            guard.data(), wav_len)) << "code id >= codebook size must be rejected";
    }
    {
        std::vector<int32_t> bad = codes;
        bad[0] = -1;                                   // negative id
        EXPECT_FALSE(gpt_sovits::sovits_session_forward(
            session, m, bad.data(), T_codes, text.data(), (int64_t) text.size(),
            guard.data(), wav_len)) << "negative code id must be rejected";
    }
    {
        std::vector<int32_t> bad_text = text;
        bad_text[0] = static_cast<int32_t>(n_text_vocab);
        EXPECT_FALSE(gpt_sovits::sovits_session_forward(
            session, m, codes.data(), T_codes, bad_text.data(),
            (int64_t) bad_text.size(), guard.data(), wav_len))
            << "text id >= vocab must be rejected";
    }
    {
        // 25hz doubling / *640 overflow: rejected before any allocation.
        EXPECT_FALSE(gpt_sovits::sovits_session_forward(
            session, m, codes.data(), INT64_MAX / 2, text.data(),
            (int64_t) text.size(), guard.data(), wav_len))
            << "T_codes causing length overflow must be rejected";
    }
    for (float g : guard) {
        ASSERT_EQ(g, 7.0f) << "rejected forward wrote to the output buffer";
    }

    gpt_sovits::sovits_session_free(session);
}

// wav_cap exactly equals the required length (succeeds, no guard overwrite) and
// wav_cap one short (returns false, writes nothing).
TEST_P(SoVITSV2ProParity, ForwardCapacityBoundarySentinel) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));
    if (!file_exists(ref("quantizer_codes")) || !file_exists(ref("enc_p_input_text"))) {
        GTEST_SKIP() << label() << ": missing codes/text dump";
    }

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const auto codes_all = load_npy_as_i32(ref("quantizer_codes"));
    const auto text      = load_npy_as_i32(ref("enc_p_input_text"));
    ASSERT_GT(codes_all.size(), 0u);

    const int64_t T_codes = std::min<int64_t>(4, (int64_t) codes_all.size());
    const std::vector<int32_t> codes(codes_all.begin(), codes_all.begin() + T_codes);

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 3u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        session, m, {{rd.spec.data(), rd.T, rd.sv.data()}}));

    const int64_t T_ssl   = 2 * T_codes;
    const int64_t wav_len = T_ssl * kFrameMul;
    const float   sentinel = 12345.0f;

    // Exact capacity: succeeds and must not write past wav_len.
    {
        std::vector<float> buf(static_cast<size_t>(wav_len) + 8, sentinel);
        ASSERT_TRUE(gpt_sovits::sovits_session_forward(
            session, m, codes.data(), T_codes, text.data(), (int64_t) text.size(),
            buf.data(), wav_len));
        for (int64_t i = wav_len; i < wav_len + 8; ++i) {
            EXPECT_EQ(buf[static_cast<size_t>(i)], sentinel)
                << "wrote past wav_cap at index " << i;
        }
        std::vector<float> out(buf.begin(), buf.begin() + wav_len);
        EXPECT_TRUE(vec_all_finite(out)) << "forward output not finite";
    }

    // One short: fails and writes nothing.
    {
        std::vector<float> buf(static_cast<size_t>(wav_len) + 8, sentinel);
        EXPECT_FALSE(gpt_sovits::sovits_session_forward(
            session, m, codes.data(), T_codes, text.data(), (int64_t) text.size(),
            buf.data(), wav_len - 1));
        for (size_t i = 0; i < buf.size(); ++i) {
            ASSERT_EQ(buf[i], sentinel) << "buffer modified at index " << i;
        }
    }

    gpt_sovits::sovits_session_free(session);
}

// ge_enc must equal an independent host matmul of the session's ge through the
// model's ge_to512 weights (catches a wrong projection / wrong weight layout).
TEST_P(SoVITSV2ProParity, GeEncMatchesIndependentMatrixReference) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 5u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        session, m, {{rd.spec.data(), rd.T, rd.sv.data()}}));

    const auto ge     = read_tensor_f32(session.ge);
    const auto ge_enc = read_tensor_f32(session.ge_enc);
    ASSERT_EQ(ge.size(), static_cast<size_t>(kGin));
    ASSERT_EQ(ge_enc.size(), static_cast<size_t>(kEncPGeDim));

    // linear_2d: y = mul_mat(weight, x); weight is {in=gin, out=512}, stored
    // ne0-fastest, so W(i,o) = w[i + o*gin]. ggml's F16-weight matmul converts
    // the F32 activation to F16 first (vec_dot_type = F16); depending on arch
    // the products may then be accumulated in F32 (x86) or F16 (some ARM NEON
    // paths), and the reduction order differs from a scalar loop. The reference
    // therefore rounds ge to f16 and accumulates in f64, and only checks
    // structural agreement at a loose tolerance: a wrong weight or a transposed
    // layout would be O(1)-O(100), not the ~1e-3 relative seen here. A strict,
    // accumulation-independent projection check is GeTo512StrictSmallProjection.
    const auto w = read_tensor_any_f32(m.v2pro.ge_to512_w);
    const auto b = read_tensor_any_f32(m.v2pro.ge_to512_b);
    ASSERT_EQ(w.size(), static_cast<size_t>(kGin * kEncPGeDim));
    ASSERT_EQ(b.size(), static_cast<size_t>(kEncPGeDim));

    std::vector<float> ge_f16(ge.size());
    for (size_t i = 0; i < ge.size(); ++i) {
        ge_f16[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(ge[i]));
    }

    double max_abs = 0.0, sum_sq = 0.0;
    for (int64_t o = 0; o < kEncPGeDim; ++o) {
        double acc = static_cast<double>(b[o]);
        for (int64_t i = 0; i < kGin; ++i) {
            acc += static_cast<double>(w[i + o * kGin]) * static_cast<double>(ge_f16[i]);
        }
        const double e = std::abs(acc - static_cast<double>(ge_enc[o]));
        max_abs = std::max(max_abs, e);
        sum_sq += e * e;
    }
    const double rmse = std::sqrt(sum_sq / static_cast<double>(kEncPGeDim));
    printf("[%s ge_enc] vs independent matmul max_abs=%.3e rmse=%.3e\n",
           label(), max_abs, rmse);
    EXPECT_LT(max_abs, 8.0e-2);
    EXPECT_LT(rmse,    3.0e-2);

    gpt_sovits::sovits_session_free(session);
}

// Strict projection check that does not depend on the accumulation mode: feed
// ge_to512 a sparse input whose nonzero entries are exactly representable in
// f16 (no activation rounding) and few enough that even an F16 accumulator is
// exact to within a few ulp. The host reference is then held to a tight
// tolerance, so a transposed / wrong-weight projection fails loudly.
TEST_P(SoVITSV2ProParity, GeTo512StrictSmallProjection) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    ASSERT_TRUE(m.v2pro.ge_to512_w != nullptr);

    // Sparse, exactly-f16-representable input.
    std::vector<float> ge_in(static_cast<size_t>(kGin), 0.0f);
    ge_in[0]    = 1.0f;
    ge_in[5]    = -2.0f;
    ge_in[100]  = 4.0f;
    ge_in[1023] = 0.5f;

    const size_t max_nodes = 4096;
    GraphContext gctx(max_nodes);
    struct ggml_tensor * ge_t =
        ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kGin, 1);
    ggml_set_input(ge_t);
    struct ggml_tensor * out = sovits_ge_to512_forward(gctx, ge_t, m.v2pro);
    ASSERT_NE(out, nullptr);
    out = ggml_cont(gctx, out);
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);

    std::vector<float> actual;
    ASSERT_TRUE(run_graph(shared_cpu_backend(), gf, out, {
        {ge_t, {ge_in.data(), ge_in.size() * sizeof(float)}},
    }, actual));
    ASSERT_EQ(actual.size(), static_cast<size_t>(kEncPGeDim));
    // Guard against NaN silently poisoning max_abs (std::max drops NaN).
    ASSERT_TRUE(vec_all_finite(actual)) << "ge_to512 output has non-finite values";

    const auto w = read_tensor_any_f32(m.v2pro.ge_to512_w);
    const auto b = read_tensor_any_f32(m.v2pro.ge_to512_b);
    ASSERT_EQ(w.size(), static_cast<size_t>(kGin * kEncPGeDim));

    double max_abs = 0.0;
    for (int64_t o = 0; o < kEncPGeDim; ++o) {
        double acc = static_cast<double>(b[o]);
        for (int64_t i = 0; i < kGin; ++i) {
            acc += static_cast<double>(w[i + o * kGin]) * static_cast<double>(ge_in[i]);
        }
        max_abs = std::max(max_abs, std::abs(acc - static_cast<double>(actual[o])));
    }
    printf("[%s ge_to512 strict] sparse f32 max_abs=%.3e\n", label(), max_abs);
    EXPECT_LT(max_abs, 1.0e-3);
}

// Multi-refer semantics: element-wise mean of the per-slice ge, independent of
// order, and a duplicated slice reproduces the single-slice ge.
TEST_P(SoVITSV2ProParity, MultiReferOrderMeanDuplicate) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const int64_t half = rd.T / 2;
    ASSERT_GE(half, 4);
    const float * specA = rd.spec.data();
    const int64_t TA    = half;
    const float * specB = rd.spec.data() + half * kMelChannels;
    const int64_t TB    = rd.T - half;
    const float * sv    = rd.sv.data();

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 9u));

    auto ge_of = [&](const std::vector<gpt_sovits::sovits_refer_view> & views) {
        EXPECT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, views));
        return read_tensor_f32(session.ge);
    };

    const auto geA  = ge_of({{specA, TA, sv}});
    const auto geB  = ge_of({{specB, TB, sv}});
    const auto geAB = ge_of({{specA, TA, sv}, {specB, TB, sv}});
    const auto geBA = ge_of({{specB, TB, sv}, {specA, TA, sv}});
    const auto geAA = ge_of({{specA, TA, sv}, {specA, TA, sv}});

    std::vector<float> meanAB(geA.size());
    for (size_t i = 0; i < geA.size(); ++i) {
        meanAB[i] = 0.5f * (geA[i] + geB[i]);
    }

    const auto e_mean = compute_errors(geAB, meanAB);
    const auto e_ord  = compute_errors(geAB, geBA);
    const auto e_dup  = compute_errors(geAA, geA);
    printf("[%s multi-refer] mean max_abs=%.3e order max_abs=%.3e dup max_abs=%.3e\n",
           label(), e_mean.max_abs, e_ord.max_abs, e_dup.max_abs);
    EXPECT_LT(e_mean.max_abs, 1.0e-2) << "multi-refer mean mismatch";
    EXPECT_LT(e_ord.max_abs,  1.0e-6) << "refer order changed the result";
    EXPECT_LT(e_dup.max_abs,  1.0e-6) << "duplicate slice changed the result";

    gpt_sovits::sovits_session_free(session);
}

// Switching references (A -> B -> A) replaces the cache and reproduces the
// earlier result deterministically.
TEST_P(SoVITSV2ProParity, GeCacheUpdateAToB) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const int64_t half = rd.T / 2;
    ASSERT_GE(half, 4);
    const float * specA = rd.spec.data();
    const int64_t TA    = half;
    const float * specB = rd.spec.data() + half * kMelChannels;
    const int64_t TB    = rd.T - half;
    const float * sv    = rd.sv.data();

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 11u));

    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA = read_tensor_f32(session.ge);
    ASSERT_NE(session.ge, session.ge_enc);

    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}));
    const auto geB = read_tensor_f32(session.ge);
    EXPECT_GT(compute_errors(geB, geA).max_abs, 1.0)
        << "A and B produced the same ge; update did not happen";

    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA2 = read_tensor_f32(session.ge);
    EXPECT_EQ(compute_errors(geA2, geA).max_abs, 0.0) << "A -> B -> A not deterministic";

    gpt_sovits::sovits_session_free(session);
}

// Input errors return false and leave the previous cache byte-for-byte intact;
// a subsequent valid call still works (retry).
TEST_P(SoVITSV2ProParity, GeFailureKeepsCacheAndRetry) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const int64_t half = rd.T / 2;
    ASSERT_GE(half, 4);
    const float * specA = rd.spec.data();
    const int64_t TA    = half;
    const float * specB = rd.spec.data() + half * kMelChannels;
    const int64_t TB    = rd.T - half;
    const float * sv    = rd.sv.data();

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, shared_cpu_backend(), 0.5f, 13u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA = read_tensor_f32(session.ge);
    const GeCacheSnapshot snap = snapshot_ge_cache(session);

    // Empty list.
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {}));
    expect_ge_cache_unchanged(session, snap, "empty refer list");
    // Null spec.
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{nullptr, 4, sv}}));
    expect_ge_cache_unchanged(session, snap, "null spec");
    // Non-positive T.
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, 0, sv}}));
    expect_ge_cache_unchanged(session, snap, "T<=0");
    // v2Pro requires a per-slice sv.
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, nullptr}}));
    expect_ge_cache_unchanged(session, snap, "null sv");

    // Retry with a valid, different reference.
    EXPECT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}));
    const auto geB = read_tensor_f32(session.ge);
    EXPECT_TRUE(vec_all_finite(geB));
    EXPECT_GT(compute_errors(geB, geA).max_abs, 1.0) << "retry did not update the cache";

    gpt_sovits::sovits_session_free(session);
}

// Audio-driven ge (fbank + sv_forward3 in-process) must match the explicit
// path where the same SV embedding is passed in directly. Reuses the cached
// model + a cached SV model.
TEST_P(SoVITSV2ProParity, AudioVsExplicitSVMatch) {
    const std::string model    = model_path();
    const std::string sv_model = kTestDir + "models/sv-eres2netv2-f32.gguf";
    ASSERT_TRUE(file_exists(model));
    ASSERT_TRUE(file_exists(sv_model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);

    const auto audio = load_npy_with_shape(kRefDir + "sv_audio16k.npy");
    ASSERT_FALSE(audio.data.empty());
    // Short slice (~0.5s) to keep the in-process SV forward cheap; both paths
    // use the same cached model + cached SV model.
    const int64_t n_use = std::min<int64_t>(8000, (int64_t) audio.data.size());
    ASSERT_GT(n_use, 0);

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;
    gpt_sovits::sv_model * sm_ptr = cached_sv_model(sv_model);
    ASSERT_NE(sm_ptr, nullptr);
    gpt_sovits::sv_model & sm = *sm_ptr;
    ASSERT_TRUE(m.hparams.is_v2pro);

    // Explicit SV path.
    std::vector<float> feats;
    const int m_frames = gpt_sovits::sv_fbank(audio.data.data(), n_use, feats);
    ASSERT_GT(m_frames, 0);
    std::vector<float> sv_vec(static_cast<size_t>(sm.hparams.out_dim), 0.0f);
    ASSERT_TRUE(gpt_sovits::sv_forward3(sm, feats.data(), m_frames, sv_vec.data()));

    gpt_sovits::sovits_session s_explicit{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(s_explicit, shared_cpu_backend(), 0.5f, 21u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(
        s_explicit, m, {{rd.spec.data(), rd.T, sv_vec.data()}}));

    // Audio path (computes the same sv in-process).
    gpt_sovits::sovits_session s_audio{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(s_audio, shared_cpu_backend(), 0.5f, 21u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge_audio(
        s_audio, m, sm, {{rd.spec.data(), rd.T, audio.data.data(), n_use}}));

    const auto ge_e  = read_tensor_f32(s_explicit.ge);
    const auto ge_a  = read_tensor_f32(s_audio.ge);
    const auto enc_e = read_tensor_f32(s_explicit.ge_enc);
    const auto enc_a = read_tensor_f32(s_audio.ge_enc);
    const auto e_ge  = compute_errors(ge_a, ge_e);
    const auto e_enc = compute_errors(enc_a, enc_e);
    printf("[%s audio-vs-sv] ge max_abs=%.3e ge_enc max_abs=%.3e\n",
           label(), e_ge.max_abs, e_enc.max_abs);
    EXPECT_EQ(e_ge.max_abs,  0.0) << "audio ge differs from explicit-sv ge";
    EXPECT_EQ(e_enc.max_abs, 0.0) << "audio ge_enc differs from explicit-sv ge_enc";

    gpt_sovits::sovits_session_free(s_audio);
    gpt_sovits::sovits_session_free(s_explicit);
}

// Negative failure injection through a test backend wrapper: a graph_compute
// failure in compute_ge must return false and leave the old cache untouched,
// for both the MelStyleEncoder stage and the ge_to512 stage; a retry succeeds
// once the backend recovers.
TEST_P(SoVITSV2ProParity, FailureInjectionKeepsCache) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const int64_t half = rd.T / 2;
    ASSERT_GE(half, 4);
    const float * specA = rd.spec.data();
    const int64_t TA    = half;
    const float * specB = rd.spec.data() + half * kMelChannels;
    const int64_t TB    = rd.T - half;
    const float * sv    = rd.sv.data();

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;

    FailInjection * inj = nullptr;
    ggml_backend_t wb = wrap_failing_backend(shared_cpu_backend(), &inj);
    ASSERT_NE(wb, nullptr);
    ASSERT_NE(inj, nullptr);

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, wb, 0.5f, 17u));
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA = read_tensor_f32(session.ge);
    const GeCacheSnapshot snapA = snapshot_ge_cache(session);

    // Fail the next graph_compute (the MelStyleEncoder forward of the next call).
    inj->fail_at_call = inj->calls + 1;
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}));
    expect_ge_cache_unchanged(session, snapA, "mel-style stage failure");

    // Backend recovers: the retry must succeed and update the cache.
    inj->fail_at_call = -1;
    EXPECT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}));
    const auto geB = read_tensor_f32(session.ge);
    EXPECT_GT(compute_errors(geB, geA).max_abs, 1.0);
    const GeCacheSnapshot snapB = snapshot_ge_cache(session);

    // Now fail the 2nd graph_compute of the call (the ge_to512 projection),
    // leaving the MelStyleEncoder stage to succeed.
    inj->fail_at_call = inj->calls + 2;
    EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    expect_ge_cache_unchanged(session, snapB, "ge_to512 stage failure");

    // Recover and confirm the cache updates again.
    inj->fail_at_call = -1;
    EXPECT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA2 = read_tensor_f32(session.ge);
    EXPECT_EQ(compute_errors(geA2, geA).max_abs, 0.0);

    gpt_sovits::sovits_session_free(session);
    ggml_backend_free(wb);   // frees the wrapper only; the CPU backend is shared
}

// Negative allocation injection through a test backend/buft wrapper: an
// allocation failure (null) or a std::bad_alloc thrown at a chosen stage of
// compute_ge must return false, leave ge AND ge_enc untouched, and leak no
// backend buffer (counted). Covers the graph buffer, the persistent ge buffer,
// and the ge_enc buffer allocated after ge has already succeeded. No production
// test hook.
TEST_P(SoVITSV2ProParity, AllocFailureInjectionKeepsCache) {
    const std::string model = model_path();
    ASSERT_TRUE(file_exists(model));
    const std::string refer_path = ref("refer");
    if (!file_exists(refer_path)) GTEST_SKIP() << label() << ": no refer dump";
    ASSERT_TRUE(file_exists(ref("sv_emb")));

    ReferSlice rd = load_refer_slice(refer_path, ref("sv_emb"));
    ASSERT_GT(rd.T, 0);
    const int64_t half = rd.T / 2;
    ASSERT_GE(half, 4);
    const float * specA = rd.spec.data();
    const int64_t TA    = half;
    const float * specB = rd.spec.data() + half * kMelChannels;
    const int64_t TB    = rd.T - half;
    const float * sv    = rd.sv.data();

    gpt_sovits::sovits_model * m_ptr = cached_sovits_model(model);
    ASSERT_NE(m_ptr, nullptr);
    gpt_sovits::sovits_model & m = *m_ptr;

    AllocInjector * inj = nullptr;
    ggml_backend_t wb = wrap_inject_backend(shared_cpu_backend(), &inj);
    ASSERT_NE(wb, nullptr);
    ASSERT_NE(inj, nullptr);

    gpt_sovits::sovits_session session{};
    ASSERT_TRUE(gpt_sovits::sovits_session_init(session, wb, 0.5f, 23u));

    // Probe: one successful compute records the size of every alloc_buffer call
    // so each stage can be targeted regardless of how many buffers a graph
    // allocates.
    inj->sizes.clear();
    ASSERT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specA, TA, sv}}));
    const auto geA = read_tensor_f32(session.ge);
    const GeCacheSnapshot snapA = snapshot_ge_cache(session);
    EXPECT_EQ(inj->live_buffers, 2) << "expected ge + ge_enc persistent buffers";

    const size_t align    = ggml_backend_buft_get_alignment(inj->real_buft);
    const size_t ge_size  = GGML_PAD(static_cast<size_t>(kGin) * sizeof(float), align);
    const size_t enc_size = GGML_PAD(static_cast<size_t>(kEncPGeDim) * sizeof(float), align);
    // Persistent ge / ge_enc are allocated after the graph buffers within a
    // compute, so take the last matching call index for each.
    int ge_idx = -1, enc_idx = -1;
    for (size_t i = 0; i < inj->sizes.size(); ++i) {
        if (inj->sizes[i] == ge_size)  ge_idx  = static_cast<int>(i) + 1;
        if (inj->sizes[i] == enc_size) enc_idx = static_cast<int>(i) + 1;
    }
    ASSERT_GE(ge_idx, 1)  << "could not locate the persistent ge allocation";
    ASSERT_GE(enc_idx, 1) << "could not locate the persistent ge_enc allocation";

    auto run_alloc_fail = [&](int rel_alloc_idx, const char * what) {
        inj->fail_alloc_at_call = inj->alloc_calls + rel_alloc_idx;
        EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}))
            << what;
        expect_ge_cache_unchanged(session, snapA, what);
        EXPECT_EQ(inj->live_buffers, 2) << what << ": buffer leak";
        inj->fail_alloc_at_call = -1;
    };
    // A compute_ge of one v2Pro slice runs two graph_computes: the
    // MelStyleEncoder forward (1) then the ge_to512 projection (2). Throwing
    // from the compute trampoline (a C++ boundary) exercises the session's
    // std::bad_alloc catch at both progress points.
    auto run_throw = [&](int rel_compute_idx, const char * what) {
        inj->throw_compute_at_call = inj->compute_calls + rel_compute_idx;
        EXPECT_FALSE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}))
            << what;
        expect_ge_cache_unchanged(session, snapA, what);
        EXPECT_EQ(inj->live_buffers, 2) << what << ": buffer leak";
        inj->throw_compute_at_call = -1;
    };

    run_alloc_fail(1,       "graph buffer alloc failure");
    run_alloc_fail(ge_idx,  "persistent ge alloc failure");
    run_alloc_fail(enc_idx, "persistent ge_enc alloc failure (ge already succeeded)");
    run_throw(1, "bad_alloc during the MelStyleEncoder graph_compute");
    run_throw(2, "bad_alloc during the ge_to512 graph_compute (ge computed)");

    // Retry after injection is disarmed: cache updates, still no leak.
    EXPECT_TRUE(gpt_sovits::sovits_session_compute_ge(session, m, {{specB, TB, sv}}));
    const auto geB = read_tensor_f32(session.ge);
    EXPECT_GT(compute_errors(geB, geA).max_abs, 1.0);
    EXPECT_EQ(inj->live_buffers, 2);

    gpt_sovits::sovits_session_free(session);
    EXPECT_EQ(inj->live_buffers, 0) << "session_free leaked backend buffers";
    ggml_backend_free(wb);   // frees the wrapper/buft/device; CPU backend is shared
}

} // namespace
