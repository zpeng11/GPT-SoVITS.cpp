// tests/sovits/test_sovits_helpers.cpp
//
// Tests for the two pipeline-stateless helpers added to block.cpp:
//   - sovits_quantized_double_25hz_forward: time-doubling via repeat_interleave
//   - sovits_sample_z_p_forward:            z_p = m + randn * exp(logs)
//
// Both helpers are pure (no weights, no GGUF I/O), so tests construct
// deterministic inputs and verify the math directly.

#include <gtest/gtest.h>

#include "gpt_sovits/sovits.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "test_backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

static constexpr int64_t kRVQDim        = 768;
static constexpr int64_t kFlowChannels  = 192;

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

    ~GraphContext() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;

    operator struct ggml_context *() { return ctx; } // NOLINT
};

} // namespace

// ---------------------------------------------------------------------------
// sovits_quantized_double_25hz_forward
// ---------------------------------------------------------------------------

TEST(SoVITSHelpers, QuantizedDouble25hzMatchesRepeatInterleave) {
    const int64_t T = 7;
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    GraphContext gctx(64);
    ASSERT_NE(gctx.ctx, nullptr);

    // Build host input: q[c, t] = t * 1000 + c (distinct per cell so any
    // indexing mistake surfaces immediately).
    std::vector<float> q_host(static_cast<size_t>(kRVQDim * T));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kRVQDim; ++c) {
            q_host[static_cast<size_t>(t * kRVQDim + c)] =
                static_cast<float>(t * 1000 + c);
        }
    }

    // Host-built index [0, 0, 1, 1, ..., T-1, T-1].
    std::vector<int32_t> idx_host(static_cast<size_t>(2 * T));
    for (int64_t t = 0; t < T; ++t) {
        idx_host[static_cast<size_t>(2 * t)]     = static_cast<int32_t>(t);
        idx_host[static_cast<size_t>(2 * t + 1)] = static_cast<int32_t>(t);
    }

    struct ggml_tensor * q = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kRVQDim, T);
    ggml_set_name(q, "q");
    ggml_set_input(q);

    struct ggml_tensor * idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 2 * T);
    ggml_set_name(idx, "idx");
    ggml_set_input(idx);

    struct ggml_tensor * out =
        gpt_sovits::sovits_quantized_double_25hz_forward(gctx, q, idx);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ne[0], kRVQDim);
    EXPECT_EQ(out->ne[1], 2 * T);

    ggml_set_output(out);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 64, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(q, q_host.data(),
                            0, q_host.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_host.data(),
                            0, idx_host.size() * sizeof(int32_t));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    std::vector<float> out_host(static_cast<size_t>(kRVQDim * 2 * T));
    ggml_backend_tensor_get(out, out_host.data(),
                            0, out_host.size() * sizeof(float));

    // Expected: out[:, 2t] == out[:, 2t+1] == q[:, t].
    for (int64_t t = 0; t < T; ++t) {
        const float expected_a = static_cast<float>(t * 1000);     // c == 0
        const float expected_b = static_cast<float>(t * 1000 + (kRVQDim - 1)); // last c
        const float got_a0 = out_host[static_cast<size_t>(2 * t * kRVQDim)];
        const float got_a1 = out_host[static_cast<size_t>((2 * t + 1) * kRVQDim)];
        const float got_b0 = out_host[static_cast<size_t>(2 * t * kRVQDim + (kRVQDim - 1))];
        const float got_b1 = out_host[static_cast<size_t>((2 * t + 1) * kRVQDim + (kRVQDim - 1))];
        EXPECT_FLOAT_EQ(got_a0, expected_a) << "t=" << t << " r=0 c=0";
        EXPECT_FLOAT_EQ(got_a1, expected_a) << "t=" << t << " r=1 c=0";
        EXPECT_FLOAT_EQ(got_b0, expected_b) << "t=" << t << " r=0 c=last";
        EXPECT_FLOAT_EQ(got_b1, expected_b) << "t=" << t << " r=1 c=last";
    }

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// sovits_sample_z_p_forward
// ---------------------------------------------------------------------------

TEST(SoVITSHelpers, SampleZPMatchesFormula) {
    const int64_t T = 5;
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    GraphContext gctx(64);
    ASSERT_NE(gctx.ctx, nullptr);

    // Deterministic inputs so we can check exact values.
    // m[c, t] = c * 0.01 + t * 0.1
    // logs[c, t] = -1.0 (constant, so exp(logs) = 1/e)
    // randn[c, t] = (c == t) ? noise_scale : 0   (single non-zero per time step)
    // -> z_p = m + randn * exp(logs)
    const float noise_scale = 0.5f;
    const float logs_val    = -1.0f;
    const float sigma       = std::exp(logs_val);

    std::vector<float> m_host(static_cast<size_t>(kFlowChannels * T));
    std::vector<float> logs_host(static_cast<size_t>(kFlowChannels * T), logs_val);
    std::vector<float> randn_host(static_cast<size_t>(kFlowChannels * T), 0.0f);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kFlowChannels; ++c) {
            m_host[static_cast<size_t>(t * kFlowChannels + c)] =
                static_cast<float>(c) * 0.01f + static_cast<float>(t) * 0.1f;
        }
        // Only one channel per time step receives noise_scale; the rest stay 0.
        if (t < kFlowChannels) {
            randn_host[static_cast<size_t>(t * kFlowChannels + t)] = noise_scale;
        }
    }

    struct ggml_tensor * m = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, T);
    ggml_set_name(m, "m");
    ggml_set_input(m);

    struct ggml_tensor * logs = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, T);
    ggml_set_name(logs, "logs");
    ggml_set_input(logs);

    struct ggml_tensor * randn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, kFlowChannels, T);
    ggml_set_name(randn, "randn");
    ggml_set_input(randn);

    struct ggml_tensor * z_p =
        gpt_sovits::sovits_sample_z_p_forward(gctx, m, logs, randn);
    ASSERT_NE(z_p, nullptr);
    EXPECT_EQ(z_p->ne[0], kFlowChannels);
    EXPECT_EQ(z_p->ne[1], T);

    ggml_set_output(z_p);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 64, false);
    ggml_build_forward_expand(gf, z_p);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(alloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    ggml_backend_tensor_set(m, m_host.data(),
                            0, m_host.size() * sizeof(float));
    ggml_backend_tensor_set(logs, logs_host.data(),
                            0, logs_host.size() * sizeof(float));
    ggml_backend_tensor_set(randn, randn_host.data(),
                            0, randn_host.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, gf), GGML_STATUS_SUCCESS);

    std::vector<float> z_p_host(static_cast<size_t>(kFlowChannels * T));
    ggml_backend_tensor_get(z_p, z_p_host.data(),
                            0, z_p_host.size() * sizeof(float));

    // Expected: z_p[c, t] = m[c, t] + randn[c, t] * sigma
    // All entries match m exactly except for the (t, t) diagonal, where they
    // differ by noise_scale * sigma.
    double max_abs_err = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kFlowChannels; ++c) {
            const float m_val     = static_cast<float>(c) * 0.01f + static_cast<float>(t) * 0.1f;
            const float randn_val = (c == t) ? noise_scale : 0.0f;
            const float expected  = m_val + randn_val * sigma;
            const float got       = z_p_host[static_cast<size_t>(t * kFlowChannels + c)];
            const double err      = std::fabs(static_cast<double>(got) - expected);
            max_abs_err = std::max(max_abs_err, err);
        }
    }
    EXPECT_LT(max_abs_err, 1e-5);

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
}
