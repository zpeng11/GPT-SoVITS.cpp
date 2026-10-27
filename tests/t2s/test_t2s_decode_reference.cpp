// Teacher-forced CPU/F32 comparison; raw logits are projected from hidden
// states independently of the sampler. Paths and scope are in README.md.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "decode_reference.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifndef T2S_DECODE_REFERENCE_DIR
#error "T2S_DECODE_REFERENCE_DIR must be defined by the test target"
#endif
#ifndef T2S_DECODE_MODEL_PATH
#error "T2S_DECODE_MODEL_PATH must be defined by the test target"
#endif

namespace {

// CPU/F32 bounds are component-specific; quantized weights/KV are not covered.
constexpr double kHiddenMaxAbs = 2e-4;
constexpr double kHiddenRmse   = 1e-5;
constexpr double kLogitsMaxAbs = 1e-4;
constexpr double kLogitsRmse   = 2e-5;
constexpr double kKvMaxAbs     = 1e-4;
constexpr double kKvRmse       = 1e-5;

struct BackendGuard {
    ggml_backend_t ptr = nullptr;
    ~BackendGuard() { reset(); }
    void reset(ggml_backend_t next = nullptr) {
        if (ptr) ggml_backend_free(ptr);
        ptr = next;
    }
};
struct ModelGuard {
    gpt_sovits::t2s_model model;
    bool loaded = false;
    ~ModelGuard() { reset(); }
    bool load(const std::string & path, ggml_backend_t backend) {
        reset();
        return loaded = gpt_sovits::t2s_model_load(path, model, backend);
    }
    void reset() {
        if (loaded) gpt_sovits::t2s_model_free(model);
        loaded = false;
    }
};
struct SessionGuard {
    gpt_sovits::t2s_session session;
    ~SessionGuard() { reset(); }
    bool init(const gpt_sovits::t2s_hparams & hp, ggml_backend_t backend,
              uint32_t n_batch, uint32_t slot_size,
              const gpt_sovits::t2s_sampler_config & cfg, enum ggml_type kv_type) {
        reset();
        return gpt_sovits::t2s_session_init(
            session, hp, backend, n_batch, slot_size, cfg, kv_type);
    }
    void reset() { gpt_sovits::t2s_session_free(session); }
};
struct FlexGuard {
    gpt_sovits::t2s_flex_graph graph;
    ~FlexGuard() { reset(); }
    bool build(gpt_sovits::t2s_session & session,
               const gpt_sovits::t2s_model & model,
               const gpt_sovits::t2s_batch_plan & plan) {
        reset();
        graph = gpt_sovits::t2s_session_build_flex_graph(session, model, plan);
        return graph.ctx != nullptr;
    }
    void reset() { gpt_sovits::t2s_flex_graph_free(graph); }
};

struct ErrorStats {
    double max_abs = 0.0;
    double rmse    = 0.0;
    size_t max_at  = 0;
};

static ErrorStats compute_errors(const float * actual, const float * expected, size_t n) {
    ErrorStats stats;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::abs(static_cast<double>(actual[i]) -
                                  static_cast<double>(expected[i]));
        if (e > stats.max_abs) {
            stats.max_abs = e;
            stats.max_at  = i;
        }
        sum_sq += e * e;
    }
    stats.rmse = n > 0 ? std::sqrt(sum_sq / static_cast<double>(n)) : 0.0;
    return stats;
}

static bool all_finite(const std::vector<float> & values) {
    for (float v : values) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

static void check_floats(const std::string & label,
                         const std::vector<float> & actual,
                         const float * expected, size_t n,
                         double max_abs_tol, double rmse_tol) {
    if (actual.size() != n) {
        ADD_FAILURE() << label << ": size " << actual.size() << " != " << n;
        return;
    }
    const ErrorStats stats = compute_errors(actual.data(), expected, n);
    const bool finite = all_finite(actual);
    std::fprintf(stderr,
                 "  [%s] max_abs=%.6g rmse=%.6g finite=%d max_at=%zu\n",
                 label.c_str(), stats.max_abs, stats.rmse, finite ? 1 : 0,
                 stats.max_at);
    EXPECT_TRUE(finite) << label << ": non-finite value";
    EXPECT_LT(stats.max_abs, max_abs_tol)
        << label << ": max_abs=" << stats.max_abs << " at " << stats.max_at;
    EXPECT_LT(stats.rmse, rmse_tol) << label << ": rmse=" << stats.rmse;
}

struct ArgmaxInfo {
    int32_t index  = -1;
    int32_t second = -1;
    double margin  = 0.0;
};

static ArgmaxInfo argmax_info(const float * values, int64_t n) {
    ArgmaxInfo info;
    float top1 = -std::numeric_limits<float>::infinity();
    float top2 = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        const float v = values[i];
        if (v > top1) {
            top2 = top1;
            info.second = info.index;
            top1 = v;
            info.index = static_cast<int32_t>(i);
        } else if (v > top2) {
            top2 = v;
            info.second = static_cast<int32_t>(i);
        }
    }
    info.margin = static_cast<double>(top1) - static_cast<double>(top2);
    return info;
}

static int32_t argmax_index(const std::vector<float> & values) {
    return argmax_info(values.data(), static_cast<int64_t>(values.size())).index;
}

// Primary comparison is numerical; argmax equality is also strict. The
// diagnostic includes the reference top-2 gap so an argmax mismatch can be
// distinguished from a near tie without weakening the assertion.
static void check_argmax(const std::string & label,
                         const float * reference_logits, int64_t vocab,
                         int32_t expected, int32_t actual) {
    const ArgmaxInfo info = argmax_info(reference_logits, vocab);
    std::fprintf(stderr,
                 "  [%s] argmax=%d actual=%d second=%d margin=%.6g\n",
                 label.c_str(), expected, actual, info.second, info.margin);
    EXPECT_EQ(info.index, expected)
        << label << ": reference argmax metadata disagrees with logits";
    EXPECT_EQ(actual, expected)
        << label << ": argmax mismatch (reference second=" << info.second
        << ", gap=" << info.margin << ")";
}

static bool read_f32(ggml_tensor * tensor, std::vector<float> & out,
                     size_t count, size_t element_offset = 0) {
    if (!tensor || tensor->type != GGML_TYPE_F32) return false;
    const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
    if (element_offset > elements || count > elements - element_offset) return false;
    out.resize(count);
    ggml_backend_tensor_get(tensor, out.data(),
                            element_offset * sizeof(float),
                            count * sizeof(float));
    return true;
}

static bool read_i32(ggml_tensor * tensor, std::vector<int32_t> & out,
                     size_t count, size_t element_offset = 0) {
    if (!tensor || tensor->type != GGML_TYPE_I32) return false;
    const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
    if (element_offset > elements || count > elements - element_offset) return false;
    out.resize(count);
    ggml_backend_tensor_get(tensor, out.data(),
                            element_offset * sizeof(int32_t),
                            count * sizeof(int32_t));
    return true;
}

static bool upload_f32(ggml_tensor * tensor, const std::vector<float> & data) {
    if (!tensor || tensor->type != GGML_TYPE_F32) return false;
    if (static_cast<size_t>(ggml_nelements(tensor)) != data.size()) return false;
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    return true;
}

static bool upload_i32_scalar(ggml_tensor * tensor, int32_t value) {
    if (!tensor || tensor->type != GGML_TYPE_I32) return false;
    if (ggml_nelements(tensor) != 1) return false;
    ggml_backend_tensor_set(tensor, &value, 0, sizeof(value));
    return true;
}

// Host projection must use the PyTorch/GGML weight layout {hidden, vocab},
// i.e. contiguous weights[v * H + c], not [c * V + v].
static std::vector<float> project_logits(const std::vector<float> & weights,
                                         const float * hidden,
                                         int64_t hidden_dim, int64_t vocab) {
    std::vector<float> logits(static_cast<size_t>(vocab));
    for (int64_t v = 0; v < vocab; ++v) {
        const float * row = weights.data() + static_cast<size_t>(v * hidden_dim);
        // Float accumulation deliberately mirrors ggml_mul_mat.
        float acc = 0.0f;
        for (int64_t c = 0; c < hidden_dim; ++c) {
            acc += hidden[c] * row[c];
        }
        logits[static_cast<size_t>(v)] = acc;
    }
    return logits;
}

static bool load_reference(const std::string & path,
                           t2s_reference::Reference & out,
                           std::string & error) {
    try {
        out = t2s_reference::load(path);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
    } catch (...) {
        error = "unknown exception";
    }
    return false;
}

// F32 baseline is mandatory: decode touches these tensors, and any non-F32
// weight would invalidate the reference comparison.
static bool require_f32_model(const gpt_sovits::t2s_model & model,
                              std::string & error) {
    auto need = [&](const ggml_tensor * tensor, const std::string & name) {
        if (!tensor) {
            error = name + " is null";
            return false;
        }
        if (tensor->type != GGML_TYPE_F32) {
            error = name + " is not F32";
            return false;
        }
        return true;
    };

    const auto & w = model.weights;
    if (!need(w.lm_head_w, "lm_head_w")) return false;
    if (!need(w.embed.text_embedding, "text_embedding") ||
        !need(w.embed.bert_proj_w, "bert_proj_w") ||
        !need(w.embed.bert_proj_b, "bert_proj_b") ||
        !need(w.embed.text_pos_alpha, "text_pos_alpha") ||
        !need(w.embed.audio_embedding, "audio_embedding") ||
        !need(w.embed.audio_pos_alpha, "audio_pos_alpha")) {
        return false;
    }
    if (!need(w.extract_latent.ssl_proj_w, "ssl_proj_w") ||
        !need(w.extract_latent.ssl_proj_b, "ssl_proj_b") ||
        !need(w.extract_latent.codebook, "codebook")) {
        return false;
    }
    if (static_cast<int>(w.attention.size()) != static_cast<int>(model.hparams.n_layer)) {
        error = "attention weight count != n_layer";
        return false;
    }
    for (size_t i = 0; i < w.attention.size(); ++i) {
        const auto & a = w.attention[i];
        const std::string p = "attention[" + std::to_string(i) + "].";
        if (!need(a.qkv_w, p + "qkv_w") || !need(a.qkv_b, p + "qkv_b") ||
            !need(a.out_proj_w, p + "out_proj_w") ||
            !need(a.out_proj_b, p + "out_proj_b") ||
            !need(a.ln1_w, p + "ln1_w") || !need(a.ln1_b, p + "ln1_b") ||
            !need(a.ffn_up_w, p + "ffn_up_w") ||
            !need(a.ffn_up_b, p + "ffn_up_b") ||
            !need(a.ffn_down_w, p + "ffn_down_w") ||
            !need(a.ffn_down_b, p + "ffn_down_b") ||
            !need(a.ln2_w, p + "ln2_w") || !need(a.ln2_b, p + "ln2_b")) {
            return false;
        }
    }
    return true;
}

class T2SDecodeReference : public ::testing::Test {
protected:
    void SetUp() override {
        reference_dir_ = T2S_DECODE_REFERENCE_DIR;
        model_path_    = T2S_DECODE_MODEL_PATH;
        if (reference_dir_.empty()) {
            GTEST_SKIP() << "T2S_DECODE_REFERENCE_DIR is empty; configure "
                            "GPTSOVITS_T2S_DECODE_REFERENCE";
        }
        ASSERT_FALSE(model_path_.empty())
            << "T2S_DECODE_MODEL_PATH is empty; configure "
               "GPTSOVITS_T2S_DECODE_MODEL";

        std::string error;
        if (!load_reference(reference_dir_, ref_, error)) {
            FAIL() << "cannot load reference " << reference_dir_ << ": " << error;
        }

        backend_.reset(ggml_backend_cpu_init());
        ASSERT_NE(backend_.ptr, nullptr) << "CPU backend init failed";

        ASSERT_TRUE(model_.load(model_path_, backend_.ptr))
            << "cannot load F32 model " << model_path_;

        try {
            t2s_reference::require_model(ref_, model_.model);
        } catch (const std::exception & e) {
            FAIL() << "reference/model mismatch: " << e.what();
        } catch (...) {
            FAIL() << "reference/model mismatch: unknown exception";
        }

        ASSERT_TRUE(require_f32_model(model_.model, error)) << error;

        hidden_ = static_cast<int64_t>(ref_.hparams.hidden_dim);
        layers_ = static_cast<int64_t>(ref_.hparams.n_layer);
        vocab_  = static_cast<int64_t>(ref_.hparams.vocab_size);

        ASSERT_TRUE(read_f32(model_.model.weights.lm_head_w, lm_head_,
                             static_cast<size_t>(hidden_ * vocab_)))
            << "cannot read lm_head weights";

        sampler_cfg_.top_k              = 1;  // deterministic rank-1 token
        sampler_cfg_.top_p              = 1.0f;
        sampler_cfg_.temperature        = 1.0f;
        sampler_cfg_.repetition_penalty = 1.0f;

        // One slot is enough; keep headroom above the reference final context.
        const uint32_t slot_size = static_cast<uint32_t>(ref_.final + 16);
        ASSERT_TRUE(session_.init(model_.model.hparams, backend_.ptr, 1,
                                  slot_size, sampler_cfg_, GGML_TYPE_F32));
        ASSERT_EQ(session_.session.kv_cache_type, GGML_TYPE_F32);
        ASSERT_FALSE(session_.session.k_caches.empty());
        ASSERT_EQ(session_.session.k_caches[0]->type, GGML_TYPE_F32);

        // The reference treats all text as reference text and all audio as the
        // prompt; no separate input text participates in this decode case.
        session_.session.ref_T_ref    = ref_.text;
        session_.session.ref_T_prompt = ref_.prompt;
    }

    void TearDown() override {
        // Strict reverse order: graph -> session -> model -> backend.
        flex_.reset();
        session_.reset();
        model_.reset();
        backend_.reset();
    }

    void check_logits(int64_t step, const std::vector<float> & actual) {
        const std::string label = "logits step " + std::to_string(step);
        const float * expected = ref_.logits.data() + step * vocab_;
        check_floats(label, actual, expected, static_cast<size_t>(vocab_),
                     kLogitsMaxAbs, kLogitsRmse);
        check_argmax(label + " greedy", expected, vocab_,
                     ref_.argmax[static_cast<size_t>(step)],
                     argmax_index(actual));
    }

    void check_prefill_mask(ggml_tensor * mask) {
        const int64_t slot_size = session_.session.slot_size;
        const int64_t n_kv      = session_.session.n_batch * slot_size;
        if (!mask || mask->type != GGML_TYPE_F16 ||
            mask->ne[0] != n_kv || mask->ne[1] != ref_.prefill) {
            ADD_FAILURE() << "unexpected prefill mask tensor";
            return;
        }
        std::vector<ggml_fp16_t> host(static_cast<size_t>(n_kv * ref_.prefill));
        ggml_backend_tensor_get(mask, host.data(), 0, host.size() * sizeof(ggml_fp16_t));

        size_t bad = 0;
        int64_t first_col = -1, first_row = -1;
        float first_value = 0.0f;
        for (int64_t col = 0; col < ref_.prefill; ++col) {
            for (int64_t row = 0; row < slot_size; ++row) {
                const bool expect_attend =
                    col < ref_.text ? row < ref_.text : row <= col;
                const float value = ggml_fp16_to_fp32(
                    host[static_cast<size_t>(col * n_kv + row)]);
                const bool attend = value == 0.0f;
                const bool masked = std::isinf(value) && value < 0.0f;
                if ((expect_attend && !attend) || (!expect_attend && !masked)) {
                    if (bad == 0) {
                        first_col = col;
                        first_row = row;
                        first_value = value;
                    }
                    ++bad;
                }
            }
        }
        std::fprintf(stderr,
                     "  [prefill mask] mismatches=%zu first=(col=%lld,row=%lld,value=%g)\n",
                     bad, (long long) first_col, (long long) first_row,
                     first_value);
        EXPECT_EQ(bad, 0u) << "prefill three-part mask is incorrect";
    }

    void check_kv(const std::string & phase, int64_t rows,
                  const std::vector<float> & expected_k,
                  const std::vector<float> & expected_v) {
        const size_t n = static_cast<size_t>(hidden_ * rows);
        for (int64_t layer = 0; layer < layers_; ++layer) {
            const size_t base = static_cast<size_t>(layer) * n;
            std::vector<float> actual;
            ASSERT_TRUE(read_f32(session_.session.k_caches[layer], actual, n));
            check_floats(phase + " K layer " + std::to_string(layer), actual,
                         expected_k.data() + base, n, kKvMaxAbs, kKvRmse);
            ASSERT_TRUE(read_f32(session_.session.v_caches[layer], actual, n));
            check_floats(phase + " V layer " + std::to_string(layer), actual,
                         expected_v.data() + base, n, kKvMaxAbs, kKvRmse);
        }
    }

    // Compare the newly written decode row. The full history (including the
    // prefill prefix) is compared separately at the end.
    void check_new_kv_row(int64_t step) {
        const int64_t row = ref_.prefill + step;
        const size_t offset = static_cast<size_t>(row * hidden_);
        const std::string label = "step " + std::to_string(step);
        const size_t row_n = static_cast<size_t>(hidden_);
        for (int64_t layer = 0; layer < layers_; ++layer) {
            const size_t base = static_cast<size_t>(layer) *
                                static_cast<size_t>(ref_.final) * row_n + offset;
            std::vector<float> actual;
            ASSERT_TRUE(read_f32(session_.session.k_caches[layer], actual,
                                 row_n, offset));
            check_floats(label + " K layer " + std::to_string(layer), actual,
                         ref_.final_k.data() + base, row_n, kKvMaxAbs, kKvRmse);
            ASSERT_TRUE(read_f32(session_.session.v_caches[layer], actual,
                                 row_n, offset));
            check_floats(label + " V layer " + std::to_string(layer), actual,
                         ref_.final_v.data() + base, row_n, kKvMaxAbs, kKvRmse);
        }
    }

    void run_prefill() {
        gpt_sovits::t2s_batch_plan plan;
        plan.n_query = {static_cast<int>(ref_.prefill)};
        ASSERT_TRUE(flex_.build(session_.session, model_.model, plan));
        auto & graph = flex_.graph;
        ASSERT_EQ(graph.N, static_cast<int>(ref_.prefill));
        ASSERT_EQ(graph.n_decode, 0);
        ASSERT_EQ(graph.n_active, 1);
        ASSERT_NE(graph.x_prefill, nullptr);
        ASSERT_NE(graph.exp_noise, nullptr);

        // Upload the oracle's positioned embeddings; do not re-embed tokens.
        ASSERT_TRUE(upload_f32(graph.x_prefill, ref_.prefill_input));
        const std::vector<float> ones(static_cast<size_t>(vocab_), 1.0f);
        ASSERT_TRUE(upload_f32(graph.exp_noise, ones));
        if (graph.seen_mask) {
            ASSERT_TRUE(upload_f32(graph.seen_mask, std::vector<float>(
                static_cast<size_t>(vocab_ * graph.n_active), 0.0f)));
        }

        gpt_sovits::t2s_session_flex_advance(session_.session, plan, graph);

        EXPECT_EQ(session_.session.slots[0].n_pos, ref_.prefill);
        EXPECT_EQ(session_.session.slots[0].audio_offset, ref_.text);
        EXPECT_EQ(session_.session.ref_T_ref, ref_.text);
        EXPECT_EQ(session_.session.ref_T_prompt, ref_.prompt);
        EXPECT_EQ(ref_.context_lengths[0], ref_.prefill);
        check_prefill_mask(graph.mask);

        ASSERT_EQ(ggml_backend_graph_compute(backend_.ptr, graph.gf),
                  GGML_STATUS_SUCCESS);

        std::vector<float> hidden;
        ASSERT_TRUE(read_f32(graph.y, hidden,
                             static_cast<size_t>(hidden_ * ref_.prefill)));
        check_floats("prefill hidden", hidden, ref_.prefill_hidden.data(),
                     hidden.size(), kHiddenMaxAbs, kHiddenRmse);

        const float * last = hidden.data() +
            static_cast<size_t>((ref_.prefill - 1) * hidden_);
        check_logits(0, project_logits(lm_head_, last, hidden_, vocab_));

        std::vector<int32_t> token;
        ASSERT_TRUE(read_i32(graph.greedy, token, 1));
        check_argmax("prefill greedy token", ref_.logits.data(), vocab_,
                     ref_.argmax[0], token[0]);
        int32_t greedy = token[0];
        ASSERT_TRUE(read_i32(graph.sampled, token, 1));
        EXPECT_EQ(token[0], greedy) << "top-k=1/noise=1 must match greedy";

        check_kv("prefill", ref_.prefill, ref_.prefill_k, ref_.prefill_v);
        flex_.reset();
    }

    void run_flex_decode() {
        for (int64_t step = 0; step < ref_.steps; ++step) {
            gpt_sovits::t2s_batch_plan plan;
            plan.n_query = {1};
            ASSERT_TRUE(flex_.build(session_.session, model_.model, plan));
            auto & graph = flex_.graph;
            ASSERT_EQ(graph.n_decode, 1);
            ASSERT_NE(graph.decode_token_ids, nullptr);
            ASSERT_NE(graph.decode_positions, nullptr);

            // Teacher forcing: never feed a generated token.
            ASSERT_TRUE(upload_i32_scalar(graph.decode_token_ids,
                                          ref_.forced_tokens[step]));
            const std::vector<float> ones(static_cast<size_t>(vocab_), 1.0f);
            ASSERT_TRUE(upload_f32(graph.exp_noise, ones));

            const int old_n_pos = session_.session.slots[0].n_pos;
            EXPECT_EQ(old_n_pos, ref_.prefill + step);
            gpt_sovits::t2s_session_flex_advance(session_.session, plan, graph);
            EXPECT_EQ(session_.session.slots[0].n_pos, old_n_pos + 1);
            EXPECT_EQ(session_.session.slots[0].audio_offset, ref_.text);
            EXPECT_EQ(session_.session.slots[0].n_pos,
                      ref_.context_lengths[step + 1]);

            std::vector<int32_t> value;
            ASSERT_TRUE(read_i32(graph.decode_positions, value, 1));
            EXPECT_EQ(value[0], ref_.audio_positions[step]);
            EXPECT_EQ(value[0], ref_.prompt + step);
            ASSERT_TRUE(read_i32(graph.kv_pos, value, 1));
            EXPECT_EQ(value[0], old_n_pos);

            ASSERT_EQ(ggml_backend_graph_compute(backend_.ptr, graph.gf),
                      GGML_STATUS_SUCCESS);

            // Read graph.y fresh for this build; flex graphs are rebuilt per
            // step and their allocator memory is reused.
            std::vector<float> hidden;
            ASSERT_TRUE(read_f32(graph.y, hidden, static_cast<size_t>(hidden_)));
            check_logits(step + 1,
                         project_logits(lm_head_, hidden.data(), hidden_, vocab_));

            ASSERT_TRUE(read_i32(graph.greedy, value, 1));
            check_argmax("step " + std::to_string(step) + " greedy token",
                         ref_.logits.data() + (step + 1) * vocab_, vocab_,
                         ref_.argmax[step + 1], value[0]);
            const int32_t greedy = value[0];
            ASSERT_TRUE(read_i32(graph.sampled, value, 1));
            EXPECT_EQ(value[0], greedy) << "top-k=1/noise=1 must match greedy";

            check_new_kv_row(step);
            flex_.reset();
        }
        EXPECT_EQ(session_.session.slots[0].n_pos, ref_.final);
        check_kv("final", ref_.final, ref_.final_k, ref_.final_v);
    }

    void run_persistent_decode() {
        ASSERT_TRUE(gpt_sovits::t2s_session_build_decode_graph(
            session_.session, model_.model));
        EXPECT_EQ(session_.session.decode_model, &model_.model);

        ggml_cgraph * const graph =
            gpt_sovits::t2s_session_get_decode_graph(session_.session);
        ggml_tensor * const token_id =
            gpt_sovits::t2s_session_get_token_id(session_.session);
        ggml_tensor * const position =
            gpt_sovits::t2s_session_get_position(session_.session);
        ggml_tensor * const exp_noise =
            gpt_sovits::t2s_session_get_exp_noise(session_.session);
        ASSERT_NE(graph, nullptr);
        ASSERT_NE(token_id, nullptr);
        ASSERT_NE(position, nullptr);
        ASSERT_NE(exp_noise, nullptr);

        const std::vector<float> ones(static_cast<size_t>(vocab_), 1.0f);
        for (int64_t step = 0; step < ref_.steps; ++step) {
            ASSERT_TRUE(upload_i32_scalar(token_id, ref_.forced_tokens[step]));
            ASSERT_TRUE(upload_f32(exp_noise, ones));

            const int old_n_pos = session_.session.slots[0].n_pos;
            EXPECT_EQ(old_n_pos, ref_.prefill + step);
            gpt_sovits::t2s_session_decode_advance(session_.session);
            EXPECT_EQ(session_.session.slots[0].n_pos, old_n_pos + 1);
            EXPECT_EQ(session_.session.slots[0].n_pos,
                      ref_.context_lengths[step + 1]);

            // Persistent graph and input tensors must be reused, not rebuilt.
            EXPECT_EQ(gpt_sovits::t2s_session_get_decode_graph(session_.session),
                      graph);
            EXPECT_EQ(gpt_sovits::t2s_session_get_token_id(session_.session),
                      token_id);
            EXPECT_EQ(gpt_sovits::t2s_session_get_position(session_.session),
                      position);
            EXPECT_EQ(gpt_sovits::t2s_session_get_exp_noise(session_.session),
                      exp_noise);

            std::vector<int32_t> value;
            ASSERT_TRUE(read_i32(position, value, 1));
            EXPECT_EQ(value[0], ref_.audio_positions[step]);
            EXPECT_EQ(value[0], ref_.prompt + step);
            ASSERT_TRUE(read_i32(session_.session.kv_pos, value, 1));
            EXPECT_EQ(value[0], old_n_pos);

            ASSERT_EQ(ggml_backend_graph_compute(backend_.ptr, graph),
                      GGML_STATUS_SUCCESS);

            ASSERT_TRUE(read_i32(gpt_sovits::t2s_session_get_greedy(
                                     session_.session), value, 1));
            check_argmax("persistent step " + std::to_string(step) + " greedy",
                         ref_.logits.data() + (step + 1) * vocab_, vocab_,
                         ref_.argmax[step + 1], value[0]);
            const int32_t greedy = value[0];
            ASSERT_TRUE(read_i32(gpt_sovits::t2s_session_get_sampled(
                                     session_.session), value, 1));
            EXPECT_EQ(value[0], greedy) << "top-k=1/noise=1 must match greedy";

            check_new_kv_row(step);
        }
        EXPECT_EQ(session_.session.slots[0].n_pos, ref_.final);
        check_kv("final", ref_.final, ref_.final_k, ref_.final_v);
    }

    BackendGuard backend_;
    ModelGuard model_;
    SessionGuard session_;
    FlexGuard flex_;

    t2s_reference::Reference ref_;
    std::vector<float> lm_head_;
    int64_t hidden_ = 0;
    int64_t layers_ = 0;
    int64_t vocab_  = 0;
    gpt_sovits::t2s_sampler_config sampler_cfg_;
    std::string reference_dir_;
    std::string model_path_;
};

TEST_F(T2SDecodeReference, FlexCpuF32) {
    run_prefill();
    if (HasFatalFailure()) return;
    run_flex_decode();
}

TEST_F(T2SDecodeReference, PersistentCpuF32) {
    run_prefill();
    if (HasFatalFailure()) return;
    run_persistent_decode();
}

}  // namespace
