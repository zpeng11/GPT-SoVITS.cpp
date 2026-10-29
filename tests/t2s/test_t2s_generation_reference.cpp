// Only the reference embeddings and prompt tokens are installed by the test;
// target embedding and closed-loop generation use the public API. Policy and
// scope are in README.md.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "generation_reference.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#ifndef T2S_GENERATION_REFERENCE_DIR
#error "T2S_GENERATION_REFERENCE_DIR must be defined by the test target"
#endif
#ifndef T2S_DECODE_MODEL_PATH
#error "T2S_DECODE_MODEL_PATH must be defined by the test target"
#endif

namespace {

namespace fs = std::filesystem;
using gpt_sovits::t2s_generate_result;
using gpt_sovits::t2s_stop_reason;

struct BackendGuard {
    ggml_backend_t ptr = nullptr;
    ~BackendGuard() { if (ptr) ggml_backend_free(ptr); }
};
struct ModelGuard {
    gpt_sovits::t2s_model model;
    bool loaded = false;
    ~ModelGuard() { if (loaded) gpt_sovits::t2s_model_free(model); }
};
struct SessionGuard {
    gpt_sovits::t2s_session session;
    ~SessionGuard() { gpt_sovits::t2s_session_free(session); }
};

static bool upload_f32(ggml_tensor * t, const std::vector<float> & v) {
    if (!t || t->type != GGML_TYPE_F32 ||
        static_cast<size_t>(ggml_nelements(t)) != v.size()) return false;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    return true;
}

static std::vector<unsigned char> bytes_of(ggml_tensor * t) {
    std::vector<unsigned char> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

static bool bytes_equal(ggml_tensor * t, const std::vector<unsigned char> & expected,
                        size_t & diff) {
    if (ggml_nbytes(t) != expected.size()) return false;
    const std::vector<unsigned char> actual = bytes_of(t);
    for (diff = 0; diff < actual.size(); ++diff)
        if (actual[diff] != expected[diff]) return false;
    diff = actual.size();
    return true;
}

// All F32: iterating ctx_w avoids a second, drift-prone weight list.
static bool require_f32_weights(const gpt_sovits::t2s_model & model, std::string & error) {
    if (!model.ctx_w) { error = "model ctx_w is null"; return false; }
    size_t n = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
         t = ggml_get_next_tensor(model.ctx_w, t)) {
        ++n;
        if (t->type != GGML_TYPE_F32) {
            error = std::string(t->name[0] ? t->name : "<unnamed>") + " is not F32";
            return false;
        }
    }
    if (!n) { error = "model has no weight tensors"; return false; }
    return true;
}

// Cache exported positioned embeddings exactly as compute_ref_emb would.
static bool install_ref(gpt_sovits::t2s_session & session,
                        const gpt_sovits::t2s_model & model,
                        int64_t T_ref, int64_t T_prompt,
                        const std::vector<int32_t> & prompt_tokens,
                        const std::vector<float> & text,
                        const std::vector<float> & audio, std::string & error) {
    const int64_t d = model.hparams.hidden_dim;
    if (T_ref <= 0 || T_prompt <= 0 ||
        static_cast<int64_t>(prompt_tokens.size()) != T_prompt ||
        static_cast<int64_t>(text.size()) != d * T_ref ||
        static_cast<int64_t>(audio.size()) != d * T_prompt) {
        error = "cached reference embedding shape mismatch";
        return false;
    }
    struct ggml_init_params params = { ggml_tensor_overhead() * 3, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { error = "ggml_init() for reference cache failed"; return false; }
    ggml_tensor * text_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, T_ref);
    ggml_tensor * audio_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, T_prompt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, session.backend);
    if (!text_t || !audio_t || !buf || !upload_f32(text_t, text) ||
        !upload_f32(audio_t, audio)) {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        error = "cached reference allocation/upload failed";
        return false;
    }
    session.ctx_ref = ctx;
    session.buf_ref = buf;
    session.ref_text_emb = text_t;
    session.ref_audio_emb = audio_t;
    session.ref_T_ref = T_ref;
    session.ref_T_prompt = T_prompt;
    // The prompt tokens travel with the cached embeddings so generation can
    // seed the repetition history from the original prompt.
    session.ref_prompt_tokens = prompt_tokens;
    session.ref_model = &model;
    return true;
}

static bool parse_stop(const std::string & s, t2s_stop_reason & out) {
    if (s == "eos") { out = t2s_stop_reason::eos; return true; }
    if (s == "max_new_tokens") { out = t2s_stop_reason::max_new_tokens; return true; }
    return false;
}
static const char * stop_name(t2s_stop_reason r) {
    return r == t2s_stop_reason::eos ? "eos" : "max_new_tokens";
}

struct PenaltyChoice { int32_t argmax = -1; double gap = 0.0; };

// Reconstruct the greedy token from raw logits plus the binary repetition
// history, which starts from the original prompt.  Diagnostics only.
static PenaltyChoice penalty_choice(const t2s_generation_reference::Reference & ref,
                                    size_t step, int64_t vocab, float rp) {
    std::vector<unsigned char> seen(static_cast<size_t>(vocab), 0);
    for (int32_t id : ref.input.prompt_tokens)
        if (id >= 0 && id < vocab)
            seen[static_cast<size_t>(id)] = 1;
    for (size_t i = 0; i < step && i < ref.codes.size(); ++i)
        if (ref.codes[i] >= 0 && ref.codes[i] < vocab)
            seen[static_cast<size_t>(ref.codes[i])] = 1;
    const float * row = ref.logits.data() + step * static_cast<size_t>(vocab);
    float top1 = -std::numeric_limits<float>::infinity(), top2 = top1;
    PenaltyChoice c;
    for (int64_t v = 0; v < vocab; ++v) {
        float x = row[static_cast<size_t>(v)];
        if (seen[static_cast<size_t>(v)]) x = x < 0.0f ? x * rp : x * (1.0f / rp);
        if (x > top1) { top2 = top1; top1 = x; c.argmax = static_cast<int32_t>(v); }
        else if (x > top2) top2 = x;
    }
    c.gap = static_cast<double>(top1) - static_cast<double>(top2);
    return c;
}

static void compare_codes(const std::string & label,
                          const t2s_generation_reference::Reference & ref,
                          const std::vector<int32_t> & expected,
                          const t2s_generate_result & actual, float rp) {
    const size_t common = std::min(expected.size(), actual.codes.size());
    size_t first = common;
    for (size_t i = 0; i < common; ++i)
        if (expected[i] != actual.codes[i]) { first = i; break; }
    if (first == common) {
        if (expected.size() != actual.codes.size())
            ADD_FAILURE() << label << ": code count " << actual.codes.size()
                          << " != expected " << expected.size();
        return;
    }
    const int64_t vocab = static_cast<int64_t>(ref.input.hparams.vocab_size);
    const size_t rows = vocab > 0 ? ref.logits.size() / static_cast<size_t>(vocab) : 0;
    if (first < rows) {
        const PenaltyChoice pc = penalty_choice(ref, first, vocab, rp);
        ADD_FAILURE() << label << ": first code mismatch at step " << first
                      << ": expected " << expected[first]
                      << ", actual " << actual.codes[first]
                      << ", penalty-argmax=" << pc.argmax
                      << ", top1-gap=" << pc.gap;
    } else {
        ADD_FAILURE() << label << ": first code mismatch at step " << first
                      << ": expected " << expected[first]
                      << ", actual " << actual.codes[first]
                      << " (no reference logits row)";
    }
}

static void check_stop(const std::string & label, const t2s_generate_result & actual,
                       t2s_stop_reason expected) {
    EXPECT_EQ(actual.stop_reason, expected)
        << label << ": stop_reason " << stop_name(actual.stop_reason)
        << " != " << stop_name(expected);
}

class T2SGenerationReference : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = T2S_GENERATION_REFERENCE_DIR;
        model_path_ = T2S_DECODE_MODEL_PATH;
        if (root_.empty()) {
            GTEST_SKIP() << "T2S_GENERATION_REFERENCE_DIR is empty; configure "
                            "GPTSOVITS_T2S_GENERATION_REFERENCE";
        }
    }

    void run_case(const std::string & subdir, float expected_rp) {
        reference_path_ = (fs::path(root_) / subdir).string();
        try {
            ref_ = t2s_generation_reference::load(reference_path_);
        } catch (const std::exception & e) {
            FAIL() << "cannot load reference " << reference_path_ << ": " << e.what();
        } catch (...) {
            FAIL() << "cannot load reference " << reference_path_;
        }
        if (HasFatalFailure()) return;
        validate_reference(expected_rp);
        if (HasFatalFailure()) return;

        backend_.ptr = ggml_backend_cpu_init();
        ASSERT_NE(backend_.ptr, nullptr) << "CPU backend init failed";
        model_.loaded = gpt_sovits::t2s_model_load(model_path_, model_.model,
                                                   backend_.ptr);
        ASSERT_TRUE(model_.loaded) << "cannot load F32 model " << model_path_;
        try {
            t2s_generation_reference::require_model(ref_, model_.model);
        } catch (const std::exception & e) {
            FAIL() << "reference/model mismatch: " << e.what();
        } catch (...) {
            FAIL() << "reference/model mismatch";
        }
        std::string error;
        ASSERT_TRUE(require_f32_weights(model_.model, error)) << error;
        if (HasFatalFailure()) return;
        init_session();
        if (HasFatalFailure()) return;
        install_reference();
        if (!HasFatalFailure()) run_sequence();
    }

    void validate_reference(float expected_rp) {
        const auto & in = ref_.input;
        H_ = in.hparams.hidden_dim;
        V_ = in.hparams.vocab_size;
        EOS_ = in.hparams.eos;
        R_ = ref_.reference_text;
        T_ = ref_.target_text;
        P_ = in.prompt;
        N_ = R_ + T_ + P_;
        ASSERT_TRUE(H_ > 0 && V_ > 0 && EOS_ == V_ - 1 && EOS_ > 0);
        ASSERT_TRUE(R_ > 0 && T_ > 0 && P_ > 0);
        ASSERT_EQ(in.text, R_ + T_);
        ASSERT_EQ(in.prefill, N_);
        ASSERT_EQ(static_cast<int64_t>(in.phones.size()), R_ + T_);
        ASSERT_EQ(static_cast<int64_t>(in.bert.size()), (R_ + T_) * 1024);
        ASSERT_EQ(static_cast<int64_t>(in.prompt_tokens.size()), P_);
        ASSERT_EQ(static_cast<int64_t>(in.prefill_input.size()), N_ * H_);
        ASSERT_GT(ref_.max_new_tokens, 0);
        ASSERT_LE(ref_.max_new_tokens,
                  static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
        ASSERT_TRUE(std::isfinite(ref_.repetition_penalty));
        ASSERT_GT(ref_.repetition_penalty, 0.0f);
        EXPECT_FLOAT_EQ(ref_.repetition_penalty, expected_rp);
        ASSERT_EQ(ref_.logits.size() % static_cast<size_t>(V_), 0u);
        ASSERT_GE(ref_.logits.size() / static_cast<size_t>(V_), ref_.codes.size());
        const int64_t needed = N_ + ref_.max_new_tokens - 1;
        ASSERT_GT(needed, 0);
        ASSERT_LT(needed, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
        slot_size_ = static_cast<uint32_t>(needed);
    }

    void init_session() {
        sampler_cfg_.top_k = 1;
        sampler_cfg_.top_p = 1.0f;
        sampler_cfg_.temperature = 1.0f;
        sampler_cfg_.repetition_penalty = ref_.repetition_penalty;
        ASSERT_TRUE(gpt_sovits::t2s_session_init(
            session_.session, model_.model.hparams, backend_.ptr, 1, slot_size_,
            sampler_cfg_, GGML_TYPE_F32));
        ASSERT_EQ(session_.session.kv_cache_type, GGML_TYPE_F32);
        ASSERT_EQ(session_.session.n_batch, 1u);
        ASSERT_EQ(session_.session.slots.size(), 1u);
        ASSERT_FALSE(session_.session.k_caches.empty());
        ASSERT_EQ(session_.session.k_caches[0]->type, GGML_TYPE_F32);
        ASSERT_EQ(session_.session.v_caches[0]->type, GGML_TYPE_F32);
        // Unused KV entries stay deterministic; masked reads cannot leak them.
        for (size_t i = 0; i < session_.session.k_caches.size(); ++i) {
            ggml_backend_tensor_memset(session_.session.k_caches[i], 0, 0,
                                       ggml_nbytes(session_.session.k_caches[i]));
            ggml_backend_tensor_memset(session_.session.v_caches[i], 0, 0,
                                       ggml_nbytes(session_.session.v_caches[i]));
        }
    }

    void install_reference() {
        const auto & in = ref_.input;
        target_phones_.assign(in.phones.begin() + R_, in.phones.end());
        target_bert_.assign(in.bert.begin() + R_ * 1024,
                            in.bert.begin() + (R_ + T_) * 1024);
        const std::vector<float> text(
            in.prefill_input.begin(),
            in.prefill_input.begin() + static_cast<size_t>(R_) * H_);
        const std::vector<float> audio(
            in.prefill_input.begin() + static_cast<size_t>(R_ + T_) * H_,
            in.prefill_input.begin() + static_cast<size_t>(R_ + T_ + P_) * H_);
        std::string error;
        ASSERT_TRUE(install_ref(session_.session, model_.model, R_, P_,
                                in.prompt_tokens, text, audio, error)) << error;
        ASSERT_EQ(session_.session.ref_T_ref, R_);
        ASSERT_EQ(session_.session.ref_T_prompt, P_);
        ASSERT_EQ(session_.session.ref_prompt_tokens, in.prompt_tokens);
        ASSERT_EQ(session_.session.ref_model, &model_.model);
        ref_text_ptr_ = session_.session.ref_text_emb;
        ref_audio_ptr_ = session_.session.ref_audio_emb;
        ref_text_bytes_ = bytes_of(ref_text_ptr_);
        ref_audio_bytes_ = bytes_of(ref_audio_ptr_);
        ref_prompt_tokens_ = session_.session.ref_prompt_tokens;
    }

    void generate(int64_t budget, uint32_t seed, const std::string & label,
                  t2s_generate_result & result) {
        gpt_sovits::t2s_generate_config config;
        config.max_new_tokens = static_cast<int32_t>(budget);
        config.seed = seed;
        ASSERT_TRUE(gpt_sovits::t2s_session_generate(
            session_.session, model_.model, target_phones_.data(), T_,
            target_bert_.data(), result, config)) << label << ": generate failed";
    }

    void expect_released(const std::string & label) {
        ASSERT_EQ(session_.session.slots.size(), 1u);
        EXPECT_EQ(session_.session.slots[0].n_pos, 0) << label << ": n_pos";
        EXPECT_EQ(session_.session.slots[0].audio_offset, 0)
            << label << ": audio_offset";
    }

    void expect_ref_unchanged(const std::string & label) {
        ASSERT_EQ(session_.session.ref_text_emb, ref_text_ptr_);
        ASSERT_EQ(session_.session.ref_audio_emb, ref_audio_ptr_);
        ASSERT_EQ(session_.session.ref_T_ref, R_);
        ASSERT_EQ(session_.session.ref_T_prompt, P_);
        EXPECT_EQ(session_.session.ref_prompt_tokens, ref_prompt_tokens_)
            << label << ": ref_prompt_tokens changed";
        EXPECT_EQ(session_.session.ref_model, &model_.model);
        size_t diff = 0;
        EXPECT_TRUE(bytes_equal(ref_text_ptr_, ref_text_bytes_, diff))
            << label << ": ref_text_emb changed at byte " << diff;
        EXPECT_TRUE(bytes_equal(ref_audio_ptr_, ref_audio_bytes_, diff))
            << label << ": ref_audio_emb changed at byte " << diff;
    }

    void run_sequence() {
        t2s_stop_reason expected_stop = t2s_stop_reason::max_new_tokens;
        ASSERT_TRUE(parse_stop(ref_.stop_reason, expected_stop))
            << "unsupported reference stop_reason '" << ref_.stop_reason << "'";

        // Full budget.  No requirement on EOS versus the length limit.
        t2s_generate_result full;
        generate(ref_.max_new_tokens, 42, "full", full);
        if (HasFatalFailure()) return;
        compare_codes("full", ref_, ref_.codes, full, ref_.repetition_penalty);
        check_stop("full", full, expected_stop);
        expect_released("full");
        expect_ref_unchanged("full");
        ggml_cgraph * const decode_graph = session_.session.gf_dec;
        if (ref_.selected.size() > 1) {
            EXPECT_NE(decode_graph, nullptr);
            EXPECT_EQ(session_.session.decode_model, &model_.model);
        } else {
            EXPECT_EQ(decode_graph, nullptr);
            EXPECT_EQ(session_.session.decode_model, nullptr);
        }
        std::fprintf(stderr, "  [%s] full: codes=%zu stop=%s max_new_tokens=%lld\n",
                     fs::path(reference_path_).filename().string().c_str(),
                     full.codes.size(), stop_name(full.stop_reason),
                     static_cast<long long>(ref_.max_new_tokens));

        // A shorter budget must truncate exactly and report the limit, even if
        // the golden full run's next event would have been EOS.
        const size_t count = ref_.codes.size();
        if (count > 0) {
            const size_t short_budget = std::min<size_t>(5, count);
            t2s_generate_result shortened;
            generate(static_cast<int64_t>(short_budget), 42, "short", shortened);
            if (HasFatalFailure()) return;
            const std::vector<int32_t> expected(ref_.codes.begin(),
                                                ref_.codes.begin() + short_budget);
            compare_codes("short", ref_, expected, shortened,
                          ref_.repetition_penalty);
            check_stop("short", shortened, t2s_stop_reason::max_new_tokens);
            expect_released("short");
            expect_ref_unchanged("short");
            EXPECT_EQ(session_.session.gf_dec, decode_graph);
        }
        if (HasFatalFailure()) return;

        // top-k=1 is RNG-independent: another seed reproduces the codes and
        // reuses the persistent decode graph.
        t2s_generate_result repeated;
        generate(ref_.max_new_tokens, 7, "repeat", repeated);
        if (HasFatalFailure()) return;
        compare_codes("repeat", ref_, ref_.codes, repeated,
                      ref_.repetition_penalty);
        check_stop("repeat", repeated, expected_stop);
        EXPECT_EQ(repeated.codes, full.codes) << "top-k=1 codes changed across seeds";
        EXPECT_EQ(repeated.stop_reason, full.stop_reason);
        expect_released("repeat");
        expect_ref_unchanged("repeat");
        EXPECT_EQ(session_.session.gf_dec, decode_graph)
            << "completed run did not reuse the decode graph";
    }

    BackendGuard backend_;
    ModelGuard model_;
    SessionGuard session_;
    t2s_generation_reference::Reference ref_;
    gpt_sovits::t2s_sampler_config sampler_cfg_;
    std::string root_, reference_path_, model_path_;
    int64_t H_ = 0, V_ = 0, EOS_ = 0, R_ = 0, T_ = 0, P_ = 0, N_ = 0;
    uint32_t slot_size_ = 0;
    std::vector<int32_t> target_phones_;
    std::vector<float> target_bert_;
    std::vector<int32_t> ref_prompt_tokens_;
    ggml_tensor * ref_text_ptr_ = nullptr;
    ggml_tensor * ref_audio_ptr_ = nullptr;
    std::vector<unsigned char> ref_text_bytes_, ref_audio_bytes_;
};

TEST_F(T2SGenerationReference, CpuGreedyNoPenalty) {
    run_case("rp1", 1.0f);
}
TEST_F(T2SGenerationReference, CpuGreedyRepetitionPenalty) {
    run_case("rp135", 1.35f);
}

}  // namespace
