// tests/t2s/test_t2s_session.cpp
//
// Tests for the T2S session module: KV cache allocation, slot management,
// decode graph building, and accessor helpers.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "ggml-backend.h"

#include <cmath>
#include <string>

static const std::string kTestDir = T2S_TEST_DIR;

static ggml_backend_t create_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    return backend;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void expect_shape(ggml_tensor * t, const std::vector<int64_t> & expected) {
    ASSERT_NE(t, nullptr);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t exp = (size_t)i < expected.size() ? expected[i] : 1;
        EXPECT_EQ(t->ne[i], exp) << "dim " << i << " mismatch";
    }
}

// ---------------------------------------------------------------------------
// Init / Free
// ---------------------------------------------------------------------------

TEST(T2SSession, InitAndFree) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 4;
    const uint32_t slot_size = 64;

    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // Check tensor shapes
    const int64_t d_model = hparams.hidden_dim;
    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    ASSERT_EQ(session.k_caches.size(), hparams.n_layer);
    ASSERT_EQ(session.v_caches.size(), hparams.n_layer);

    for (uint32_t i = 0; i < hparams.n_layer; i++) {
        expect_shape(session.k_caches[i], {d_model, max_ctx});
        expect_shape(session.v_caches[i], {d_model, max_ctx});
    }

    expect_shape(session.kv_pos, {n_batch});
    expect_shape(session.mask,   {max_ctx, n_batch});
    expect_shape(session.x_dec,  {d_model, n_batch});

    EXPECT_EQ(session.n_batch, n_batch);
    EXPECT_EQ(session.slot_size, slot_size);
    EXPECT_EQ(session.slots.size(), (size_t)n_batch);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, InitRejectsZeroBatch) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    EXPECT_FALSE(gpt_sovits::t2s_session_init(session, hparams, backend, 0, 64));

    ggml_backend_free(backend);
}

TEST(T2SSession, InitRejectsZeroSlotSize) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    EXPECT_FALSE(gpt_sovits::t2s_session_init(session, hparams, backend, 4, 0));

    ggml_backend_free(backend);
}

TEST(T2SSession, DoubleFreeSafe) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 2, 32));
    gpt_sovits::t2s_session_free(session);
    // Second free should be a no-op (all pointers already null).
    gpt_sovits::t2s_session_free(session);

    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Slot management
// ---------------------------------------------------------------------------

// Test helper: activate a slot with given n_pos for testing purposes.
// In production code, activation happens via t2s_session_advance.
static void test_activate_slot(gpt_sovits::t2s_session & session, int slot_id, int n_pos) {
    const int64_t max_ctx = (int64_t) session.n_batch * session.slot_size;
    const int slot_start  = slot_id * session.slot_size;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);

    session.slots[slot_id].in_use = true;
    session.slots[slot_id].n_pos  = n_pos;

    for (int k = 0; k < n_pos; k++) {
        session.mask_host[slot_id * max_ctx + slot_start + k] = zero;
    }
}

TEST(T2SSession, FindFreeSlot) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch = 3;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, 32));

    // Find and activate all slots.
    std::vector<int> ids;
    for (uint32_t i = 0; i < n_batch; i++) {
        int id = gpt_sovits::t2s_session_find_free_slot(session);
        EXPECT_EQ(id, (int)i);
        test_activate_slot(session, id, 0);
        ids.push_back(id);
    }

    // No more free slots.
    EXPECT_EQ(gpt_sovits::t2s_session_find_free_slot(session), -1);

    // Release one and find again.
    gpt_sovits::t2s_session_slot_release(session, ids[1]);
    int id = gpt_sovits::t2s_session_find_free_slot(session);
    EXPECT_EQ(id, (int)ids[1]);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotReuseAfterRelease) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 2, 16));

    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_GE(s0, 0);
    test_activate_slot(session, s0, 0);

    // Release and find again — should get the same slot back.
    gpt_sovits::t2s_session_slot_release(session, s0);
    int s1 = gpt_sovits::t2s_session_find_free_slot(session);
    EXPECT_EQ(s1, s0);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

TEST(T2SSession, Accessors) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 16;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // kv_pos
    struct ggml_tensor * kv_pos = gpt_sovits::t2s_session_get_kv_pos(session);
    ASSERT_NE(kv_pos, nullptr);
    EXPECT_EQ(kv_pos->ne[0], (int64_t)n_batch);

    // mask
    struct ggml_tensor * mask = gpt_sovits::t2s_session_get_mask(session);
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ(mask->ne[0], (int64_t)(n_batch * slot_size));
    EXPECT_EQ(mask->ne[1], (int64_t)n_batch);

    // n_kv
    EXPECT_EQ(gpt_sovits::t2s_session_get_n_kv(session), (int)(n_batch * slot_size));

    // layer caches
    for (uint32_t i = 0; i < hparams.n_layer; i++) {
        auto caches = gpt_sovits::t2s_session_get_layer_caches(session, (int)i);
        ASSERT_NE(caches.k, nullptr);
        ASSERT_NE(caches.v, nullptr);
        EXPECT_EQ(caches.k->ne[0], (int64_t)hparams.hidden_dim);
        EXPECT_EQ(caches.v->ne[0], (int64_t)hparams.hidden_dim);
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Decode graph build (requires model weights)
// ---------------------------------------------------------------------------

TEST(T2SSession, BuildDecodeGraph) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 32;

    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, backend, n_batch, slot_size));
    ASSERT_TRUE(gpt_sovits::t2s_session_build_decode_graph(session, model));

    // Verify graph outputs
    struct ggml_tensor * x_dec = gpt_sovits::t2s_session_get_x_dec(session);
    struct ggml_tensor * y_dec = gpt_sovits::t2s_session_get_y_dec(session);
    struct ggml_cgraph * gf    = gpt_sovits::t2s_session_get_decode_graph(session);

    ASSERT_NE(x_dec, nullptr);
    ASSERT_NE(y_dec, nullptr);
    ASSERT_NE(gf, nullptr);

    const int64_t d_model = model.hparams.hidden_dim;
    expect_shape(x_dec, {d_model, n_batch});
    expect_shape(y_dec, {d_model, n_batch});

    // alloc_dec should have been created
    EXPECT_NE(session.alloc_dec, nullptr);

    // Cleanup order: session first (references model weights), then model.
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Mask management
// ---------------------------------------------------------------------------

static float fp16_to_fp32(ggml_fp16_t v) {
    return ggml_fp16_to_fp32(v);
}

static ggml_fp16_t fp32_to_fp16(float v) {
    return ggml_fp32_to_fp16(v);
}

TEST(T2SSession, MaskInitAllInf) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;
    ASSERT_EQ(session.mask_host.size(), (size_t)(max_ctx * n_batch));

    const float neg_inf = -INFINITY;
    for (size_t i = 0; i < session.mask_host.size(); i++) {
        float val = fp16_to_fp32(session.mask_host[i]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "mask_host[" << i << "] = " << val << ", expected -inf";
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotMaskSetup) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    // Activate slot 0 with 3 prefilled tokens via test helper.
    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 3);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, s0), 3);

    // Column 0: rows [0, 3) should be 0, rest -inf.
    for (int k = 0; k < 3; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_EQ(val, 0.0f) << "mask_host[0][" << k << "] should be 0";
    }
    for (int k = 3; k < max_ctx; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "mask_host[0][" << k << "] should be -inf";
    }

    // Column 1 should still be all -inf.
    for (int k = 0; k < max_ctx; k++) {
        float val = fp16_to_fp32(session.mask_host[1 * max_ctx + k]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "mask_host[1][" << k << "] should be -inf";
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotDecodeStepMask) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    // Activate slot 0 with 2 prefilled tokens, then do 3 decode steps.
    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 2);

    for (int step = 0; step < 3; step++) {
        gpt_sovits::t2s_session_decode_advance(session);
    }

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, s0), 5);

    // Column 0: rows [0, 5) should be 0, rest -inf.
    for (int k = 0; k < 5; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_EQ(val, 0.0f) << "after decode steps, mask_host[0][" << k << "] should be 0";
    }
    for (int k = 5; k < max_ctx; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "after decode steps, mask_host[0][" << k << "] should be -inf";
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotReleaseMask) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    // Activate, decode, then release.
    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 2);
    gpt_sovits::t2s_session_decode_advance(session);
    gpt_sovits::t2s_session_slot_release(session, s0);

    // Column 0 should be all -inf after release.
    for (int k = 0; k < max_ctx; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "after release, mask_host[0][" << k << "] should be -inf";
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotReuseMask) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    // Activate with 3 tokens, decode 1 step, release, then re-activate with 1 token.
    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 3);
    gpt_sovits::t2s_session_decode_advance(session);
    gpt_sovits::t2s_session_slot_release(session, s0);

    test_activate_slot(session, 0, 1);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 1);

    // Column 0: only row 0 should be 0 (from test_activate_slot n_pos=1), rest -inf.
    float val0 = fp16_to_fp32(session.mask_host[0 * max_ctx + 0]);
    EXPECT_EQ(val0, 0.0f) << "after re-alloc, mask_host[0][0] should be 0";

    for (int k = 1; k < max_ctx; k++) {
        float val = fp16_to_fp32(session.mask_host[0 * max_ctx + k]);
        EXPECT_TRUE(std::isinf(val) && val < 0)
            << "after re-alloc, mask_host[0][" << k << "] should be -inf";
    }

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// kv_pos management
// ---------------------------------------------------------------------------

static int32_t read_kv_pos(const gpt_sovits::t2s_session & session, int slot_id) {
    int32_t val = -1;
    ggml_backend_tensor_get(session.kv_pos, &val,
                            slot_id * sizeof(int32_t), sizeof(int32_t));
    return val;
}

TEST(T2SSession, SlotDecodeStepKvPos) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 16;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 3);
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(read_kv_pos(session, s0), 3);

    // Second decode step: kv_pos[0] = 0 * slot_size + 4 = 4
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(read_kv_pos(session, s0), 4);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, MultiSlotKvPos) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 16;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 2);
    int s1 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s1, 1);
    test_activate_slot(session, s1, 5);

    gpt_sovits::t2s_session_decode_advance(session);

    // slot 0: kv_pos[0] = 0 * 16 + 2 = 2
    EXPECT_EQ(read_kv_pos(session, s0), 2);
    // slot 1: kv_pos[1] = 1 * 16 + 5 = 21
    EXPECT_EQ(read_kv_pos(session, s1), 21);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Quantized KV cache
// ---------------------------------------------------------------------------

TEST(T2SSession, InitQuantizedKVCache) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 16;

    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend,
                                              n_batch, slot_size, GGML_TYPE_Q8_0));

    const int64_t d_model = hparams.hidden_dim;
    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    // Shapes are unchanged (ne[] counts elements, not bytes).
    for (uint32_t i = 0; i < hparams.n_layer; i++) {
        expect_shape(session.k_caches[i], {d_model, max_ctx});
        expect_shape(session.v_caches[i], {d_model, max_ctx});
        EXPECT_EQ(session.k_caches[i]->type, GGML_TYPE_Q8_0);
        EXPECT_EQ(session.v_caches[i]->type, GGML_TYPE_Q8_0);
    }

    EXPECT_EQ(session.kv_cache_type, GGML_TYPE_Q8_0);

    // Slot management should work the same.
    int s0 = gpt_sovits::t2s_session_find_free_slot(session);
    ASSERT_EQ(s0, 0);
    test_activate_slot(session, s0, 3);
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(read_kv_pos(session, s0), 3);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Reference embedding caching
// ---------------------------------------------------------------------------

TEST(T2SSession, RefEmbNotCachedInitially) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 2, 16));

    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_emb(session), nullptr);
    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_ref(session), 0);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, ComputeRefEmbWithModel) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) GTEST_SKIP() << "Model file not found: " << path;
    fclose(f);

    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    // Skip if model extract_latent weights are incompatible (e.g. kernel_size != 2).
    if (model.weights.extract_latent.ssl_proj_w->ne[0] != 2) {
        gpt_sovits::t2s_model_free(model);
        ggml_backend_free(backend);
        GTEST_SKIP() << "Model extract_latent weights have incompatible shape";
    }

    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, backend, 2, 32));

    const int64_t T_ref       = 5;
    const int64_t T_hub       = 50;
    const int64_t d_model     = model.hparams.hidden_dim;
    const int64_t bert_dim    = 1024;
    const int64_t hubert_dim  = 768;

    std::vector<int32_t> ref_tokens(T_ref, 1);
    std::vector<float>   ref_bert(bert_dim * T_ref, 0.1f);
    std::vector<float>   hubert(hubert_dim * T_hub, 0.5f);

    bool ok = gpt_sovits::t2s_session_compute_ref_emb(
        session, model,
        ref_tokens.data(), T_ref,
        ref_bert.data(),
        hubert.data(), T_hub);
    ASSERT_TRUE(ok);

    // Verify cached tensor shape.
    struct ggml_tensor * ref_emb = gpt_sovits::t2s_session_get_ref_emb(session);
    ASSERT_NE(ref_emb, nullptr);
    EXPECT_EQ(ref_emb->ne[0], d_model);

    EXPECT_EQ(gpt_sovits::t2s_session_get_ref_T_ref(session), T_ref);

    // T_prompt = floor((T_hub - kernel_size) / stride) + 1 = (50-2)/2+1 = 25
    int64_t T_prompt = (T_hub - 2) / 2 + 1;
    EXPECT_EQ(ref_emb->ne[1], T_ref + T_prompt);

    // Verify data is non-zero after computation.
    const size_t nbytes = ggml_nbytes(ref_emb);
    std::vector<float> data(nbytes / sizeof(float));
    ggml_backend_tensor_get(ref_emb, data.data(), 0, nbytes);
    bool any_nonzero = false;
    for (float v : data) {
        if (v != 0.0f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "ref_emb should have non-zero values after computation";

    // Cleanup.
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SSession, SessionFreeCleansRefEmb) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) GTEST_SKIP() << "Model file not found: " << path;
    fclose(f);

    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    // Skip if model extract_latent weights are incompatible.
    if (model.weights.extract_latent.ssl_proj_w->ne[0] != 2) {
        gpt_sovits::t2s_model_free(model);
        ggml_backend_free(backend);
        GTEST_SKIP() << "Model extract_latent weights have incompatible shape";
    }

    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, backend, 2, 32));

    const int64_t T_ref      = 3;
    const int64_t T_hub      = 20;
    const int64_t bert_dim   = 1024;
    const int64_t hubert_dim = 768;

    std::vector<int32_t> tokens(T_ref, 1);
    std::vector<float>   bert(bert_dim * T_ref, 0.1f);
    std::vector<float>   hubert(hubert_dim * T_hub, 0.5f);

    ASSERT_TRUE(gpt_sovits::t2s_session_compute_ref_emb(
        session, model, tokens.data(), T_ref, bert.data(), hubert.data(), T_hub));

    EXPECT_NE(gpt_sovits::t2s_session_get_ref_emb(session), nullptr);

    // session_free should clean up ref_emb without crash.
    gpt_sovits::t2s_session_free(session);

    // Double free should also be safe.
    gpt_sovits::t2s_session_free(session);

    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Flexible computation graph (t2s_session_build_graph)
// ---------------------------------------------------------------------------

TEST(T2SBatchPlan, Total) {
    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {10, 0, 20, 1};
    EXPECT_EQ(plan.total(), 31);

    plan.n_query = {0, 0, 0, 0};
    EXPECT_EQ(plan.total(), 0);
}

TEST(T2SBuildGraph, RejectsEmptyPlan) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 4, 16));

    gpt_sovits::t2s_model model;
    model.hparams = hparams;

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {0, 0, 0, 0};
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    EXPECT_EQ(graph.ctx, nullptr) << "should fail for empty plan";

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SBuildGraph, RejectsWrongPlanSize) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 4, 16));

    gpt_sovits::t2s_model model;
    model.hparams = hparams;

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 0};  // wrong size
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    EXPECT_EQ(graph.ctx, nullptr) << "should fail for wrong plan size";

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

// Helper: create a minimal model with zero-initialized weight tensors for graph building.
static gpt_sovits::t2s_model create_minimal_model(gpt_sovits::t2s_hparams hparams, ggml_backend_t backend) {
    gpt_sovits::t2s_model model;
    model.hparams = hparams;

    const int64_t d_model = hparams.hidden_dim;
    const int64_t d_ff    = hparams.linear_units;
    const int     n_layer = (int) hparams.n_layer;

    struct ggml_init_params wparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(n_layer * 12 + 3),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model.backend = backend;
    model.ctx_w   = ggml_init(wparams);

    model.weights.attention.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        auto & w = model.weights.attention[i];
        w.qkv_w       = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, d_model, 3 * d_model);
        w.qkv_b       = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, 3 * d_model);
        w.out_proj_w  = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, d_model, d_model);
        w.out_proj_b  = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
        w.ln1_w       = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
        w.ln1_b       = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
        w.ffn_up_w    = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, d_model, d_ff);
        w.ffn_up_b    = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_ff);
        w.ffn_down_w  = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, d_ff, d_model);
        w.ffn_down_b  = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
        w.ln2_w       = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
        w.ln2_b       = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, d_model);
    }
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, backend);

    return model;
}

TEST(T2SBuildGraph, BuildsGraphWithCorrectShapes) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) GTEST_SKIP() << "Model file not found: " << path;
    fclose(f);

    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    const uint32_t n_batch   = 4;
    const uint32_t slot_size = 16;

    gpt_sovits::t2s_session session;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, model.hparams, backend, n_batch, slot_size));

    const int64_t d_model = model.hparams.hidden_dim;
    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    for (uint32_t i = 0; i < n_batch; i++) {
        int id = gpt_sovits::t2s_session_find_free_slot(session);
        test_activate_slot(session, id, 0);
    }

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {5, 0, 3, 1};

    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);
    EXPECT_EQ(graph.N, 9);

    expect_shape(graph.x,      {d_model, 9});
    expect_shape(graph.y,      {d_model, 9});
    expect_shape(graph.kv_pos, {9});
    expect_shape(graph.mask,   {max_ctx, 9});
    EXPECT_NE(graph.alloc, nullptr);
    EXPECT_NE(graph.gf, nullptr);

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SBuildGraph, KvPosCorrectness) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 3;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // slot 0: n_pos=5, slot 1: n_pos=3 (idle in plan), slot 2: idle
    test_activate_slot(session, 0, 5);
    test_activate_slot(session, 1, 3);

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 0, 4};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    // kv_pos: slot0*8 + 5 = 5, slot2*8 + 0..3 = 16,17,18,19
    const int N = 5;
    ASSERT_EQ(graph.N, N);
    std::vector<int32_t> kv_pos_read(N);
    ggml_backend_tensor_get(graph.kv_pos, kv_pos_read.data(), 0, N * sizeof(int32_t));

    EXPECT_EQ(kv_pos_read[0], 5);
    EXPECT_EQ(kv_pos_read[1], 16);
    EXPECT_EQ(kv_pos_read[2], 17);
    EXPECT_EQ(kv_pos_read[3], 18);
    EXPECT_EQ(kv_pos_read[4], 19);

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SBuildGraph, MaskCorrectness) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    // Both slots idle — prefill scenario.
    // slot 0: idle, n_query=2; slot 1: idle, n_query=3.

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {2, 3};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    const int N = 5;
    ASSERT_EQ(graph.N, N);

    std::vector<ggml_fp16_t> mask_read((size_t) max_ctx * N);
    ggml_backend_tensor_get(graph.mask, mask_read.data(), 0, max_ctx * N * sizeof(ggml_fp16_t));

    // mask(row, col) at mask_read[col * max_ctx + row]
    // Slot 0 region: rows [0,8), slot 1 region: rows [8,16)

    // Column 0: slot 0, j=0, n_pos=0 -> attend row 0 only
    EXPECT_EQ(fp16_to_fp32(mask_read[0 * max_ctx + 0]), 0.0f);
    for (int r = 1; r < max_ctx; r++) {
        float val = fp16_to_fp32(mask_read[0 * max_ctx + r]);
        EXPECT_TRUE(std::isinf(val) && val < 0) << "col0 row" << r;
    }

    // Column 1: slot 0, j=1, n_pos=0 -> attend rows 0,1
    EXPECT_EQ(fp16_to_fp32(mask_read[1 * max_ctx + 0]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(mask_read[1 * max_ctx + 1]), 0.0f);

    // Column 2: slot 1, j=0, n_pos=0 -> attend row 8 only
    EXPECT_EQ(fp16_to_fp32(mask_read[2 * max_ctx + 8]), 0.0f);
    for (int r = 0; r < 8; r++) {
        EXPECT_TRUE(std::isinf(fp16_to_fp32(mask_read[2 * max_ctx + r])) && fp16_to_fp32(mask_read[2 * max_ctx + r]) < 0)
            << "col2 row" << r;
    }
    for (int r = 9; r < max_ctx; r++) {
        EXPECT_TRUE(std::isinf(fp16_to_fp32(mask_read[2 * max_ctx + r])) && fp16_to_fp32(mask_read[2 * max_ctx + r]) < 0)
            << "col2 row" << r;
    }

    // Column 3: slot 1, j=1, n_pos=0 -> attend rows 8,9
    EXPECT_EQ(fp16_to_fp32(mask_read[3 * max_ctx + 8]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(mask_read[3 * max_ctx + 9]), 0.0f);

    // Column 4: slot 1, j=2, n_pos=0 -> attend rows 8,9,10
    EXPECT_EQ(fp16_to_fp32(mask_read[4 * max_ctx + 8]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(mask_read[4 * max_ctx + 9]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(mask_read[4 * max_ctx + 10]), 0.0f);

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SBuildGraph, GraphFreeSafe) {
    gpt_sovits::t2s_graph graph;  // default-constructed
    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_graph_free(graph);  // double free
}

// ---------------------------------------------------------------------------
// t2s_session_advance
// ---------------------------------------------------------------------------

TEST(T2SAdvance, UpdatesNPos) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 3;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // slot 0: active, n_pos=2 (decode); slot 1: idle (prefill); slot 2: idle
    test_activate_slot(session, 0, 2);

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 3, 0};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 2);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 3);   // 2 + 1 (decode)
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 3);   // 0 + 3 (prefill, activated)
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 2), 0);   // idle, unchanged

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, UpdatesPersistentMask) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    // slot 0: idle — will be prefilled by advance

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {3, 0};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    // Positions [0,3) should now be revealed in persistent mask
    EXPECT_EQ(fp16_to_fp32(session.mask_host[0 * max_ctx + 0]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(session.mask_host[0 * max_ctx + 1]), 0.0f);
    EXPECT_EQ(fp16_to_fp32(session.mask_host[0 * max_ctx + 2]), 0.0f);
    // Position 3 still masked
    EXPECT_TRUE(std::isinf(fp16_to_fp32(session.mask_host[0 * max_ctx + 3])) &&
                fp16_to_fp32(session.mask_host[0 * max_ctx + 3]) < 0);

    // n_pos should be 3
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 3);

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, InterleavesWithDecodeAdvance) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    // Both slots active
    test_activate_slot(session, 0, 2);
    test_activate_slot(session, 1, 3);

    // Simulate persistent decode advance on slot 0
    // decode_advance reveals position at current n_pos and increments
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 3);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 4);

    // Now use flexible graph for decode on both slots
    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 1};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 4);   // 3 + 1 (decode)
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 5);   // 4 + 1 (decode)

    // Persistent mask: slot 0 has [0,4), slot 1 has [8,13)
    for (int r = 0; r < 4; r++) {
        EXPECT_EQ(fp16_to_fp32(session.mask_host[0 * max_ctx + r]), 0.0f)
            << "slot0 row" << r;
    }
    EXPECT_TRUE(std::isinf(fp16_to_fp32(session.mask_host[0 * max_ctx + 4])) &&
                fp16_to_fp32(session.mask_host[0 * max_ctx + 4]) < 0);

    for (int r = 0; r < 5; r++) {
        EXPECT_EQ(fp16_to_fp32(session.mask_host[1 * max_ctx + 8 + r]), 0.0f)
            << "slot1 row" << r;
    }
    EXPECT_TRUE(std::isinf(fp16_to_fp32(session.mask_host[1 * max_ctx + 13])) &&
                fp16_to_fp32(session.mask_host[1 * max_ctx + 13]) < 0);

    // Resume persistent decode: decode_advance should work correctly
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 5);   // 4 + 1
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 6);   // 5 + 1

    // kv_pos should point to the correct positions
    std::vector<int32_t> kv_pos_read(2);
    ggml_backend_tensor_get(session.kv_pos, kv_pos_read.data(), 0, 2 * sizeof(int32_t));
    EXPECT_EQ(kv_pos_read[0], 4);   // slot 0: write at pos 4
    EXPECT_EQ(kv_pos_read[1], 13);  // slot 1: write at pos 13

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, ActivatesIdleSlot) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 3;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t) n_batch * slot_size;

    // Slot 0 active, slots 1 and 2 idle
    test_activate_slot(session, 0, 2);

    // Plan uses slot 1 (idle) for prefill and slot 0 for decode
    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 4, 0};  // slot 0: decode, slot 1: prefill (activates), slot 2: idle

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    // Slot 1 should be idle before advance
    EXPECT_FALSE(session.slots[1].in_use);

    gpt_sovits::t2s_session_advance(session, plan, graph);

    // Slot 1 should now be active
    EXPECT_TRUE(session.slots[1].in_use);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 3);   // 2 + 1
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 4);   // 0 + 4 (activated)
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 2), 0);   // still idle

    // Persistent mask: slot 1's positions [8, 12) should be revealed
    for (int k = 0; k < 4; k++) {
        EXPECT_EQ(fp16_to_fp32(session.mask_host[1 * max_ctx + 8 + k]), 0.0f)
            << "slot1 row" << k;
    }
    EXPECT_TRUE(std::isinf(fp16_to_fp32(session.mask_host[1 * max_ctx + 12])) &&
                fp16_to_fp32(session.mask_host[1 * max_ctx + 12]) < 0);

    // decode_advance should now include slot 1
    gpt_sovits::t2s_session_decode_advance(session);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 0), 4);   // 3 + 1
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, 1), 5);   // 4 + 1

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, RejectsDecodeOnIdleSlot) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // Slot 0 idle, try to decode on it (n_query=1) — should assert.
    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 0};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    EXPECT_DEATH(gpt_sovits::t2s_session_advance(session, plan, graph), "");

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, RejectsDecodeOnEmptySlot) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // Slot 0 active but n_pos=0, try to decode — should assert.
    test_activate_slot(session, 0, 0);

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {1, 0};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    EXPECT_DEATH(gpt_sovits::t2s_session_advance(session, plan, graph), "");

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SAdvance, RejectsPrefillOnActiveSlot) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;
    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    // Slot 0 active with n_pos=3, try to prefill (n_query>1) — should assert.
    test_activate_slot(session, 0, 3);

    gpt_sovits::t2s_batch_plan plan;
    plan.n_query = {5, 0};

    auto model = create_minimal_model(hparams, backend);
    auto graph = gpt_sovits::t2s_session_build_graph(session, model, plan);
    ASSERT_NE(graph.ctx, nullptr);

    EXPECT_DEATH(gpt_sovits::t2s_session_advance(session, plan, graph), "");

    gpt_sovits::t2s_graph_free(graph);
    gpt_sovits::t2s_session_free(session);
    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

