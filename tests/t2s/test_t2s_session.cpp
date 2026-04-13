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

TEST(T2SSession, SlotAllocRelease) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch = 3;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, 32));

    // Allocate all slots.
    std::vector<int> ids;
    for (uint32_t i = 0; i < n_batch; i++) {
        int id = gpt_sovits::t2s_session_slot_alloc(session, 0);
        EXPECT_EQ(id, (int)i);
        EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, id), 0);
        ids.push_back(id);
    }

    // No more slots.
    EXPECT_EQ(gpt_sovits::t2s_session_slot_alloc(session, 0), -1);

    // Release one and re-allocate.
    gpt_sovits::t2s_session_slot_release(session, ids[1]);
    int id = gpt_sovits::t2s_session_slot_alloc(session, 0);
    EXPECT_EQ(id, (int)ids[1]);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, id), 0);

    gpt_sovits::t2s_session_free(session);
    ggml_backend_free(backend);
}

TEST(T2SSession, SlotReuseAfterRelease) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, 2, 16));

    int s0 = gpt_sovits::t2s_session_slot_alloc(session, 0);
    ASSERT_GE(s0, 0);

    // Release and re-alloc — should get the same slot back.
    gpt_sovits::t2s_session_slot_release(session, s0);
    int s1 = gpt_sovits::t2s_session_slot_alloc(session, 0);
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

TEST(T2SSession, SlotAllocMask) {
    ggml_backend_t backend = create_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_hparams hparams;
    gpt_sovits::t2s_session session;

    const uint32_t n_batch   = 2;
    const uint32_t slot_size = 8;
    ASSERT_TRUE(gpt_sovits::t2s_session_init(session, hparams, backend, n_batch, slot_size));

    const int64_t max_ctx = (int64_t)n_batch * slot_size;

    // Allocate slot 0 with 3 prefilled tokens.
    int s0 = gpt_sovits::t2s_session_slot_alloc(session, 3);
    ASSERT_EQ(s0, 0);
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

    // Allocate slot 0 with 2 prefilled tokens, then do 3 decode steps.
    int s0 = gpt_sovits::t2s_session_slot_alloc(session, 2);
    ASSERT_EQ(s0, 0);

    for (int step = 0; step < 3; step++) {
        gpt_sovits::t2s_session_slot_decode_step(session, s0);
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

    // Allocate, decode, then release.
    int s0 = gpt_sovits::t2s_session_slot_alloc(session, 2);
    ASSERT_EQ(s0, 0);
    gpt_sovits::t2s_session_slot_decode_step(session, s0);
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

    // Allocate with 3 tokens, decode 1 step, release, then re-alloc with 1 token.
    int s0 = gpt_sovits::t2s_session_slot_alloc(session, 3);
    ASSERT_EQ(s0, 0);
    gpt_sovits::t2s_session_slot_decode_step(session, s0);
    gpt_sovits::t2s_session_slot_release(session, s0);

    int s1 = gpt_sovits::t2s_session_slot_alloc(session, 1);
    ASSERT_EQ(s1, 0);
    EXPECT_EQ(gpt_sovits::t2s_session_slot_n_pos(session, s1), 1);

    // Column 0: only row 0 should be 0 (from new n_pos=1), rest -inf.
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
