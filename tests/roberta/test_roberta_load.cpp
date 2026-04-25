// tests/roberta/test_roberta_load.cpp
//
// Tests that the GGUF loader correctly loads RoBERTa model files and populates
// all weight pointers with expected tensor shapes.

#include <gtest/gtest.h>

#include "gpt_sovits/roberta.h"

#include "test_backend.h"

#include <string>

static const std::string kTestDir    = ROBERTA_MODEL_DIR;
static const std::string kModelF32   = kTestDir + "chinese-roberta-wwm-ext-large-f32.gguf";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_weight_completeness(const gpt_sovits::roberta_model_block_weights & w) {
    // Embeddings
    const auto & emb = w.embeddings;
    ASSERT_NE(emb.word_embeddings,       nullptr);
    ASSERT_NE(emb.position_embeddings,   nullptr);
    ASSERT_NE(emb.token_type_embeddings, nullptr);
    ASSERT_NE(emb.layer_norm_w,          nullptr);
    ASSERT_NE(emb.layer_norm_b,          nullptr);

    // All 24 encoder layers
    for (int i = 0; i < 24; i++) {
        const auto & layer = w.encoder.layers[i];
        const auto & attn  = layer.attention;

        EXPECT_NE(attn.q_w,   nullptr) << "layer " << i;
        EXPECT_NE(attn.q_b,   nullptr) << "layer " << i;
        EXPECT_NE(attn.k_w,   nullptr) << "layer " << i;
        EXPECT_NE(attn.k_b,   nullptr) << "layer " << i;
        EXPECT_NE(attn.v_w,   nullptr) << "layer " << i;
        EXPECT_NE(attn.v_b,   nullptr) << "layer " << i;
        EXPECT_NE(attn.out_w, nullptr) << "layer " << i;
        EXPECT_NE(attn.out_b, nullptr) << "layer " << i;

        EXPECT_NE(layer.attn_ln_w,  nullptr) << "layer " << i;
        EXPECT_NE(layer.attn_ln_b,  nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_up_w,   nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_up_b,   nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_down_w, nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_down_b, nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_ln_w,   nullptr) << "layer " << i;
        EXPECT_NE(layer.ffn_ln_b,   nullptr) << "layer " << i;
    }
}

static void check_tensor_shapes(const gpt_sovits::roberta_model_block_weights & w) {
    // Embeddings
    // word_embeddings: PyTorch (21128, 1024) -> ggml ne = {1024, 21128}
    EXPECT_EQ(w.embeddings.word_embeddings->ne[0], 1024);
    EXPECT_EQ(w.embeddings.word_embeddings->ne[1], 21128);

    // position_embeddings: PyTorch (512, 1024) -> ggml ne = {1024, 512}
    EXPECT_EQ(w.embeddings.position_embeddings->ne[0], 1024);
    EXPECT_EQ(w.embeddings.position_embeddings->ne[1],  512);

    // token_type_embeddings: PyTorch (2, 1024) -> ggml ne = {1024, 2}
    EXPECT_EQ(w.embeddings.token_type_embeddings->ne[0], 1024);
    EXPECT_EQ(w.embeddings.token_type_embeddings->ne[1],    2);

    // LayerNorm: {1024}
    EXPECT_EQ(w.embeddings.layer_norm_w->ne[0], 1024);
    EXPECT_EQ(w.embeddings.layer_norm_b->ne[0], 1024);

    // Encoder layer 0 attention Q weight: (1024, 1024) -> ggml ne = {1024, 1024}
    EXPECT_EQ(w.encoder.layers[0].attention.q_w->ne[0], 1024);
    EXPECT_EQ(w.encoder.layers[0].attention.q_w->ne[1], 1024);

    // FFN up: (4096, 1024) -> ggml ne = {1024, 4096}
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[0], 1024);
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[1], 4096);

    // FFN down: (1024, 4096) -> ggml ne = {4096, 1024}
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[0], 4096);
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[1], 1024);
}

// Helper: skip test if model file not found.
#define ASSERT_MODEL_EXISTS(path) do { \
    FILE * f = fopen(path.c_str(), "rb"); \
    if (!f) GTEST_SKIP() << "Model file not found: " << path; \
    fclose(f); \
} while (0)

// ---------------------------------------------------------------------------
// Load tests (f32)
// ---------------------------------------------------------------------------

TEST(RobertaLoad, LoadsSuccessfully) {
    ASSERT_MODEL_EXISTS(kModelF32);
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::roberta_model model{};
    ASSERT_TRUE(gpt_sovits::roberta_model_load(kModelF32, model, backend));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w,   nullptr);
    EXPECT_NE(model.ctx_w,   nullptr);
    gpt_sovits::roberta_model_free(model);
    ggml_backend_free(backend);
}

TEST(RobertaLoad, WeightPointersComplete) {
    ASSERT_MODEL_EXISTS(kModelF32);
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::roberta_model model{};
    ASSERT_TRUE(gpt_sovits::roberta_model_load(kModelF32, model, backend));
    check_weight_completeness(model.weights);
    gpt_sovits::roberta_model_free(model);
    ggml_backend_free(backend);
}

TEST(RobertaLoad, TensorShapesCorrect) {
    ASSERT_MODEL_EXISTS(kModelF32);
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::roberta_model model{};
    ASSERT_TRUE(gpt_sovits::roberta_model_load(kModelF32, model, backend));
    check_tensor_shapes(model.weights);
    gpt_sovits::roberta_model_free(model);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Edge-case tests (no model file needed)
// ---------------------------------------------------------------------------

TEST(RobertaLoad, NonExistentFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);
    gpt_sovits::roberta_model model{};
    EXPECT_FALSE(gpt_sovits::roberta_model_load("/nonexistent/path.gguf", model, backend));
    ggml_backend_free(backend);
}

TEST(RobertaLoad, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::roberta_model model{};
    gpt_sovits::roberta_model_free(model);
}
