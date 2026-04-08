// tests/hubert/test_hubert_load.cpp
//
// Tests that the GGUF loader correctly loads HuBERT model files and populates
// all weight pointers with expected tensor shapes.

#include <gtest/gtest.h>

#include "gpt_sovits/gpt_sovits.h"

#include <string>

static const std::string kTestDir  = HUBERT_TEST_DIR;
static const std::string kModelF32 = kTestDir + "models/chinese-hubert-base-f32.gguf";
// ---------------------------------------------------------------------------
// Basic load / free
// ---------------------------------------------------------------------------

TEST(HubertLoad, F32ModelLoadsSuccessfully) {
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(kModelF32, model));
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);
    gpt_sovits::hubert_model_free(model);
}

TEST(HubertLoad, NonExistentFileReturnsFalse) {
    gpt_sovits::hubert_model model{};
    EXPECT_FALSE(gpt_sovits::hubert_model_load("/nonexistent/path.gguf", model));
}

TEST(HubertLoad, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::hubert_model model{};
    // Should not crash or leak.
    gpt_sovits::hubert_model_free(model);
}

// ---------------------------------------------------------------------------
// Weight pointer completeness
// ---------------------------------------------------------------------------

TEST(HubertLoad, ModelStructureIsMappedCorrectly) {
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(kModelF32, model));

    const auto & w = model.weights;

    // Feature encoder: 7 conv kernels + GroupNorm
    for (int i = 0; i < 7; i++) {
        EXPECT_NE(w.feature_encoder.conv_w[i], nullptr) << "conv_w[" << i << "]";
    }
    EXPECT_NE(w.feature_encoder.conv0_norm_w, nullptr);
    EXPECT_NE(w.feature_encoder.conv0_norm_b, nullptr);

    // Feature projection
    EXPECT_NE(w.feature_projection.layer_norm_w, nullptr);
    EXPECT_NE(w.feature_projection.layer_norm_b, nullptr);
    EXPECT_NE(w.feature_projection.projection_w, nullptr);
    EXPECT_NE(w.feature_projection.projection_b, nullptr);

    // Encoder positional conv
    EXPECT_NE(w.encoder.pos_conv.weight_v, nullptr);
    EXPECT_NE(w.encoder.pos_conv.weight_g, nullptr);
    EXPECT_NE(w.encoder.pos_conv.bias, nullptr);

    // Encoder layer norm
    EXPECT_NE(w.encoder.layer_norm_w, nullptr);
    EXPECT_NE(w.encoder.layer_norm_b, nullptr);

    // All 12 encoder layers (spot-check every field)
    for (int i = 0; i < 12; i++) {
        const auto & l = w.encoder.layers[i];
        EXPECT_NE(l.attention.q_proj_w,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.q_proj_b,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.k_proj_w,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.k_proj_b,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.v_proj_w,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.v_proj_b,   nullptr) << "layer " << i;
        EXPECT_NE(l.attention.out_proj_w, nullptr) << "layer " << i;
        EXPECT_NE(l.attention.out_proj_b, nullptr) << "layer " << i;
        EXPECT_NE(l.ln1_w,               nullptr) << "layer " << i;
        EXPECT_NE(l.ln1_b,               nullptr) << "layer " << i;
        EXPECT_NE(l.ffn_up_w,            nullptr) << "layer " << i;
        EXPECT_NE(l.ffn_up_b,            nullptr) << "layer " << i;
        EXPECT_NE(l.ffn_down_w,          nullptr) << "layer " << i;
        EXPECT_NE(l.ffn_down_b,          nullptr) << "layer " << i;
        EXPECT_NE(l.ln2_w,               nullptr) << "layer " << i;
        EXPECT_NE(l.ln2_b,               nullptr) << "layer " << i;
    }

    gpt_sovits::hubert_model_free(model);
}

// ---------------------------------------------------------------------------
// Tensor shapes
// ---------------------------------------------------------------------------

TEST(HubertLoad, TensorShapesCorrect) {
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(kModelF32, model));

    const auto & w = model.weights;

    // Feature encoder conv0: PyTorch (512, 1, 10) -> no transpose -> ggml ne = {10, 1, 512}
    EXPECT_EQ(w.feature_encoder.conv_w[0]->ne[0],  10);
    EXPECT_EQ(w.feature_encoder.conv_w[0]->ne[1],   1);
    EXPECT_EQ(w.feature_encoder.conv_w[0]->ne[2], 512);

    // Feature encoder conv1-4: PyTorch (512, 512, 3) -> ggml ne = {3, 512, 512}
    for (int i = 1; i <= 4; i++) {
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[0],   3) << "conv_w[" << i << "]";
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[1], 512) << "conv_w[" << i << "]";
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[2], 512) << "conv_w[" << i << "]";
    }

    // Feature encoder conv5-6: PyTorch (512, 512, 2) -> ggml ne = {2, 512, 512}
    for (int i = 5; i <= 6; i++) {
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[0],   2) << "conv_w[" << i << "]";
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[1], 512) << "conv_w[" << i << "]";
        EXPECT_EQ(w.feature_encoder.conv_w[i]->ne[2], 512) << "conv_w[" << i << "]";
    }

    // GroupNorm: {512}
    EXPECT_EQ(w.feature_encoder.conv0_norm_w->ne[0], 512);
    EXPECT_EQ(w.feature_encoder.conv0_norm_b->ne[0], 512);

    // Feature projection: LN {512}, projection PyTorch (768,512) -> ggml ne = {512, 768}
    EXPECT_EQ(w.feature_projection.layer_norm_w->ne[0], 512);
    EXPECT_EQ(w.feature_projection.projection_w->ne[0], 512);
    EXPECT_EQ(w.feature_projection.projection_w->ne[1], 768);
    EXPECT_EQ(w.feature_projection.projection_b->ne[0], 768);

    // Positional conv weight_v: PyTorch (768,48,128) -> no transpose -> ggml ne = {128, 48, 768}
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[0], 128);
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[1],  48);
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[2], 768);

    // Encoder layer 0 attention: Q/K/V/out PyTorch (768,768) -> ggml ne = {768, 768}
    EXPECT_EQ(w.encoder.layers[0].attention.q_proj_w->ne[0], 768);
    EXPECT_EQ(w.encoder.layers[0].attention.q_proj_w->ne[1], 768);

    // FFN up: PyTorch (3072,768) -> ggml ne = {768, 3072}
    // FFN down: PyTorch (768,3072) -> ggml ne = {3072, 768}
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[0],   768);
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[1],  3072);
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[0], 3072);
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[1],  768);

    gpt_sovits::hubert_model_free(model);
}
