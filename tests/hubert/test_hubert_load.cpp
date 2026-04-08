// tests/hubert/test_hubert_load.cpp
//
// Tests that the GGUF loader correctly loads HuBERT model files and populates
// all weight pointers with expected tensor shapes, across all quantization
// types (f32, f16, q8, q5, q4).

#include <gtest/gtest.h>

#include "gpt_sovits/gpt_sovits.h"

#include <string>
#include <vector>

static const std::string kTestDir = HUBERT_TEST_DIR;

// ---------------------------------------------------------------------------
// Model paths
// ---------------------------------------------------------------------------

struct ModelVariant {
    std::string name;
    std::string path;
};

static const std::vector<ModelVariant> kModelVariants = {
    {"F32", kTestDir + "models/chinese-hubert-base-f32.gguf"},
    {"F16", kTestDir + "models/chinese-hubert-base-f16.gguf"},
    {"Q8",  kTestDir + "models/chinese-hubert-base-q8.gguf"},
    {"Q5",  kTestDir + "models/chinese-hubert-base-q5.gguf"},
    {"Q4",  kTestDir + "models/chinese-hubert-base-q4.gguf"},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Verify that every expected weight pointer is non-null for a loaded model.
static void check_weight_completeness(const gpt_sovits::hubert_model_block_weights & w) {
    // Feature encoder: 7 conv kernels + GroupNorm
    for (int i = 0; i < 7; i++) {
        ASSERT_NE(w.feature_encoder.conv_w[i], nullptr) << "conv_w[" << i << "]";
    }
    ASSERT_NE(w.feature_encoder.conv0_norm_w, nullptr);
    ASSERT_NE(w.feature_encoder.conv0_norm_b, nullptr);

    // Feature projection
    ASSERT_NE(w.feature_projection.layer_norm_w, nullptr);
    ASSERT_NE(w.feature_projection.layer_norm_b, nullptr);
    ASSERT_NE(w.feature_projection.projection_w, nullptr);
    ASSERT_NE(w.feature_projection.projection_b, nullptr);

    // Encoder positional conv
    ASSERT_NE(w.encoder.pos_conv.weight_v, nullptr);
    ASSERT_NE(w.encoder.pos_conv.weight_g, nullptr);
    ASSERT_NE(w.encoder.pos_conv.bias, nullptr);

    // Encoder layer norm
    ASSERT_NE(w.encoder.layer_norm_w, nullptr);
    ASSERT_NE(w.encoder.layer_norm_b, nullptr);

    // All 12 encoder layers
    for (int i = 0; i < 12; i++) {
        const auto & l = w.encoder.layers[i];
        ASSERT_NE(l.attention.q_proj_w,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.q_proj_b,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.k_proj_w,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.k_proj_b,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.v_proj_w,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.v_proj_b,   nullptr) << "layer " << i;
        ASSERT_NE(l.attention.out_proj_w, nullptr) << "layer " << i;
        ASSERT_NE(l.attention.out_proj_b, nullptr) << "layer " << i;
        ASSERT_NE(l.ln1_w,               nullptr) << "layer " << i;
        ASSERT_NE(l.ln1_b,               nullptr) << "layer " << i;
        ASSERT_NE(l.ffn_up_w,            nullptr) << "layer " << i;
        ASSERT_NE(l.ffn_up_b,            nullptr) << "layer " << i;
        ASSERT_NE(l.ffn_down_w,          nullptr) << "layer " << i;
        ASSERT_NE(l.ffn_down_b,          nullptr) << "layer " << i;
        ASSERT_NE(l.ln2_w,               nullptr) << "layer " << i;
        ASSERT_NE(l.ln2_b,               nullptr) << "layer " << i;
    }
}

// Verify tensor shapes.  For quantized weights, ne[0] reflects the logical
// (float) dimension — ggml stores the quantized bytes but keeps the original
// shape in the GGUF header.
static void check_tensor_shapes(const gpt_sovits::hubert_model_block_weights & w) {
    // Feature encoder conv0: PyTorch (512, 1, 10) -> ggml ne = {10, 1, 512}
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

    // Feature projection: LN {512}, projection (768,512) -> ggml ne = {512, 768}
    EXPECT_EQ(w.feature_projection.layer_norm_w->ne[0], 512);
    EXPECT_EQ(w.feature_projection.projection_w->ne[0], 512);
    EXPECT_EQ(w.feature_projection.projection_w->ne[1], 768);
    EXPECT_EQ(w.feature_projection.projection_b->ne[0], 768);

    // Positional conv weight_v: (768,48,128) -> ggml ne = {128, 48, 768}
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[0], 128);
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[1],  48);
    EXPECT_EQ(w.encoder.pos_conv.weight_v->ne[2], 768);

    // Encoder layer 0 attention: Q/K/V/out (768,768) -> ggml ne = {768, 768}
    EXPECT_EQ(w.encoder.layers[0].attention.q_proj_w->ne[0], 768);
    EXPECT_EQ(w.encoder.layers[0].attention.q_proj_w->ne[1], 768);

    // FFN up: (3072,768) -> ggml ne = {768, 3072}
    // FFN down: (768,3072) -> ggml ne = {3072, 768}
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[0],   768);
    EXPECT_EQ(w.encoder.layers[0].ffn_up_w->ne[1],  3072);
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[0], 3072);
    EXPECT_EQ(w.encoder.layers[0].ffn_down_w->ne[1],  768);
}

// ---------------------------------------------------------------------------
// Parameterized load test
// ---------------------------------------------------------------------------

class HubertLoadAll : public ::testing::TestWithParam<ModelVariant> {};

TEST_P(HubertLoadAll, LoadsSuccessfully) {
    const auto & variant = GetParam();
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(variant.path, model))
        << "Failed to load " << variant.path;
    EXPECT_NE(model.backend, nullptr);
    EXPECT_NE(model.buf_w, nullptr);
    EXPECT_NE(model.ctx_w, nullptr);
    gpt_sovits::hubert_model_free(model);
}

TEST_P(HubertLoadAll, WeightPointersComplete) {
    const auto & variant = GetParam();
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(variant.path, model))
        << "Failed to load " << variant.path;
    check_weight_completeness(model.weights);
    gpt_sovits::hubert_model_free(model);
}

TEST_P(HubertLoadAll, TensorShapesCorrect) {
    const auto & variant = GetParam();
    gpt_sovits::hubert_model model{};
    ASSERT_TRUE(gpt_sovits::hubert_model_load(variant.path, model))
        << "Failed to load " << variant.path;
    check_tensor_shapes(model.weights);
    gpt_sovits::hubert_model_free(model);
}

INSTANTIATE_TEST_SUITE_P(
    HubertLoadAll,
    HubertLoadAll,
    ::testing::ValuesIn(kModelVariants),
    [](const ::testing::TestParamInfo<ModelVariant> & info) {
        return info.param.name;
    });

// ---------------------------------------------------------------------------
// Edge-case tests (f32 only)
// ---------------------------------------------------------------------------

TEST(HubertLoad, NonExistentFileReturnsFalse) {
    gpt_sovits::hubert_model model{};
    EXPECT_FALSE(gpt_sovits::hubert_model_load("/nonexistent/path.gguf", model));
}

TEST(HubertLoad, FreeOnDefaultInitializedModelIsSafe) {
    gpt_sovits::hubert_model model{};
    gpt_sovits::hubert_model_free(model);
}
