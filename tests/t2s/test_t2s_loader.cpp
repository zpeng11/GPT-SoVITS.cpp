// tests/t2s/test_t2s_loader.cpp
//
// Tests that the GGUF loader correctly loads T2S model files produced by
// convert_t2s_to_gguf.py and populates all weight pointers with expected
// tensor shapes.

#include <gtest/gtest.h>

#include "gpt_sovits/t2s.h"

#include "test_backend.h"

#include <string>

static const std::string kTestDir = T2S_TEST_DIR;

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
// Tests
// ---------------------------------------------------------------------------

TEST(T2SLoader, LoadF16Model) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    // --- hparams ---
    EXPECT_EQ(model.hparams.embedding_dim, 512u);
    EXPECT_EQ(model.hparams.hidden_dim, 512u);
    EXPECT_EQ(model.hparams.n_head, 16u);
    EXPECT_EQ(model.hparams.n_layer, 24u);
    EXPECT_EQ(model.hparams.vocab_size, 1025u);
    EXPECT_EQ(model.hparams.eos, 1024u);

    // --- extract latent ---
    // ssl_proj: Conv1d(768, 768, kernel=2, stride=2) from top-level ssl_proj.weight
    auto & el = model.weights.extract_latent;
    expect_shape(el.ssl_proj_w, {2, 768, 768});
    expect_shape(el.ssl_proj_b, {768});
    expect_shape(el.codebook,   {768, 1024});

    // --- encoder ---
    auto & enc = model.weights.embed;
    expect_shape(enc.text_embedding, {512, 732});
    expect_shape(enc.bert_proj_w,    {1024, 512});
    expect_shape(enc.bert_proj_b,    {512});
    expect_shape(enc.text_pos_alpha, {1});
    expect_shape(enc.audio_embedding, {512, 1025});
    expect_shape(enc.audio_pos_alpha, {1});

    // --- attention layers ---
    ASSERT_EQ(model.weights.attention.size(), 24u);

    // Layer 0: check all shapes
    auto & a0 = model.weights.attention[0];
    expect_shape(a0.qkv_w,       {512, 1536});
    expect_shape(a0.qkv_b,       {1536});
    expect_shape(a0.out_proj_w,  {512, 512});
    expect_shape(a0.out_proj_b,  {512});
    expect_shape(a0.ln1_w,       {512});
    expect_shape(a0.ln1_b,       {512});
    expect_shape(a0.ffn_up_w,    {512, 2048});
    expect_shape(a0.ffn_up_b,    {2048});
    expect_shape(a0.ffn_down_w,  {2048, 512});
    expect_shape(a0.ffn_down_b,  {512});
    expect_shape(a0.ln2_w,       {512});
    expect_shape(a0.ln2_b,       {512});

    // Spot-check last layer
    auto & a23 = model.weights.attention[23];
    expect_shape(a23.qkv_w, {512, 1536});

    // --- sampler ---
    expect_shape(model.weights.lm_head_w, {512, 1025});

    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SLoader, AllAttentionWeightsNonNull) {
    const std::string path = kTestDir + "models/s1v3-s2Gv2ProPlus-f16.gguf";
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    ASSERT_TRUE(gpt_sovits::t2s_model_load(path, model, backend));

    for (size_t i = 0; i < model.weights.attention.size(); i++) {
        auto & layer = model.weights.attention[i];
        EXPECT_NE(layer.qkv_w, nullptr)       << "layer " << i;
        EXPECT_NE(layer.qkv_b, nullptr)       << "layer " << i;
        EXPECT_NE(layer.out_proj_w, nullptr)  << "layer " << i;
        EXPECT_NE(layer.out_proj_b, nullptr)  << "layer " << i;
        EXPECT_NE(layer.ln1_w, nullptr)       << "layer " << i;
        EXPECT_NE(layer.ln1_b, nullptr)       << "layer " << i;
        EXPECT_NE(layer.ffn_up_w, nullptr)    << "layer " << i;
        EXPECT_NE(layer.ffn_up_b, nullptr)    << "layer " << i;
        EXPECT_NE(layer.ffn_down_w, nullptr)  << "layer " << i;
        EXPECT_NE(layer.ffn_down_b, nullptr)  << "layer " << i;
        EXPECT_NE(layer.ln2_w, nullptr)       << "layer " << i;
        EXPECT_NE(layer.ln2_b, nullptr)       << "layer " << i;
    }

    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}

TEST(T2SLoader, MissingFileReturnsFalse) {
    ggml_backend_t backend = create_test_backend();
    ASSERT_NE(backend, nullptr);

    gpt_sovits::t2s_model model;
    EXPECT_FALSE(gpt_sovits::t2s_model_load("/nonexistent/path.gguf", model, backend));

    gpt_sovits::t2s_model_free(model);
    ggml_backend_free(backend);
}
