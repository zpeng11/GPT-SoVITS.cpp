# Chinese HuBERT-Base 模型架构详解

## 模型概况

`chinese-hubert-base` 是一个基于 HuBERT 架构的语音自监督预训练模型，模型类型为 `HubertModel`（纯 encoder，无 CTC head），权重以 float16 存储。

- 总参数量: **94,371,712 (~94.4M)**
- 模型文件大小: float16 ~180 MB, float32 ~360 MB
- 下采样率: 320x (20ms@16kHz)

## 整体流水线

```
Raw Waveform → Feature Encoder (7层CNN) → Feature Projection → Positional Conv Embedding → Transformer Encoder (12层) → Hidden States
```

## 模块详解

### 1. Feature Encoder（特征编码器）

7 层 1D 卷积，将原始波形编码为特征序列。`conv_bias=false`，所有卷积层无 bias。

| 层 | 输入维度 | 输出维度 | kernel | stride | 参数量 |
|----|---------|---------|--------|--------|--------|
| 0  | 1       | 512     | 10     | 5      | 6,144  |
| 1  | 512     | 512     | 3      | 2      | 786,432 |
| 2  | 512     | 512     | 3      | 2      | 786,432 |
| 3  | 512     | 512     | 3      | 2      | 786,432 |
| 4  | 512     | 512     | 3      | 2      | 786,432 |
| 5  | 512     | 512     | 2      | 2      | 524,288 |
| 6  | 512     | 512     | 2      | 2      | 524,288 |

- **第 0 层**: Conv1d(1, 512, k=10, bias=False) + GroupNorm(512) → 5,120 + 1,024 = 6,144
- **第 1-4 层**: Conv1d(512, 512, k=3, bias=False) × 4 → 3,145,728
- **第 5-6 层**: Conv1d(512, 512, k=2, bias=False) × 2 → 1,048,576
- **归一化**: GroupNorm（仅第 0 层），第 1-6 层无归一化
- **激活函数**: GELU
- **总下采样率**: 5 × 2^6 = 320，即每 320 个采样点（20ms@16kHz）输出一帧特征
- **输出形状**: `(batch, seq_len/320, 512)`
- **子模块参数量**: **3,676,160**

权重 key 格式: `feature_extractor.conv_layers.{0-6}.conv.weight`，第 0 层额外有 `feature_extractor.conv_layers.0.layer_norm.weight/bias`

### 2. Feature Projection（特征投影）

```
Feature Encoder output (512) → LayerNorm(512) → Linear(512 → 768) → Dropout(0.0)
```

| 层 | 计算 | 参数量 |
|----|------|--------|
| LayerNorm(512) | 512 + 512 | 1,024 |
| Linear(512 → 768) | 512 × 768 + 768 | 393,984 |

- **子模块参数量**: **395,008**

权重 key:
- `feature_projection.layer_norm.weight/bias` — LayerNorm(512)
- `feature_projection.projection.weight/bias` — Linear(768, 512)

### 3. Positional Conv Embedding（卷积位置编码）

```
Conv1d(768, 768, kernel_size=128, groups=16) + weight_norm + GELU
```

| 参数 | 计算 | 参数量 |
|------|------|--------|
| weight_v (Conv权重) | 768 × 48 × 128 | 4,718,592 |
| weight_g (norm缩放) | 1 × 1 × 128 | 128 |
| bias | 768 | 768 |

- `groups=16`，即每个分组处理 768/16 = 48 个通道
- 使用 weight normalization（拆分为 weight_g 和 weight_v）
- **子模块参数量**: **4,719,488**

权重 key:
- `encoder.pos_conv_embed.conv.weight_v` — (768, 48, 128)
- `encoder.pos_conv_embed.conv.weight_g` — (1, 1, 128)
- `encoder.pos_conv_embed.conv.bias` — (768,)

### 4. Encoder LayerNorm（编码器顶层归一化）

在 Transformer Encoder 输出后还有一个最终的 LayerNorm。

| 参数 | 参数量 |
|------|--------|
| LayerNorm(768) | 1,536 |

权重 key: `encoder.layer_norm.weight/bias`

### 5. Transformer Encoder（核心编码器）

**12 层** `HubertEncoderLayer`，标准 **post-norm** 模式（`do_stable_layer_norm=false`）。

每层结构:

```
x → Multi-Head Self-Attention → Dropout → Residual Add → LayerNorm(768)
  → FFN (768→3072→768, GELU) → Dropout → Residual Add → LayerNorm(768)
```

#### 每层参数明细

| 子模块 | 层 | 计算 | 参数量 |
|--------|-----|------|--------|
| **Attention** | | | |
| q_proj | Linear(768→768) | 768×768 + 768 | 590,592 |
| k_proj | Linear(768→768) | 768×768 + 768 | 590,592 |
| v_proj | Linear(768→768) | 768×768 + 768 | 590,592 |
| out_proj | Linear(768→768) | 768×768 + 768 | 590,592 |
| layer_norm | LayerNorm(768) | 768 + 768 | 1,536 |
| **Feed-Forward** | | | |
| intermediate_dense | Linear(768→3072) | 768×3072 + 3072 | 2,362,368 |
| output_dense | Linear(3072→768) | 3072×768 + 768 | 2,360,064 |
| final_layer_norm | LayerNorm(768) | 768 + 768 | 1,536 |
| **每层合计** | | | **7,087,872** |

#### Attention 细节
- 注意力头数: 12（head_dim = 768/12 = 64）
- Attention Dropout: 0.1

#### FFN 细节
- 中间维度: 3072（4 × hidden_size）
- 激活函数: GELU
- Hidden Dropout: 0.1, Activation Dropout: 0.1

#### LayerNorm
- eps = 1e-5
- 模式: post-norm（attention/FFN 后做 LayerNorm）

#### 权重 key 格式（以第 i 层为例）

```
encoder.layers.{i}.attention.q_proj.weight/bias       — (768, 768) / (768,)
encoder.layers.{i}.attention.k_proj.weight/bias       — (768, 768) / (768,)
encoder.layers.{i}.attention.v_proj.weight/bias       — (768, 768) / (768,)
encoder.layers.{i}.attention.out_proj.weight/bias     — (768, 768) / (768,)
encoder.layers.{i}.layer_norm.weight/bias             — (768,) / (768,)
encoder.layers.{i}.feed_forward.intermediate_dense.weight/bias — (3072, 768) / (3072,)
encoder.layers.{i}.feed_forward.output_dense.weight/bias     — (768, 3072) / (768,)
encoder.layers.{i}.final_layer_norm.weight/bias       — (768,) / (768,)
```

- **12 层总计参数量**: 12 × 7,087,872 = **85,054,464**

### 6. Masked Spec Embed（掩码嵌入）

训练时用于 SpecAugment 的可学习掩码 embedding，推理时不使用。

- 形状: (768,)
- 参数量: **768**

权重 key: `masked_spec_embed`

## 参数量汇总

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Feature Encoder (7层CNN) | 3,676,160 | 3.9% |
| Feature Projection | 395,008 | 0.4% |
| Pos Conv Embed | 4,719,488 | 5.0% |
| Encoder LayerNorm | 1,536 | 0.0% |
| Transformer Encoder (12层) | 85,054,464 | 90.1% |
| Masked Spec Embed | 768 | 0.0% |
| **总计** | **94,371,712** | **100%** |

## 配置参数速查表

| 参数 | 值 |
|-----|-----|
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| head_dim | 64 |
| intermediate_size (FFN) | 3072 |
| CNN 特征维度 | 512 |
| CNN 层数 | 7 |
| 下采样率 | 320 (20ms) |
| 激活函数 | GELU |
| vocab_size | 32 |
| LayerNorm eps | 1e-5 |
| dropout | 0.1 |
| attention_dropout | 0.1 |
| activation_dropout | 0.1 |
| feat_proj_dropout | 0.0 |
| layerdrop | 0.1 |
| do_stable_layer_norm | false (post-norm) |
| conv_bias | false |
| feat_extract_norm | group |
| num_conv_pos_embeddings | 128 |
| num_conv_pos_embedding_groups | 16 |

## 在 GPT-SoVITS 中的用途

在 GPT-SoVITS 流水线中，chinese-hubert-base 作为语音特征提取器:

1. 输入原始音频波形
2. 经过 Feature Encoder + Feature Projection 得到 768 维特征序列
3. 送入 Transformer Encoder 提取深层语音表示
4. 该表示送入后续的 SoVITS / T2S 模型使用
