"""Regenerate tiny, deterministic torch.save fixtures; requires PyTorch."""

from collections import OrderedDict
from pathlib import Path

import torch


def main():
    output = Path(__file__).resolve().parent / "fixtures"
    output.mkdir(exist_ok=True)
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    state = OrderedDict([
        ("weight", base),
        ("bias", torch.arange(4, dtype=torch.float32)),
        ("transposed", base.t()),
        ("sliced", base[1::2, 1::2]),
        ("broadcast", base[1, :4].expand(3, 4)),
        ("scalar", torch.tensor(7.5)),
        ("empty", torch.empty(2, 0, 3)),
        ("half", torch.tensor([0.5, -1.25], dtype=torch.float16)),
        ("double", torch.tensor([1.125, -2.5], dtype=torch.float64)),
        ("long", torch.tensor([2**40, -3], dtype=torch.int64)),
        ("int", torch.tensor([123, -45], dtype=torch.int32)),
        ("short", torch.tensor([12, -3], dtype=torch.int16)),
        ("byte", torch.tensor([0, 255], dtype=torch.uint8)),
        ("bool", torch.tensor([True, False])),
    ])
    torch.save(state, output / "bare.pt")
    torch.save({"weight": state, "config": {"version": 2}, "info": "fixture"},
               output / "wrapped.pt")
    torch.save({}, output / "empty.pt")
    torch.save({"weight": {}, "config": {"version": 2}}, output / "empty_wrapped.pt")

    linear = torch.arange(192, dtype=torch.float32).sub(96).div(32).reshape(64, 3).t()
    torch.save({
        "encoder.layers.0.attention.q_proj.weight": linear,
        "encoder.layers.0.attention.q_proj.bias": torch.tensor([0.5, -1.25, 2.0]),
        "feature_extractor.conv_layers.0.conv.weight": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "encoder.pos_conv_embed.conv.weight_g": torch.arange(4, dtype=torch.float32).reshape(1, 1, 4),
        "masked_spec_embed": torch.ones(3),
    }, output / "hubert.pt")
    torch.save({
        "bert.encoder.layer.0.attention.self.query.weight": linear,
        "bert.encoder.layer.0.attention.self.query.bias": torch.tensor([0.5, -1.25, 2.0]),
        "bert.embeddings.word_embeddings.weight": torch.arange(64, dtype=torch.float32).reshape(2, 32) / 16,
        "bert.embeddings.position_ids": torch.arange(32).reshape(1, 32),
        "bert.pooler.dense.weight": torch.ones(3, 3),
        "cls.predictions.bias": torch.ones(3),
    }, output / "roberta.pt")
    print(f"Generated fixtures in {output} with torch {torch.__version__}")


if __name__ == "__main__":
    main()
