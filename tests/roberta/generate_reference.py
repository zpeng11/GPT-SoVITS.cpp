#!/usr/bin/env python3
"""Generate reference outputs for chinese-roberta-wwm-ext-large parity testing.

Runs the HuggingFace BertModel on deterministic token IDs and saves
intermediate and final outputs as raw binary files whose memory
layout matches the ggml convention (ne[0] contiguous).

Usage:
    python generate_reference.py <model_dir> [--output-dir <dir>] [--length <tokens>]

Output files (raw, little-endian):
    ref_input_ids.bin       - input token IDs          {T}        (i32)
    ref_embeddings.bin      - embeddings output         {1024, T}  (f32)
    ref_encoder_layer0.bin  - encoder layer 0 output    {1024, T}  (f32)
    ref_model_output.bin    - final encoder output      {1024, T}  (f32)
"""

import argparse
import os

import numpy as np
import torch


def save_raw_bin(path: str, data: np.ndarray) -> None:
    """Save a numpy array as raw binary."""
    data = np.ascontiguousarray(data)
    with open(path, "wb") as f:
        f.write(data.tobytes())
    size_kb = data.nbytes / 1024
    print(f"  Saved {os.path.basename(path):40s} shape={str(data.shape):20s} dtype={data.dtype} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RoBERTa reference outputs for C++ parity testing")
    parser.add_argument("model_dir",
                        help="Path to HF chinese-roberta-wwm-ext-large directory "
                             "(containing pytorch_model.bin and config.json)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Output directory for reference files (default: cwd)")
    parser.add_argument("--length", "-l", type=int, default=32,
                        help="Number of input tokens (default: 32)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducible input (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate deterministic token IDs
    # ------------------------------------------------------------------
    print(f"Generating input: {args.length} tokens, seed={args.seed}")
    rng = np.random.default_rng(args.seed)
    vocab_size = 21128
    input_ids = rng.integers(1, vocab_size, size=args.length).astype(np.int32)

    save_raw_bin(os.path.join(args.output_dir, "ref_input_ids.bin"), input_ids)

    # ------------------------------------------------------------------
    # 2. Load BertModel (eager attention, no dropout)
    # ------------------------------------------------------------------
    print(f"\nLoading BertModel from {args.model_dir} ...")
    from transformers import BertConfig, BertModel  # noqa: delayed import

    config = BertConfig.from_pretrained(args.model_dir, local_files_only=True)
    config._attn_implementation = "eager"
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0

    model = BertModel.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
    )
    model.eval()
    model = model.float()

    # ------------------------------------------------------------------
    # 3. Forward pass with hidden states
    # ------------------------------------------------------------------
    print("Running forward pass ...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_ids).unsqueeze(0)  # [1, T]
        outputs = model(input_tensor, output_hidden_states=True)

        # embeddings: outputs.hidden_states[0]  [1, T, 1024]
        # encoder layer 0: outputs.hidden_states[1]  [1, T, 1024]
        # final: outputs.last_hidden_state  [1, T, 1024]

    # ------------------------------------------------------------------
    # 4. Convert to ggml memory layout and save
    #
    #    ggml {ne0, ne1} stores ne1 contiguous blocks of ne0 elements.
    #    In numpy C-order that corresponds to shape (ne1, ne0).
    #
    #    PyTorch [1, T, 1024] -> squeeze -> [T, 1024] -> numpy
    #    ggml layout for {1024, T} in numpy is shape (T, 1024).
    # ------------------------------------------------------------------
    print("\nSaving reference outputs ...")

    # Embeddings output
    embeddings = outputs.hidden_states[0].squeeze(0).contiguous().numpy()  # [T, 1024]
    save_raw_bin(os.path.join(args.output_dir, "ref_embeddings.bin"), embeddings)

    # Encoder layer 0 output
    encoder_layer0 = outputs.hidden_states[1].squeeze(0).contiguous().numpy()  # [T, 1024]
    save_raw_bin(os.path.join(args.output_dir, "ref_encoder_layer0.bin"), encoder_layer0)

    # Full model output
    model_out = outputs.last_hidden_state.squeeze(0).contiguous().numpy()  # [T, 1024]
    save_raw_bin(os.path.join(args.output_dir, "ref_model_output.bin"), model_out)

    T = args.length
    print(f"\nSequence length: T = {T}")
    print("Done!")


if __name__ == "__main__":
    main()
