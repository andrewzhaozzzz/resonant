#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    toks = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(toks.size()).float()
    return torch.sum(toks * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--df",           required=True,
                   help="Path to cleaned df pickle (same order as embeddings).")
    p.add_argument("--shard-dir",    required=True,
                   help="Directory containing *_shard0_embeddings.npy, *_shard1_embeddings.npy")
    p.add_argument("--save-name",    required=True,
                   help="Base name of shards, e.g. 'who_leads_model_shard'")
    p.add_argument("--model-dir",    required=True,
                   help="Directory where model is saved")
    p.add_argument("--tokenizer-dir",required=True,
                   help="Directory where tokenizer is saved (usually same as model-dir)")
    p.add_argument("--full-output",  required=False,
                   help="If provided, will save merged embeddings here (.npy)")
    p.add_argument("--n-samples",    type=int, default=10,
                   help="How many random docs to re-embed for testing")
    p.add_argument("--tol",          type=float, default=1e-5,
                   help="Tolerance for float comparison")
    p.add_argument("--max-length",   type=int, default=128,
                   help="Max token length when re-embedding")
    args = p.parse_args()

    # 1) Load original DataFrame
    print("Loading DataFrame from", args.df)
    df = pd.read_pickle(args.df)
    total_docs = len(df)
    print("→ Found", total_docs, "documents in the DF")

    # 2) Load model & tokenizer to get hidden_size
    print("Loading model from", args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    hidden_size = model.config.hidden_size
    print(f"Model hidden_size = {hidden_size}")

    # 3) Locate shard files
    shard_files = sorted([
        f for f in os.listdir(args.shard_dir)
        if f.startswith(args.save_name) and f.endswith("_embeddings.npy")
    ])
    if not shard_files:
        raise RuntimeError("No shard embeddings found in " + args.shard_dir)
    print("Found shards:", shard_files)

    # 4) Memory-map and stack shards
    shards = []
    for fn in shard_files:
        path = os.path.join(args.shard_dir, fn)
        size_bytes = os.path.getsize(path)
        num_floats = size_bytes // 4
        rows = num_floats // hidden_size
        if rows * hidden_size != num_floats:
            raise ValueError(f"Shard {fn} size {size_bytes} not divisible by {hidden_size*4}")
        print(f"Memmapping {fn}: shape = ({rows}, {hidden_size})")
        shards.append(np.memmap(path, dtype='float32', mode='r', shape=(rows, hidden_size)))
    embeddings = np.vstack(shards)
    print(f"→ Merged embeddings shape: {embeddings.shape}")

    # 5) Optionally save the full .npy
    if args.full_output:
        print("Saving merged embeddings to", args.full_output)
        np.save(args.full_output, embeddings)

    # 6) Sanity check length
    if embeddings.shape[0] != total_docs:
        raise AssertionError(
            f"Embeddings rows ({embeddings.shape[0]}) != df rows ({total_docs})"
        )

    # 7) Spot-check random samples
    print("Spot-checking embeddings against model re-encodings")
    rng = np.random.RandomState(42)
    sample_idxs = rng.choice(total_docs, size=min(args.n_samples, total_docs), replace=False)
    for idx in sample_idxs:
        text = df.iloc[idx]["post_text"]
        enc = tokenizer(text,
                        truncation=True,
                        padding=True,
                        max_length=args.max_length,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        new_emb = mean_pooling(out, enc["attention_mask"])
        new_emb = F.normalize(new_emb, p=2, dim=1).cpu().numpy()[0]
        saved_emb = embeddings[idx]
        diff = np.abs(saved_emb - new_emb).max()
        if diff > args.tol:
            raise AssertionError(f"Mismatch at idx {idx}: max diff {diff:.2e} > tol {args.tol}")
        print(f"✔ idx {idx} match (max diff {diff:.2e})")

    print("All checks passed. Embeddings aligned correctly.")

if __name__ == "__main__":
    main()
