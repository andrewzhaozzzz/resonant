#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import json

def main():
    p = argparse.ArgumentParser(
        description="Split a big DF + .npy embeddings into N padded shards"
    )
    p.add_argument("--df",           required=True,
                   help="Path to cleaned df pickle (must have a 'date' column)")
    p.add_argument("--embeddings",   required=True,
                   help="Path to full merged .npy embeddings")
    p.add_argument("--num-shards",   type=int, default=8,
                   help="How many pieces to split into")
    p.add_argument("--window-days",  type=int, default=14,
                   help="Days of padding on each end of every shard")
    p.add_argument("--out-dir",      required=True,
                   help="Directory where shard files will be written")
    p.add_argument("--prefix",       default="who_leads_model",
                   help="Filename prefix for each shard")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- load and sort by date
    print("Loading DF…")
    df = pd.read_pickle(args.df)
    df['date'] = pd.to_datetime(df['date'])
    order = np.argsort(df['date'].values)
    df = df.iloc[order].reset_index(drop=True)

    print("Loading embeddings (as memmap)…")
    emb = np.load(args.embeddings, mmap_mode="r")
    N = len(df)
    if emb.shape[0] != N:
        raise ValueError(f"DF rows ({N}) != emb rows ({emb.shape[0]})")

    print("Assigning embeddings to DataFrame...")
    df["embedding"] = [emb[i] for i in order]

    print("Done. DataFrame shape:", df.shape)
    df.to_pickle("/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df_and_embeddings.pkl")



if __name__ == "__main__":
    main()
