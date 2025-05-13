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

    dates = df['date'].values.astype('datetime64[D]')
    size  = N // args.num_shards

    for k in range(args.num_shards):
        # global slice [S:E)
        S = k * size
        E = N if (k == args.num_shards - 1) else (k + 1) * size

        # compute padded window
        first_date = dates[S]
        last_date  = dates[E-1]
        min_date   = first_date - np.timedelta64(args.window_days, 'D')
        max_date   = last_date  + np.timedelta64(args.window_days, 'D')

        # binary‐search into sorted array
        min_idx = np.searchsorted(dates, min_date, side='left')
        max_idx = np.searchsorted(dates, max_date, side='right') - 1

        # slice out just the padded region
        df_k  = df.iloc[min_idx:max_idx+1].reset_index(drop=True)
        emb_k = emb[min_idx:max_idx+1]

        # compute local offsets
        start_off = S - min_idx
        end_off   = E - min_idx

        # build filenames
        base     = os.path.join(args.out_dir, f"{args.prefix}_shard{k}")
        df_path  = base + "_df.pkl"
        emb_path = base + "_emb.npy"
        cfg_path = base + "_config.json"

        # write them
        print(f"Writing shard {k}: global[{S}:{E}) → padded[{min_idx}:{max_idx+1})")
        df_k.to_pickle(df_path)
        np.save(emb_path, emb_k)
        # cast to int so JSON can handle it
        json.dump({
            "start_offset": int(start_off),
            "end_offset":   int(end_off)
        }, open(cfg_path, "w"), indent=2)

        print("  ", df_path)
        print("  ", emb_path)
        print("  ", cfg_path)

if __name__ == "__main__":
    main()
