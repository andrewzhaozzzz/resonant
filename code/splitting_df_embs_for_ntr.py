#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import json

def main():
    p = argparse.ArgumentParser(description="Split a single merged DF with embeddings into N padded shards")
    p.add_argument("--df", required=True, help="Path to merged DF with 'embedding' column and 'date'")
    p.add_argument("--num-shards", type=int, default=8, help="How many shards to create")
    p.add_argument("--window-days", type=int, default=14, help="Days of padding on each side of each shard")
    p.add_argument("--out-dir", required=True, help="Directory to write shard files")
    p.add_argument("--prefix", default="who_leads_model", help="Output filename prefix for each shard")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading merged DataFrame...")
    df = pd.read_pickle(args.df)
    if 'date' not in df.columns or 'embedding' not in df.columns:
        raise ValueError("Input DataFrame must contain 'date' and 'embedding' columns")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        raise ValueError("Null or unparseable dates found in 'date' column")

    df = df.sort_values("date").reset_index(drop=True)
    dates = df['date'].values.astype("datetime64[D]")
    N = len(df)

    print(f"Total documents: {N}")
    shard_indices = np.array_split(np.arange(N), args.num_shards)

    for k, shard_idx in enumerate(shard_indices):
        S = shard_idx[0]
        E = shard_idx[-1] + 1

        first_date = dates[S]
        last_date  = dates[E - 1]
        min_date = first_date - np.timedelta64(args.window_days, 'D')
        max_date = last_date + np.timedelta64(args.window_days, 'D')

        min_idx = np.searchsorted(dates, min_date, side='left')
        max_idx = np.searchsorted(dates, max_date, side='right') - 1
        df_k = df.iloc[min_idx:max_idx + 1].reset_index(drop=True)

        start_offset = S - min_idx
        end_offset   = E - min_idx

        base = os.path.join(args.out_dir, f"{args.prefix}_shard{k}")
        df_path = base + ".pkl"
        cfg_path = base + "_config.json"

        print(f"[Shard {k}] Global range: [{S}:{E}) | Padded range: [{min_idx}:{max_idx+1}] | Offsets: [{start_offset}:{end_offset}]")
        df_k.to_pickle(df_path)
        with open(cfg_path, "w") as f:
            json.dump({"start_offset": int(start_offset), "end_offset": int(end_offset)}, f, indent=2)

        print(f"  → Saved {df_path}")
        print(f"  → Saved {cfg_path}")

if __name__ == "__main__":
    main()
