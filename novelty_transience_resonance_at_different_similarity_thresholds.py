#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def compute_ntr_variants(
    df: pd.DataFrame,
    emb: torch.Tensor,
    window_days: int,
    thresholds: list,
    save_every: int,
    out_path: str,
    start_idx: int,
    end_idx: int
) -> pd.DataFrame:
    """
    Compute novelty/transience/resonance at various threshold intervals on padded shards.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])  # full datetime
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    dates = df['date'].values  # use numpy array for performance
    n = len(df)

    emb = emb.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = emb.device

    novelty    = {tau: np.full(n, np.nan, dtype=float) for tau in thresholds}
    transience = {tau: np.full(n, np.nan, dtype=float) for tau in thresholds}
    resonance  = {tau: np.full(n, np.nan, dtype=float) for tau in thresholds}

    prog_file    = out_path + ".progress.json"
    partial_file = out_path + ".partial.pkl"
    if os.path.exists(prog_file):
        processed = json.load(open(prog_file)).get("processed_idx", start_idx)
        processed = max(processed, start_idx)
        print(f"Resuming at index {processed}")
    else:
        processed = start_idx

    if processed >= end_idx:
        print("Nothing to do")
        sys.exit(0)

    for i in tqdm(range(processed, end_idx), desc=f"NTR {start_idx}-{end_idx}"):
        date_i = dates[i]
        emb_i  = emb[i]

        mask_b = (dates < date_i) & (dates >= date_i - np.timedelta64(window_days, 'D'))
        idx_b  = np.nonzero(mask_b)[0]
        if idx_b.size:
            emb_b = emb[idx_b]
            sims_b = emb_b @ emb_i
            for tau in thresholds:
                prop_similar = (sims_b >= tau).float().mean().item()
                novelty[tau][i] = 1 - prop_similar

        mask_a = (dates > date_i) & (dates <= date_i + np.timedelta64(window_days, 'D'))
        idx_a  = np.nonzero(mask_a)[0]
        if idx_a.size:
            emb_a = emb[idx_a]
            sims_a = emb_a @ emb_i
            for tau in thresholds:
                prop_similar = (sims_a >= tau).float().mean().item()
                transience[tau][i] = 1 - prop_similar

        for tau in thresholds:
            n0 = novelty[tau][i]
            t0 = transience[tau][i]
            if not np.isnan(n0) and not np.isnan(t0):
                resonance[tau][i] = n0 - t0

        if ((i+1) % save_every == 0) or (i+1 == end_idx):
            json.dump({"processed_idx": i+1}, open(prog_file, "w"))
            part = df.iloc[start_idx:i+1].copy()
            for tau in thresholds:
                part[f'novelty_{tau}']    = novelty[tau][start_idx:i+1]
                part[f'transience_{tau}'] = transience[tau][start_idx:i+1]
                part[f'resonance_{tau}']  = resonance[tau][start_idx:i+1]
            part.to_pickle(partial_file)
            print(f"Checkpoint at {i+1} → {partial_file}")

    out_full = pd.DataFrame({
        'date': df['date'],
        'user_type': df['user_type'],
    })
    for tau in thresholds:
        out_full[f'novelty_{tau}']    = novelty[tau]
        out_full[f'transience_{tau}'] = transience[tau]
        out_full[f'resonance_{tau}']  = resonance[tau]

    result = out_full.iloc[start_idx:end_idx].reset_index(drop=True)
    if os.path.exists(prog_file):    os.remove(prog_file)
    if os.path.exists(partial_file): os.remove(partial_file)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--df",          required=True,
                   help="padded shard DF (.pkl)")
    p.add_argument("--embeddings",  required=True,
                   help="padded shard embeddings (.npy)")
    p.add_argument("--config",      required=True,
                   help="shard config JSON (with start/end offsets)")
    p.add_argument("--out",         required=True,
                   help="where to write results (.pkl)")
    p.add_argument("--window-days", type=int, default=14,
                   help="± days window")
    p.add_argument("--thresholds",  nargs="+", type=float,
                   default=[0.7,0.85,0.90,0.95,0.97],
                   help="cosine-sim cutoffs")
    p.add_argument("--save-every",  type=int, default=1_000_000,
                   help="checkpoint every N docs")
    args = p.parse_args()

    df   = pd.read_pickle(args.df)
    emb  = torch.from_numpy(np.load(args.embeddings)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cfg  = json.load(open(args.config))
    start = int(cfg["start_offset"])
    end   = int(cfg["end_offset"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = compute_ntr_variants(df, emb, args.window_days,
                               args.thresholds, args.save_every,
                               args.out, start, end)
    out.to_pickle(args.out)
    print("Done:", args.out)

if __name__ == "__main__":
    main()
