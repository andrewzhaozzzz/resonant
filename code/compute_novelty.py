import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def compute_novelty(
    df, 
    embs_t,
    dates,
    user_codes, # ?
    start_idx,
    end_idx,
    prog_file,
    partial_file,
    taus = np.full(N, np.nan, dtype=np.float32),
    window_days = 14,
    save_every = 10000
):
    """
    For each post i in [start_idx, end_idx), find raw tauᵢ = max cosine(emb_i, emb_j) over all
    j in the prior window.  If there are no prior posts, tauᵢ = np.nan.  Store raw tauᵢ in taus[i].
    Checkpoint every `save_every` rows by writing a partial pickled DataFrame that includes
    the 'min_tau' column (which is just the raw tau up to that point).
    """
    prog = json.load(open(prog_file)) if os.path.exists(prog_file) else {}
    window_delta = np.timedelta64(window_days, 'D')
    left = 0
    right = start_idx

    for i in tqdm(range(start_idx, end_idx), desc="novelty pass"):
        d0 = dates[i]

        # advance left pointer so that dates[left] >= d0 - window_delta
        min_date = d0 - window_delta
        while left < i and dates[left] < min_date:
            left += 1

        # advance right pointer so that dates[right] < d0
        while right < i and dates[right] < d0:
            right += 1

        # now all posts j in [left, right) are within the prior window
        if left >= right:
            tau_raw = np.nan
        else:
            idxs = np.arange(left, right)
            sims = (embs_t[idxs] @ embs_t[i]).cpu().numpy()
            tau_raw = sims.max()

        taus[i] = tau_raw

        # checkpoint JSON + partial .pkl
        if ((i + 1) % save_every == 0) or (i + 1 == end_idx):
            prog["novelty_idx"] = i + 1
            with open(prog_file, 'w') as f:
                json.dump(prog, f)

            part = df.iloc[:i + 1].copy()
            part["min_tau"] = taus[:i + 1]
            part.to_pickle(partial_file)
            
