import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def compute_resonance(
    df,
    embedding_path,
    taus,                            # raw tauᵢ from compute_novelty
    start_idx,
    end_idx,
    prog_file,
    partial_file,
    dataset_path = False, 
    date_col = "date",
    user_col = "user_type",
    window_days = 14,
    min_tau = 0.7,
    save_every = 1000000,
    group_by_users = True
):
    """
    For each post i in [start_idx, end_idx):
      1. Look up raw tauᵢ = taus[i].  If np.isnan(tauᵢ), skip (no prior).
      2. Define tau_threshold = max(tauᵢ, min_tau).
      3. Let window = all posts j with date_j ∈ (date_i, date_i + window_days].
         - total_posts_after[i] = count of all such j
         - total_counts_per_ut[i, u] = count of those j with user_code == u
      4. Among those j, find “resonant” j where sim(emb_j, emb_i) > tau_threshold.
         - num_resonant_posts[i] = total resonant j
         - resonant_counts_per_ut[i, u] = count of resonant j with user_code == u
         - impact_per_ut[i, u] = sum_{j resonant with ut == u} (sim(emb_j, emb_i) – tauᵢ)
      5. overall_impact[i] = sum across all u of impact_per_ut[i, u].
      6. Every `save_every`, checkpoint a partial DataFrame with columns:
         'min_tau', 'total_posts_after', 'num_resonant_posts', 'overall_impact', and
         per‐user‐type columns 'total_posts_after_<ut>', 'resonant_posts_<ut>', 'impact_<ut>'.

    Parameters
    ----------
    df : string, DataFrame
        If dataset_path is True, then it should be a string indicating the path of the pickle file containing the original dataset.
        If dataset_path is False, then it should be the original dataset in DataFrame form.

    embedding_path : string
        The path directory of the memmap npy file containing calculated embeddings.

    taus : array_like
        The min tau calculated by compute_novelty. Should be the min tau column in the compute_novelty output DataFrame.

    start_idx : int
        The row index of the first document in dataset to consider in novelty calculation. Note that the first post might have np.nan min tau, but could affect following posts.

    end_idx : int
        The row index of the last document in dataset to consider in novelty calculation.

    prog_file : string
        A self-selected path directory for saving and reading progress file. Used to record the progress of resonance calculation, and resume from the latest progress.

    partial_file : string
        A self-selected path directory for saving min tau results. The file would be the original dataset plus additional columns with impact information.

    date_col : string, optional
        The name of the dataset column that contains date information.

    user_col : string, optional
        The name of the dataset column that contains user group types information.

    window_days : int, optional
        The number of days to consider in the prior window. Documents with an earlier date than the date of current document - window_days would not be considered in novelty calculation.

    min_tau : scalar, optional
        For each document, if its calculated min tau is lower than the specified min_tau, use the specified min_tau instead when calculating resonance.

    save_every : int, optional
        If specified, save the min tau information to the partial file when every save_every row has been processed.

    group_by_users : bool, optional
        If True, resonance and impact would be calculated within user groups.
        If False, resonance and impact would be calculated among all user groups.
    
    """
    if dataset_path == True:
        df = pd.read_pickle(df)
    date_raw = list(df[date_col])
    date_np = [np.datetime64(i, "D") for i in date_raw]
    dates = np.array(date_np)

    ut_list = list(df[user_col].unique())
    ut_to_code = {ut: i for i, ut in enumerate(ut_list)}
    user_codes = np.array([ut_to_code[u] for u in df[user_col]], dtype=int)
    
    prog = json.load(open(prog_file)) if os.path.exists(prog_file) else {}
    if group_by_users == True:
        n_ut = len(ut_list)
    window_delta = np.timedelta64(window_days, 'D')
    right = start_idx
    left = start_idx

    N = len(df)

    fp = np.memmap(embedding_path, dtype = "float32", mode = "r")
    fp = fp.reshape(N, -1)
    embs_t = torch.from_numpy(fp)
    
    # total_posts_after[i] = total # of posts in (date_i, date_i + window_days]
    total_posts_after = np.zeros(N, dtype=int)
    # total_counts_per_ut[i, u] = # of all posts by user type u in that window
    total_counts_per_ut = np.zeros((N, n_ut), dtype=int)
    # num_resonant_posts[i] = # of posts j in window where sim(j, i) > tau_threshold
    num_resonant_posts = np.zeros(N, dtype=int)
    # resonant_counts_per_ut[i, u] = # of resonant posts by user type u
    resonant_counts_per_ut = np.zeros((N, n_ut), dtype=int)
    # impact_per_ut[i, u] = sum over resonant j of (sim(j, i) – raw tauᵢ) for ut == u
    impact_per_ut = np.zeros((N, n_ut), dtype=float)
    # overall_impact[i] = sum_u impact_per_ut[i, u]
    overall_impact = np.zeros(N, dtype=float)

    for i in tqdm(range(start_idx, end_idx), desc="resonance pass"):
        tau_raw = taus[i]
        if np.isnan(tau_raw):
            # no prior posts => skip resonance entirely
            continue

        # build tau_threshold for filtering
        tau_threshold = tau_raw if tau_raw >= min_tau else min_tau

        d0 = dates[i]
        max_date = d0 + window_delta

        # advance right to include all posts with date <= max_date
        if right < i + 1:
            right = i + 1
        while right < len(df) and dates[right] <= max_date:
            right += 1

        # advance left to exclude posts with date <= d0
        if left < i + 1:
            left = i + 1
        while left < len(df) and dates[left] <= d0:
            left += 1
        
        if left >= right and i + 1 != end_idx:
            continue

        idxs = np.arange(left, right)  # all posts strictly after i within window
        total_posts_after[i] = idxs.size

        if idxs.size:
            # count total posts by user type in window
            if group_by_users == True:
                for u in range(n_ut):
                    total_counts_per_ut[i, u] = np.sum(user_codes[idxs] == u)
            sims_fwd = (embs_t[idxs] @ embs_t[i]).cpu().numpy()
            above_mask = sims_fwd > tau_threshold
            above_idxs = idxs[above_mask]
            num_resonant_posts[i] = above_idxs.size

            if above_idxs.size:
                sims_above = sims_fwd[above_mask]

                # group by user type
                if group_by_users == True:
                    ut_codes = user_codes[above_idxs]  # array of length = #resonant
                    for u in range(n_ut):
                        sel_ut = (ut_codes == u)
                        resonant_counts_per_ut[i, u] = np.sum(sel_ut)
                        if np.any(sel_ut):
                            # impact contribution = sum( sim_j – tauᵢ ) over those j
                            impact_per_ut[i, u] = np.sum(sims_above[sel_ut] - tau_raw)

                overall_impact[i] = impact_per_ut[i].sum()
                
        # checkpoint JSON + partial .pkl
        if ((i + 1) % save_every == 0) or (i + 1 == end_idx):
            prog["resonance_idx"] = i + 1
            with open(prog_file, 'w') as f:
                json.dump(prog, f)

            # Always rebuild a fresh partial of exactly (i+1) rows
            part = df.iloc[:i + 1].copy()
            part["min_tau"] = taus[:i + 1]

            part["total_posts_after"] = total_posts_after[:i + 1]
            part["num_resonant_posts"] = num_resonant_posts[:i + 1]
            part["overall_impact"] = overall_impact[:i + 1]

            # attach per‐user‐type columns
            if group_by_users == True:
                for u, ut_name in enumerate(ut_list):
                    part[f"total_posts_after_{ut_name}"] = total_counts_per_ut[:i + 1, u]
                    part[f"resonant_posts_{ut_name}"]    = resonant_counts_per_ut[:i + 1, u]
                    part[f"impact_{ut_name}"]            = impact_per_ut[:i + 1, u]

            part.to_pickle(partial_file)
