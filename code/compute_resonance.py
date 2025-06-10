import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def compute_resonance(
    df,
    embs_t,
    dates,
    user_codes,
    window_days,
    min_tau,
    taus,                            # raw tauᵢ from compute_novelty
    total_counts_per_ut,             # shape (N, n_ut): total posts in window by ut (user_type)
    resonant_counts_per_ut,          # shape (N, n_ut): resonant posts by ut
    impact_per_ut,                   # shape (N, n_ut): impact contributions by ut
    overall_impact,                  # shape (N,): sum of impact_per_ut[i, :]
    total_posts_after,               # shape (N,): all posts in window
    num_resonant_posts,              # shape (N,): count of resonant posts total
    ut_list,                         # list of user‐type names, length = n_ut
    start_idx,
    end_idx,
    save_every,
    prog_file,
    partial_file
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
    """
    prog = json.load(open(prog_file)) if os.path.exists(prog_file) else {}
    n_ut = len(ut_list)
    window_delta = np.timedelta64(window_days, 'D')
    right = start_idx
    left = start_idx

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

        idxs = np.arange(left, right)  # all posts strictly after i within window
        total_posts_after[i] = idxs.size

        if idxs.size:
            # count total posts by user type in window
            for u in range(n_ut):
                total_counts_per_ut[i, u] = np.sum(user_codes[idxs] == u)

            sims_fwd = (embs_t[idxs] @ embs_t[i]).cpu().numpy()
            above_mask = sims_fwd > tau_threshold
            above_idxs = idxs[above_mask]
            num_resonant_posts[i] = above_idxs.size

            if above_idxs.size:
                sims_above = sims_fwd[above_mask]

                # group by user type
                ut_codes = user_codes[above_idxs]  # array of length = #resonant
                for u in range(n_ut):
                    sel_ut = (ut_codes == u)
                    resonant_counts_per_ut[i, u] = np.sum(sel_ut)
                    if np.any(sel_ut):
                        # impact contribution = sum( sim_j – tauᵢ ) over those j
                        impact_per_ut[i, u] = np.sum(sims_above[sel_ut] - tau_raw)

                overall_impact[i] = impact_per_ut[i].sum()

                # *************************************************************
                ## PRINT OUT FOR DEBUGGING ##
                
                # if above_idxs.size >= 4:
                #     # debug print
                #     print(f"\nResonant post idx={i}, tau={tau_raw:.3f}, echoes={above_idxs.size}")
                #     print(f"  Text: {df.at[i,'post_text']}\n")
                #     for j,s in zip(above_idxs, sims_above):
                #         print(f"    echo sim={s:.3f} text={df.at[j,'post_text']}")
                
                #     # top10 past
                #     mask_bwd = (dates < d0) & (dates >= d0 - np.timedelta64(window_days, 'D'))
                #     idxs_bwd = np.where(mask_bwd)[0]
                #     if idxs_bwd.size:
                #         embs_i = embs_t[i]
                #         sims_bwd = (embs_t[idxs_bwd] @ embs_i).detach().cpu().numpy()
                #         top10 = np.argsort(-sims_bwd)[:10]
                #         print("\n  Top 10 past posts:")
                #         for rank, k in enumerate(top10, 1):
                #             j = idxs_bwd[k]
                #             print(f"    {rank:2d}. sim={sims_bwd[k]:.3f} text={df.at[j,'post_text']}")
                # *************************************************************

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
            for u, ut_name in enumerate(ut_list):
                part[f"total_posts_after_{ut_name}"] = total_counts_per_ut[:i + 1, u]
                part[f"resonant_posts_{ut_name}"]    = resonant_counts_per_ut[:i + 1, u]
                part[f"impact_{ut_name}"]            = impact_per_ut[:i + 1, u]

            part.to_pickle(partial_file)
