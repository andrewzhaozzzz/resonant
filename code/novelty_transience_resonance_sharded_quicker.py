#!/usr/bin/env python3
# File: novelty_transience_resonance_sharded_quicker.py

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
    user_codes,
    window_days,
    taus,                # will store raw tauᵢ (np.nan if no prior posts)
    start_idx,
    end_idx,
    save_every,
    prog_file,
    partial_file
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file", required=True, help="padded shard .pkl"
    )
    p.add_argument(
        "--config", required=True, help="shard config JSON"
    )
    p.add_argument(
        "--output_file", required=True, help="where to write enriched .pkl"
    )
    p.add_argument(
        "--window-days", type=int, default=14
    )
    p.add_argument(
        "--min_tau", type=float, default=0.7
    )
    p.add_argument(
        "--min-words", type=int, default=2
    )
    p.add_argument(
        "--save-every", type=int, default=250000
    )
    args = p.parse_args()

    prog_file = args.output_file + ".progress.json"
    partial_file = args.output_file + ".partial.pkl"

    # --- load & filter
    df = pd.read_pickle(args.file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["post_text"] = df["post_text"].fillna("").astype(str)
    df = df[df["post_text"].str.split().str.len() >= args.min_words].reset_index(drop=True)

    # --- clamp shard offsets
    cfg = json.load(open(args.config))
    start = int(cfg["start_offset"])
    end = int(cfg["end_offset"])
    N = len(df)
    start = max(0, min(start, N))
    end = max(start, min(end, N))

    # --- AUTO‐EXCLUDE first `window_days` for shard 0 only
    # If this shard’s start_offset == 0, find the index where date >= (earliest_date + window_days)
    if start == 0:
        dates_all = df["date"].values.astype("datetime64[D]")
        threshold_date = dates_all[0] + np.timedelta64(args.window_days, "D")
        new_start = int(np.searchsorted(dates_all, threshold_date, side="left"))
        # ensure don’t exceed end
        start = min(new_start, end)

    # At this point, `start` is the first index to actually process in both novelty & resonance
    start = max(0, start)
    end = max(start, end)

    # --- prepare arrays & codes
    embs_np = np.stack(df["embedding"].values)  # (N, embedding_dim)
    dates = df["date"].values.astype("datetime64[D]")
    ut_list = list(df["user_type"].unique())
    ut_to_code = {ut: i for i, ut in enumerate(ut_list)}
    user_codes = np.array([ut_to_code[u] for u in df["user_type"]], dtype=int)
    n_ut = len(ut_list)

    N = len(df)

    # raw tauᵢ (np.nan if no prior)
    taus = np.full(N, np.nan, dtype=np.float32)
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

    device = torch.device("cuda")
    embs_t = torch.from_numpy(embs_np).to(device)

    # --- resume from checkpoint
    prog = json.load(open(prog_file)) if os.path.exists(prog_file) else {}
    nov_start = max(start, prog.get("novelty_idx", start))
    res_start = max(start, prog.get("resonance_idx", start))
    nov_start = min(nov_start, end)
    res_start = min(res_start, end)

    print(f"[novelty]   rows {nov_start}→{end}")
    compute_novelty(
        df,
        embs_t,
        dates,
        user_codes,
        args.window_days,
        taus,
        nov_start,
        end,
        args.save_every,
        prog_file,
        partial_file
    )

    # ─── remove novelty’s partial/progress before resonance ───
    if os.path.exists(partial_file):
        os.remove(partial_file)
    if os.path.exists(prog_file):
        os.remove(prog_file)

    print(f"[resonance] rows {res_start}→{end}")
    compute_resonance(
        df,
        embs_t,
        dates,
        user_codes,
        args.window_days,
        args.min_tau,
        taus,
        total_counts_per_ut,
        resonant_counts_per_ut,
        impact_per_ut,
        overall_impact,
        total_posts_after,
        num_resonant_posts,
        ut_list,
        res_start,
        end,
        args.save_every,
        prog_file,
        partial_file
    )

    # --- final write in bulk to avoid fragmentation
    df["min_tau"] = taus
    df["total_posts_after"] = total_posts_after
    df["num_resonant_posts"] = num_resonant_posts
    df["overall_impact"] = overall_impact

    for u, ut_name in enumerate(ut_list):
        df[f"total_posts_after_{ut_name}"] = total_counts_per_ut[:, u]
        df[f"resonant_posts_{ut_name}"] = resonant_counts_per_ut[:, u]
        df[f"impact_{ut_name}"] = impact_per_ut[:, u]

    # optional defragmentation
    df = df.copy()

    df.to_pickle(args.output_file)
    print(f"Wrote full output → {args.output_file}")


if __name__ == "__main__":
    main()
