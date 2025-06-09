#!/usr/bin/env python3
"""
inspect_max_impact_full.py

For a completed shard output (final .pkl), find the top N posts by overall_impact
(authored by a given user_type), and for each:
  • print its impact and the count of resonant echoes (sim > max(raw τᵢ, min_tau))
  • top 3 “novelty” neighbors before it
  • top 5 “resonant” echoes after it
"""
import argparse
import numpy as np
import pandas as pd

def cosine_sims(embs, vec):
    return embs.dot(vec)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--final-file", required=True,
        help="Path to the final shard .pkl (must contain columns: embedding, min_tau, overall_impact, date, post_text, user_type)"
    )
    p.add_argument(
        "--window-days", type=int, default=14,
        help="Number of days for look-back/look-forward window (default: 14)"
    )
    p.add_argument(
        "--min-tau", type=float, default=0.7,
        help="Minimum tau threshold for identifying resonant echoes (default: 0.7)"
    )
    p.add_argument(
        "--user-type", type=str, required=True,
        help="Which user_type to filter on"
    )
    p.add_argument(
        "--top-n", type=int, default=10,
        help="How many top-impact posts to show (default: 10)"
    )
    p.add_argument(
        "--prior-nbrs", type=int, default=3,
        help="How many prior neighbors to list per post (default: 3)"
    )
    p.add_argument(
        "--echo-nbrs", type=int, default=5,
        help="How many resonant echoes to list per post (default: 5)"
    )
    args = p.parse_args()

    # --- load and sort
    df = pd.read_pickle(args.final_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- ensure required columns exist
    expected = {"embedding", "min_tau", "overall_impact", "date", "post_text", "user_type"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in final file: {missing}")

    # --- filter by user_type, pick top-N by overall_impact
    hits = df[df["user_type"] == args.user_type]
    if hits.empty:
        print(f"No posts by user_type='{args.user_type}'")
        return

    top_hits = hits.nlargest(args.top_n, "overall_impact")
    embs_all = np.stack(df["embedding"].values)

    # --- for each top hit
    for rank, (idx, post) in enumerate(top_hits.iterrows(), 1):
        raw_tau = post.min_tau
        tau_threshold = max(raw_tau, args.min_tau)
        d0 = post.date
        td = pd.Timedelta(days=args.window_days)

        print(f"=== #{rank} Impact={post.overall_impact:.3f} "
              f"(global idx={idx}) ===")
        print(f"Date: {d0.date()}  User: {post.user_type}  raw τ={raw_tau:.3f}  "
              f"threshold={tau_threshold:.3f}")
        print("Text:")
        print(post.post_text, "\n")

        # --- top prior neighbors (no threshold)
        nov_mask = (df["date"] <  d0) & (df["date"] >= d0 - td)
        idxs_nov = np.where(nov_mask)[0]
        if idxs_nov.size:
            sims = cosine_sims(embs_all[idxs_nov], embs_all[idx])
            topm = np.argsort(-sims)[: args.prior_nbrs]
            print(f"Top {args.prior_nbrs} prior neighbors:")
            for i in topm:
                gi = idxs_nov[i]
                sim_val = sims[i]
                date_i = df.at[gi, "date"].date()
                text_i = df.at[gi, "post_text"]
                print(f" • idx={gi} date={date_i} sim={sim_val:.3f}")
                print(f"   {text_i}\n")
        else:
            print("No prior posts in window.\n")

        # --- all future candidates within window
        res_mask = (df["date"] >  d0) & (df["date"] <= d0 + td)
        idxs_res = np.where(res_mask)[0]
        if idxs_res.size:
            sims_fwd = cosine_sims(embs_all[idxs_res], embs_all[idx])
            keep = sims_fwd > tau_threshold
            idxs_e = idxs_res[keep]
            sims_e = sims_fwd[keep]

            resonant_count = idxs_e.size
            impact_sum = np.sum(sims_e - raw_tau) if resonant_count > 0 else 0.0
            print(f"Resonant count={resonant_count}  recalculated impact={impact_sum:.3f}")

            if idxs_e.size:
                # sort by (sim - raw_tau) descending
                deltas = sims_e - raw_tau
                order = np.argsort(-deltas)[: args.echo_nbrs]
                print(f"Top {args.echo_nbrs} resonant echoes:")
                for i in order:
                    gi = idxs_e[i]
                    sim_val = sims_e[i]
                    delta = sim_val - raw_tau
                    date_i = df.at[gi, "date"].date()
                    text_i = df.at[gi, "post_text"]
                    print(f" • idx={gi} date={date_i} sim={sim_val:.3f} Δ={delta:.3f}")
                    print(f"   {text_i}\n")
            else:
                print("No resonant echoes in window.\n")
        else:
            print("No future posts in window.\n")

        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
