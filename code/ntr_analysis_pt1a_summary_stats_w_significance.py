#!/usr/bin/env python3
"""
Loads a full Resonance/Impact DataFrame, computes group‐level resonance metrics,
and writes a summary CSV with, for each user type:
  - total posts
  - percentage of posts with at least `res_threshold` resonant posts
  - median overall impact (among posts meeting the threshold)
  - 90th‐percentile overall impact (among posts meeting the threshold)

Additionally, performs statistical significance tests:
  • z‐tests for differences in % Posts Resonant
  • Kruskal–Wallis test on overall impact among resonant posts
  • Pairwise Mann–Whitney U tests on overall impact

Usage:
  python resonance_group_summary_tidy.py \
    --input-file PATH/to/output.pkl \
    --res-threshold 1 \
    --output-dir /path/to/plots_tables
"""
import os
import argparse

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import kruskal, mannwhitneyu

def main():
    p = argparse.ArgumentParser(
        description="Group‐level resonance summary"
    )
    p.add_argument(
        "--input-file", required=True,
        help="Path to the full NTR DataFrame pickle"
    )
    p.add_argument(
        "--res-threshold", type=int, default=1,
        help="Minimum number of resonant posts to count as ‘resonant’ (default: 1)"
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory where the summary CSV will be written"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_pickle(args.input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date").reset_index(drop=True)

    total_rows = len(df)
    print(f"Loaded full DataFrame: {total_rows} rows")
    print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # Define posts meeting the resonance threshold
    df['has_resonance'] = df['num_resonant_posts'] >= args.res_threshold

    # Group by user_type
    grp = df.groupby('user_type')

    summary = pd.DataFrame({
        'Num Posts': grp.size(),
        '% Posts Resonant': grp['has_resonance'].mean()
    })

    # Filter to posts that meet the resonance threshold
    resonant_only = df[df['has_resonance']]
    g2 = resonant_only.groupby('user_type')

    summary['Median Impact'] = g2['overall_impact'].median()
    summary['90th% Impact'] = g2['overall_impact'].quantile(0.9)

    # Round values and write to CSV
    summary = summary.round(3)
    out_fn = os.path.basename(args.input_file).replace('.pkl', '_group_summary.csv')
    out_csv = os.path.join(args.output_dir, out_fn)
    summary.to_csv(out_csv)

    print(f"Summary written to {out_csv}\n")
    print(summary)

    # ─── Statistical significance tests ───

    # 1) Proportion tests for % Posts Resonant (z-test)
    print("\n–– Proportion tests for % Posts Resonant (z-test) ––")
    for u1, u2 in combinations(summary.index, 2):
        c1 = int(df.loc[df.user_type == u1, 'has_resonance'].sum())
        n1 = int(summary.loc[u1, 'Num Posts'])
        c2 = int(df.loc[df.user_type == u2, 'has_resonance'].sum())
        n2 = int(summary.loc[u2, 'Num Posts'])
        z, p = proportions_ztest([c1, c2], [n1, n2])
        print(f"{u1} vs {u2}: z={z:.3f}, p={p:.3f}")

    # 2) Kruskal–Wallis test on overall impact among resonant posts
    print("\n–– Kruskal–Wallis on Impact ––")
    impact_samples = [
        resonant_only.loc[resonant_only.user_type == u, 'overall_impact'].dropna().values
        for u in summary.index
    ]
    H, p_kw = kruskal(*impact_samples)
    print(f"H={H:.3f}, p={p_kw:.3f}")

    # 3) Pairwise Mann–Whitney U tests on overall impact among resonant posts
    print("\n–– Pairwise Mann–Whitney U for Impact ––")
    for u1, u2 in combinations(summary.index, 2):
        x1 = resonant_only.loc[resonant_only.user_type == u1, 'overall_impact'].dropna().values
        x2 = resonant_only.loc[resonant_only.user_type == u2, 'overall_impact'].dropna().values
        if len(x1) and len(x2):
            U, p_mw = mannwhitneyu(x1, x2, alternative='two-sided')
            print(f"{u1} vs {u2}: U={U:.1f}, p={p_mw:.3f}")
        else:
            print(f"{u1} vs {u2}: insufficient data for Mann–Whitney U test")

if __name__ == "__main__":
    main()
