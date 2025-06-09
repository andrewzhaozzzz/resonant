#!/usr/bin/env python3
"""
Bubble‐plot of Percentage of Resonant Posts vs Median Impact, all bubbles
in Oxford Blue, with labels placed just outside their own bubbles.
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    "dem_house":   "House Democrats",
    "dem_senate":  "Senate Democrats",
    "democrats":   "Democratic Supporters",
    "media":       "Media",
    "public":      "Informed Public",
    "random":      "Random",
    "rep_house":   "House Republicans",
    "rep_senate":  "Senate Republicans",
    "republicans": "Republican Supporters",
}

OXFORD_BLUE = "#002147"

def main():
    p = argparse.ArgumentParser(
        description="Bubble‐plot of % Resonant Posts vs Median Impact by group"
    )
    p.add_argument(
        "--input-file", required=True,
        help="pickled DataFrame with columns 'user_type', 'num_resonant_posts', 'overall_impact', etc."
    )
    p.add_argument(
        "--output-file", required=True,
        help="where to save the PNG"
    )
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # --- load & compute summary ---
    df = pd.read_pickle(args.input_file)
    # determine for each post whether it was resonant (i.e., num_resonant_posts > 0)
    df['echoed'] = df['num_resonant_posts'] > 0

    grp = df.groupby('user_type')
    summary = pd.DataFrame({
        'NumPosts':  grp.size(),
        'PctEchoed': grp['echoed'].mean() * 100,
        'MedianImpact': grp.apply(
            lambda g: g.loc[g.echoed, 'overall_impact'].median() if g.echoed.any() else 0
        )
    })

    # --- prepare bubble sizes ---
    max_posts = summary.NumPosts.max()
    areas = (summary.NumPosts / max_posts) * 2000.0  # pt^2

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for ut, row in summary.iterrows():
        ax.scatter(
            row.PctEchoed,
            row.MedianImpact,
            s      = areas.loc[ut],
            color  = OXFORD_BLUE,
            edgecolor='k',
            alpha  = 0.8,
            zorder = 3
        )

    ax.set_xlabel("Percentage of Resonant Posts (%)", fontsize=12)
    ax.set_ylabel("Median Post Impact", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    lo, hi = summary.PctEchoed.min(), summary.PctEchoed.max()
    pad = 0.2 * (hi - lo)
    ax.set_xlim(max(0, lo - pad), min(100, hi + pad))

    # --- labels just outside each bubble ---
    x_mid = 0.5 * (lo + hi)
    for i, (ut, row) in enumerate(summary.iterrows()):
        x, y = row.PctEchoed, row.MedianImpact
        area_pts2 = areas.loc[ut]
        r_pts     = np.sqrt(area_pts2 / np.pi)
        pad_pts   = r_pts + 4

        # choose left/right
        if x < x_mid:
            dx, ha = +pad_pts, "left"
        else:
            dx, ha = -pad_pts, "right"

        # slight up/down wiggle
        dy = 3 if (i % 2) == 0 else -3
        va = "bottom" if dy > 0 else "top"

        # special tweak: nudge Senate Democrats further down
        if ut == "dem_senate":
            dy -= 4
            va = "top"

        ax.annotate(
            LABELS.get(ut, ut),
            xy         = (x, y),
            xytext     = (dx, dy),
            textcoords = "offset points",
            ha         = ha,
            va         = va,
            fontsize   = 9,
            zorder     = 4,
        )

    plt.tight_layout()
    fig.savefig(args.output_file, dpi=300)
    print("Saved →", args.output_file)

if __name__ == "__main__":
    main()
