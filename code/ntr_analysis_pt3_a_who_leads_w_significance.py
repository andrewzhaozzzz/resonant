#!/usr/bin/env python3
"""
Compute and plot a “who‐influences‐whom” heatmap of average impact
and annotate cells whose median impact is significantly > 0
(one‐sample Wilcoxon signed‐rank; *, p<.05; **, p<.01; ***, p<.001).

Rows = author, columns = audience,
values = mean(post_impact) per author–audience cell.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file",  required=True,
                   help="full NTR DataFrame pickle")
    p.add_argument("--output-file", default="who_leads_heatmap_impact.png",
                   help="where to save the heatmap")
    args = p.parse_args()

    df = pd.read_pickle(args.input_file)
    authors = sorted(df["user_type"].unique())
    audience = authors

    mat = np.zeros((len(authors), len(audience)), dtype=float)
    stars = np.full(mat.shape, "", dtype=object)

    for i, auth in enumerate(authors):
        sub = df[df.user_type == auth]

        for j, aud in enumerate(audience):
            if sub.empty or f"impact_{aud}" not in sub:
                mat[i, j] = np.nan
                continue

            impacts = sub[f"impact_{aud}"].values
            if impacts.size == 0:
                mat[i, j] = np.nan
            else:
                mat[i, j] = np.nanmean(impacts)

            # Select only posts with at least one resonant echo from this audience
            if f"resonant_posts_{aud}" in sub:
                arr = sub.loc[sub[f"resonant_posts_{aud}"] > 0, f"impact_{aud}"].values
            else:
                arr = np.array([])

            if len(arr) >= 5:
                try:
                    _, p = wilcoxon(arr, zero_method="wilcox", alternative="greater")
                except ValueError:
                    p = 1.0
            else:
                p = 1.0

            if p < 0.001:
                stars[i, j] = "***"
            elif p < 0.01:
                stars[i, j] = "**"
            elif p < 0.05:
                stars[i, j] = "*"

    cmap = LinearSegmentedColormap.from_list("oxford_cmu",
                                             ["#002147", "#C8102E"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_ylabel("Author")
    ax.set_xlabel("Audience")
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels([LABELS[a] for a in authors])
    ax.set_xticks(range(len(audience)))
    ax.set_xticklabels([LABELS[a] for a in audience],
                       rotation=45, ha="right")

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isnan(mat[y, x]):
                txt = ""
            else:
                txt = f"{mat[y, x]:.2f}{stars[y, x]}"
            ax.text(x, y, txt, ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Impact Score")

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Heatmap saved to {args.output_file}")

if __name__ == "__main__":
    main()
