import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

def create_heatmap(df_path, output_path, 
                   user_col = "user_type", 
                   prob_thresholds = [0.001, 0.01, 0.05],
                   dpi = 300,
                   messages = True,
                   **heatmap_options):

    df = pd.read_pickle(df_path)
    authors = sorted(df[user_col].unique())
    audience = authors

    mat = np.zeros((len(authors), len(audience)), dtype = float)
    stars = np.full(mat.shape, "", dtype=object)

    for i, auth in enumerate(authors):
        sub = df[df[user_col] == auth]

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

            if p < prob_thresholds[0]:
                stars[i, j] = "***"
            elif p < prob_thresholds[1]:
                stars[i, j] = "**"
            elif p < prob_thresholds[2]:
                stars[i, j] = "*"

    cmap = LinearSegmentedColormap.from_list(heatmap_options["heatmap_name"],
                                             ["#002147", "#C8102E"])
    fig, ax = plt.subplots(figsize = heatmap_options["figsize"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_ylabel(heatmap_options["ylabel"])
    ax.set_xlabel(heatmap_options["xlabel"])
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels([LABELS[a] for a in authors])
    ax.set_xticks(range(len(audience)))
    ax.set_xticklabels([LABELS[a] for a in audience],
                       rotation = 45, ha = "right")

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isnan(mat[y, x]):
                txt = ""
            else:
                txt = f"{mat[y, x]:.2f}{stars[y, x]}"
            ax.text(x, y, txt, ha = "center", va = "center", 
                    color = "white", fontsize = 8)

    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label("Average Impact Score")

    plt.tight_layout()
    plt.savefig(output_path, dpi = dpi)

    if messages == True:
        print(f"Heatmap saved to {output_path}")

def add_heatmap_options(heatmap_options):
    if "heatmap_name" not in heatmap_options.keys():
        heatmap_options["heatmap_name"] = "oxford_cmu"
    if "figsize" not in heatmap_options.keys():
        heatmap_options["figsize"] = (8, 6)
    if "ylabel" not in heatmap_options.keys():
        heatmap_options["ylabel"] = "Author"
    if "xlabel" not in heatmap_options.keys():
        heatmap_options["xlabel"] = "Audience"
