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
                                             heatmap_options["heatmap_color"])
    fig, ax = plt.subplots(figsize = heatmap_options["figsize"])
    im = ax.imshow(mat, aspect = "auto", cmap = cmap)

    ax.set_ylabel(heatmap_options["ylabel"])
    ax.set_xlabel(heatmap_options["xlabel"])
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels([LABELS[a] for a in authors])
    ax.set_xticks(range(len(audience)))
    ax.set_xticklabels([LABELS[a] for a in audience],
                       rotation = heatmap_options["rotation"], 
                       ha = heatmap_options["xlabel_ha"])

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isnan(mat[y, x]):
                txt = ""
            else:
                txt = f"{mat[y, x]:.2f}{stars[y, x]}"
            ax.text(x, y, txt, ha = heatmap_options["text_ha"], 
                    va = heatmap_options["text_va"], 
                    color = heatmap_options["text_color"], 
                    fontsize = heatmap_options["text_fontsize"])

    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label("Average Impact Score")

    plt.tight_layout()
    plt.savefig(output_path, dpi = dpi)

    if messages == True:
        print(f"Heatmap saved to {output_path}")

def add_heatmap_options(heatmap_options):
    if "heatmap_name" not in heatmap_options.keys():
        heatmap_options["heatmap_name"] = "oxford_cmu"
    if "heatmap_color" not in heatmap_options.keys():
        heatmap_options["heatmap_color"] = ["#002147", "#C8102E"]
    if "figsize" not in heatmap_options.keys():
        heatmap_options["figsize"] = (8, 6)
    if "rotation" not in heatmap_options.keys():
        heatmap_options["rotation"] = 45
    if "xlabel_ha" not in heatmap_options.keys():
        heatmap_options["xlabel_ha"] = "right"
    if "ylabel" not in heatmap_options.keys():
        heatmap_options["ylabel"] = "Author"
    if "xlabel" not in heatmap_options.keys():
        heatmap_options["xlabel"] = "Audience"
    if "text_ha" not in heatmap_options.keys():
        heatmap_options["text_ha"] = "center"
    if "text_va" not in heatmap_options.keys():
        heatmap_options["text_va"] = "center"
    if "text_color" not in heatmap_options.keys():
        heatmap_options["text_color"] = "white"
    if "text_fontsize" not in heatmap_options.keys():
        heatmap_options["text_fontsize"] = 8
    return heatmap_options


def create_bubbleplot(df_path, output_path, 
                      user_col = "user_type",
                      dpi = 300,
                      messages = True,
                      **bubbleplot_options):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- load & compute summary ---
    df = pd.read_pickle(df_path)
    # determine for each post whether it was resonant (i.e., num_resonant_posts > 0)
    df['echoed'] = df['num_resonant_posts'] > 0

    grp = df.groupby(user_col)
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
    fig, ax = plt.subplots(figsize = bubbleplot_options["figsize"])
    for ut, row in summary.iterrows():
        ax.scatter(
            row.PctEchoed,
            row.MedianImpact,
            s      = areas.loc[ut],
            color  = bubbleplot_options["color"],
            edgecolor = bubbleplot_options["edgecolor"],
            alpha  = bubbleplot_options["scatter_alpha"],
            zorder = bubbleplot_options["scatter_zorder"]
        )

    ax.set_xlabel(bubbleplot_options["xlabel"], fontsize = bubbleplot_options["fontsize"])
    ax.set_ylabel(bubbleplot_options["ylabel"], fontsize = bubbleplot_options["fontsize"])
    ax.grid(True, linestyle = bubbleplot_options["grid_linestyle"], 
            alpha = bubbleplot_options["grid_alpha"], 
            zorder = bubbleplot_options["grid_zorder"])

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
    fig.savefig(output_path, dpi = dpi)

    if messages == True:
      print("Saved →", output_path)

def add_bubbleplot_options(bubbleplot_options):
    if "color" not in bubbleplot_options.keys():
        bubbleplot_options["color"] = "#002147"
    if "edgecolor" not in bubbleplot_options.keys():
        bubbleplot_options["edgecolor"] = "k"
    if "scatter_alpha" not in bubbleplot_options.keys():
        bubbleplot_options["scatter_alpha"] = 0.8
    if "scatter_zorder" not in bubbleplot_options.keys():
        bubbleplot_options["scatter_zorder"] = 3
    if "xlabel" not in bubbleplot_options.keys():
        bubbleplot_options["xlabel"] = "Percentage of Resonant Posts (%)"
    if "ylabel" not in bubbleplot_options.keys():
        bubbleplot_options["ylabel"] = "Median Post Impact"
    if "fontsize" not in bubbleplot_options.keys():
        bubbleplot_options["fontsize"] = 12
    if "grid_linestyle" not in bubbleplot_options.keys():
        bubbleplot_options["grid_linestyle"] = "--"
    if "grid_alpha" not in bubbleplot_options.keys():
        bubbleplot_options["grid_alpha"] = 0.3
    if "grid_zorder" not in bubbleplot_options.keys():
        bubbleplot_options["grid_zorder"] = 1
    return bubbleplot_options

def add_training_args(training_args):
  if "learning_rate" not in training_args.keys():
    training_args["learning_rate"] = 1e-5
  if "num_train_epochs" not in training_args.keys():
    training_args["num_train_epochs"] = 1
  if "weight_decay" not in training_args.keys():
    training_args["weight_decay"] = 0.01
  if "save_steps" not in training_args.keys():
    training_args["save_steps"] = 100000
  if "per_device_train_batch_size" not in training_args.keys():
    training_args["per_device_train_batch_size"] = 16
  if "gradient_accumulation_steps" not in training_args.keys():
    training_args["gradient_accumulation_steps"] = 4
  return training_args

