import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

def cosine_sims(embs, vec):
    return embs.dot(vec)

def example_posts(df_path, embedding_path, window_days = 14, 
                  min_tau = 0.7, 
                  date_col = "date",
                  user_col = "user_type",
                  text_col = "post_text",
                  user_filter_type = ["all"], 
                  top_n = 10, 
                  prior_nbrs = 3,
                  echo_nbrs = 5):

    # --- load and sort
    df = pd.read_pickle(df_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    '''
    # --- ensure required columns exist
    expected = {"embedding", "min_tau", "overall_impact", "date", "post_text", "user_type"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in final file: {missing}")
    '''

    if user_filter_type == ["all"]:
      user_filter_type = list(pd.unique(df[user_col]))

    for i in user_filter_type:
      # --- filter by user_type, pick top-N by overall_impact
      hits = df[df[user_col] == i]
      if hits.empty:
          print(f"No posts by user_type='{i}'")
          return
      else:
          print(f"Filtering by user_type='{i}'\n")

      top_hits = hits.nlargest(top_n, "overall_impact")
      fp = np.memmap(embedding_path, dtype = "float32", mode = "r")
      fp = fp.reshape(len(df), -1)
      embs_all = fp

      # --- for each top hit
      for rank, (idx, post) in enumerate(top_hits.iterrows(), 1):
          raw_tau = post.min_tau
          tau_threshold = max(raw_tau, min_tau)
          d0 = post[date_col]
          td = pd.Timedelta(days=window_days)

          print(f"=== #{rank} Impact={post.overall_impact:.3f} "
                f"(global idx={idx}) ===")
          print(f"Date: {d0.date()}  User: {post[user_col]}  raw τ={raw_tau:.3f}  "
                f"threshold={tau_threshold:.3f}")
          print("Text:")
          print(post.post_text, "\n")

          # --- top prior neighbors (no threshold)
          nov_mask = (df[date_col] <  d0) & (df[date_col] >= d0 - td)
          idxs_nov = np.where(nov_mask)[0]
          if idxs_nov.size:
              sims = cosine_sims(embs_all[idxs_nov], embs_all[idx])
              topm = np.argsort(-sims)[: prior_nbrs]
              print(f"Top {prior_nbrs} prior neighbors:")
              for i in topm:
                  gi = idxs_nov[i]
                  sim_val = sims[i]
                  date_i = df.at[gi, date_col].date()
                  text_i = df.at[gi, text_col]
                  print(f" • idx={gi} date={date_i} sim={sim_val:.3f}")
                  print(f"   {text_i}\n")
          else:
              print("No prior posts in window.\n")

          # --- all future candidates within window
          res_mask = (df[date_col] >  d0) & (df[date_col] <= d0 + td)
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
                  order = np.argsort(-deltas)[: echo_nbrs]
                  print(f"Top {echo_nbrs} resonant echoes:")
                  for i in order:
                      gi = idxs_e[i]
                      sim_val = sims_e[i]
                      delta = sim_val - raw_tau
                      date_i = df.at[gi, date_col].date()
                      text_i = df.at[gi, text_col]
                      print(f" • idx={gi} date={date_i} sim={sim_val:.3f} Δ={delta:.3f}")
                      print(f"   {text_i}\n")
              else:
                  print("No resonant echoes in window.\n")
          else:
              print("No future posts in window.\n")

          print("-" * 60 + "\n")


def summary_stats(df_path, output_path, date_col = "date",
                  user_col = "user_type",
                  res_threshold = 1,
                  proportion_test = True,
                  kruskal_wallis = True,
                  mann_whitney_u = True,
                  messages = True):
    
    df = pd.read_pickle(df_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    total_rows = len(df)
    if messages == True:
        print(f"Loaded full DataFrame: {total_rows} rows")
        print(f"Date range: {df[date_col].iloc[0]} → {df[date_col].iloc[-1]}")

    # Define posts meeting the resonance threshold
    df['has_resonance'] = df['num_resonant_posts'] >= res_threshold

    # Group by user_type
    grp = df.groupby(user_col)

    summary = pd.DataFrame({
        'Num Posts': grp.size(),
        '% Posts Resonant': grp['has_resonance'].mean()
    })

    # Filter to posts that meet the resonance threshold
    resonant_only = df[df['has_resonance']]
    g2 = resonant_only.groupby(user_col)

    summary['Median Impact'] = g2['overall_impact'].median()
    summary['90th% Impact'] = g2['overall_impact'].quantile(0.9)

    # Round values and write to CSV
    summary = summary.round(3)
    out_fn = os.path.basename(df_path).replace('.pkl', '_group_summary.csv')
    out_csv = os.path.join(output_path, out_fn)
    summary.to_csv(out_csv)

    if messages == True:
        print(f"Summary written to {out_csv}\n")
        print(summary) #?

    # ─── Statistical significance tests ───

    # 1) Proportion tests for % Posts Resonant (z-test)
    if proportion_test == True:
        print("\n–– Proportion tests for % Posts Resonant (z-test) ––")
        for u1, u2 in combinations(summary.index, 2):
            c1 = int(df.loc[df[user_col] == u1, 'has_resonance'].sum())
            n1 = int(summary.loc[u1, 'Num Posts'])
            c2 = int(df.loc[df[user_col] == u2, 'has_resonance'].sum())
            n2 = int(summary.loc[u2, 'Num Posts'])
            z, p = proportions_ztest([c1, c2], [n1, n2])
            print(f"{u1} vs {u2}: z={z:.3f}, p={p:.3f}")

    # 2) Kruskal–Wallis test on overall impact among resonant posts
    if kruskal_wallis == True:
        print("\n–– Kruskal–Wallis on Impact ––")
        impact_samples = [
            resonant_only.loc[resonant_only[user_col] == u, 'overall_impact'].dropna().values
            for u in summary.index
        ]
        H, p_kw = kruskal(*impact_samples)
        print(f"H={H:.3f}, p={p_kw:.3f}")

    # 3) Pairwise Mann–Whitney U tests on overall impact among resonant posts
    if mann_whitney_u == True:
        print("\n–– Pairwise Mann–Whitney U for Impact ––")
        for u1, u2 in combinations(summary.index, 2):
            x1 = resonant_only.loc[resonant_only[user_col] == u1, 'overall_impact'].dropna().values
            x2 = resonant_only.loc[resonant_only[user_col] == u2, 'overall_impact'].dropna().values
            if len(x1) and len(x2):
                U, p_mw = mannwhitneyu(x1, x2, alternative='two-sided')
                print(f"{u1} vs {u2}: U={U:.1f}, p={p_mw:.3f}")
            else:
                print(f"{u1} vs {u2}: insufficient data for Mann–Whitney U test")

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
    
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
    if "figsize" not in bubbleplot_options.keys():
        bubbleplot_options["figsize"] = (8, 6)
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

