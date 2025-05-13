#!/usr/bin/env python3
"""
Merge padded-shard NTR results back onto the original DataFrame and run QC analysis,
reading config JSONs from any filename pattern in the given config directory.
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_original(df_path):
    df = pd.read_pickle(df_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def init_results(df, flavor_cols):
    res = pd.DataFrame(index=df.index)
    res['date'] = df['date']
    res['user_type'] = df['user_type']
    for col in flavor_cols:
        res[col] = np.nan
    return res


def validate_ranges(results, flavor_cols):
    for col in flavor_cols:
        col_data = results[col].dropna()
        if col.startswith(('novelty_','transience_')):
            if col_data.min() < 0 or col_data.max() > 1:
                print(f"Warning: {col} out of [0,1]: min={col_data.min()}, max={col_data.max()}")
        if col.startswith('resonance_'):
            if col_data.min() < -1 or col_data.max() > 1:
                print(f"Warning: {col} out of [-1,1]: min={col_data.min()}, max={col_data.max()}")


def merge_shards(df_meta, results_dir, config_dir, prefix, num_shards):
    N = len(df_meta)
    size = N // num_shards

    # Discover config JSONs by scanning config_dir
    cfg_paths = glob.glob(os.path.join(config_dir, '*_config.json'))
    configs = {}
    for p in cfg_paths:
        fname = os.path.basename(p)
        m = re.search(r'(?:\D)?(\d+)_config\.json$', fname)
        if m:
            k = int(m.group(1))
            configs[k] = p
    missing = [k for k in range(num_shards) if k not in configs]
    if missing:
        raise FileNotFoundError(f"Missing config JSON(s) for shard(s): {missing}")

    # Identify flavor columns from any result file
    sample = glob.glob(os.path.join(results_dir, f"{prefix}0*.pkl"))
    if not sample:
        raise FileNotFoundError(f"No .pkl shards in {results_dir}")
    flavor_cols = [c for c in pd.read_pickle(sample[0]).columns if c not in ('date','user_type')]

    results = init_results(df_meta, flavor_cols)
    assigned = np.zeros(N, dtype=bool)
    total_loaded = 0

    for k in range(num_shards):
        S = k * size
        E = N if k == num_shards - 1 else (k + 1) * size

        cfg = json.load(open(configs[k]))
        start_off = int(cfg['start_offset'])
        end_off   = int(cfg['end_offset'])

        final_pkl   = os.path.join(results_dir,  f"{prefix}{k}.pkl")
        partial_pkl = final_pkl + ".partial.pkl"
        prog_json   = final_pkl + ".progress.json"

        if os.path.exists(final_pkl):
            df_shard = pd.read_pickle(final_pkl)
            idx = np.arange(len(df_shard)) + S
        elif os.path.exists(partial_pkl):
            df_shard = pd.read_pickle(partial_pkl)
            if os.path.exists(prog_json):
                proc = json.load(open(prog_json)).get('processed_idx')
                expect = proc - start_off if proc is not None else None
                if expect is not None and len(df_shard) != expect:
                    print(f"Warn shard{k}: partial len {len(df_shard)} != {expect}")
            missing_cols = set(flavor_cols) - set(df_shard.columns)
            for c in missing_cols:
                df_shard[c] = np.nan
            j = df_shard.index.to_numpy()
            idx = j - start_off + S
        else:
            print(f"Warn: shard{k} missing both final & partial; skipping")
            continue

        # Validate index bounds
        if idx.min() < S or idx.max() >= E:
            raise ValueError(f"Shard{k} idx out of bounds [{S},{E}): {idx.min()}–{idx.max()}")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"Shard{k} idx out of total bounds [0,{N}): {idx.min()}–{idx.max()}")

        # Assign
        for col in flavor_cols:
            results.loc[idx, col] = df_shard[col].values
        assigned[idx] = True
        total_loaded += len(idx)
        print(f"Shard{k}: assigned {len(idx)} rows → [{S}:{E})")

    unique = assigned.sum()
    print(f"Total loaded (sum): {total_loaded}")
    print(f"Unique assigned: {unique}/{N}")
    if total_loaded != unique:
        print("Warning: overlaps detected")
    print(f"Unassigned rows: {N - unique}")
    print("Nulls per flavor:\n", results[flavor_cols].isna().sum())

    return results, flavor_cols


def analyze(results, flavor_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # split flavors into metric groups
    novelty    = sorted([c for c in flavor_cols if c.startswith('novelty_')])
    transience = sorted([c for c in flavor_cols if c.startswith('transience_')])
    resonance  = sorted([c for c in flavor_cols if c.startswith('resonance_')])

    # aggregate daily: novelty/transience -> mean, resonance -> sum
    agg = {c: 'mean' for c in novelty + transience}
    agg.update({c: 'sum' for c in resonance})
    daily = (results.groupby(['date','user_type'])[list(agg)]
             .agg(agg).unstack('user_type').sort_index())

    # save daily CSVs
    for c in agg:
        daily[c].to_csv(os.path.join(out_dir, f"daily_{c}.csv"))

    # histograms, medians, correlation matrix (existing QC)
    for c in flavor_cols:
        plt.figure(); plt.hist(results[c].dropna(), bins=50)
        plt.title(c); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{c}_hist.png"), dpi=150); plt.close()

    med = results.groupby('user_type')[flavor_cols].median()
    fig, ax = plt.subplots(figsize=(12,6)); med.plot.bar(ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "medians_by_group.png"), dpi=150); plt.close()

    corr = results[flavor_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6)); im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=150); plt.close()

    # ------------------------------------------------------------------
    # 1) Overlayed line-plots for selected user types
    thresholds = [float(c.split('_')[1]) for c in resonance]
    user_types = ['democrats', 'republicans']
    for ut in user_types:
        plt.figure(figsize=(10,5))
        for tau in thresholds:
            col = f'resonance_{tau}'
            plt.plot(daily.index, daily[(col, ut)], label=f'τ={tau}')
        plt.title(f'Daily resonance for {ut.capitalize()} at multiple thresholds')
        plt.xlabel('Date')
        plt.ylabel('Sum of resonance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'overlayed_resonance_{ut}.png'))
        plt.close()

    # 2) Bar chart of top N days at τ=0.90
    topn = 5
    tau0 = '0.90'
    col0 = f'resonance_{tau0}'
    for ut in daily[col0].columns:
        series = daily[(col0, ut)].dropna()
        top = series.nlargest(topn)
        plt.figure(figsize=(8,4))
        plt.bar(top.index.strftime('%Y-%m-%d'), top.values)
        plt.title(f'Top {topn} days of resonance at τ={tau0} for {ut}')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Date')
        plt.ylabel('Resonance sum')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'top{topn}_resonance_{tau0}_{ut}.png'))
        plt.close()

    # 3) Heatmap of positive vs. negative day counts per threshold/user_type
    diff = {}
    for tau in thresholds:
        col = f'resonance_{tau}'
        # count positive minus negative days for each user_type
        pos = (daily[col] > 0).sum()
        neg = (daily[col] < 0).sum()
        diff[tau] = pos - neg
    diff_df = pd.DataFrame(diff).T  # index=thresholds, columns=user_types
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(diff_df.values, aspect='auto',
                   cmap='bwr', vmin=-diff_df.abs().values.max(), vmax=diff_df.abs().values.max())
    ax.set_xticks(range(len(diff_df.columns)))
    ax.set_xticklabels(diff_df.columns, rotation=90)
    ax.set_yticks(range(len(diff_df.index)))
    ax.set_yticklabels([f'τ={t}' for t in diff_df.index])
    ax.set_title('(# positive days − # negative days) by threshold and user_type')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_pos_neg_diff.png'))
    plt.close()

    # 4) Summary statistics for methods section
    # mean & std for τ=0.90
    means = daily[col0].mean()
    stds  = daily[col0].std()
    # correlations between thresholds
    corr_85_90 = daily['resonance_0.85'].corrwith(daily['resonance_0.90'])
    corr_90_97 = daily['resonance_0.90'].corrwith(daily['resonance_0.97'])
    with open(os.path.join(out_dir,'summary_stats.txt'), 'w') as f:
        f.write('Mean and std of daily resonance at τ=0.90 per user_type:\n')
        for ut in means.index:
            f.write(f'{ut}: mean={means[ut]:.3f}, std={stds[ut]:.3f}\n')
        f.write('\nCorrelations between thresholds (τ=0.85 vs 0.90):\n')
        for ut, val in corr_85_90.items():
            f.write(f'{ut}: {val:.3f}\n')
        f.write('Correlations between thresholds (τ=0.90 vs 0.97):\n')
        for ut, val in corr_90_97.items():
            f.write(f'{ut}: {val:.3f}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--df",          required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--config-dir",  default=None,
                   help="Folder containing *_config.json; defaults to results-dir")
    p.add_argument("--prefix",      default="twitter_ntr_scores_shard")
    p.add_argument("--num-shards",  type=int, default=8)
    p.add_argument("--out-dir",     required=True,
                   help="Directory for CSVs & plots")
    args = p.parse_args()

    cfg_dir = args.config_dir or args.results_dir
    df_meta = load_original(args.df)
    results, flavors = merge_shards(df_meta,
                                    args.results_dir,
                                    cfg_dir,
                                    args.prefix,
                                    args.num_shards)
    validate_ranges(results, flavors)
    analyze(results, flavors, args.out_dir)


if __name__=='__main__':
    main()
