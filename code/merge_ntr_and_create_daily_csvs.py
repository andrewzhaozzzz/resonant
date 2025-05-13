#!/usr/bin/env python3
"""
1) Merge shard results and write daily novelty/transience/resonance CSVs,
   trimming the first/last window_days and replacing empty-group zeros with NaN.
2) Extract top-N resonant posts per threshold plus their neighbor posts,
   including their full text for manual inspection.
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_original(df_path):
    print("Loading original dataframe from", df_path)
    df = pd.read_pickle(df_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if 'post_text' not in df.columns:
        raise KeyError("Your data frame must include a 'post_text' column.")
    print(f"→ Loaded {len(df)} rows")
    return df


def init_results(df, flavor_cols):
    print(f"Initializing results DataFrame with {len(df)} rows and {len(flavor_cols)} score columns")
    res = pd.DataFrame(index=df.index)
    res['date']      = df['date']
    res['user_type'] = df['user_type']
    for c in flavor_cols:
        res[c] = np.nan
    return res


def merge_shards(df_meta, results_dir, config_dir, prefix, num_shards):
    print("Merging shard results from", results_dir)
    N    = len(df_meta)
    size = N // num_shards

    # load shard configs
    cfg_paths = glob.glob(os.path.join(config_dir, '*_config.json'))
    configs = {}
    for p in cfg_paths:
        m = re.search(r'(\d+)_config\.json$', os.path.basename(p))
        if m:
            configs[int(m.group(1))] = p
    missing = [k for k in range(num_shards) if k not in configs]
    if missing:
        raise FileNotFoundError(f"Missing config JSON(s): {missing}")

    # flavor columns from one sample
    sample_paths = glob.glob(os.path.join(results_dir, f"{prefix}0.pkl")) + \
                   glob.glob(os.path.join(results_dir, f"{prefix}0.pkl.partial.pkl"))
    if not sample_paths:
        raise FileNotFoundError("No shard result files found in results_dir")
    sample_df   = pd.read_pickle(sample_paths[0])
    flavor_cols = [c for c in sample_df.columns if c.startswith(('novelty_','transience_','resonance_'))]

    results  = init_results(df_meta, flavor_cols)
    assigned = np.zeros(N, bool)

    for k in tqdm(range(num_shards), desc="Merging shards"):
        S = k * size
        E = N if k == num_shards - 1 else (k + 1) * size

        cfg       = json.load(open(configs[k]))
        start_off = int(cfg['start_offset'])

        base    = os.path.join(results_dir, f"{prefix}{k}.pkl")
        final   = base
        partial = base + ".partial.pkl"

        if os.path.exists(final):
            df_sh = pd.read_pickle(final)
            idx   = np.arange(len(df_sh)) + S
        elif os.path.exists(partial):
            df_sh = pd.read_pickle(partial)
            for c in flavor_cols:
                if c not in df_sh:
                    df_sh[c] = np.nan
            idx = df_sh.index.to_numpy() - start_off + S
        else:
            print(f"⚠️ Skipping missing shard {k}")
            continue

        if idx.min() < S or idx.max() >= E:
            raise ValueError(f"Shard{k} out of bounds [{S},{E}): {idx.min()}–{idx.max()}")

        for c in flavor_cols:
            results.loc[idx, c] = df_sh[c].astype(float).values
        assigned[idx] = True
        print(f"→ Shard{k}: assigned {len(idx)} rows to results[{S}:{E}]")

    if not assigned.all():
        print(f"⚠️  Warning: only {assigned.sum()}/{N} rows assigned; {N-assigned.sum()} unassigned")
    else:
        print("All rows successfully assigned.")
    return results, flavor_cols


def write_daily(results, flavor_cols, out_dir, window_days):
    print("Computing daily aggregates")
    os.makedirs(out_dir, exist_ok=True)
    novelty    = sorted([c for c in flavor_cols if c.startswith('novelty_')])
    transience = sorted([c for c in flavor_cols if c.startswith('transience_')])
    resonance  = sorted([c for c in flavor_cols if c.startswith('resonance_')])

    agg = {c: 'mean' for c in novelty + transience}
    agg.update({c: 'sum' for c in resonance})

    daily = ( results
            .groupby(['date','user_type'])[list(agg)]
            .agg(agg)
            .unstack('user_type')
            .sort_index() )

    print(f"Trimming first/last {window_days} days of each series")
    daily = daily.iloc[window_days:-window_days]

    print("Replacing zero-count days with NaN in daily metrics")
    counts = results.groupby(['date','user_type']).size().unstack('user_type').sort_index()
    counts = counts.iloc[window_days:-window_days]
    for metric in agg:
        for ut in counts.columns:
            mask = counts[ut] == 0
            if mask.any():
                daily.loc[mask, (metric, ut)] = np.nan

    for metric in agg:
        fn   = f"daily_{metric}.csv"
        path = os.path.join(out_dir, fn)
        daily[metric].to_csv(path)
        print(f"✓ Wrote {fn} ({path})")


def build_full_embeddings(df_meta, embeddings_source, config_dir, num_shards):
    """
    Returns an (N, D) array of L2-normalized embeddings,
    correctly aligned to df_meta via each shard's start/end offsets.
    """
    print("Building full embedding matrix…")

    # grab all shard-configs
    cfg_paths = glob.glob(os.path.join(config_dir, '*_config.json'))
    if not cfg_paths:
        raise FileNotFoundError(f"No config JSONs in {config_dir}")
    # infer embedding‐shard prefix: drop "<shard#>_config.json"
    sample_cfg = os.path.basename(cfg_paths[0])
    emb_prefix = re.sub(r'\d+_config\.json$', '', sample_cfg)

    N = len(df_meta)
    # load one shard to get embedding dimensionality
    emb0_path = os.path.join(embeddings_source, f"{emb_prefix}0_emb.npy")
    if not os.path.exists(emb0_path):
        raise FileNotFoundError(f"No embeddings found at {emb0_path}")
    sample_emb = np.load(emb0_path, mmap_mode='r')
    D = sample_emb.shape[1]
    emb_full = np.full((N, D), np.nan, dtype=float)

    size = N // num_shards
    for k in tqdm(range(num_shards), desc="Loading embeddings"):
        cfg_path = os.path.join(config_dir, f"{emb_prefix}{k}_config.json")
        if not os.path.exists(cfg_path):
            print(f"⚠️ Skipping missing config for shard {k}")
            continue
        cfg       = json.load(open(cfg_path))
        start_off = int(cfg.get('start_offset', 0))

        emb_path = os.path.join(embeddings_source, f"{emb_prefix}{k}_emb.npy")
        if not os.path.exists(emb_path):
            print(f"⚠️ Skipping embeddings shard {k}: file not found")
            continue

        arr = np.load(emb_path)
        L   = arr.shape[0]
        S   = k * size
        E   = N if k == num_shards - 1 else (k + 1) * size

        # full vs partial
        if L == (E - S):
            idx = np.arange(L) + S
        else:
            idx = np.arange(L) - start_off + S

        # clip any out‐of‐bounds
        valid = (idx >= 0) & (idx < N)
        if not valid.all():
            print(f"⚠️ Warning: shard {k} indices out of bounds; clipping")
        idx = idx[valid]
        arr = arr[valid]

        # normalize and insert
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        emb_full[idx] = arr / norms

    return emb_full


def extract_examples(df_meta, results, embeddings_source,
                     window_days, thresholds, top_n, out_dir,
                     num_shards, config_dir):
    print("Extracting examples (with aligned embeddings)…")
    emb = build_full_embeddings(df_meta, embeddings_source, config_dir, num_shards)

    examples = []
    for tau in tqdm(thresholds, desc="Thresholds"):
        col     = f"resonance_{tau}"
        top_idx = results[col].nlargest(top_n).index.to_numpy()
        for i in tqdm(top_idx, desc=f"Top-{top_n} idx for τ={tau}"):
            focus_text = df_meta.at[i, 'post_text']
            date_i     = df_meta.at[i, 'date']
            ut_i       = df_meta.at[i, 'user_type']
            res_i      = results.at[i, col]
            delta      = np.timedelta64(window_days, 'D')
            mask = (
                (df_meta['date'] >= date_i - delta) &
                (df_meta['date'] <= date_i + delta) &
                (df_meta.index != i)
            )
            neigh = df_meta.index[mask].to_numpy()
            if neigh.size == 0:
                continue

            sims = emb[neigh] @ emb[i]
            hits = neigh[sims >= tau]
            for j in hits:
                examples.append({
                    'threshold':          tau,
                    'focus_idx':          int(i),
                    'focus_date':         date_i,
                    'focus_user_type':    ut_i,
                    'focus_resonance':    res_i,
                    'focus_text':         focus_text,
                    'neighbor_idx':       int(j),
                    'neighbor_date':      df_meta.at[j, 'date'],
                    'neighbor_user_type': df_meta.at[j, 'user_type'],
                    'neighbor_resonance': float(results.at[j, col]),
                    'neighbor_text':      df_meta.at[j, 'post_text'],
                    'similarity':         float(sims[neigh == j])
                })

    df_ex = pd.DataFrame(examples)
    fn    = os.path.join(out_dir, "resonant_examples.csv")
    df_ex.to_csv(fn, index=False)
    print(f"✓ Wrote resonant_examples.csv ({fn})")

    if not df_ex.empty:
        print("\nSample for inspection:")
        print(df_ex[['threshold','focus_text','neighbor_text','similarity']]
              .head(10).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--df",           required=True)
    p.add_argument("--results-dir",  required=True)
    p.add_argument("--config-dir",   default=None)
    p.add_argument("--prefix",       default="twitter_ntr_scores_shard")
    p.add_argument("--num-shards",   type=int, default=8)
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--embeddings",   required=True)
    p.add_argument("--window-days",  type=int, default=14)
    p.add_argument("--thresholds",   nargs="+", type=float,
                   default=[0.7, 0.85, 0.90, 0.95, 0.97])
    p.add_argument("--top-n",        type=int, default=10)
    args = p.parse_args()

    cfg_dir   = args.config_dir or args.results_dir
    df_meta   = load_original(args.df)
    results, flavor_cols = merge_shards(
        df_meta, args.results_dir, cfg_dir, args.prefix, args.num_shards
    )

    write_daily(results, flavor_cols, args.out_dir, args.window_days)

    extract_examples(
        df_meta, results,
        args.embeddings, args.window_days,
        args.thresholds, args.top_n,
        args.out_dir, args.num_shards,
        cfg_dir
    )


if __name__ == "__main__":
    main()
