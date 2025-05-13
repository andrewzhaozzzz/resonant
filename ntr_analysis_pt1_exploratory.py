#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os, sys

def main():
    p = argparse.ArgumentParser(
        description="1) Aggregate daily NTR & 2) QC plots"
    )
    p.add_argument("--ntr-pkl",  required=True,
                   help="Merged tweet-level NTR pickle")
    p.add_argument("--out-dir",  required=True,
                   help="Where to write daily CSVs and QC figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- load full tweet-level NTR
    try:
        df = pd.read_pickle(args.ntr_pkl)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find `{args.ntr_pkl}`")
    df['date'] = pd.to_datetime(df['date'])

    # --- identify your 5 resonance flavors
    flavors = sorted([c for c in df.columns if c.startswith("resonance_")])

    # 1) DAILY GROUP-LEVEL AGGREGATION
    for flavor in flavors:
        daily = (
            df
            .groupby(['date','user_type'])[flavor]
            .mean()
            .unstack('user_type')
            .sort_index()
        )
        daily_csv = os.path.join(args.out_dir, f"daily_{flavor}.csv")
        daily.to_csv(daily_csv)
        print(f"[agg] {flavor} → {daily_csv}")

    # 2) DISTRIBUTION HISTOGRAMS
    for flavor in flavors:
        plt.figure(figsize=(6,4))
        sns.histplot(df[flavor].dropna(), kde=True)
        plt.title(flavor)
        plt.tight_layout()
        png = os.path.join(args.out_dir, f"{flavor}_hist.png")
        plt.savefig(png, dpi=150)
        plt.close()
        print(f"[hist] {flavor} → {png}")

    # 3) MEDIANS BY GROUP
    med = df.groupby('user_type')[flavors].median()
    fig, axes = plt.subplots((len(flavors)+2)//3, 3, figsize=(12,8))
    axes = axes.flatten()
    for ax, flavor in zip(axes, flavors):
        med[flavor].plot.bar(ax=ax)
        ax.set_title(flavor)
        ax.set_xlabel("")
    # hide any unused subplots
    for ax in axes[len(flavors):]:
        ax.axis('off')
    plt.tight_layout()
    png = os.path.join(args.out_dir, "flavor_medians_by_group.png")
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"[medians] → {png}")

    # 4) FLAVOR CORRELATION MATRIX
    corr = df[flavors].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Flavor Correlations")
    plt.tight_layout()
    png = os.path.join(args.out_dir, "flavor_correlation_matrix.png")
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"[corr] → {png}")

if __name__=="__main__":
    main()
