#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, sys
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

def load_daily(daily_dir, tau):
    path = os.path.join(daily_dir, f"daily_resonance_{tau}.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: Missing daily file {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)

def main():
    p = argparse.ArgumentParser(
        description="Static congruence, Granger, VAR+IRF on daily NTR"
    )
    p.add_argument("--daily-dir", required=True,
                   help="Where daily_{flavor}.csv files live")
    p.add_argument("--out-dir",   required=True,
                   help="Where to write tables & figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # detect all tau values
    all_files = [f for f in os.listdir(args.daily_dir)
                 if f.startswith("daily_resonance_") and f.endswith(".csv")]
    taus = [f.replace("daily_resonance_","").replace(".csv","") for f in all_files]
    if not taus:
        sys.exit("ERROR: No daily_resonance_*.csv found in "+args.daily_dir)

    # 1) STATIC PEARSON CORRELATIONS
    for tau in taus:
        daily = load_daily(args.daily_dir, tau)
        corr = daily.corr()
        corr.to_csv(os.path.join(args.out_dir, f"pearson_corr_{tau}.csv"))
        plt.figure(figsize=(6,5))
        plt.imshow(corr, cmap="vlag", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=45)
        plt.yticks(range(len(corr)), corr.index)
        plt.title(f"Pearson Corr. (resonance_{tau})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"pearson_corr_{tau}.png"), dpi=150)
        plt.close()

    # 2) GRANGER-CAUSALITY (p-values)
    maxlag = 7
    for tau in taus:
        daily = load_daily(args.daily_dir, tau).dropna()
        groups = daily.columns.tolist()
        pmat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for g_to in groups:
            for g_from in groups:
                if g_to==g_from: continue
                data = daily[[g_to, g_from]]
                try:
                    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    pvals = [res[l][0]['ssr_ftest'][1] for l in res]
                    pmat.loc[g_from, g_to] = min(pvals)
                except:
                    pmat.loc[g_from, g_to] = np.nan
        pmat.to_csv(os.path.join(args.out_dir, f"granger_p_{tau}.csv"))
        plt.figure(figsize=(6,5))
        plt.imshow(pmat, cmap="Reds", vmin=0, vmax=0.1)
        plt.colorbar()
        plt.xticks(range(len(groups)), groups, rotation=45)
        plt.yticks(range(len(groups)), groups)
        plt.title(f"Granger p-values (resonance_{tau})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"granger_p_{tau}.png"), dpi=150)
        plt.close()

    # 3) VAR + CUMULATIVE IRFs on resonance_decay
    if "decay" not in taus:
        sys.exit("ERROR: 'decay' flavor missing; cannot run VAR.")
    daily = load_daily(args.daily_dir, "decay").dropna()
    model = VAR(daily)
    res   = model.fit(7)
    irf   = res.irf(15)
    groups = daily.columns.tolist()
    legis  = [g for g in groups if g.startswith("dem_") or g.startswith("rep_")]
    for shock in groups:
        plt.figure(figsize=(6,4))
        for lg in legis:
            idx_s = groups.index(shock)
            idx_r = groups.index(lg)
            series = irf.cum_effects[:, idx_s, idx_r]
            plt.plot(range(16), series, label=lg)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.legend()
        plt.title(f"IRF: shock to {shock}")
        plt.xlabel("Days after shock")
        plt.ylabel("Cumulative response")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"irf_{shock}.png"), dpi=150)
        plt.close()

    # 4) save raw IRF tensor
    np.save(os.path.join(args.out_dir, "irf_cum_effects.npy"),
            irf.cum_effects)
    print("[VAR] IRFs and plots saved to", args.out_dir)

if __name__=="__main__":
    main()
