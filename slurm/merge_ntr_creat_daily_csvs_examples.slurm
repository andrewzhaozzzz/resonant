#!/bin/bash
#SBATCH --job-name=merge_ntr_full
#SBATCH --partition=cmist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/merge_ntr_full.%j.out
#SBATCH --error=/home/export/hbailey/logs/merge_ntr_full.%j.err

CODE=/home/export/hbailey/github/embedding_resonance/code/merge_ntr_and_create_daily_csvs.py
DF=/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl
SHARD=/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards
RESULTS=/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/ntr_results
OUT=/home/export/hbailey/github/embedding_resonance/plots_tables/early_analysis
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

echo "[$(date)] Starting merge + daily CSVS + examples"
srun $PYTHON $CODE \
  --df           $DF \
  --results-dir  $RESULTS \
  --config-dir   $SHARD \
  --prefix       twitter_ntr_scores_shard \
  --num-shards   8 \
  --out-dir      $OUT \
  --embeddings   $SHARD \
  --window-days  14 \
  --thresholds   0.7 0.85 0.90 0.95 0.97 \
  --top-n        10
echo "[$(date)] Done."
