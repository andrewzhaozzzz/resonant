#!/bin/bash
#SBATCH --job-name=merge_ntr_qc
#SBATCH --partition=phil_condo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/merge_ntr_qc.%j.out
#SBATCH --error=/home/export/hbailey/logs/merge_ntr_qc.%j.err

SHARD_DIR=/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards
RESULTS_DIR=/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/ntr_results
PLOTS_DIR=/home/export/hbailey/github/embedding_resonance/plots_tables
CODE_DIR=/home/export/hbailey/github/embedding_resonance/code
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

echo "[$(date)] Launching merge & QC"
cd ${CODE_DIR}

srun ${PYTHON} early_ntr_analysis_pt1.py \
  --df           /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl \
  --results-dir  ${RESULTS_DIR} \
  --config-dir   ${SHARD_DIR} \
  --prefix       twitter_ntr_scores_shard \
  --num-shards   8 \
  --out-dir      ${PLOTS_DIR}

echo "[$(date)] Done."
