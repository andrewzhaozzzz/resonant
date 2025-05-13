#!/bin/bash
#SBATCH --job-name=agg_explore
#SBATCH --partition=statds_condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/agg_explore.%j.out
#SBATCH --error=/home/export/hbailey/logs/agg_explore.%j.err

echo "[$(date)] Starting aggregation & exploration"

cd /home/export/hbailey/github/embedding_resonance/code
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

$PYTHON ntr_analysis_pt1_exploratory.py \
  --ntr-pkl /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/twitter_ntr_scores_full.pkl \
  --out-dir /home/export/hbailey/github/embedding_resonance/plots_tables

echo "[$(date)] Done aggregation & exploration"
