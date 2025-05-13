#!/bin/bash
#SBATCH --job-name=time_series
#SBATCH --partition=statds_condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/time_series.%j.out
#SBATCH --error=/home/export/hbailey/logs/time_series.%j.err

echo "[$(date)] Starting time-series analysis"

cd /home/export/hbailey/github/embedding_resonance/code
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

$PYTHON ntr_analysis_pt2_time_series.py \
  --daily-dir /home/export/hbailey/github/embedding_resonance/plots_tables \
  --out-dir   /home/export/hbailey/github/embedding_resonance/plots_tables

echo "[$(date)] Done time-series analysis"
