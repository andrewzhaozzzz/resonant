#!/bin/bash
#SBATCH --job-name=twitter_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --partition=statds_condo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/%x.%j.out
#SBATCH --error=/home/export/hbailey/logs/%x.%j.err

echo "Starting job on $(hostname) at $(date)"

# go to the directory where your plotting script lives
cd /home/export/hbailey/github/embedding_resonance/code

# invoke the exact Python binary from your torch env
/home/export/hbailey/.conda/envs/torch/bin/python descriptive_data_plot.py

echo "Job finished at $(date)"
