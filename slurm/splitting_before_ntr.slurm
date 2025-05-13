#!/bin/bash
#SBATCH --job-name=split_shards
#SBATCH --partition=cmist_condo,phil_condo,statds_condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/split_shards.%j.out
#SBATCH --error=/home/export/hbailey/logs/split_shards.%j.err

echo "Starting shard split on $(hostname) at $(date)"

cd /home/export/hbailey/github/embedding_resonance/code
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

$PYTHON splitting_df_embs_for_ntr.py \
  --df /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl \
  --embeddings /home/export/hbailey/models/embedding_resonance/who_leads_model_final/embedding_output/who_leads_model_embeddings.npy \
  --num-shards 8 \
  --window-days 14 \
  --out-dir /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards \
  --prefix who_leads_model

echo "Shard split done at $(date)"
