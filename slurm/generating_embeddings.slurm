#!/bin/bash
#SBATCH --job-name=emb_shard1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=statds_condo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/emb_shard1.%j.out
#SBATCH --error=/home/export/hbailey/logs/emb_shard1.%j.err

echo "Running shard 1 on statds_condo: $CUDA_VISIBLE_DEVICES"
nvidia-smi

PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python
cd /home/export/hbailey/github/embedding_resonance/code
$PYTHON generate_embeddings.py --shard 1 --num_shards 2
