#!/bin/bash
#SBATCH --job-name=emb_shard0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=cmist
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/emb_shard0.%j.out
#SBATCH --error=/home/export/hbailey/logs/emb_shard0.%j.err

echo "Running shard 0 on $SLURM_JOB_NODELIST: $CUDA_VISIBLE_DEVICES"
nvidia-smi

PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python
cd /home/export/hbailey/github/embedding_resonance/code
$PYTHON generate_embeddings.py --shard 0 --num_shards 2

