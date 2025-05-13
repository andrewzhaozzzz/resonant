#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=generating_embeddings_pos_pairs
#SBATCH --gres=gpu:1 
#SBATCH --partition=cmist
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
## output files
#SBATCH --output=/home/export/hbailey/logs/%x.%j.out
#SBATCH --error=/home/export/hbailey/logs/%x.%j.err

# Useful job diagnostics
echo "CUDA Devices(s) allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi
export WANDB_API_KEY="d4e82b44dd10afe34c6a1eb4741ec2f8bd4fcb8d"


cd /home/export/hbailey/github/embedding_resonance/code
/home/export/hbailey/.conda/envs/torch/bin/python generating_embeddings_pospairs.py
