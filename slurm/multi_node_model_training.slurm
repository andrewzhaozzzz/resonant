#!/bin/bash
#SBATCH --job-name=finetune_embedding_resonance
#SBATCH --partition=cmist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8       # spawn 8 processes (one per GPU)
#SBATCH --gres=gpu:8              # allocate all 8 GPUs on n03
#SBATCH --cpus-per-task=8         # CPU cores per process
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/%x.%j.out
#SBATCH --error=/home/export/hbailey/logs/%x.%j.err

echo "Running on node: $SLURM_NODELIST"
scontrol show hostnames $SLURM_NODELIST

# Point to your env's Python executable
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

# Provide your W&B API key so init() won’t prompt for login
export WANDB_API_KEY="d4e82b44dd10afe34c6a1eb4741ec2f8bd4fcb8d"

# Set up distributed master address & port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345

# Change into your code directory
cd /home/export/hbailey/github/embedding_resonance/code

# Launch DDP training across all 8 GPUs
$PYTHON -m torch.distributed.run \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    training_model.py
