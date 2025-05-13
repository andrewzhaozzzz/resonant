#!/bin/bash
#SBATCH --job-name=validate_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=statds_condo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/%x.%j.out
#SBATCH --error=/home/export/hbailey/logs/%x.%j.err

echo "Starting validation on $(hostname) at $(date)"

# go to the code directory (absolute path)
cd /home/export/hbailey/github/embedding_resonance/code

echo "Running in $(pwd)"

# show GPUs if available
if command -v nvidia-smi &> /dev/null; then
  echo "CUDA devices on this node:"
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# use the exact Python
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

$PYTHON validate_embeddings.py \
  --df /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl \
  --shard-dir /home/export/hbailey/models/embedding_resonance/who_leads_model_final/embedding_output \
  --save-name who_leads_model_shard \
  --model-dir /home/export/hbailey/models/embedding_resonance/who_leads_model_final \
  --tokenizer-dir /home/export/hbailey/models/embedding_resonance/who_leads_model_final \
  --full-output /home/export/hbailey/models/embedding_resonance/who_leads_model_final/embedding_output/who_leads_model_embeddings.npy \
  --n-samples 10

echo "Validation finished at $(date)"
