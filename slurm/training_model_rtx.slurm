#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=200:00:00
#SBATCH --job-name=finetune_embedding_resonance
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --cluster=htc
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hannah.bailey@oii.ox.ac.uk
## output files
#SBATCH --output=/home/hert5583/logs/%x.%j.out
#SBATCH --error=/home/hert5583/logs/%x.%j.err

module load Anaconda3/2021.11
module load CUDA/11.1.1-GCC-10.2.0
module load cuDNN/8.0.4.30-CUDA-11.1.1

# Useful job diagnostics
echo "CUDA Devices(s) allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source activate torch
cd /home/hert5583/Github/embedding_resonance/code
/home/hert5583/.conda/envs/torch/bin/python3 training_model.py