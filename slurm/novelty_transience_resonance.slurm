#!/bin/bash
#SBATCH --job-name=NTR_array
#SBATCH --partition=cmist_condo,phil_condo,statds_condo
#SBATCH --nodes=1
#SBATCH --ntasks=1     
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1               # one GPU per array task
#SBATCH --time=48:00:00
#SBATCH --array=0-7               # 8 shards: IDs 0…7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbailey@andrew.cmu.edu
#SBATCH --output=/home/export/hbailey/logs/%x.%A_%a.out
#SBATCH --error=/home/export/hbailey/logs/%x.%A_%a.err

# set up threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# go to code directory
cd /home/export/hbailey/github/embedding_resonance/code

# Python env
PYTHON=/home/export/hbailey/.conda/envs/torch/bin/python

echo "Processing shard: "
echo $SLURM_ARRAY_TASK_ID

# Each of the 8 tasks runs one copy of script
srun $PYTHON novelty_transience_resonance_at_different_similarity_thresholds.py \
  --df /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards/who_leads_model_shard${SLURM_ARRAY_TASK_ID}_df.pkl \
  --embeddings /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards/who_leads_model_shard${SLURM_ARRAY_TASK_ID}_emb.npy \
  --config /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/shards/who_leads_model_shard${SLURM_ARRAY_TASK_ID}_config.json \
  --out /home/export/hbailey/data/embedding_resonance/who_leads_who_follows/ntr_results/twitter_ntr_scores_shard${SLURM_ARRAY_TASK_ID}.pkl \
  --window-days 14 \
  --thresholds 0.7 0.85 0.90 0.95 0.97 \
  --save-every 1000000


echo "All tasks finished at $(date)"