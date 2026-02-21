#!/bin/bash
#SBATCH --job-name=v2-discovery
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=results_v2/logs/discovery_%A_%a.log
#SBATCH --array=0-2

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Discovery: seed=$SEED ==="
python -u experiments/v2_experiment.py \
    --phase discovery \
    --seed $SEED \
    --top_k 200 \
    --output_dir results_v2
