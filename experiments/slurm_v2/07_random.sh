#!/bin/bash
#SBATCH --job-name=v2-random
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results_v2/logs/random_%A_%a.log
#SBATCH --array=0-2

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Random Ablation Control: seed=$SEED ==="
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods random_ablation \
    --n_random_samples 5 \
    --seed $SEED \
    --output_dir results_v2
