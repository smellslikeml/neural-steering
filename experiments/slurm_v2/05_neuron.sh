#!/bin/bash
#SBATCH --job-name=v2-neuron
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results_v2/logs/neuron_%A_%a.log
#SBATCH --array=0-2

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Neuron Steering Sweep: seed=$SEED ==="
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods neuron_steering \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0 \
    --seed $SEED \
    --output_dir results_v2
