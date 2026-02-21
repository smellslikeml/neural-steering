#!/bin/bash
#SBATCH --job-name=v2-temp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=results_v2/logs/temperature_%A_%a.log
#SBATCH --array=0-2

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Temperature Sweep: seed=$SEED ==="
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods temperature \
    --temperatures 0.3,0.5,0.7,1.0 \
    --seed $SEED \
    --output_dir results_v2
