#!/bin/bash
#SBATCH --job-name=v2-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=results_v2/logs/analysis_%j.log

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

echo "=== Statistical Analysis ==="
python -u experiments/v2_experiment.py \
    --phase analysis \
    --n_bootstrap 10000 \
    --output_dir results_v2
