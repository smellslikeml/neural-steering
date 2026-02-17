#!/bin/bash
#SBATCH --job-name=mult_sweep
#SBATCH --output=logs/multiplier_sweep_%j.out
#SBATCH --error=logs/multiplier_sweep_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --partition=batch

# Multiplier Sweep Experiment
# Sweeps alpha from 0.0 to 3.0 for refusal and capitals circuits

set -euo pipefail

echo "=== Multiplier Sweep Experiment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

# Create output directories
mkdir -p results logs

# Run experiment
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
echo "Model: $MODEL"

python experiments/multiplier_sweep.py \
    --model "$MODEL" \
    --output-dir results

echo "=== Done: $(date) ==="
