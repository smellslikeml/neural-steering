#!/bin/bash
#SBATCH --job-name=layer_loc
#SBATCH --output=logs/layer_localization_%j.out
#SBATCH --error=logs/layer_localization_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --partition=batch

# Layer Localization Experiment
# Discovers circuits for 5 behaviors and analyzes layer distribution

set -euo pipefail

echo "=== Layer Localization Experiment ==="
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

python experiments/layer_localization.py \
    --model "$MODEL" \
    --output-dir results

echo "=== Done: $(date) ==="
