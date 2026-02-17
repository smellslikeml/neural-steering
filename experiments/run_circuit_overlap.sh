#!/bin/bash
#SBATCH --job-name=circuit_ovlp
#SBATCH --output=logs/circuit_overlap_%j.out
#SBATCH --error=logs/circuit_overlap_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --partition=batch

# Circuit Overlap Experiment
# Discovers 5 behavioral circuits and measures pairwise overlap

set -euo pipefail

echo "=== Circuit Overlap Experiment ==="
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

python experiments/circuit_overlap.py \
    --model "$MODEL" \
    --output-dir results

echo "=== Done: $(date) ==="
