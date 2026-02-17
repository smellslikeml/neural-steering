#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --output=logs/scaling_analysis_%j.out
#SBATCH --error=logs/scaling_analysis_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --partition=batch

# Scaling Analysis Experiment (1B vs 8B)
# Runs identical experiments on both model scales, then compares
# Note: Loads 8B first (needs more memory), then frees and loads 1B

set -euo pipefail

echo "=== Scaling Analysis Experiment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

# Create output directories
mkdir -p results logs

# Run experiment
# Use --model-only to run just one model if needed (e.g., for separate SLURM jobs)
if [ "${1:-}" = "8b" ]; then
    echo "Running 8B model only"
    python experiments/scaling_analysis.py --output-dir results --model-only 8b
elif [ "${1:-}" = "1b" ]; then
    echo "Running 1B model only"
    python experiments/scaling_analysis.py --output-dir results --model-only 1b
else
    echo "Running both models sequentially"
    python experiments/scaling_analysis.py --output-dir results
fi

echo "=== Done: $(date) ==="
