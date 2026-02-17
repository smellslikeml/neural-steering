#!/bin/bash
# Submit all experiments to SLURM
# Usage: bash experiments/run_all.sh

set -euo pipefail

cd ~/cc/neuron-steering
mkdir -p logs

echo "Submitting all neuron steering experiments..."

JOB1=$(sbatch experiments/run_layer_localization.sh | awk '{print $4}')
echo "  Layer localization: job $JOB1"

JOB2=$(sbatch experiments/run_circuit_overlap.sh | awk '{print $4}')
echo "  Circuit overlap: job $JOB2"

JOB3=$(sbatch experiments/run_scaling_analysis.sh | awk '{print $4}')
echo "  Scaling analysis: job $JOB3"

JOB4=$(sbatch experiments/run_multiplier_sweep.sh | awk '{print $4}')
echo "  Multiplier sweep: job $JOB4"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in results/"
