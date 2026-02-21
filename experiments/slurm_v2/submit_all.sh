#!/bin/bash
# V2 Rigorous Anti-Slop Experiment — SLURM Submission Pipeline
# Submits all jobs with proper dependency chains.
# Usage: bash slurm_v2/submit_all.sh

set -euo pipefail
cd ~/cc/neuron-steering

echo "=== V2 Rigorous Anti-Slop Experiment ==="
echo "Submitting all jobs..."

# Create output directories
mkdir -p results_v2/{circuits,generations,perplexity,analysis,logs}

# ---------------------------------------------------------------
# Job 1: Circuit Discovery (array job, 3 seeds)
# ---------------------------------------------------------------
JOB1=$(sbatch --parsable slurm_v2/01_discovery.sh)
echo "Job 1 (discovery): $JOB1"

# ---------------------------------------------------------------
# Jobs 2-7: Generation (all depend on discovery)
# ---------------------------------------------------------------
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/02_baseline.sh)
echo "Job 2 (baseline+prompt_eng): $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/03_temperature.sh)
echo "Job 3 (temperature): $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/04_logit.sh)
echo "Job 4 (logit_suppression): $JOB4"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/05_neuron.sh)
echo "Job 5 (neuron_steering): $JOB5"

JOB6=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/06_caa.sh)
echo "Job 6 (caa_repeng): $JOB6"

JOB7=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_v2/07_random.sh)
echo "Job 7 (random_ablation): $JOB7"

# ---------------------------------------------------------------
# Job 8: Perplexity (depends on all generation jobs)
# ---------------------------------------------------------------
JOB8=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3:$JOB4:$JOB5:$JOB6:$JOB7 slurm_v2/08_perplexity.sh)
echo "Job 8 (perplexity): $JOB8"

# ---------------------------------------------------------------
# Job 9: Statistical Analysis (depends on generation + perplexity)
# ---------------------------------------------------------------
JOB9=$(sbatch --parsable --dependency=afterok:$JOB8 slurm_v2/09_analysis.sh)
echo "Job 9 (analysis): $JOB9"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: squeue -u \$USER"
echo "Results: results_v2/"
