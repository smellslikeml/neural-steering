#!/bin/bash
#SBATCH --job-name=v2-ppl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results_v2/logs/perplexity_%A_%a.log
#SBATCH --array=0-2

source ~/cc/env/bin/activate
cd ~/cc/neuron-steering

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Perplexity Measurement: seed=$SEED ==="
python -u experiments/v2_experiment.py \
    --phase perplexity \
    --seed $SEED \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0 \
    --caa_alphas 0.5,1.0,1.5,2.0,3.0,5.0,7.0,10.0 \
    --logit_biases="-5,-10,-20" \
    --temperatures 0.3,0.5,0.7,1.0 \
    --n_random_samples 5 \
    --output_dir results_v2
