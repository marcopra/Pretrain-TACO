#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=(
    "basketball-v2" 
    "bin-picking-v2" 
    "button-press-v2" 
    "push-v2" 
    "shelf-place-v2"
    )
wandb_tag="RESNET18"

for seed in $seeds; do
    for env_name in "${env_names[@]}"; do
        echo "Submitting job: ENV_NAME=${env_name}, SEED=${seed}"
        sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",WANDB_TAG="${wandb_tag}" launchers/SLURM/RESNET/resnet18.sh
    done
done
