#!/bin/bash

seeds="1"
env_names=("button-press-v2")
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="BASELINE_BUTTONPRESS"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_MT" launchers/SLURM/TACO/ButtonPress/MT_pretrained.sh
            done
        done
    done
done
