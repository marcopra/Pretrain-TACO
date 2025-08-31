#!/bin/bash

seeds="1 1 1"
env_names=("bin-picking-v2")
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="BASELINE_BINPICKING"
no_taco="false"
freeze="true"
num_steps="1100000"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",FREEZE="${freeze}",WANDB_PROJECT="taco_multihead_long",NUM_STEPS="${num_steps}" launchers/PBS/TACO/Basketball/MT_pretrained.sh
            done
        done
    done
done