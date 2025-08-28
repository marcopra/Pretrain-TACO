#!/bin/bash

seeds="1 1 1"
env_names=(
    "basketball-v2"
    "bin-picking-v2"
    # "button-press-v2"
    "push-v2"
    "shelf-place-v2"
)

random_hand_inital="true"
random_goal_inital="false"
wandb_tag="BASELINE_BASKETBALL"
no_taco="false"
num_steps=2100000
save_snapshot="true"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_multihead_long",NUM_STEPS="${num_steps}",SAVE_SNAPSHOT="${save_snapshot}" launchers/PBS/BASELINE/taco_base.sh
            done
        done
    done
done
