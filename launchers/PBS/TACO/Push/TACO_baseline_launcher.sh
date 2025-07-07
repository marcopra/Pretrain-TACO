#!/bin/bash

seeds="1 1 1 1 1 1 1 1 1 1 1"
env_names=("push-v2")
random_hand_initial="true"
random_goal_n_object_initial="false"
# Loop through each environment name and seed   

for random_hand in $random_hand_initial; do
    for random_goal in $random_goal_n_object_initial; do
        for env_name in "${env_names[@]}"; do
            for seed in $seeds; do
                qsub -v SEED="$seed",ENV_NAME="$env_name",RANDOM_HAND="$random_hand",RANDOM_GOAL="$random_goal" launchers/TACO/Push/baseline.sh
            done
        done
    done
done
