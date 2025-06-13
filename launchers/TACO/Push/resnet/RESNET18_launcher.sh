#!/bin/bash

seeds="1"
env_names=("push-v2")
model_paths=(
    "resnet18_l5_pretrained"
    "resnet18_l4_pretrained"
    "resnet18_l3_pretrained"
    "resnet18_l5_random"
    "resnet18_l4_random"
    "resnet18_l3_random"
 
)

BATCH_SIZE="512"
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="RESNET18"

for bs in $BATCH_SIZE; do
    for random_hand in $random_hand_inital; do
        for random_goal in $random_goal_inital; do
            for seed in $seeds; do
                for env_name in "${env_names[@]}"; do
                    for model_path in "${model_paths[@]}"; do
                        echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, BATCH_SIZE=${bs}, MODEL_PATH=${model_path}"
                        qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag},BATCH_SIZE=${bs}" launchers/TACO/Push/resnet/RESNET.sh
                    done
                done
            done
        done
    done
done
