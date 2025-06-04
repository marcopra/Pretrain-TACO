#!/bin/bash

seeds="1"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/resnet50_l3.tar"
    "/home/mprattico/Pretrain-TACO/models/resnet50_l4.pth.tar"
    "resnet50_l5"
)

BATCH_SIZE="1024"
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="RESNET50"

for bs in $BATCH_SIZE; do
    for random_hand in $random_hand_inital; do
        for random_goal in $random_goal_inital; do
            for seed in $seeds; do
                for env_name in "${env_names[@]}"; do
                    for model_path in "${model_paths[@]}"; do
                        echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, BATCH_SIZE=${bs}, MODEL_PATH=${model_path}"
                        qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag},BATCH_SIZE=${bs}" launchers/TACO/resnet/multiRESNET.sh
                    done
                done
            done
        done
    done
done
