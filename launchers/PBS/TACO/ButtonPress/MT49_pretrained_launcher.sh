#!/bin/bash

seeds="1 1 1 1 1"
env_names=("button-press-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/mt/MT49/taco_MT_MT50_OOD_ButtonPress_0.0_lr=0.0005_ts=192512000_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT49"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}" launchers/PBS/TACO/ButtonPress/MT_pretrained.sh
                done
            done
        done
    done
done
