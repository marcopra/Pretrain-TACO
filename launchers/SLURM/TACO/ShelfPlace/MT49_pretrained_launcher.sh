#!/bin/bash

seeds="1 1 1 1 1"
env_names=("shelf-place-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/mt/MT49/taco_MT_MT50_OOD_ShelfPlace_0.0_lr=0.0005_ts=177766400_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT49_SHELFPLACE"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_MT" launchers/SLURM/TACO/ShelfPlace/MT_pretrained.sh
                done
            done
        done
    done
done
