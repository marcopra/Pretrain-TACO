#!/bin/bash

seeds="1"
env_names=("button-press-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/mt/MT1/taco_MT_MT1IDButtonPressTopdownWall.json_lr=0.0005_ts=35020800_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT1/taco_MT_MT1IDDrawerOpen.json_lr=0.0005_ts=162355200_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT1/taco_MT_MT1IDHandlePressSide.json_lr=0.0005_ts=65945600_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT1/taco_MT_MT1IDPushWall.json_lr=0.0005_ts=119398400_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT1/taco_MT_MT1IDSweep.json_lr=0.0005_ts=178892800_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT1_BUTTONPRESS"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_MT" launchers/SLURM/TACO/ButtonPress/MT_pretrained.sh
                done
            done
        done
    done
done
