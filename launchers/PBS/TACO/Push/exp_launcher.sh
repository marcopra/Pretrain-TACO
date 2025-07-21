#!/bin/bash

seeds="1"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/exp/33/taco_MT_MT50_OOD_Push_0.33_lr=0.0005_ts=200000512_curl_rew.pt"
    "/home/mprattico/Pretrain-TACO/models/exp/66/taco_MT_MT50_OOD_Push_0.66_lr=0.0005_ts=200000512_curl_rew.pt"
    "/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_Push_0.99_lr=0.0005_ts=11980800_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("33_PUSH" "66_PUSH" "99_PUSH")
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for i in "${!model_paths[@]}"; do
                    model_path="${model_paths[$i]}"
                    wandb_tag="${wandb_tags[$i]}"
                    echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, TAG=${wandb_tag}"
                    qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="TACO_EXP" launchers/PBS/TACO/Push/MT_pretrained.sh
                done
            done
        done
    done
done
