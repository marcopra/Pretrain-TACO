#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=("basketball-v2")
model_paths=(
    # "/home/mprattico/Pretrain-TACO/models/multiheads/exp/0.33/taco_MT_exp_0.33_MT49OODBasketball.json_lr=0.0005_ts=195532800_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/multiheads/exp/0.66/taco_MT_exp_0.66_MT49OODBasketball.json_lr=0.0005_ts=195993600_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/multiheads/exp/0.99/taco_MT_exp_0.99_MT49OODBasketball.json_lr=0.0005_ts=11724800_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("33_BASKETBALL" "66_BASKETBALL" "99_BASKETBALL")
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for i in "${!model_paths[@]}"; do
                    model_path="${model_paths[$i]}"
                    wandb_tag="${wandb_tags[$i]}"
                    echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, TAG=${wandb_tag}"
                    qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_multihead" launchers/PBS/TACO/Basketball/MT_pretrained.sh
                done
            done
        done
    done
done
