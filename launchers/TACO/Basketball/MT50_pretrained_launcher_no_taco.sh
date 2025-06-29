#!/bin/bash

seeds="1 1 1 1 1 1 1 1"
env_names=("Basketball-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Basketball_0.0_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Basketball_0.33_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Basketball_0.66_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Basketball_0.99_lr=0.0005_ts=50000896_curl_rew.pt" 
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT50"
no_taco="true"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}"NO_TACO="${no_taco}" launchers/TACO/Basketball/MT_pretrained.sh
                done
            done
        done
    done
done
