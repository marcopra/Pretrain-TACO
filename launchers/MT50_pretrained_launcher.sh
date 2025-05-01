#!/bin/bash

seeds="0 1 2"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=40_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=80_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=60_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=20_1OOD_push_config_ts=200000512.pt"
)
random_hand_goal_n_object_initial="false true"

for random_hand in $random_hand_goal_n_object_initial; do
    for seed in $seeds; do
        for env_name in "${env_names[@]}"; do
            for model_path in "${model_paths[@]}"; do
            echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_hand}, SEED=${seed}"
            qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_hand}",MODEL_PATH="${model_path}" launchers/MT_pretrained.sh
            done
        done
    done
done
