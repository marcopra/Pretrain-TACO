#!/bin/bash

seeds="0 1 2"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=80_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=60_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=40_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=20_1OOD_push_config_ts=200000512.pt"
)
random_hand_initial="true false"
random_goal_n_object_initial="true false"

for random_hand in $random_hand_initial; do
    for random_goal in $random_goal_n_object_initial; do
        for env_name in "${env_names[@]}"; do
            for seed in $seeds; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}" launchers/baseline.sh
            done
        done
    done
done


