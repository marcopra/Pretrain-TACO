#!/bin/bash

seeds="1 1 1"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/ST/taco_push-v2_mod2_fs3_ar2_exp=80__ds=100000_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/ST/taco_push-v2_mod2_fs3_ar2_exp=60__ds=100000_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/ST/taco_push-v2_mod2_fs3_ar2_exp=40__ds=100000_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/ST/taco_push-v2_mod2_fs3_ar2_exp=20__ds=100000_ts=200000512.pt"
)

random_hand_goal_n_object_initial="false true"

for random_hand in $random_hand_goal_n_object_initial; do
    for seed in $seeds; do
        for env_name in "${env_names[@]}"; do
            for model_path in "${model_paths[@]}"; do
            echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_hand}, SEED=${seed}"
            qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_hand}",MODEL_PATH="${model_path}" launchers/PBS/TACO/Push/MT_pretrained.sh
            done
        done
    done

done


