#!/bin/bash

seeds="0 1 2"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=80_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=60_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=40_1OOD_push_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/taco_MT_exp=20_1OOD_push_config_ts=200000512.pt"
)

for seed in $seeds; do
    for env_name in "${env_names[@]}"; do
        for model_path in "${model_paths[@]}"; do
            echo "Submitting job: ENV_NAME=${env_name}, MODEL_PATH=${model_path}, SEED=${seed}"
            qsub -v SEED="${seed}",MODEL_PATH="${model_path}",ENV_NAME="${env_name}" launchers/MT_pretrained.sh
        done
    done
done


