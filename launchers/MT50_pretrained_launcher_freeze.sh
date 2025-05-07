#!/bin/bash

seeds="1 1 1"
env_names=("push-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=20_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=40_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=60_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt"
    "/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT50_FREEZE"
freeze="true"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",freeze="${FREEZE}" launchers/MT_pretrained.sh
                done
            done
        done
    done
done
