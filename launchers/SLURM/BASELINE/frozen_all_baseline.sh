#!/bin/bash

seeds="1 1 1 1 1"
env_names=(
    "basketball-v2" 
    "bin-picking-v2" 
    "button-press-v2" 
    "push-v2" 
    "shelf-place-v2"
    )
model_paths=(
    "r3m"
    "/home/mprattico/Pretrain-TACO/models/moco_aug.pth.tar"

)
   
random_hand_inital="true"
random_goal_inital="false"
no_taco="false"
freeze="true"

freeze="false"
agent="taco_resnet"
for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",FREEZE="${freeze}",NO_TACO="${no_taco}",agent="${}",WANDB_PROJECT="taco_frozen_baselines" launchers/SLURM/BASELINE/baseline_pretrained.sh
                done
            done
        done
    done
done

