#!/bin/bash

seeds="1 1 1"
env_names=("basketball-v2")
model_paths=(
"/home/mprattico/Pretrain-TACO/models_checkpoint/Basketball/encoder_basketball-v2_0_step15000_ep59_rew7.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/Basketball/encoder_basketball-v2_0_step45000_ep179_rew103.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/Basketball/encoder_basketball-v2_0_step100000_ep399_rew3868.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("8k_BASKETBALL" "45k_BASKETBALL" "100k_BASKETBALL")
no_taco="false"
num_steps=300000
freeze="true"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for i in "${!model_paths[@]}"; do
                    model_path="${model_paths[$i]}"
                    wandb_tag="${wandb_tags[$i]}"
                    echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, TAG=${wandb_tag}"
                    qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="checkpoint",NUM_STEPS="${num_steps}",FREEZE="${freeze}" launchers/PBS/Checkpoint/Basketball/MT_pretrained.sh
                done
            done
        done
    done
done