#!/bin/bash

seeds="1 1 1"
env_names=("button-press-v2")
model_paths=(
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_step2000_ep7_rew106.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_step5000_ep19_rew246.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_step15000_ep59_rew589.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_step30000_ep119_rew3475.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("2k_BUTTONPRESS" "5k_BUTTONPRESS" "15k_BUTTONPRESS" "30k_BUTTONPRESS")
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
                    qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="checkpoint",NUM_STEPS="${num_steps}",FREEZE="${freeze}" launchers/PBS/Checkpoint/ButtonPress/MT_pretrained.sh
                done
            done
        done
    done
done