#!/bin/bash

seeds="1"
env_names=("button-press-v2")
model_paths=(
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_20pct_rew890_step195000_ep780.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_40pct_rew3632_step220000_ep880.pt"
"/home/mprattico/Pretrain-TACO/models_checkpoint/ButtonPress/encoder_button-press-v2_0_60pct_rew3632_step220000_ep880.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("20_BUTTON_PRESS" "40_BUTTON_PRESS" "60_BUTTON_PRESS")
no_taco="false"
num_steps=2100000

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for i in "${!model_paths[@]}"; do
                    model_path="${model_paths[$i]}"
                    wandb_tag="${wandb_tags[$i]}"
                    echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, TAG=${wandb_tag}"
                    sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="checkpoint",NUM_STEPS="${num_steps}" launchers/SLURM/Checkpoint/ButtonPress/MT_pretrained.sh
                done
            done
        done
    done
done
