#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=("bin-picking-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/multiheads/exp/0.33/taco_MT_exp_0.33_MT49OODBinPicking.json_lr=0.0005_ts=194508800_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/multiheads/exp/0.99/taco_MT_exp_0.99_MT49OODBinPicking.json_lr=0.0005_ts=23808000_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tags=("0.33_BINPICKING" "0.99_BINPICKING")
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for i in "${!model_paths[@]}"; do
                    model_path="${model_paths[$i]}"
                    wandb_tag="${wandb_tags[$i]}"
                    echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}, TAG=${wandb_tag}"
                    sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_multiheads" launchers/SLURM/TACO/BinPicking/MT_pretrained.sh
                done
            done
        done
    done
done
