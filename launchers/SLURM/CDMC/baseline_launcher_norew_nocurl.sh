#!/bin/bash

seeds="1 1 1"
task_names=("point_mass_maze_reach_bottom_left")

wandb_tag="BASELINE_noreward_nocurl"
reward="false"
curl="false"



for seed in $seeds; do
    for env_name in "${task_names[@]}"; do
        echo "Submitting job: ENV_NAME=${env_name}, SEED=${seed}"
        sbatch --export=SEED="${seed}",TASK="${env_name}",REWARD="${reward}",CURL="${curl}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_cdmc" ../MT_pretrained.sh
    done
done

