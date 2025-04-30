#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Parameters passed via qsub -v
ENV_LIST=${ENV_LIST:-"push-v2"}
# TASKS=${TASKS:-"0"}
DATASET_SIZE=${DATASET_SIZE:-1000}
EXPERT_PROBS=${EXPERT_PROBS:-"0.0"}
SAVE_PATH=${SAVE_PATH:-"data/mt50/"}
RANDOM_INIT=${RANDOM_INIT:-"true"}
RANDOM_GOAL=${RANDOM_GOAL:-"true"}

# Load environment
source ~/.bashrc
conda activate metataco

# Prepare random init and random goal flags
RANDOM_INIT_FLAG="--random_init"
if [ "$RANDOM_INIT" = "false" ]; then
    RANDOM_INIT_FLAG="--no_random_init"
fi

RANDOM_GOAL_FLAG="--randomize_goal"
if [ "$RANDOM_GOAL" = "false" ]; then
    RANDOM_GOAL_FLAG="--no_randomize_goal"
fi

# Run the data collection script
python metaworld_image_dataset.py \
    --env_names "$ENV_LIST" \
    --dataset_size "$DATASET_SIZE" \
    --expert_probs "$EXPERT_PROBS" \
    --save_path "$SAVE_PATH" \
    $RANDOM_INIT_FLAG \
    $RANDOM_GOAL_FLAG

echo "Data collection completed for env_list=$ENV_LIST, expert_probs=$EXPERT_PROBS"
