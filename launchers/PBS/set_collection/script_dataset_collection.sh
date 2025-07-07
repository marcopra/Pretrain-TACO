#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Parameters passed via qsub -v
ENV_LIST=${ENV_LIST:-"push-v2"}
TASKS=${TASKS:-"0"}
EPISODES=${EPISODES:-10}
EXPERT_PROBS=${EXPERT_PROBS:-"0.0"}
CONFIG=${CONFIG:-"data/mt50/"}
RANDOM_INIT=${RANDOM_INIT:-"true"}
RANDOM_GOAL=${RANDOM_GOAL:-"true"}

# Restore original commas if they were escaped
ENV_LIST=$(echo "$ENV_LIST" | sed 's/\\,/,/g')
EXPERT_PROBS=$(echo "$EXPERT_PROBS" | sed 's/\\,/,/g')
TASKS=$(echo "$TASKS" | sed 's/\\,/,/g')

echo "Running dataset collection script with the following parameters:"
echo "ENV_LIST: $ENV_LIST"
echo "TASKS: $TASKS"
echo "EPISODES: $EPISODES"
echo "EXPERT_PROBS: $EXPERT_PROBS"
echo "CONFIG: $CONFIG"
echo "RANDOM_INIT: $RANDOM_INIT"
echo "RANDOM_GOAL: $RANDOM_GOAL"

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
python collect_metaworld_episodes.py \
    --env_names "$ENV_LIST" \
    --num_episodes $EPISODES \
    --expert_probs "$EXPERT_PROBS" \
    --config_name "$CONFIG" \
    --log_level DEBUG \
    --task "$TASKS" \
    $RANDOM_INIT_FLAG \
    $RANDOM_GOAL_FLAG

echo "Data collection completed for env_list=$ENV_LIST, expert_probs=$EXPERT_PROBS"
