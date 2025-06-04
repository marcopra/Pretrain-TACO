#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use an optional experiment argument passed via qsub -v EXPERIMENT=...
SEED=${SEED:-0}
ENV_NAME=${ENV_NAME:-"push-v2"}
RANDOM_HAND=${RANDOM_HAND:-0}
RANDOM_GOAL=${RANDOM_GOAL:-0}
MODEL_PATH=${MODEL_PATH:-"none"}
WANDB_TAG=${WANDB_TAG:-"none"}
FREEZE=${FREEZE:-"false"}
BATCH_SIZE=${BATCH_SIZE:-512}

# Set seed argument based on SEED value
if [ "$SEED" -eq 1 ]; then
    SEED_ARG=$(($RANDOM % 10000)) 
else
    SEED_ARG=$SEED
fi

export HYDRA_FULL_ERROR=1

# Create the experiment name once to ensure consistency
EXP_NAME="RESNET_${MODEL_PATH}_BS${BATCH_SIZE}_${SEED_ARG}_random_init_${RANDOM_HAND}_random_goal_${RANDOM_GOAL}_freeze_${FREEZE}"

# Define cleanup function
cleanup() {
    echo "Performing cleanup: Removing experiment folder"
    rm -rf exp_local/metaworld/$EXP_NAME
    echo "exp_local/metaworld/$EXP_NAME"
    echo "Cleanup completed"
}

# Set trap to ensure cleanup happens on job termination (including wall time limit)
trap cleanup EXIT HUP INT TERM

# Load environment
source ~/.bashrc
conda activate metataco



# Use quotes and escape model path appropriately
python3 train_metaworld.py agent="taco_resnet" agent.pretrained_path=\"${MODEL_PATH}\" exp_name=\"${EXP_NAME}\" seed=$SEED_ARG env_name=$ENV_NAME random_init=$RANDOM_HAND random_goal=$RANDOM_GOAL wandb_tag=$WANDB_TAG num_train_frames=300000 agent.freeze_encoder=$FREEZE batch_size=$BATCH_SIZE

# Cleanup will be triggered automatically by the trap
rm -rf exp_local/metaworld/$EXP_NAME*