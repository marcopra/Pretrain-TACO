#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use environment variables passed via qsub
SEED=${SEED:-0}
ENV_NAME=${ENV_NAME:-"basketball-v2"}
RANDOM_HAND=${RANDOM_HAND:-0}
RANDOM_GOAL=${RANDOM_GOAL:-0}
MODEL_PATH=${MODEL_PATH:-"none"}
WANDB_TAG=${WANDB_TAG:-"none"}
FREEZE=${FREEZE:-"false"}
NO_TACO=${NO_TACO:-"false"}
WANDB_PROJECT=${WANDB_PROJECT:-"taco_metaworld"}
AGENT=${AGENT:-"taco"}

# Set seed argument based on SEED value
if [ "$SEED" -eq 1 ]; then
    SEED_ARG=$(($RANDOM % 10000)) 
else
    SEED_ARG=$SEED
fi

export HYDRA_FULL_ERROR=1

# Validate NO_TACO value
case $NO_TACO in
    true|false)
        # Valid value
        ;;
    *)
        echo "Error: Invalid NO_TACO value. Allowed values are true or false."
        exit 1
        ;;
esac

# Create the experiment name once to ensure consistency
EXP_NAME="${MODEL_PATH}_${SEED_ARG}"

# Define cleanup function
cleanup() {
    echo "Performing cleanup: Removing experiment folder"
    rm -rf exp_local/metaworld/$EXP_NAME
    echo "exp_local/metaworld/$EXP_NAME"
    echo "Cleanup completed"
}

# Set trap to ensure cleanup happens on job termination
trap cleanup EXIT HUP INT TERM

# Load environment
source ~/.bashrc
conda activate metataco

# Use quotes and escape model path appropriately
python3 train_metaworld.py agent=taco_resnet agent.pretrained_path=\"${MODEL_PATH}\" exp_name=\"${EXP_NAME}\" seed=$SEED_ARG env_name=$ENV_NAME random_init=$RANDOM_HAND random_goal=$RANDOM_GOAL wandb_tag=$WANDB_TAG wandb_project=$WANDB_PROJECT num_train_frames=2002000 agent.freeze_encoder=$FREEZE agent.no_taco=$NO_TACO agent=$AGENT batch_size=256

# Cleanup will be triggered automatically by the trap
rm -rf exp_local/metaworld/$EXP_NAME*
