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
NO_TACO=${NO_TACO:-"false"}

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

# Set trap to ensure cleanup happens on job termination (including wall time limit)
trap cleanup EXIT HUP INT TERM

# Load environment
source ~/.bashrc
conda activate metataco


# Use quotes and escape model path appropriately
python3 train_metaworld.py agent.pretrained_path=\"${MODEL_PATH}\" exp_name=\"${EXP_NAME}\" seed=$SEED_ARG env_name=$ENV_NAME random_init=$RANDOM_HAND random_goal=$RANDOM_GOAL wandb_tag=$WANDB_TAG num_train_frames=300000 agent.freeze_encoder=$FREEZE agent.no_taco=$NO_TACO wandb_project=taco_MT

# Cleanup will be triggered automatically by the trap
rm -rf exp_local/metaworld/$EXP_NAME*