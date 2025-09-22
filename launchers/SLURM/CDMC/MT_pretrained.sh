#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd $SLURM_SUBMIT_DIR

# Use environment variables passed via sbatch
SEED=${SEED:-0}
TASK=${TASK:-"point_mass_maze_reach_bottom_left"}
MODEL_PATH=${MODEL_PATH:-"none"}
WANDB_TAG=${WANDB_TAG:-"none"}
FREEZE=${FREEZE:-"false"}
NO_TACO=${NO_TACO:-"false"}
REWARD=${REWARD:-"true"}
CURL=${CURL:-"true"}
WANDB_PROJECT=${WANDB_PROJECT:-"taco_cdmc"}
AGENT=${AGENT:-"taco"}
NUM_STEPS=${NUM_STEPS:-220000}

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
EXP_NAME="${TASK}_${MODEL_PATH}_${SEED_ARG}"

# Define cleanup function
cleanup() {
    echo "Performing cleanup: Removing experiment folder"
    rm -rf exp_local/$EXP_NAME
    echo "exp_local/$EXP_NAME"
    echo "Cleanup completed"
}

# Set trap to ensure cleanup happens on job termination
trap cleanup EXIT HUP INT TERM

# Load environment
source ~/.bashrc
conda activate metataco

# Use quotes and escape model path appropriately
python3 train_cdmc.py agent=$AGENT agent.pretrained_path=\"${MODEL_PATH}\" exp_name=\"${EXP_NAME}\" seed=$SEED_ARG task=$TASK wandb_tag=$WANDB_TAG wandb_project=$WANDB_PROJECT num_train_frames=$NUM_STEPS agent.freeze_encoder=$FREEZE agent.no_taco=$NO_TACO  agent.reward=$REWARD agent.curl=$CURL

# Cleanup will be triggered automatically by the trap
rm -rf exp_local/$EXP_NAME*
