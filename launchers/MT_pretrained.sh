#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use an optional experiment argument passed via qsub -v EXPERIMENT=...
SEED=${SEED:-0}
ENV_NAME=${ENV_NAME:-"push-v2"}
# Extract just the filename part for experiment naming to avoid Hydra parsing issues
MODEL_FILENAME=$(basename "${MODEL_PATH}")

# Define cleanup function
cleanup() {
    echo "Performing cleanup: Removing experiment folder"
    rm -rf exp_local/metaworld/${MODEL_FILENAME}_${ENV_NAME}
    echo "Cleanup completed"
}

# Set trap to ensure cleanup happens on job termination (including wall time limit)
trap cleanup EXIT HUP INT TERM

# Load environment
source ~/.bashrc
conda activate metataco

# Use quotes and escape model path appropriately
python3 train_metaworld.py agent.pretrained_path="${MODEL_PATH}" exp_name="${MODEL_FILENAME}" seed=$SEED env_name="${ENV_NAME}"

# Cleanup will be triggered automatically by the trap
