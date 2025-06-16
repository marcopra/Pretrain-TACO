#!/bin/bash

# Check if correct number of arguments is provided
if [ $# -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <dataset_name> <cuda_device> <exp>"
    echo "dataset_name: basketball, binpicking, buttonpress, shelfplace"
    echo "cuda_device: 0 to 7"
    echo "exp: 0, 33, 66, 99"
    exit 1
fi

dataset_name_input=$1
cuda_device=$2
exp_input=$3

# Map simplified dataset names to full names
case $dataset_name_input in
    basketball)
        dataset_name="MT50_OOD_Basketball"
        ;;
    binpicking)
        dataset_name="MT50_OOD_BinPicking"
        ;;
    buttonpress)
        dataset_name="MT50_OOD_ButtonPress"
        ;;
    shelfplace)
        dataset_name="MT50_OOD_ShelfPlace"
        ;;
    *)
        echo "Error: Invalid dataset_name value. Allowed values are basketball, binpicking, buttonpress, shelfplace."
        exit 1
        ;;
esac

# Validate cuda_device value
case $cuda_device in
    0|1|2|3|4|5|6|7)
        # Valid value
        ;;
    *)
        echo "Error: Invalid cuda_device value. Allowed values are 0 to 7."
        exit 1
        ;;
esac

# Map integer exp values to decimal format
case $exp_input in
    0)
        exp="0.0"
        ;;
    33)
        exp="0.33"
        ;;
    66)
        exp="0.66"
        ;;
    99)
        exp="0.99"
        ;;
    *)
        echo "Error: Invalid exp value. Allowed values are 0, 33, 66, 99."
        exit 1
        ;;
esac

echo "Running pretraining with dataset: ${dataset_name}, cuda_device: ${cuda_device}, exp: ${exp}"

# Set CUDA device environment variable
export CUDA_VISIBLE_DEVICES=${cuda_device}

# Run the Python command with the appropriate arguments
python pretrain_taco_multi_task_episodes_from_checkpoint.py --dataset_config "data_episodes/${dataset_name}/${exp}" --use_wandb --total_steps 200_000_000 --checkpoint "100_000_000, 200_000_000" --lr 5e-4
