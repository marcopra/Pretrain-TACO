#!/bin/bash

# Check if cuda_device argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No cuda_device argument provided"
    echo "Usage: $0 <cuda_device>"
    exit 1
fi

cuda_device=$1

# Validate cuda_device value
case $cuda_device in
    0|1|2|3|4|5|6|7)
        # Valid value
        ;;
    *)
        echo "Error: Invalid cuda_device value. Allowed values are 0, 1, 2, or 3."
        exit 1
        ;;
esac

# Number of runs
num_runs=9

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py exp_name=baseline${cuda_device} seed=1 env_name=button-press-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=300000 device=cuda:${cuda_device}

    # Cleanup command
    rm -rf baseline${cuda_device}*
done


