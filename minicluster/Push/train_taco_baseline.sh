#!/bin/bash

# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco]"
    echo "cuda_device: 0 to 7"
    echo "no_taco: true or false (default: false)"
    exit 1
fi

cuda_device=$1
no_taco=${2:-false}

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

# Validate no_taco value
case $no_taco in
    true|false)
        # Valid value
        ;;
    *)
        echo "Error: Invalid no_taco value. Allowed values are true or false."
        exit 1
        ;;
esac

# Number of runs
num_runs=35

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py exp_name=baseline${cuda_device} seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=220000 device=cuda:${cuda_device} agent.no_taco=${no_taco} wandb_project="taco_MT"

    # Cleanup command
    rm -rf baseline${cuda_device}*
done


