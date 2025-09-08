#!/bin/bash

# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco] [batch_size]"
    echo "cuda_device: 0 to 7"
    echo "no_taco: true or false (default: false)"
    echo "batch_size: positive integer (default: 256)"
    exit 1
fi

cuda_device=$1
no_taco=${2:-false}
batch_size=${3:-1024}

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

# # Validate batch_size value
# if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || [ "$batch_size" -le 0 ]; then
#     echo "Error: Invalid batch_size value. Must be a positive integer."
#     exit 1
# fi

# Number of runs
num_runs=5

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    SEED=$(($RANDOM % 10000)) 
    # Run the Python command with the appropriate arguments
    python3 train_metaworld_checkpoint.py agent=taco_checkpoint exp_name=baseline${cuda_device}${SEED} seed=${SEED} env_name=push-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=300000 device=cuda:${cuda_device} agent.no_taco=${no_taco} batch_size=${batch_size} wandb_project="checkpoint" encoder_checkpoint_steps=[8000,45000,600000,1000000]

done

