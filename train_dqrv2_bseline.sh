#!/bin/bash

# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [freeze_encoder]"
    echo "freeze_encoder: true or false (default: false)"
    exit 1
fi

cuda_device=$2

# Set freeze_encoder default value to false if not provided
freeze_encoder=${1:-false}

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

# Validate freeze_encoder value
case $freeze_encoder in
    true|false)
        # Valid value
        ;;
    *)
        echo "Error: Invalid freeze_encoder value. Allowed values are true or false."
        exit 1
        ;;
esac

# Set suffix for experiment name based on freeze_encoder value
if [ "$freeze_encoder" = "true" ]; then
    freeze_suffix="_frozen"
else
    freeze_suffix=""
fi

# Run the Python command with the appropriate arguments
python3 train_metaworld.py --config-name config_drqv2metaworld exp_name=baseline${cuda_device}${freeze_suffix} seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=${freeze_encoder}

# Cleanup command
rm -rf exp_local/metaworld/baseline${cuda_device}${freeze_suffix}*

# Run the Python command with the appropriate arguments
python3 train_metaworld.py --config-name config_drqv2metaworld exp_name=baseline${cuda_device}${freeze_suffix} seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=${freeze_encoder}

# Cleanup command
rm -rf exp_local/metaworld/baseline${cuda_device}${freeze_suffix}*

# Run the Python command with the appropriate arguments
python3 train_metaworld.py --config-name config_drqv2metaworld exp_name=baseline${cuda_device}${freeze_suffix} seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=baseline num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=${freeze_encoder}

# Cleanup command
rm -rf exp_local/metaworld/baseline${cuda_device}${freeze_suffix}*
