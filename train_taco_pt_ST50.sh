#!/bin/bash

# Check if exp argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No exp argument provided"
    echo "Usage: $0 <exp>"
    exit 1
fi

exp=$1

# Determine CUDA device based on exp value
case $exp in
    0)
        cuda_device=0
        ;;
    33)
        cuda_device=1
        ;;
    66)
        cuda_device=2
        ;;
    99)
        cuda_device=3
        ;;
    *)
        echo "Error: Invalid exp value. Allowed values are 0, 33, 66, 99."
        exit 1
        ;;
esac

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt"  exp_name="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=ST50 num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=false

# Cleanup command
rm -rf exp_local/metaworld//home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt"  exp_name="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=ST50-FREEZE num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=true

# Cleanup command
rm -rf exp_local/metaworld//home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt"  exp_name="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=ST50 num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=false

# Cleanup command
rm -rf exp_local/metaworld//home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt"  exp_name="/home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=ST50-FREEZE num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=true

# Cleanup command
rm -rf exp_local/metaworld//home/mprattico/Pretrain-TACO/models/taco_MT_ST50_0.${exp}_lr=0.0005_ts\=50000896.pt