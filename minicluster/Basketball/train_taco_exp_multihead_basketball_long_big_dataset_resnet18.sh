#!/bin/bash

# Number of runs
num_runs=3

# Check if arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> <dataset_size> [freeze_encoder]"
    echo "cuda_device: 0 to 7"
    echo "dataset_size: 500k, 1M"
    echo "freeze_encoder: true or false (default: false)"
    exit 1
fi

cuda_device=$1
dataset_size=$2
freeze_encoder=${3:-false}

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

# Validate dataset_size value and set model path
case $dataset_size in
    500k)
        model_path="models/big_dataset/resnet18_l5/taco_MT_resnet18_l5_MT49OODBasketball_ds\=392000_bs\=512_ts\=38348800_lr\=0.0005_curl_rew_best.pt"
        wandb_tag="0_BASKETBALL_500K"
        ;;
    1M)
        model_path="models/big_dataset/resnet18_l5/taco_MT_resnet18_l5_MT49OODBasketball_ds\=943250_bs\=512_ts\=23475200_lr\=0.0005_curl_rew_best.pt"
        wandb_tag="0_BASKETBALL_1M"
        ;;
    *)
        echo "Error: Invalid dataset_size value. Allowed values are 0, 33, 66, 99."
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
    wandb_suffix="-FROZEN"
    wandb_tag="${wandb_tag}-FROZEN"
fi

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    SEED=$(($RANDOM % 10000)) 
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py agent=taco_resnet agent.pretrained_path="/home/mprattico/Pretrain-TACO/${model_path}" exp_name="/home/mprattico/Pretrain-TACO/${model_path}${SEED}" seed=${SEED} env_name=basketball-v2 random_init=true random_goal=false wandb_tag=${wandb_tag} wandb_project="taco_multihead_long" num_train_frames=1100000 device=cuda:${cuda_device} agent.freeze_encoder=${freeze_encoder} batch_size=512

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/${model_path}${SEED}*
done
