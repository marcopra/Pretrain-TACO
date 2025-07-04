#!/bin/bash

# Check if correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> <env_name>"
    echo "cuda_device: 0 to 7"
    echo "env_name: push, basketball, shelf-place, bin-picking, button-press"
    exit 1
fi

cuda_device=$1
env_name=$2

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

# Validate env_name value
case $env_name in
    push|basketball|shelf-place|bin-picking|button-press)
        # Valid value
        ;;
    *)
        echo "Error: Invalid env_name value. Allowed values are push, basketball, shelf-place, bin-picking, button-press."
        exit 1
        ;;
esac

# Add -v2 suffix to environment name
env_name_full="${env_name}-v2"

# Set pretrained model path based on environment
case $env_name in
    basketball)
        model_path="/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_Basketball_0.99_lr\=0.0005_ts\=3430400_curl_rew_best.pt"
        ;;
    bin-picking)
        model_path="/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_BinPicking_0.99_lr\=0.0005_ts\=153600_curl_rew_best.pt"
        ;;
    button-press)
        model_path="/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_ButtonPress_0.99_lr\=0.0005_ts\=30515200_curl_rew_best.pt"
        ;;
    push)
        model_path="/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_Push_0.99_lr\=0.0005_ts\=11980800_curl_rew_best.pt"
        ;;
    shelf-place)
        model_path="/home/mprattico/Pretrain-TACO/models/exp/99/taco_MT_MT50_OOD_ShelfPlace_0.99_lr\=0.0005_ts\=24166400_curl_rew_best.pt"
        ;;
esac

# Number of runs
num_runs=7

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs for environment $env_name_full"
    
    # Generate random seed between 0 and 10000
    seed=$((RANDOM % 10001))
    echo "Using seed: $seed"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py agent.pretrained_path="$model_path" exp_name="${model_path}_seed${seed}" seed=$seed env_name=$env_name_full random_init=true random_goal=false wandb_tag=MT50-debug wandb_project=taco_metaworld_debug num_train_frames=300000 device=cuda:${cuda_device} agent.no_taco=false

    # Cleanup command
    rm -rf exp_local/metaworld/${model_path}_seed${seed}
done
