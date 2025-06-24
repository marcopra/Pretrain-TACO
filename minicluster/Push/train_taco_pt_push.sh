#!/bin/bash


# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco] [exp]"
    echo "cuda_device: 0 to 7"
    echo "no_taco: true or false (default: false)"
    echo "exp: 0, 33, 66, 99 (default: 0)"
    exit 1
fi

no_taco=${1:-false}
cuda_device=$2
exp=$3

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

# Validate exp value
case $exp in
    0|33|66|99)
        # Valid value
        ;;
    *)
        echo "Error: Invalid exp value. Allowed values are 0, 33, 66, 99."
        exit 1
        ;;
esac

# Set suffix for experiment name based on no_taco value
if [ "$no_taco" = "true" ]; then
    wandb_suffix="-NOTACO_PUSH"
else
    wandb_suffix="_PUSH"
fi

# Set ts value based on exp
case $exp in
    0)
        ts_value="177766400"
        ;;
    33)
        ts_value="153600"
        ;;
    66)
        ts_value="153139200"
        ;;
    99)
        ts_value="43315200"
        ;;
esac

# Number of runs
num_runs=9

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/${model_path}"  exp_name="/home/mprattico/Pretrain-TACO/models/${model_path}" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=MT${model_num}${wandb_suffix} num_train_frames=300000 device=cuda:${cuda_device} agent.no_taco=${no_taco}

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_ShelfPlace_0.${exp}_lr\=0.0005_ts\=${ts_value}_curl_rew_best.pt
done