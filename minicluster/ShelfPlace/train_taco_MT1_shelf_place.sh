#!/bin/bash


# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco]"
    echo "cuda_device: 0 to 4"
    echo "no_taco: true or false (default: false)"
    exit 1
fi

cuda_device=$1
no_taco=${2:-false}

# Validate cuda_device value and set model path
case $cuda_device in
    0)
        model_path="models/mt/MT1/taco_MT_MT1IDButtonPressTopdownWall.json_lr\=0.0005_ts\=35020800_curl_rew_best.pt"
        ;;
    1)
        model_path="models/mt/MT1/taco_MT_MT1IDDrawerOpen.json_lr\=0.0005_ts\=162355200_curl_rew_best.pt"
        ;;
    2)
        model_path="models/mt/MT1/taco_MT_MT1IDHandlePressSide.json_lr\=0.0005_ts\=65945600_curl_rew_best.pt"
        ;;
    3)
        model_path="models/mt/MT1/taco_MT_MT1IDPushWall.json_lr\=0.0005_ts\=119398400_curl_rew_best.pt"
        ;;
    4)
        model_path="models/mt/MT1/taco_MT_MT1IDSweep.json_lr\=0.0005_ts\=178892800_curl_rew_best.pt"
        ;;
    *)
        echo "Error: Invalid cuda_device value. Allowed values are 0 to 4."
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

# Set suffix for experiment name based on no_taco value
if [ "$no_taco" = "true" ]; then
    wandb_suffix="-NOTACO_SHELFPLACE"
else
    wandb_suffix="_SHELFPLACE"
fi

# Number of runs
num_runs=9

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/${model_path}"  exp_name="/home/mprattico/Pretrain-TACO/${model_path}" seed=1 env_name=shelf-place-v2 random_init=true random_goal=false wandb_tag=MT1${wandb_suffix} num_train_frames=300000 device=cuda:${cuda_device} agent.no_taco=${no_taco}

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/${model_path}
done