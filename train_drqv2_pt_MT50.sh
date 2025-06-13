#!/bin/bash

# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [freeze_encoder] [exp]"
    echo "cuda_device: 0 to 7"
    echo "freeze_encoder: true or false (default: false)"
    echo "exp: 0, 33, 66, 99 (default: 0)"
    exit 1
fi

freeze_encoder=${1:-false}
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

# Set suffix for experiment name based on freeze_encoder value
if [ "$freeze_encoder" = "true" ]; then
    wandb_suffix="-FREEZE"
else
    wandb_suffix=""
fi

# Set ts value based on exp
if [ "$exp" = "99" ]; then
    ts_value="50000896"
else
    ts_value="200000512"
fi

# Number of runs
num_runs=9

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py --config-name config_drqv2metaworld agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Push_0.${exp}_lr\=0.0005_ts\=${ts_value}_curl_rew.pt" exp_name="/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Push_0.${exp}_lr\=0.0005_ts\=${ts_value}_curl_rew.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=MT50${wandb_suffix} num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=${freeze_encoder}

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Push_0.${exp}_lr\=0.0005_ts\=${ts_value}_curl_rew.pt
done
