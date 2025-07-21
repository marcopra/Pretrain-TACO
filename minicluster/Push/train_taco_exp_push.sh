#!/bin/bash

# Number of runs
num_runs=28

# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco]"
    echo "cuda_device: 0 to 5 (0,3->33%, 1,4->66%, 2,5->99%)"
    echo "no_taco: true or false (default: false)"
    exit 1
fi

cuda_device=$1
no_taco=${2:-false}

# Validate cuda_device value and set model path and gpu mapping
case $cuda_device in
    0|3)
        model_path="models/exp/33/taco_MT_MT50_OOD_Push_0.33_lr\=0.0005_ts\=200000512_curl_rew.pt"
        wandb_tag="33_PUSH"
        gpu_device="$cuda_device"
        ;;
    1|4)
        model_path="models/exp/66/taco_MT_MT50_OOD_Push_0.66_lr\=0.0005_ts\=200000512_curl_rew.pt"
        wandb_tag="66_PUSH"
        gpu_device="$cuda_device"
        ;;
    2|5)
        model_path="models/exp/99/taco_MT_MT50_OOD_Push_0.99_lr\=0.0005_ts\=11980800_curl_rew_best.pt"
        wandb_tag="99_PUSH"
        gpu_device="$cuda_device"
        ;;
    *)
        echo "Error: Invalid cuda_device value. Allowed values are 0 to 5."
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
    wandb_suffix="-NOTACO"
    wandb_tag="${wandb_tag}-NOTACO"
fi

# Run experiments in a loop
for i in $(seq 1 $num_runs); do
    echo "Running experiment $i of $num_runs"
    
    # Run the Python command with the appropriate arguments
    python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/${model_path}" exp_name="/home/mprattico/Pretrain-TACO/${model_path}" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=${wandb_tag} wandb_project="TACO_EXP" num_train_frames=220000 device=cuda:${gpu_device} agent.no_taco=${no_taco}

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/${model_path}
done
