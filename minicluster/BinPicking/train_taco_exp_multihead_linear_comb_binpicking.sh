#!/bin/bash

# Number of runs
num_runs=7

# Check if arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> <model_exp> [no_taco]"
    echo "cuda_device: 0 to 7"
    echo "model_exp: 0, 33, 66, 99"
    echo "no_taco: true or false (default: false)"
    exit 1
fi

cuda_device=$1
model_exp=$2
no_taco=${3:-false}

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

# Validate model_exp value and set model path
case $model_exp in
    0)
        model_path="models/multiheads/exp/0.0/taco_MT_MT49OODBinPicking.json_lr\=0.0005_ts\=121753600_curl_rew_best.pt"
        wandb_tag="0_BINPICKING"
        ;;
    33)
        model_path="models/multiheads/exp/0.33/taco_MT_exp_0.33_MT49OODBinPicking.json_lr\=0.0005_ts\=194508800_curl_rew_best.pt"
        wandb_tag="33_BINPICKING"
        ;;
    66)
        model_path="models/multiheads/exp/0.66/taco_MT_exp_0.66_MT49OODBinPicking.json_lr\=0.0005_ts\=194406400_curl_rew_best.pt"
        wandb_tag="66_BINPICKING"
        ;;
    99)
        model_path="models/multiheads/exp/0.99/taco_MT_exp_0.99_MT49OODBinPicking.json_lr\=0.0005_ts\=23808000_curl_rew_best.pt"
        wandb_tag="99_BINPICKING"
        ;;
    *)
        echo "Error: Invalid model_exp value. Allowed values are 0, 33, 66, 99."
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
    python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/${model_path}" exp_name="/home/mprattico/Pretrain-TACO/${model_path}" seed=1 env_name=bin-picking-v2 random_init=true random_goal=false wandb_tag=${wandb_tag} wandb_project="taco_multihead_linear_comb" num_train_frames=220000 device=cuda:${cuda_device} agent.no_taco=${no_taco}  agent=taco_multiheads_linear_comb

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/${model_path}
done
