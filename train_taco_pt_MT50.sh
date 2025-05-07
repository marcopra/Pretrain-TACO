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
    20)
        cuda_device=0
        ;;
    40)
        cuda_device=1
        ;;
    60)
        cuda_device=2
        ;;
    80)
        cuda_device=3
        ;;
    *)
        echo "Error: Invalid exp value. Allowed values are 20, 40, 60, or 80."
        exit 1
        ;;
esac

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" exp_name="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=MT50-FREEZE num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=true

# Cleanup command
rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt_push-v2

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" exp_name="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=MT50-FREEZE num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=true

# Cleanup command
rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt_push-v2

# Run the Python command with the appropriate arguments
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" exp_name="/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts\=200000512.pt" seed=1 env_name=push-v2 random_init=true random_goal=false wandb_tag=MT50-FREEZE num_train_frames=300000 device=cuda:${cuda_device} agent.freeze_encoder=true

# Cleanup command
rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/MT50/taco_MT_MT50_exp\=${exp}_1OOD_push_fs3_ar2_ri1_rg0_config_ts=200000512.pt_push-v2


