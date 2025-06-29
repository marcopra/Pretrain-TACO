#!/bin/bash


# Check if cuda_device argument is provided
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <cuda_device> [no_taco] [model_num]"
    echo "cuda_device: 0 to 7"
    echo "no_taco: true or false (default: false)"
    echo "model_num: 1, 10, 20, 30, 40, 45 (default: 10)"
    exit 1
fi

no_taco=${1:-false}
cuda_device=$2
model_num=${3:-10}

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

# Validate model_num value and set model path
case $model_num in
    1)
        model_path="taco_MT_MT1_49OOD_ID_DrawerOpen_0.0_lr\=0.0005_ts\=25344000_curl_rew_best.pt"
        ;;
    10)
        model_path="taco_MT_MT10_40OOD_ID_ButtonPressTopdownWallDisassembleDoorOpenDrawerOpenHandlePressLeverPullPlateSlideBackSidePlateSlideStickPullSweep_0.0_lr\=0.0005_ts\=69171200_curl_rew_best.pt"
        ;;
    20)
        model_path="taco_MT_MT20_30OOD_ID_ButtonPressTopdownBPTWallCoffeeButtonDisassembleDoorLockDOpenDrawerCloseDOpenHandInsertHPressHPullLeverPPegUnplugSidePlateSSidePlateSSoccerStickPullSweepIntoS_0.0_lr\=0.0005_ts\=77977600_curl_rew_best.pt"
        ;;
    30)
        model_path="taco_MT_MT30_20OOD_AssemblyBasketballBinPickingButtonPressCoffeePullCPushDialTurnFaucetCloseHammerHandlePressSidePickOutHolePPlacePPWallPushPWallReachRWallShelfPlaceWindowCloseWOpen_0.0_lr\=0.0005_ts\=149043200_curl_rew_best.pt"
        ;;
    40)
        model_path="taco_MT_MT40_10OOD_BasketballBinPickingButtonPressCoffeePushHammerPickPlacePushReachShelfPlaceWindowClose_0.0_lr\=0.0005_ts\=155699200_curl_rew_best.pt"
        ;;
    45)
        model_path="taco_MT_MT45_5OOD_BasketballBinPickingButtonPressPushShelfPlace_0.0_lr\=0.0005_ts\=154931200_curl_rew_best.pt"
        ;;
    *)
        echo "Error: Invalid model_num value. Allowed values are 1, 10, 20, 30, 40, 45."
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
    python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/${model_path}"  exp_name="/home/mprattico/Pretrain-TACO/models/${model_path}" seed=1 env_name=shelf-place-v2 random_init=true random_goal=false wandb_tag=MT${model_num}${wandb_suffix} num_train_frames=300000 device=cuda:${cuda_device} agent.no_taco=${no_taco}

    # Cleanup command
    rm -rf exp_local/metaworld/home/mprattico/Pretrain-TACO/models/${model_path}
done