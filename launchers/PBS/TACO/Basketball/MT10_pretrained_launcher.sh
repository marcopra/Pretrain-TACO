#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=("basketball-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/mt/MT10/taco_MT_MT10IDAssemblyButtonPressTopdownDoorUnlockFaucetCloseHandlePressHandlePressSideHandlePullPickOutOfHolePickPlacePushWall.json_lr=0.0005_ts=57856000_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT10/taco_MT_MT10IDBoxCloseDoorUnlockDrawerCloseFaucetCloseFaucetOpenPickOutOfHolePushBackSweepSweepIntoWindowClose.json_lr=0.0005_ts=75417600_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT10/taco_MT_MT10IDBoxCloseDrawerCloseHandlePressSideHandlePullSideLeverPullPickOutOfHolePlateSlideBackSidePlateSlideSideReachSweepInto.json_lr=0.0005_ts=51200_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT10/taco_MT_MT10IDButtonPressTopdownCoffeeButtonCoffeePushDisassembleDoorLockDoorOpenDrawerOpenPickOutOfHolePickPlaceStickPull.json_lr=0.0005_ts=183961600_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT10/taco_MT_MT10IDCoffeeButtonCoffeePushDoorUnlockFaucetOpenHandlePressHandlePullSidePlateSlidePlateSlideBackPlateSlideSideWindowOpen.json_lr=0.0005_ts=36966400_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT10_BASKETBALL"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_MT" launchers/PBS/TACO/Basketball/MT_pretrained.sh
                done
            done
        done
    done
done
