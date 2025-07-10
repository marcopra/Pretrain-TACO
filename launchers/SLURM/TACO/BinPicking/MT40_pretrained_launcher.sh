#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=("bin-picking-v2")
model_paths=(
    "/home/mprattico/Pretrain-TACO/models/mt/MT40/taco_MT_MT40OODBasketballBinPickingButtonPressButtonPressTopdownWallDialTurnDoorCloseDrawerOpenPushShelfPlaceStickPush.json_lr=0.0005_ts=93644800_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT40/taco_MT_MT40OODBasketballBinPickingButtonPressCoffeePullDoorCloseDoorOpenPushPushBackPushWallShelfPlace.json_lr=0.0005_ts=84992000_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT40/taco_MT_MT40OODBasketballBinPickingButtonPressDoorCloseHandInsertLeverPullPegUnplugSidePushReachShelfPlace.json_lr=0.0005_ts=87859200_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT40/taco_MT_MT40OODBasketballBinPickingButtonPressDoorLockDoorUnlockHandlePullPushShelfPlaceSweepIntoWindowClose.json_lr=0.0005_ts=51200_curl_rew_best.pt"
    "/home/mprattico/Pretrain-TACO/models/mt/MT40/taco_MT_MT40OODBasketballBinPickingButtonPressDrawerCloseDrawerOpenFaucetOpenPickOutOfHolePushShelfPlaceWindowClose.json_lr=0.0005_ts=11417600_curl_rew_best.pt"
)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT40_BINPICKING"
no_taco="false"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                sbatch --export=SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}",NO_TACO="${no_taco}",WANDB_PROJECT="taco_MT" launchers/SLURM/TACO/BinPicking/MT_pretrained.sh
                done
            done
        done
    done
done
