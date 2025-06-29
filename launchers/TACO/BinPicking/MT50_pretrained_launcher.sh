#!/bin/bash

seeds="1 1 1 1 1 1 1"
env_names=("bin-picking-v2")
model_paths=(
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_BinPicking_0.0_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_BinPicking_0.33_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_BinPicking_0.66_lr=0.0005_ts=200000512_curl_rew.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_BinPicking_0.99_lr=0.0005_ts=50000896_curl_rew.pt" 
    "/home/mprattico/Pretrain-TACO/models/taco_MT_MT20_30OOD_ID_ButtonPressTopdownBPTWallCoffeeButtonDisassembleDoorLockDOpenDrawerCloseDOpenHandInsertHPressHPullLeverPPegUnplugSidePlateSSidePlateSSoccerStickPullSweepIntoS_0.0_lr=0.0005_ts=77977600_curl_rew_best.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT1_49OOD_ID_DrawerOpen_0.0_lr=0.0005_ts=25344000_curl_rew_best.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT5_45OOD_ID_DisassembleDrawerOpenLeverPullPlateSlideSweep_0.0_lr=0.0005_ts=1894400_curl_rew_best.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT10_40OOD_ID_ButtonPressTopdownWallDisassembleDoorOpenDrawerOpenHandlePressLeverPullPlateSlideBackSidePlateSlideStickPullSweep_0.0_lr=0.0005_ts=69171200_curl_rew_best.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT30_20OOD_AssemblyBasketballBinPickingButtonPressCoffeePullCPushDialTurnFaucetCloseHammerHandlePressSidePickOutHolePPlacePPWallPushPWallReachRWallShelfPlaceWindowCloseWOpen_0.0_lr=0.0005_ts=149043200_curl_rew_best.pt"
    # "/home/mprattico/Pretrain-TACO/models/taco_MT_MT40_10OOD_BasketballBinPickingButtonPressCoffeePushHammerPickPlacePushReachShelfPlaceWindowClose_0.0_lr=0.0005_ts=155699200_curl_rew_best.pt"

)
   
random_hand_inital="true"
random_goal_inital="false"
wandb_tag="MT50"

for random_hand in $random_hand_inital; do
    for random_goal in $random_goal_inital; do
        for seed in $seeds; do
            for env_name in "${env_names[@]}"; do
                for model_path in "${model_paths[@]}"; do
                echo "Submitting job: ENV_NAME=${env_name}, RANDOM_HAND=${random_hand}, RANDOM_GOAL=${random_goal}, SEED=${seed}"
                qsub -v SEED="${seed}",ENV_NAME="${env_name}",RANDOM_HAND="${random_hand}",RANDOM_GOAL="${random_goal}",MODEL_PATH="${model_path}",WANDB_TAG="${wandb_tag}" launchers/TACO/BinPicking/MT_pretrained.sh
                done
            done
        done
    done
done
