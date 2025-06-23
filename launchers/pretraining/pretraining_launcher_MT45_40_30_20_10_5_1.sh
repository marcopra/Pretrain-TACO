#!/bin/bash

# Define parameter combinations
dataset_names=("MT45_5OOD_BasketballBinPickingButtonPressPushShelfPlace" "MT40_10OOD_BasketballBinPickingButtonPressCoffeePushHammerPickPlacePushReachShelfPlaceWindowClose" "MT30_20OOD_AssemblyBasketballBinPickingButtonPressCoffeePullCoffeePushDialTurnFaucetCloseHammerHandlePressSidePickOutOfHolePickPlacePickPlaceWallPushPushWallReachReachWallShelfPlaceWindowCloseWindowOpen" 
            "MT20_30OOD_ID_ButtonPressTopdownButtonPressTopdownWallCoffeeButtonDisassembleDoorLockDoorOpenDrawerCloseDrawerOpenHandInsertHandlePressHandlePullLeverPullPegUnplugSidePlateSlideSidePlateSlideSoccerStickPullSweepIntoSweep" "MT10_40OOD_ID_ButtonPressTopdownWallDisassembleDoorOpenDrawerOpenHandlePressLeverPullPlateSlideBackSidePlateSlideStickPullSweep"
            "MT5_45OOD_ID_DisassembleDrawerOpenLeverPullPlateSlideSweep" "MT1_49OOD_ID_DrawerOpen")
exps_values=("0.0")

# Submit jobs for all combinations
for dataset_name in "${dataset_names[@]}"; do
    for exps in "${exps_values[@]}"; do
        echo "Submitting job: DATASET_NAME=${dataset_name}, EXPS=${exps}"
        qsub -v DATASET_NAME="${dataset_name}",EXPS="${exps}" launchers/pretraining/pretraining.sh
    done
done
