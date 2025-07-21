#!/bin/bash

# Define dataset config paths
dataset_configs=(
    # "configs_data_MT/MT1IDButtonPressTopdownWall.json"
    # "configs_data_MT/MT1IDDrawerOpen.json"
    # "configs_data_MT/MT1IDHandlePressSide.json"
    # "configs_data_MT/MT1IDPushWall.json"
    # "configs_data_MT/MT1IDSweep.json"
    # "configs_data_MT/MT10IDAssemblyButtonPressTopdownDoorUnlockFaucetCloseHandlePressHandlePressSideHandlePullPickOutOfHolePickPlacePushWall.json"
    # "configs_data_MT/MT10IDBoxCloseDoorUnlockDrawerCloseFaucetCloseFaucetOpenPickOutOfHolePushBackSweepSweepIntoWindowClose.json"
    # "configs_data_MT/MT10IDBoxCloseDrawerCloseHandlePressSideHandlePullSideLeverPullPickOutOfHolePlateSlideBackSidePlateSlideSideReachSweepInto.json"
    # "configs_data_MT/MT10IDButtonPressTopdownCoffeeButtonCoffeePushDisassembleDoorLockDoorOpenDrawerOpenPickOutOfHolePickPlaceStickPull.json"
    # "configs_data_MT/MT10IDCoffeeButtonCoffeePushDoorUnlockFaucetOpenHandlePressHandlePullSidePlateSlidePlateSlideBackPlateSlideSideWindowOpen.json"
    # "configs_data_MT/MT30OODAsseBaskBinPBoxCButtButtButtCoffDisaDoorDoorDoorDrawHammHandPickPushPushShelStic.json"
    # "configs_data_MT/MT30OODAsseBaskBinPBoxCButtButtDoorDrawFaucPegUPickPlatPlatPushPushShelSoccSticSweeWind.json"
    # "configs_data_MT/MT30OODAsseBaskBinPBoxCButtDisaDrawFaucHandHandHandHandPegIPlatPushPushReacReacShelStic.json"
    # "configs_data_MT/MT30OODAsseBaskBinPButtCoffCoffDialDoorHandHandPegUPickPickPlatPlatPushPushReacShelSwee.json"
    # "configs_data_MT/MT30OODBaskBinPButtButtDialDoorDoorDoorHammHandPickPlatPlatPushPushReacReacShelSticSwee.json"
    # "configs_data_MT/MT40OODBasketballBinPickingButtonPressButtonPressTopdownWallDialTurnDoorCloseDrawerOpenPushShelfPlaceStickPush.json"
    # "configs_data_MT/MT40OODBasketballBinPickingButtonPressCoffeePullDoorCloseDoorOpenPushPushBackPushWallShelfPlace.json"
    # "configs_data_MT/MT40OODBasketballBinPickingButtonPressDoorCloseHandInsertLeverPullPegUnplugSidePushReachShelfPlace.json"
    # "configs_data_MT/MT40OODBasketballBinPickingButtonPressDoorLockDoorUnlockHandlePullPushShelfPlaceSweepIntoWindowClose.json"
    # "configs_data_MT/MT40OODBasketballBinPickingButtonPressDrawerCloseDrawerOpenFaucetOpenPickOutOfHolePushShelfPlaceWindowClose.json"
    "configs_data_MT/MT49OODBasketball.json"
    "configs_data_MT/MT49OODBinPicking.json"
    "configs_data_MT/MT49OODButtonPress.json"
    "configs_data_MT/MT49OODPush.json"
    "configs_data_MT/MT49OODShelfPlace.json"

)
exps_values=("0.0")

# Submit jobs for all combinations
for dataset_config in "${dataset_configs[@]}"; do
    for exps in "${exps_values[@]}"; do
        echo "Submitting PBS job: DATASET_CONFIG=${dataset_config}, EXPS=${exps}"
        qsub -v DATASET_CONFIG="${dataset_config}",EXPS="${exps}" launchers/PBS/pretraining/pretraining_MT_configs.sh
    done
done
