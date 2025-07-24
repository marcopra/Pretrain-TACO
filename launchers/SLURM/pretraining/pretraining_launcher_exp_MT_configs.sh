#!/bin/bash

# Define dataset config paths
dataset_configs=(
    # "configs_data_MT/exp/0.33/MT49OODBasketball.json"
    #  "configs_data_MT/exp/0.33/MT49OODBinPicking.json"
    #  "configs_data_MT/exp/0.33/MT49OODButtonPress.json"
    #  "configs_data_MT/exp/0.33/MT49OODPush.json"
    #  "configs_data_MT/exp/0.33/MT49OODShelfPlace.json"
    #  "configs_data_MT/exp/0.66/MT49OODBasketball.json"
    #  "configs_data_MT/exp/0.66/MT49OODBinPicking.json"
     "configs_data_MT/exp/0.66/MT49OODButtonPress.json"
     "configs_data_MT/exp/0.66/MT49OODPush.json"
     "configs_data_MT/exp/0.66/MT49OODShelfPlace.json"
     "configs_data_MT/exp/0.99/MT49OODBasketball.json"
     "configs_data_MT/exp/0.99/MT49OODBinPicking.json"
     "configs_data_MT/exp/0.99/MT49OODButtonPress.json"
     "configs_data_MT/exp/0.99/MT49OODPush.json"
     "configs_data_MT/exp/0.99/MT49OODShelfPlace.json"
)

# Submit jobs for all combinations
for dataset_config in "${dataset_configs[@]}"; do
    for exps in "${exps_values[@]}"; do
        echo "Submitting SLURM job: DATASET_CONFIG=${dataset_config}, EXPS=${exps}"
        sbatch --export=DATASET_CONFIG="${dataset_config}",EXPS="${exps}" launchers/SLURM/pretraining/pretraining.sh
    done
done
