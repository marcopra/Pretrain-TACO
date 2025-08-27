#!/bin/bash

# Define dataset config paths
dataset_configs=(
    "configs_data_MT/MT49OODBasketball.json"
    "configs_data_MT/MT49OODBinPicking.json"
    "configs_data_MT/MT49OODButtonPress.json"
    "configs_data_MT/MT49OODPush.json"
    "configs_data_MT/MT49OODShelfPlace.json"
)

# Define dataset sizes to experiment with
dataset_sizes=(30000 500000 1000000)

exps_values=("0.0")

# Submit jobs for all combinations
for dataset_config in "${dataset_configs[@]}"; do
    for dataset_size in "${dataset_sizes[@]}"; do
        for exps in "${exps_values[@]}"; do
            echo "Submitting PBS job: DATASET_CONFIG=${dataset_config}, DATASET_SIZE=${dataset_size}, EXPS=${exps}"
            qsub -v DATASET_CONFIG="${dataset_config}",DATASET_SIZE="${dataset_size}",EXPS="${exps}" launchers/PBS/pretraining/pretraining_MT_configs.sh
        done
    done
done
