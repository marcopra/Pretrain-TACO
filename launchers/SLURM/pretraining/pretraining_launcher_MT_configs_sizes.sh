#!/bin/bash

# Define dataset config paths
dataset_configs=(
    "configs_data_MT/MT49OODBasketball.json"
    # "configs_data_MT/MT49OODBinPicking.json"
    # "configs_data_MT/MT49OODButtonPress.json"
    # "configs_data_MT/MT49OODPush.json"
    # "configs_data_MT/MT49OODShelfPlace.json"
)

feature_extractors=(
    "vit_s_scratch"
    "resnet50_l5_scratch"
    # "conv"
)
# Define dataset sizes to experiment with
dataset_sizes=(1000000)

exps_values=("0.0")

# Submit jobs for all combinations
for dataset_config in "${dataset_configs[@]}"; do
    for dataset_size in "${dataset_sizes[@]}"; do
        for exps in "${exps_values[@]}"; do
            for fe in "${feature_extractors[@]}"; do
                echo "Submitting SLURM job: DATASET_CONFIG=${dataset_config}, DATASET_SIZE=${dataset_size}, EXPS=${exps}, FEATURE_EXTRACTOR=${fe}"
                sbatch --export=DATASET_CONFIG="${dataset_config}",DATASET_SIZE="${dataset_size}",EXPS="${exps}",FEATURE_EXTRACTOR="${fe}" launchers/SLURM/pretraining/pretraining_MT_configs.sh
            done
        done
    done
done