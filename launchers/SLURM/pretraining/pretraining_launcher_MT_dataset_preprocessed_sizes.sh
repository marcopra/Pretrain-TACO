#!/bin/bash

# Define dataset config paths
dataset_configs=(
    "data_episodes/MT49OODBasketball"
)

feature_extractors=(
    # "vit_s"
    "resnet18_l5"
    # "conv"
)
# Define dataset sizes to experiment with
dataset_sizes=(1200000)

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