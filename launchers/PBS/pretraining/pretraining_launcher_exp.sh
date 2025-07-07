#!/bin/bash

# Define parameter combinations
dataset_names=("MT50_OOD_Push" "MT50_OOD_Basketball" "MT50_OOD_BinPicking" "MT50_OOD_ButtonPress" "MT50_OOD_ShelfPlace")
exps_values=("0.99")

# Submit jobs for all combinations
for dataset_name in "${dataset_names[@]}"; do
    for exps in "${exps_values[@]}"; do
        echo "Submitting job: DATASET_NAME=${dataset_name}, EXPS=${exps}"
        qsub -v DATASET_NAME="${dataset_name}",EXPS="${exps}" launchers/pretraining/pretraining.sh
    done
done
