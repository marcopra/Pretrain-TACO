#!/bin/bash

# Define parameter combinations
dataset_names=("ST50" "MT50")
exps_values=("0.0" "0.33" "0.66" "0.99")

# Submit jobs for all combinations
for dataset_name in "${dataset_names[@]}"; do
    for exps in "${exps_values[@]}"; do
        echo "Submitting job: DATASET_NAME=${dataset_name}, EXPS=${exps}"
        qsub -v DATASET_NAME="${dataset_name}",EXPS="${exps}" launchers/pretraining.sh
    done
done
