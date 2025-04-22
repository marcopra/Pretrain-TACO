#!/bin/bash

seeds="0 1 2"
env_names=("push-v2" "door-open-v2" "drawer-close-v2")

for env_name in "${env_names[@]}"; do
    for seed in $seeds; do
        qsub -v SEED="$seed",ENV_NAME="$env_name" launchers/base_run.sh
    done
done