#!/bin/bash

# Define environment lists (split into two for manageability)
env_lists=(
    "push-v2,window-open-v2,door-open-v2,reach-v2,drawer-close-v2,pick-place-v2,drawer-open-v2,button-press-topdown-v2,window-close-v2,peg-insert-side-v2"
)

# Define dataset sizes
dataset_sizes=(10000)

# Define expert probabilities
expert_probs_options=("0.2,0.4" "0.6,0.8")

# Define save path
save_path="data/MT10/"

# Define randomization options
random_init_options=("true")
random_goal_options=("false")

# Loop through all combinations and submit jobs
for env_list in "${env_lists[@]}"; do
    for dataset_size in "${dataset_sizes[@]}"; do
        for expert_probs in "${expert_probs_options[@]}"; do
            for random_init in "${random_init_options[@]}"; do
                for random_goal in "${random_goal_options[@]}"; do
                    # Create a descriptive job name
                    job_name="MT10_ds${dataset_size}_ep${expert_probs//,/_}_ri${random_init}_rg${random_goal}"
                    
                    echo "Submitting job: $job_name"
                    echo "Environment list: ${env_list:0:30}..."
                    echo "Dataset size: $dataset_size"
                    echo "Expert probs: $expert_probs"
                    echo "Random init: $random_init"
                    echo "Random goal: $random_goal"
                    
                    # Submit the job with variables
                    qsub -N "$job_name" \
                         -v ENV_LIST="$env_list",DATASET_SIZE="$dataset_size",EXPERT_PROBS="$expert_probs",SAVE_PATH="$save_path",RANDOM_INIT="$random_init",RANDOM_GOAL="$random_goal" \
                         launchers/script_dataset_collection.sh
                    
                    echo "Job submitted: $job_name"
                    echo "----------------------------------------"
                done
            done
        done
    done
done

echo "All dataset collection jobs have been submitted."
