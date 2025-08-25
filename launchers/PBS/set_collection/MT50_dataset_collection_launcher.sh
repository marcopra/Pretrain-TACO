#!/bin/bash

# Define environment lists (split into two for manageability)
env_lists=(
    "assembly-v2"
    "basketball-v2" 
    "bin-picking-v2" 
    "box-close-v2" 
    "button-press-topdown-v2" 
    "button-press-topdown-wall-v2" 
    "button-press-v2" 
    "button-press-wall-v2" 
    "coffee-button-v2" 
    "coffee-pull-v2" 
    "coffee-push-v2" 
    "dial-turn-v2" 
    "disassemble-v2" 
    "door-close-v2" 
    "door-lock-v2" 
    "door-open-v2" 
    "door-unlock-v2" 
    "hand-insert-v2" 
    "drawer-close-v2" 
    "drawer-open-v2" 
    "faucet-open-v2"
    "faucet-close-v2" 
    "hammer-v2" 
    "handle-press-side-v2" 
    "handle-press-v2" 
    "handle-pull-side-v2" 
    "handle-pull-v2" 
    "lever-pull-v2" 
    "pick-place-wall-v2" 
    "pick-out-of-hole-v2" 
    "pick-place-v2" 
    "plate-slide-v2" 
    "plate-slide-side-v2" 
    "plate-slide-back-v2" 
    "plate-slide-back-side-v2" 
    "peg-insert-side-v2" 
    "peg-unplug-side-v2" 
    "soccer-v2" 
    "stick-push-v2" 
    "stick-pull-v2" 
    "push-v2" 
    "push-wall-v2" 
    "push-back-v2" 
    "reach-v2" 
    "reach-wall-v2" 
    "shelf-place-v2" 
    "sweep-into-v2" 
    "sweep-v2" 
    "window-open-v2" 
    "window-close-v2"
)

# Define dataset sizes
n_episodes=(15)

# Define expert probabilities
expert_probs_options=("0.0" "0.33" "0.66" "0.99")

# Define tasks array
tasks=(
    "0"
    )

# Define save path
config="MT50"

# Define randomization options
random_init_options=("true")
random_goal_options=("false")


# Loop through all combinations and submit jobs
for env_list in "${env_lists[@]}"; do
    for eps in "${n_episodes[@]}"; do
        for expert_probs in "${expert_probs_options[@]}"; do
            for task in "${tasks[@]}"; do
                for random_init in "${random_init_options[@]}"; do
                    for random_goal in "${random_goal_options[@]}"; do
                        # Create a descriptive job name
                        job_name="MT50_ds${eps}_exp${expert_probs//,/_}_ri${random_init}_rg${random_goal}"
                        
                        echo "Submitting job: $job_name"
                        echo "Environment list: ${env_list:0:30}..."
                        echo "Task: $task"
                        echo "N Episodes: $eps"
                        echo "Expert probs: $expert_probs"
                        echo "Random init: $random_init"
                        echo "Random goal: $random_goal"
                        
                        # Submit the job with variables - escape commas in variables
                        env_list_escaped="${env_list//,/\\,}"
                        expert_probs_escaped="${expert_probs//,/\\,}"
                        task_escaped="${task//,/\\,}"
                        
                        qsub -N "$job_name" \
                            -v ENV_LIST="$env_list_escaped",EPISODES="$eps",EXPERT_PROBS="$expert_probs_escaped",CONFIG="$config/$expert_probs_escaped",RANDOM_INIT="$random_init",RANDOM_GOAL="$random_goal",TASKS="$task_escaped" launchers/data_collection/script_dataset_collection.sh

                        echo "Job submitted: $job_name"
                        echo "----------------------------------------"
                    done
                done
            done
        done
    done
done

echo "All dataset collection jobs have been submitted."
