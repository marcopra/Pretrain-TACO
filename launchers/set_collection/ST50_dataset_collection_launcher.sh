#!/bin/bash

# Define environment lists (split into two for manageability)
env_lists=(
    "push-v2"
)

# Define dataset sizes
n_episodes=(8)

# Define expert probabilities
expert_probs_options=("0.66") # "0.0" "0.33" "0.99") #"0.0" "0.33" "0.66" "0.99")
tasks=(
       	"0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24"
        "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49"
)


# Define save path
config="ST50"

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
                        job_name="ST50_ds${eps}_exp${expert_probs//,/_}_ri${random_init}_rg${random_goal}"
                        
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
                            -v ENV_LIST="$env_list_escaped",EPISODES="$eps",EXPERT_PROBS="$expert_probs_escaped",CONFIG="$config/$expert_probs_escaped",RANDOM_INIT="$random_init",RANDOM_GOAL="$random_goal",TASKS="$task_escaped" \
                            launchers/data_collection/script_dataset_collection.sh

                        echo "Job submitted: $job_name"
                        echo "----------------------------------------"
                    done
                done
            done
        done
    done
done

echo "All dataset collection jobs have been submitted."
echo "Please check the job scheduler for status."