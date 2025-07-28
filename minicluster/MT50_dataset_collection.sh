#!/bin/bash

# MT50 Dataset Collection Script
# This script collects data from all MT50 environments using collect_metaworld_episodes.py

echo "=== MT50 Dataset Collection Script ==="
echo "Starting data collection for all MT50 environments..."

# Define all MT50 environments (converted to v2)
mt50_envs=(
    "assembly-v2" "basketball-v2" "bin-picking-v2" "box-close-v2" 
    "button-press-topdown-v2" "button-press-topdown-wall-v2" "button-press-v2" "button-press-wall-v2" 
    "coffee-button-v2" "coffee-pull-v2" "coffee-push-v2" "dial-turn-v2" "disassemble-v2" 
    "door-close-v2" "door-lock-v2" "door-open-v2" "door-unlock-v2" "hand-insert-v2" 
    "drawer-close-v2" "drawer-open-v2" "faucet-open-v2" "faucet-close-v2" "hammer-v2" 
    "handle-press-side-v2" "handle-press-v2" "handle-pull-side-v2" "handle-pull-v2" "lever-pull-v2" 
    "pick-place-wall-v2" "pick-out-of-hole-v2" "pick-place-v2" "plate-slide-v2" "plate-slide-side-v2" 
    "plate-slide-back-v2" "plate-slide-back-side-v2" "peg-insert-side-v2" "peg-unplug-side-v2" 
    "soccer-v2" "stick-push-v2" "stick-pull-v2" "push-v2" "push-wall-v2" "push-back-v2" 
    "reach-v2" "reach-wall-v2" "shelf-place-v2" "sweep-into-v2" "sweep-v2" 
    "window-open-v2" "window-close-v2"
)

# Expert probability options
expert_probs_options=("0.99") # "0.66" "0.33" "0.0")

# Number of episodes to collect
n_episodes=50

# Task (using task 0 for all environments)
task="0"

# Configuration options
random_init="true"
random_goal="false"

# Base configuration name
base_config="FullDataset"

# Total number of jobs
total_jobs=$((${#mt50_envs[@]} * ${#expert_probs_options[@]}))
current_job=0

echo "Total environments: ${#mt50_envs[@]}"
echo "Expert probabilities: ${expert_probs_options[@]}"
echo "Episodes per configuration: $n_episodes"
echo "Total jobs to run: $total_jobs"
echo "Random init: $random_init"
echo "Random goal: $random_goal"
echo ""

# Function to run data collection for a single configuration
run_collection() {
    local env_name=$1
    local expert_prob=$2
    local job_num=$3
    
    echo "[$job_num/$total_jobs] Starting collection for $env_name with expert_prob=$expert_prob"
    
    # Create config path that includes expert probability
    config_path="${base_config}/${expert_prob}"
    
    # Prepare random init and random goal flags
    local random_init_flag="--random_init"
    if [ "$random_init" = "false" ]; then
        random_init_flag="--no_random_init"
    fi
    
    local random_goal_flag="--randomize_goal"
    if [ "$random_goal" = "false" ]; then
        random_goal_flag="--no_randomize_goal"
    fi
    
    # Run the data collection
    python3 collect_metaworld_episodes.py \
        --env_names "$env_name" \
        --num_episodes $n_episodes \
        --expert_probs "$expert_prob" \
        --config_name "$config_path" \
        --tasks "$task" \
        --log_level INFO \
        --stop_on_success \
        $random_init_flag \
        $random_goal_flag
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$job_num/$total_jobs] ✓ Successfully completed $env_name with expert_prob=$expert_prob"
    else
        echo "[$job_num/$total_jobs] ✗ Failed $env_name with expert_prob=$expert_prob (exit code: $exit_code)"
    fi
    
    echo ""
    return $exit_code
}

# Main collection loop
echo "Starting data collection..."
echo "=========================================="

failed_jobs=()
successful_jobs=0

for env_name in "${mt50_envs[@]}"; do
    for expert_prob in "${expert_probs_options[@]}"; do
        current_job=$((current_job + 1))
        
        if run_collection "$env_name" "$expert_prob" "$current_job"; then
            successful_jobs=$((successful_jobs + 1))
        else
            failed_jobs+=("$env_name (expert_prob=$expert_prob)")
        fi
        
        # Add a small delay between jobs to avoid overwhelming the system
        sleep 2
    done
done

# Summary
echo "=========================================="
echo "Data collection completed!"
echo ""
echo "Summary:"
echo "  Total jobs: $total_jobs"
echo "  Successful: $successful_jobs"
echo "  Failed: ${#failed_jobs[@]}"
echo ""

if [ ${#failed_jobs[@]} -gt 0 ]; then
    echo "Failed jobs:"
    for job in "${failed_jobs[@]}"; do
        echo "  - $job"
    done
    echo ""
fi

echo "Data saved in: data_episodes/$base_config/"
echo "Organization structure:"
echo "  data_episodes/"
echo "  └── $base_config/"
echo "      ├── expert_0.0/"
echo "      │   └── pretraining_datasets/"
echo "      │       ├── env1_task0_fs3_ar2_ri1_rg0_exp=0/"
echo "      │       └── ..."
echo "      ├── expert_0.33/"
echo "      ├── expert_0.66/"
echo "      └── expert_0.99/"
echo ""

if [ ${#failed_jobs[@]} -eq 0 ]; then
    echo "🎉 All data collection jobs completed successfully!"
    exit 0
else
    echo "⚠️  Some jobs failed. Check the output above for details."
    exit 1
fi
