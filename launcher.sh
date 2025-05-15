#!/bin/bash
# Define maximum number of tmux sessions
MAX_SESSIONS=10

env_lists=(
    "push-v2"
)

# Define expert probabilities
expert_probs_options=("0.99")

tasks=(
    "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24"
    "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49"
    )

# Initialize session counter
session_counter=1

# Define save path
config="ST50"


# Create up to MAX_SESSIONS tmux sessions
# Loop through all combinations and submit jobs
for env in "${env_lists[@]}"; do
    for prob in "${expert_probs_options[@]}"; do
        for task in "${tasks[@]}"; do
            # Use modulo to cycle through sessions 1-MAX_SESSIONS
            session_id=$((session_counter % MAX_SESSIONS))
            if [ $session_id -eq 0 ]; then
                session_id=$MAX_SESSIONS
            fi
            
            # Create session if it doesn't exist yet
            if ! tmux has-session -t "session$session_id" 2>/dev/null; then
                echo "Starting session $session_id"
                tmux new-session -d -s "session$session_id"
                sleep 0.6  # ensure the session and its default pane are created
                tmux send-keys -t "session$session_id" "conda activate metataco" C-m
            fi
            expert_prob_escaped="${prob//,/\\,}"
            # Send the command with appropriate environment and probability
            echo "Running with env=$env and exp_prob=$prob on session$session_id"
            tmux send-keys -t "session$session_id" "python collect_metaworld_episodes.py --env_names $env --num_episodes 50 --expert_probs $prob --config_name "$config/$expert_prob_escaped" --log_level DEBUG  --task $task --random_init --no_randomize_goal --stop_on_success" C-m
            
            # Wait a bit before proceeding to the next command
            sleep 1
            
            # Increment counter
            session_counter=$((session_counter + 1))
        done
    done
done