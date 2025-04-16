#!/bin/bash

# Define maximum number of tmux sessions
MAX_SESSIONS=2

# Define environment names and expert probability values
envs="push-v2 window-open-v2 door-open-v2 reach-v2 drawer-close-v2 pick-place-v2 drawer-open-v2 button-press-topdown-v2 window-close-v2 peg-insert-side-v2"
exp_probs="0.4 0.2"

# Initialize session counter
session_counter=1

# Create up to MAX_SESSIONS tmux sessions
for env in $envs; do
    for prob in $exp_probs; do
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
        
        # Send the command with appropriate environment and probability
        echo "Running with env=$env and exp_prob=$prob on session$session_id"
        tmux send-keys -t "session$session_id" "python metaworld_image_dataset.py --env_names $env --dataset_size 100000 --expert_probs $prob" C-m
        
        # Wait a bit before proceeding to the next command
        sleep 1
        
        # Increment counter
        session_counter=$((session_counter + 1))
    done
done