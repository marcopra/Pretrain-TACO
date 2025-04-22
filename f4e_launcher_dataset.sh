#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco


# Define environment names and expert probability values
envs="push-v2,window-open-v2,door-open-v2,reach-v2,drawer-close-v2,pick-place-v2,drawer-open-v2,button-press-topdown-v2,window-close-v2,peg-insert-side-v2"
exp_probs="0.4,0.2"


# Create up to MAX_SESSIONS tmux sessions
for env in $envs; do
    for prob in $exp_probs; do
        python metaworld_image_dataset.py --env_names $env --dataset_size 10000 --expert_probs $prob
    done
done