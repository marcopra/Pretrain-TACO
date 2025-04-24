#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco


# Define environment names and expert probability values
envs="assembly-v2,basketball-v2,bin-picking-v2,box-close-v2,button-press-topdown-v2,button-press-topdown-wall-v2,button-press-v2,button-press-wall-v2,coffee-button-v2,coffee-pull-v2,coffee-push-v2,dial-turn-v2,disassemble-v2,door-close-v2,door-lock-v2,door-open-v2,door-unlock-v2,hand-insert-v2,drawer-close-v2,drawer-open-v2,faucet-open-v2,faucet-close-v2,hammer-v2,handle-press-side-v2,handle-press-v2,handle-pull-side-v2,handle-pull-v2,lever-pull-v2,pick-place-wall-v2,pick-out-of-hole-v2,pick-place-v2,plate-slide-v2,plate-slide-side-v2,plate-slide-back-v2,plate-slide-back-side-v2,peg-insert-side-v2,peg-unplug-side-v2,soccer-v2,stick-push-v2,stick-pull-v2,push-v2,push-wall-v2,push-back-v2,reach-v2,reach-wall-v2,shelf-place-v2,sweep-into-v2,sweep-v2,window-open-v2,window-close-v2"
exp_probs="0.4,0.2"


for prob in $exp_probs; do
    for env in $envs; do
        python metaworld2_image_dataset.py --env_names $env --dataset_size 2000 --expert_probs $prob --save_path "data/mt50/"
    done
done