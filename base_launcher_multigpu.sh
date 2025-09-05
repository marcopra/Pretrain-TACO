#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco

torchrun --nproc-per-node=4 train_metaworld_ed4ct.py batch_size=256

python train_metaworld_checkpoint.py agent.pretrained_path=none exp_name=ButtonPressCheckpoint seed=1 env_name=button-press-v2 random_init=true random_goal=false wandb_tag=baseline wandb_project=checkpoint num_train_frames=500000 device=cuda:0 agent.freeze_encoder=false
python train_metaworld_checkpoint.py agent.pretrained_path=none exp_name=BasketballCheckpoint seed=1 env_name=basketball-v2 random_init=true random_goal=false wandb_tag=baseline wandb_project=checkpoint num_train_frames=1100000 device=cuda:1 agent.freeze_encoder=false
