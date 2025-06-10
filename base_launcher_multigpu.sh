#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco

torchrun --nproc-per-node=4 train_metaworld_ed4ct.py batch_size=256