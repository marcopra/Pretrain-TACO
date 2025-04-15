#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco

# python3 train_metaworld.py agent.pretrained_path=none exp_name=taco_with_losses
python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_push-v2_img84_fs3_ar1_exp=80__ds=100000.pt" exp_name=pretrained_taco