#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco

python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_push-v2_img84_fs3_ar1_exp=60__ds=100000_ts=2000896.pt" exp_name=taco_with_losses
# python pretrain_taco.py --dataset_path "data/push-v2_img84_fs3_ar1_exp=60__ds=100000" --use_wandb --total_steps 300_000_000 --checkpoint "50_000_000, 100_000_000"
# python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/" exp_name=pretrained_taco