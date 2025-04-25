#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load environment
source ~/.bashrc
conda activate metataco

# python3 train_metaworld.py agent.pretrained_path=none exp_name=taco_with_losses random_init=true device=cuda:0
# python pretrain_taco.py --dataset_path "/fastwork/mprattico/Pretrain-TACO/data/push-v2_img84_fs3_ar2_exp=20__ds=100000" --use_wandb --total_steps 300_000_000 --checkpoint "2_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000"
# python3 train_metaworld.py agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/taco_MT_exp\=80_1OOD_push_config_ts\=200000512.pt" exp_name=pretrained_taco_80200M random_init=true device=cuda:0
python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50_exp=80_1OOD_push_config.json" --use_wandb --total_steps 300_000_000 --checkpoint "2_000_000,20_000_000,50_000_000,100_000_000,200_000_000" --lr 5e-3
