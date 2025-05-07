#!/bin/bash

python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50/MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config.json" --use_wandb --total_steps 30_000_000 --checkpoint "1_000_000,2_000_000,10_000_000,20_000_000" --lr 1e-3 --save_path "models/tuning/" 
python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50/MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config.json" --use_wandb --total_steps 30_000_000 --checkpoint "1_000_000,2_000_000,10_000_000,20_000_000" --lr 5e-3 --save_path "models/tuning/" 
python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50/MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config.json" --use_wandb --total_steps 30_000_000 --checkpoint "1_000_000,2_000_000,10_000_000,20_000_000" --lr 1e-2 --save_path "models/tuning/" 
python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50/MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config.json" --use_wandb --total_steps 30_000_000 --checkpoint "1_000_000,2_000_000,10_000_000,20_000_000" --lr 5e-2 --save_path "models/tuning/" 
python pretrain_taco_multi_task.py --dataset_config "mt_config/MT50/MT50_exp=80_1OOD_push_fs3_ar2_ri1_rg0_config.json" --use_wandb --total_steps 30_000_000 --checkpoint "1_000_000,2_000_000,10_000_000,20_000_000" --lr 1e-1 --save_path "models/tuning/" 
