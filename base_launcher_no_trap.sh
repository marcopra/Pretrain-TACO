#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q a100f

cd $PBS_O_WORKDIR


# Load environment
source ~/.bashrc
conda activate metataco

python train_metaworld.py agent=taco_resnet agent.pretrained_path="resnet50_l5" exp_name="RESNET_longexpresnet50_l5_BS240_820_random_init_true_random_goal_false_freeze_false" seed=820 env_name=push-v2 random_init=true random_goal=false wandb_tag=RESNET50-FREEZE num_train_frames=300000 agent.freeze_encoder=false batch_size=240 save_snapshot=true wandb_id=7sycw7fs
python train_metaworld.py agent=taco_resnet agent.pretrained_path="resnet50_l5" exp_name="RESNET_longexpresnet50_l5_BS240_1570_random_init_true_random_goal_false_freeze_false" seed=1570 env_name=push-v2 random_init=true random_goal=false wandb_tag=RESNET50-FREEZE num_train_frames=300000 agent.freeze_encoder=false batch_size=240 save_snapshot=true wandb_id=37fzd208