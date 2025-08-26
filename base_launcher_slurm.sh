#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

python3 train_metaworld.py agent=taco_resnet agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/moco_aug.pth.tar" exp_name="/home/mprattico/Pretrain-TACO/models/moco_aug.pth.tar_1800" seed=1800 env_name=push-v2 random_init=true random_goal=false wandb_tag=none wandb_project=taco_frozen_baselines num_train_frames=2002000 num_seed_frames=8000 agent.freeze_encoder=false agent.no_taco=false batch_size=256 save_snapshot=true
# python3 train_metaworld.py agent=taco_resnet agent.pretrained_path="r3m" exp_name="r3m_8220" seed=8220 env_name=push-v2 random_init=true random_goal=false wandb_tag=none wandb_project=taco_frozen_baselines num_train_frames=2002000 num_seed_frames=8000 agent.freeze_encoder=false agent.no_taco=false batch_size=256 save_snapshot=true