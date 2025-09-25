#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 freeze_encoder=true agent.pretrained_path="/home/mprattico/Pretrain-TACO/models_maze/checkpoint_10k/encoder_snapshot.pt" exp_name=checkpoint_10k
# python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 freeze_encoder=true agent.pretrained_path="/home/mprattico/Pretrain-TACO/models_maze/checkpoint_20k/encoder_snapshot.pt" exp_name=checkpoint_20k
# python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 freeze_encoder=true agent.pretrained_path="/home/mprattico/Pretrain-TACO/models_maze/checkpoint_400k/encoder_snapshot.pt" exp_name=checkpoint_400k
