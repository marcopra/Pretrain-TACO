#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 exp_name=checkpoint_10k
# python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 exp_name=checkpoint_20k
# python train_gym_checkpoint.py agent=taco_checkpoint save_snapshot=false seed=0 exp_name=checkpoint_400k
