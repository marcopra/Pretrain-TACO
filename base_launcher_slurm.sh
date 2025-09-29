#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

python pretrain_ST.py --dataset_config /home/mprattico/Pretrain-TACO/dataset/random_medium --save_path models/maze/random_resnet18 -fe resnet18p --total_steps 300_000_000 --checkpoint "250_000_000,300_000_000" --lr 5e-4 --use_wandb --height 224 --wodth 224
