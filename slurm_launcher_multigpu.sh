#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out          
#SBATCH --error=%j.err

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
module unload anaconda3/2023.09-0
module load anaconda3/2023.09-0
conda activate metataco
module unload anaconda3/2023.09-0

srun torchrun --nproc-per-node=4 train_metaworld_ed4ct.py batch_size=256 env_name=push-v3 wandb_tag="SLURM" agent.pretrained_path=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar wandb_mode=offline

srun torchrun --nproc-per-node=1 train_metaworld_ed4ct.py batch_size=256 env_name=push-v3 wandb_tag="SLURM" agent.pretrained_path=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar wandb_mode=offline