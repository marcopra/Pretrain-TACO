#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd $SLURM_SUBMIT_DIR

# Use environment variables passed via sbatch
# DATASET_CONFIG and EXPS should be set when submitting

# Load environment
source ~/.bashrc
conda activate metataco

echo "Running pretraining with dataset config: ${DATASET_CONFIG}, exps: ${EXPS}"
python pretrain_taco_multi_task_episodes_from_checkpoint_homogeneous.py --dataset_config "${DATASET_CONFIG}" --use_wandb --total_steps 200_000_000 --checkpoint "100_000_000, 200_000_000" --lr 5e-4 --save_path models_exp99 --homogeneous --max_size 98000
