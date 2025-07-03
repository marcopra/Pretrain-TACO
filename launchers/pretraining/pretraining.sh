#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use environment variables passed via qsub -v
# DATASET_NAME=${DATASET_NAME:-"ST50"}  # Default to ST50 if not specified
# EXPS=${EXPS:-"0.0"}  # Default to 0.0 if not specified

# Load environment
source ~/.bashrc
conda activate metataco

echo "Running pretraining with dataset: ${DATASET_NAME}, exps: ${EXPS}"
python pretrain_taco_multi_task_episodes_from_checkpoint.py --dataset_config "data_episodes/${DATASET_NAME}/${EXPS}" --use_wandb --total_steps 200_000_000 --checkpoint "100_000_000, 200_000_000" --lr 5e-4 --save_path models_exp99 --homogeneous. --max_size 98000
