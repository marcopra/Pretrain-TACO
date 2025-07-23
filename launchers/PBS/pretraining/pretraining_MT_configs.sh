#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use environment variables passed via qsub -v
# DATASET_CONFIG and EXPS should be set when submitting

# Load environment
source ~/.bashrc
conda activate metataco

echo "Running pretraining with dataset config: ${DATASET_CONFIG}"
python pretrain_taco_multi_task_episodes_from_checkpoint_homogeneous_multiheads.py --dataset_config "${DATASET_CONFIG}" --use_wandb --total_steps 200_000_000 --checkpoint "100_000_000, 200_000_000" --lr 5e-4 --save_path models_multiheads --homogeneous --max_size 98000 --wandb_project "taco-pt-multiheads" --validation_source "split"
