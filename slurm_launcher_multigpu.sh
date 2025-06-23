#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# set -x  # Enable debug mode

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
module unload anaconda3/2023.09-0
module load anaconda3/2023.09-0
conda activate metataco
module unload anaconda3/2023.09-0

echo "Node: $(hostname)"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Available GPUs: $(nvidia-smi -L)"

## NCCL configuration for InfiniBand on Leonardo
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# Set master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Set CUDA_VISIBLE_DEVICES based on SLURM_LOCALID
# This ensures each process sees only its assigned GPU as device 0
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set additional environment variables for distributed training
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

# Run with srun (no torchrun needed)
srun --unbuffered python train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline