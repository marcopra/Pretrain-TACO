#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
module unload anaconda3/2023.09-0
module load anaconda3/2023.09-0
conda activate metataco
module unload anaconda3/2023.09-0

echo "Node: $(hostname)"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Available GPUs: $(nvidia-smi -L)"

# NCCL configuration for InfiniBand HDR on Leonardo
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=30
export NCCL_IB_RETRY_CNT=10
# Aggiungi configurazioni specifiche per multi-GPU
export NCCL_CROSS_NIC=1
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_NET_GDR_READ=1
export NCCL_IB_CUDA_SUPPORT=1

# Set master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# NON impostare CUDA_VISIBLE_DEVICES qui - lascia che PyTorch gestisca le GPU
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Set environment variables for distributed training
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

# Run with srun
srun --unbuffered python train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline