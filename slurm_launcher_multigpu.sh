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

# NCCL configuration for InfiniBand - configurazione specifica per Leonardo
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# CONFIGURAZIONI SPECIFICHE PER RISOLVERE "Could not find NET with id 0"
export NCCL_NET=IB
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IGNORE_DISABLED_P2P=1

# IMPORTANTE: Usa l'interfaccia InfiniBand specifica invece di escludere
export NCCL_SOCKET_IFNAME=ib0

# Disabilita alcune ottimizzazioni che possono causare problemi
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0

# Set master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)

# Set environment variables for distributed training
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $LOCAL_RANK"

# Stampa informazioni di debug per la rete
echo "=== Network Interface Debug ==="
hostname
echo "InfiniBand interfaces:"
ip addr show ib0 | head -5
ip addr show ib1 | head -5
echo "=== InfiniBand Status ==="
ibstat 2>/dev/null | head -15 || echo "ibstat not available"
echo "==============================="

# Run with srun
srun --unbuffered python train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline