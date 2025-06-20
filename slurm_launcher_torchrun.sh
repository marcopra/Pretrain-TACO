#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=%j.err

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

# For torchrun, we need to run on each node separately
# Get node rank
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE_RANK=0
for node in $NODELIST; do
    if [[ "$node" == "$(hostname)" ]]; then
        break
    fi
    ((NODE_RANK++))
done

echo "NODE_RANK: $NODE_RANK"

# Run with torchrun (each node runs this independently)
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM_TORCHRUN" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline