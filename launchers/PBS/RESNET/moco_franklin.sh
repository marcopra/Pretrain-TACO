#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N moco_train

cd $PBS_O_WORKDIR

# Use environment variables passed via qsub -v
SEED=${SEED:-0}
ENV_NAME=${ENV_NAME:-"basketball-v2"}
WANDB_TAG=${WANDB_TAG:-"MOCO"}

# Set seed argument based on SEED value
if [ "$SEED" -eq 1 ]; then
    SEED_ARG=$(($RANDOM % 10000)) 
else
    SEED_ARG=$SEED
fi

# Load environment
source ~/.bashrc
conda activate metataco

# Setup NCCL variables for distributed training
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# Setup master node for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: 4"
echo "Number of GPUs: 4"

# Print network debug information
echo "=== Network Interface Debug ==="
hostname
echo "InfiniBand interfaces:"
echo "ib0 and ib1 interfaces (skipped for compatibility)"
echo "=== InfiniBand Status ==="
ibstat 2>/dev/null | head -15 || echo "ibstat not available"
echo "==============================="

WANDB_TAG_FINAL="${WANDB_TAG}_${ENV_NAME}"

# Run distributed training with 4 GPUs
mpirun -np 4 python train_metaworld_ed4ct.py \
    batch_size=256 \
    env_name=$ENV_NAME \
    wandb_tag="$WANDB_TAG_FINAL" \
    agent.pretrained_path=/home/mprattico/Pretrain-TACO/models/moco_aug.pth.tar \
    wandb_mode=online \
    save_snapshot=false \
    num_train_frames=220000 \
    seed=$SEED_ARG \
    exp_name="MOCO_${SEED_ARG}_$ENV_NAME"
