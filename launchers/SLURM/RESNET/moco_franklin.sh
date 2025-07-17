#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Use environment variables passed via sbatch
SEED=${SEED:-0}
ENV_NAME=${ENV_NAME:-"basketball-v2"}
WANDB_TAG=${WANDB_TAG:-"MOCO"}

# Set seed argument based on SEED value
if [ "$SEED" -eq 1 ]; then
    SEED_ARG=$(($RANDOM % 10000)) 
else
    SEED_ARG=$SEED
fi

source ~/.bashrc
conda activate metataco
ENV_NAME=${ENV_NAME:-"basketball-v2"}

# 2. Setup variabili NCCL
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# 3. Setup master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)


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

WANDB_TAG_FINAL="${WANDB_TAG}_${ENV_NAME}"
# Run with srun
srun --ntasks=4 --ntasks-per-node=4 python train_metaworld_ed4ct.py \
    batch_size=256 env_name=$ENV_NAME \
    wandb_tag="$WANDB_TAG_FINAL" \
    agent.pretrained_path=/home/mprattico/Pretrain-TACO/models/moco_aug.pth.tar \
    wandb_mode=online \
    save_snapshot=false \
    num_train_frames=220000 \
    seed=$SEED_ARG \
    exp_name="MOCO_${SEED_ARG}_$ENV_NAME"