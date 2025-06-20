#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=%j.err

set -x  # Enable debug mode

cd $SLURM_SUBMIT_DIR
module purge
module load anaconda3/2023.09-0
conda activate metataco
ble port
echo "Node: $(hostname)"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Available GPUs: $(nvidia-smi -L)"

# Get master node and find available portecho "MASTER_ADDR: $MASTER_ADDR"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')D"
JOB_NUM_NODES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"# NCCL configuration for InfiniBand
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"0,mlx5_1,mlx5_2,mlx5_3
echo "SLURM_PROCID: $SLURM_PROCID"

# NCCL configuration for InfiniBand with more conservative settingsHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_37
export NCCL_SOCKET_IFNAME=ib00
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0# Set PyTorch distributed environment variables manually
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_NET_GDR_LEVEL=0
export NCCL_TIMEOUT=1800  # 30 minutes timeoutSIZE=1

# Set PyTorch distributed environment variables manuallyecho "Starting training with RANK=$RANK, WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK"
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export RANK=$SLURM_NODEID # Launch with srun directly (no torchrun)
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

echo "Starting training with RANK=$RANK, WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK"th=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar \
echo "Python path: $(which python)"echo "Conda env: $CONDA_DEFAULT_ENV"

# Test network connectivity
echo "Testing network connectivity to master node..."
if [ "$SLURM_NODEID" -ne 0 ]; then
    nc -z $MASTER_ADDR $MASTER_PORT && echo "Connection to master successful" || echo "Connection to master failed"
fi

# Launch with srun directly (no torchrun)
echo "Launching training script..."
srun --unbuffered python train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattic/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline