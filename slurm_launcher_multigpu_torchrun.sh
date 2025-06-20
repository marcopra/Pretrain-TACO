#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=%j.err

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
module unload anaconda3/2023.09-0
module load anaconda3/2023.09-0
conda activate metataco
module unload anaconda3/2023.09-0

# Get master node and find available port
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"

# NCCL configuration for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_NET_GDR_LEVEL=0

# Launch with torchrun directly (no srun)
if [ "$SLURM_NODEID" -eq 0 ]; then
    # Only launch from the first node
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc_per_node=1 \
      --node_rank=$SLURM_NODEID \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train_metaworld_ed4ct.py \
        batch_size=128 env_name=push-v3 \
        wandb_tag="SLURM" \
        agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
        wandb_mode=offline
fi
