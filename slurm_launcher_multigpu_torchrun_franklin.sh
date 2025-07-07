#!/bin/bash
#SBATCH --job-name=metataco_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua-longrun

cd $SLURM_SUBMIT_DIR
source ~/.bashrc

conda activate metataco


# Set master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-50000 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"

## NCCL configuration for InfiniBand on Leonardo
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

SEED=$(($RANDOM % 10000)) 
# Launch torchrun on ALL nodes using srun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
    train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM_TORCHRUN" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline \
    save_snapshot=true \
    seed=$SEED \
    exp_name="SLURM_TORCHRUN_${SEED}" \
