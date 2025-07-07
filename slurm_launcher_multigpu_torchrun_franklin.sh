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

# Configurazione NCCL forzata per Ethernet (no Infiniband)
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eno12399
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_BUFFSIZE=2097152
export NCCL_TREE_THRESHOLD=0

# Evita conflitti con configurazioni IB
unset NCCL_NET_GDR_LEVEL
unset NCCL_IB_GID_INDEX
unset NCCL_IB_HCA
unset NCCL_IB_TIMEOUT
unset NCCL_IB_RETRY_CNT
unset NCCL_IB_CUDA_SUPPORT

echo "Final NCCL Configuration:"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_NET: $NCCL_NET"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_DEBUG: $NCCL_DEBUG"

SEED=$(($RANDOM % 10000)) 

# Launch torchrun on ALL nodes using srun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v2 \
    wandb_tag="SLURM_TORCHRUN" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline \
    save_snapshot=true \
    seed=$SEED \
    exp_name="SLURM_TORCHRUN_${SEED}" \