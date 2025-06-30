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

SEED=$(($RANDOM % 10000)) 

# Run with srun
srun --ntasks=8 --ntasks-per-node=4 python train_metaworld_ed4ct.py \
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/moco_aug.pth.tar \
    wandb_mode=offline \
    save_snapshot=false \
    seed=$SEED \
    exp_name="SLURM_TORCHRUN_${SEED}" 