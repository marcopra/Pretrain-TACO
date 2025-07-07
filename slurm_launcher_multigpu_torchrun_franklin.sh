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

# Debug delle interfacce di rete disponibili
echo "=== Network Interfaces ==="
ip addr show | grep -E "^[0-9]+:|inet "
echo "==========================="

# Rilevamento migliorato delle interfacce di rete
IB_INTERFACES=$(ip addr show | grep -E "ib[0-9]+" | head -1 | awk '{print $2}' | cut -d: -f1)
# Trova interfacce Ethernet attive (UP e con IP)
ETH_INTERFACES=$(ip addr show | grep -E "eno[0-9]+|eth[0-9]+|ens[0-9]+|enp[0-9]+" | grep "state UP" | head -1 | awk '{print $2}' | cut -d: -f1)

echo "IB Interfaces found: $IB_INTERFACES"
echo "Ethernet Interfaces found: $ETH_INTERFACES"

# Configurazione NCCL semplificata per Ethernet
echo "Configuring for Ethernet network (no Infiniband available)..."
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$ETH_INTERFACES
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1

# Esclusioni comuni
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME,^lo,^docker,^virbr

# Configurazioni generali per stabilità
export NCCL_BUFFSIZE=2097152
export NCCL_TREE_THRESHOLD=0

echo "Final NCCL Configuration:"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
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
    batch_size=128 env_name=push-v3 \
    wandb_tag="SLURM_TORCHRUN" \
    agent.pretrained_path=/leonardo/home/userexternal/mprattico/Pretrain-TACO/models/resnet50_l5.tar \
    wandb_mode=offline \
    save_snapshot=true \
    seed=$SEED \
    exp_name="SLURM_TORCHRUN_${SEED}" \