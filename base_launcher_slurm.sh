#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

# python pretrain_ST.py --dataset_config /home/mprattico/Pretrain-TACO/dataset/random_medium --save_path models/maze/random_resnet18 -fe resnet18 --total_steps 300_000_000 --checkpoint 250_000_000,300_000_000 --lr 5e-4 --use_wandb --resume_wandb_run hdee1avj --resume_checkpoint /home/mprattico/Pretrain-TACO/models/maze/random_resnet18/taco_ST_resnet18_l5_scratch_random_medium_lr\=0.0005_ts\=63283200_curl_rew_best.pt
# python train_gym_checkpoint.py agent=taco_checkpoint agent._target_=agents.taco_resnet.TACOAgent save_snapshot=false seed=0 freeze_encoder=true agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/maze/random_resnet18/taco_ST_resnet18_l5_scratch_random_medium_lr\=0.0005_ts\=63283200_curl_rew_best.pt" exp_name=random_resnet_frozen
python train_gym_checkpoint.py agent=taco_checkpoint agent._target_=agents.taco_resnet.TACOAgent save_snapshot=false seed=0 freeze_encoder=false agent.pretrained_path="/home/mprattico/Pretrain-TACO/models/maze/random_resnet18/taco_ST_resnet18_l5_scratch_random_medium_lr\=0.0005_ts\=63283200_curl_rew_best.pt" exp_name=random_resnet