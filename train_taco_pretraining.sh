#!/bin/bash

# Check if correct number of arguments is provided
if [ $# -ne 3 ]eval_every_frames=500; then
    echo "Error: Incorrect number of arguments provided"
    echo "Usage: $0 <dataset_name> <cuda_device> <exp>"
    echo "dataset_name: basketball, binpicking, buttonpress, shelfplace"
    echo "cuda_device: 0 to 7"
    echo "exp: 0, 33, 66, 99"
    exit 1
fi

dataset_name_input=$1
cuda_device=$2
exp_input=$3

# Map simplified dataset names to full names
case $dataset_name_input in
    basketball)
        dataset_name="MT50_OOD_Basketball"
        eval_every_frames=500;eval_every_frames=500;
    binpicking)
        dataset_name="MT50_OOD_BinPicking"
        eval_every_frames=500;eval_every_frames=500;
    buttonpress)
        dataset_name="MT50_OOD_ButtonPress"
        eval_every_frames=500;eval_every_frames=500;
    shelfplace)
        dataset_name="MT50_OOD_ShelfPlace"
        eval_every_frames=500;eval_every_frames=500;
    *)
        echo "Error: Invalid dataset_name value. Allowed values are basketball, binpicking, buttonpress, shelfplace."
        exit 1
        eval_every_frames=500;eval_every_frames=500;
esac

# Validate cuda_device value
case $cuda_device in
    0|1|2|3|4|5|6|7)
        # Valid value
        eval_every_frames=500;eval_every_frames=500;
    *)
        echo "Error: Invalid cuda_device value. Allowed values are 0 to 7."
        exit 1
        eval_every_frames=500;eval_every_frames=500;
esac

# Map integer exp values to decimal format
case $exp_input in
    0)
        exp="0.0"
        eval_every_frames=500;eval_every_frames=500;
    33)
        exp="0.33"
        eval_every_frames=500;eval_every_frames=500;
    66)
        exp="0.66"
        eval_every_frames=500;eval_every_frames=500;
    99)
        exp="0.99"
        eval_every_frames=500;eval_every_frames=500;
    *)
        echo "Error: Invalid exp value. Allowed values are 0, 33, 66, 99."
        exit 1
        eval_every_frames=500;eval_every_frames=500;
esac

echo "Running pretraining with dataset: ${dataset_name}, cuda_device: ${cuda_device}, exp: ${exp}"

# Set CUDA device environment variable
export CUDA_VISIBLE_DEVICES=${cuda_device}

# Run the Python command with the appropriate arguments
python pretrain_taco_multi_task_episodes_from_checkpoint.py --dataset_config "data_episodes/${dataset_name}/${exp}" --use_wandb --total_steps 200_000_000 --checkpoint "100_000_000, 200_000_000" --lr 5e-4



python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=101 exp_name=ptrandom_ds_rew_true1 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=102 exp_name=ptrandom_ds_rew_true2 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=103 exp_name=ptrandom_ds_rew_true3 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=104 exp_name=ptrandom_ds_rew_true4 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt"eval_every_frames=500;


python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=101 exp_name=ptrandom_ds_frozen_rew_true1 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=102 exp_name=ptrandom_ds_frozenrew_true2 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=103 exp_name=ptrandom_ds_frozen_rew_true3 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=104 exp_name=ptrandom_ds_frozen_rew_true4 wandb_tag="randompt_rew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_true/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=938035200_nocurl_rew_best.pt" agent.freeze_encoder=true eval_every_frames=500;


python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=101 exp_name=ptrandom_ds_rew_false1 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=102 exp_name=ptrandom_ds_rew_false2 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=103 exp_name=ptrandom_ds_rew_false3 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt"eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=104 exp_name=ptrandom_ds_rew_false4 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt"eval_every_frames=500;


python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=101 exp_name=ptrandom_ds_frozen_rew_false1 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=102 exp_name=ptrandom_ds_frozen_rew_false2 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=103 exp_name=ptrandom_ds_frozen_rew_false3 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
python train_gym_checkpoint.py agent=taco_proprio_states num_train_frames=220000 device=cuda:1 seed=104 exp_name=ptrandom_ds_frozen_rew_false4 wandb_tag="randompt_norew" encoder_checkpoint_steps=[] agent.pretrained_path="/home/mprattico/Pretrain-TACO/exp_local_pt/rew_false/models/maze/proprio/taco_ST_home_mprattico_Pretrain-TACO_dataset_random_medium_lr\=0.0001_ts\=784332800_nocurl_norew_best.pt" agent.freeze_encoder=true eval_every_frames=500;
