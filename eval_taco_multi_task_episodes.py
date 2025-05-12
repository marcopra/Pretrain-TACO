"""
python eval_taco_multi_task_episodes.py --pretrained_path "models/debug/taco_MT_ST50_0.0_lr=0.0005_ts=5000192.pt" --dataset_config "data_episodes/my_config/0"
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools
import wandb
import argparse
import pickle
import json
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from replay_buffer import make_replay_loader
from pathlib import Path
from agents.taco import TACOAgent


def load_unified_dataset(root_dir, batch_size=32, num_workers=4,
                         nstep=3, multistep=3, discount=0.99):
    """
    Load all episodes from all dataset subdirectories as a single unified dataset
    
    Args:
        root_dir: Root directory containing dataset subdirectories with episodes
        batch_size: Batch size for sampling
        num_workers: Number of workers for parallel loading
        nstep: N-step returns parameter
        multistep: Multistep parameter 
        discount: Discount factor
    """
    replay_dir = Path(root_dir)
    
    # Count subdirectories to give feedback
    dataset_dirs = [d for d in replay_dir.iterdir() if d.is_dir()]
    print(f"Found {len(dataset_dirs)} dataset subdirectories")
    
    # Count total episodes
    episode_count = 0
    for dataset_dir in dataset_dirs:
        episode_count += len(list(dataset_dir.glob('*.npz')))
    print(f"Total episodes across all datasets: {episode_count}")
    
    # Create loader using the root directory (will now recursively find all episodes)
    loader = make_replay_loader(
        replay_dir=replay_dir,
        max_size=1000000,  # Adjust as needed
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    
    return loader


    



# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=50, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--multistep', type=int, default=3, help='Multistep (set to 1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.json', help='Path to the dataset configuration file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='taco-mt-pretrain', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--total_steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--checkpoint', type=str, default="500000,1000000", 
                        help='Comma-separated list of steps at which to save checkpoints (e.g., "100000,500000,1000000")')
    parser.add_argument('--nstep', type=int, default=3, help='N-step returns')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained model')
    parser.add_argument('--eval_frequency', type=int, default=1000, help='Frequency of evaluation steps')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--eval_batches', type=int, default=10, help='Number of batches to use for evaluation')
    parser.add_argument('--fastwork', action='store_true', help='Prepend /home/mprattico/fastwork/ to dataset paths')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')

    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in args.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    pretraining_dataset_path = args.dataset_config + "/pretraining_datasets"
    valid_datset_path = args.dataset_config + "/test_dataset"
    # # load
    # train_dataloader = load_unified_dataset(
    #     root_dir=pretraining_dataset_path,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers
    # )

    valid_dataloader = load_unified_dataset(
        root_dir=valid_datset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "multistep": args.multistep,
            "device": args.device,
            "total_steps": args.total_steps,
            "nstep": args.nstep,
            "discount": args.discount,
            # "datasets": pretraining_datasets,
            # "num_datasets": len(pretraining_datasets),
            "dataset_config": args.dataset_config,
            "fastwork": args.fastwork,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )

    # Get a batch from training data to initialize the agent
    batch = next(iter(valid_dataloader))
    obs_shape = batch[0].shape[1:]
    action_shape = batch[1].shape[1:]

    taco_agent = TACOAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=args.device,
        lr=args.lr,
        encoder_lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        # critic_target_tau=None,
        # num_expl_steps=None,
        # update_every_steps=1,
        # stddev_schedule=None,
        # stddev_clip=None,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=1,
        stddev_schedule="linear(1.0,0.1,500000)",
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=args.multistep,
        latent_a_dim='none',
        curl=True,
        pretrained_path=args.pretrained_path,
        freeze_encoder=True
    )

    steps = 0
    epoch = 0
    valid_iterator = iter(valid_dataloader)
    
    taco_agent.train(True)
    while steps < args.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{args.total_steps}")
        
        for batch in valid_dataloader:
            # evaluate the agent
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, args.device)
            metrics = taco_agent.evaluate_taco(obs, action, action_seq, r_next_obs, reward)
            steps += args.batch_size
            
            print("Metrics:")
            print("Reward Loss: ",metrics['reward_loss'])
            print("Curl Loss: ",metrics['curl_loss'])
            print("TACO Loss: ",metrics['taco_loss'])  

            # Log training metrics to wandb
            if args.use_wandb:
                metrics['steps'] = steps
                metrics['epoch'] = epoch
                wandb.log(metrics)
        
            
            if steps >= args.total_steps:
                break
    

    if args.use_wandb:
        wandb.finish()

