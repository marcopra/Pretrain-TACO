"""
The script pretrain on dataset coming from a single task, it means that there is no target task from which we want to test. 
For this reason we will split just the pretraining dataset in train and validation set.

python pretrain_ST.py --dataset_config /home/mprattico/Pretrain-TACO/dataset/random_medium --save_path models/maze/random --total_steps 300_000_000 --checkpoint "250_000_000,300_000_000" --lr 5e-4 --use_wandb
"""
import os
import torch
import utils
import wandb
import argparse
import json
from pathlib import Path
from pretraining_utils import load_single_task_dataset


def format_pretrained_path(feature_extractor):
    """Format pretrained_path based on feature extractor type"""
    if feature_extractor == "conv":
        return None
    
    # Map feature extractor names to formatted paths
    extractor_map = {
        "vit_s": "vit_s_scratch",
        "vit_b": "vit_b_scratch",
        "vit_l": "vit_l_scratch", 
        "resnet18": "resnet18_l5_scratch",
        "resnet18p": "resnet18_l5_pretrained",
        "resnet50": "resnet50_l5_scratch",
        "resnet50p": "resnet50_l5_pretrained",
        "r3m": "r3m",
        "mvp": "mvp"
    }
    
    return extractor_map.get(feature_extractor, f"{feature_extractor}_scratch")


def extract_task_name_from_path(dataset_path):
    """Extract task name from dataset path (same logic as generate_config.py)"""
    dataset_path = Path(dataset_path)
    folder_name = dataset_path.name
    
    # Extract task name from folder name 
    parts = folder_name.split('_')
    if parts:
        # Take the first part which should contain the task name
        task_part = parts[0]
        # Remove version suffixes like -v2, -v3
        task_name = task_part.replace('-v2', '').replace('-v3', '')
        return task_name
    
    # Fallback to using the full folder name if parsing fails
    return folder_name


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
    parser.add_argument('--wandb_project', type=str, default='taco-st-pretrain', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--total_steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--checkpoint', type=str, default="500000,1000000", 
                        help='Comma-separated list of steps at which to save checkpoints (e.g., "100000,500000,1000000")')
    parser.add_argument('--nstep', type=int, default=3, help='N-step returns')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained model')
    parser.add_argument('--eval_batches', type=int, default=10, help='Number of batches to use for evaluation')
    parser.add_argument('--eval_frequency', type=int, default=50, help='Frequency of evaluation steps')
    parser.add_argument('--no_curl', action='store_true', help='Disable CURL loss (enabled by default)')
    parser.add_argument('--no_reward', action='store_true', help='Disable reward loss (enabled by default)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use for training (default: adam)')
    # New arguments for resuming training
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to the checkpoint to resume training from')
    parser.add_argument('--resume_wandb_run', type=str, default=None, help='ID of wandb run to resume')
    parser.add_argument('--continue_steps', action='store_true', default=True, help='Continue step counter from checkpoint (instead of starting from 0)')
    parser.add_argument('--max_episodes_per_dataset', type=int, default=100000000000, help='Maximum episodes to load per dataset')
    parser.add_argument('--max_size', type=int, default=None, help='Maximum size of replay buffer for training')
    parser.add_argument('--homogeneous', action='store_true', help='Load transitions evenly across datasets')
    parser.add_argument('--log_frequency', type=int, default=100, help='Log metrics every n batches')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training (rest for validation)')
    # Feature extractor argument
    parser.add_argument('--feature_extractor', '-fe', type=str, default='conv', 
                        help='Type of feature extractor to use (default: conv)')
    # Height and width arguments for feature extractors
    parser.add_argument('--height', type=int, default=84, help='Input image height')
    parser.add_argument('--width', type=int, default=84, help='Input image width')
    # Removed validation_source, validation_split_ratio, and test_eval_frequency as they're not needed for single task
    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in args.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    # Format pretrained_path based on feature extractor
    pretrained_path = format_pretrained_path(args.feature_extractor)
    print(f"Using feature extractor: {args.feature_extractor}, pretrained_path: {pretrained_path}")

    # Import the appropriate TACOAgent based on feature extractor
    if args.feature_extractor == "conv":
        from agents.taco import TACOAgent
    else:
        from agents.taco_resnet import TACOAgent

    # Handle dataset config for single task (no separate pretraining/test structure)
    config_path = Path(args.dataset_config)
    
    if config_path.suffix == '.json':
        # For JSON config, get dataset info from the config itself
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        # For single task, we expect the dataset path directly or in a simple format
        if 'dataset_path' in config_data:
            dataset_path = config_data['dataset_path']
            dataset_names = [extract_task_name_from_path(dataset_path)]
        elif 'pretraining_datasets' in config_data:
            dataset_dirs = config_data.get('pretraining_datasets', [])
            dataset_names = [extract_task_name_from_path(d) for d in dataset_dirs]
        else:
            dataset_names = ['single_task']
        print(f"Single task dataset config: {dataset_names}")
    else:
        # For folder-based config, assume the dataset is directly in the provided path
        dataset_path = args.dataset_config
        dataset_names = [extract_task_name_from_path(dataset_path)]
        print(f"Single task dataset path: {dataset_path}")
    
    # For single task, we always split the data into train/validation
    print(f"Splitting single task dataset with train ratio: {args.train_ratio}")
    print(f"Training set will respect max_episodes_per_dataset={args.max_episodes_per_dataset}, max_size={args.max_size}")
    
    # Load training data with the specified constraints
    train_dataloader = load_single_task_dataset(
        config_or_path=args.dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
        max_size=args.max_size,
        homogeneous=args.homogeneous,
        train_ratio=args.train_ratio,
        use_training_split=True  # Only get the training portion
    )
    
    # Load validation data from the remaining data (no size constraints)
    valid_dataloader = load_single_task_dataset(
        config_or_path=args.dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_episodes_per_dataset=None,  # Use all remaining episodes for validation
        max_size=None,  # Use all remaining data for validation
        homogeneous=args.homogeneous,
        train_ratio=args.train_ratio,
        use_training_split=False  # Only get the validation portion
    )
    
    # No separate test set for single task scenario
    test_dataloader = None

    # Initialize steps and epoch
    steps = 0
    epoch = 0
    
    # Initialize best model tracking
    best_eval_loss = float('inf')
    best_model_path = None
    
    # Load saved checkpoint if provided
    saved_args = None
    if args.resume_checkpoint:
        raise NotImplementedError("Resuming from checkpoint is not implemented in single task pretraining yet.")
        # print(f"Loading checkpoint from {args.resume_checkpoint}")
        # checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        # saved_args = checkpoint.get('args', {})
        
        # # Extract dataset config from checkpoint path
        # checkpoint_basename = os.path.basename(args.resume_checkpoint)
        # if "taco_MT_" in checkpoint_basename:
        #     # Extract dataset config from path format like "taco_MT_ST50_OOD_Push_0.66_lr=0.0005_ts=100352_curl_rew.pt"
        #     parts = checkpoint_basename.split("_")
        #     # Find dataset and ratio parts (e.g., "ST50" and "0.66")
        #     if len(parts) >= 4:
        #         dataset_type = "_".join(parts[2:4])  # e.g., "ST50" or "MT50" #TODO to test
        #         ratio = parts[5]  # e.g., "0.33" or "0.66"
        #         expected_config = f"data_episodes/{dataset_type}/{ratio}"
                
        #         # Check if the dataset config matches
        #         if not args.dataset_config.endswith(expected_config):
        #             raise AssertionError(
        #                 f"Dataset config mismatch! Provided: {args.dataset_config}, "
        #                 f"Expected (based on checkpoint): {expected_config}"
        #             )
        #         print(f"Verified dataset config: {args.dataset_config} matches checkpoint")
        
        # # Set steps and epoch from checkpoint if requested
        # if args.continue_steps and 'steps' in checkpoint:
        #     steps = checkpoint['steps']
        #     print(f"Resuming from step: {steps}")
        # if 'epoch' in checkpoint:
        #     epoch = checkpoint['epoch']
        #     print(f"Resuming from epoch: {epoch}")

    # Initialize wandb if enabled
    if args.use_wandb:
        print("Initializing Weights & Biases logging")
        wandb_config = {
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "batch_size": args.batch_size,
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "multistep": args.multistep,
            "device": args.device,
            "total_steps": args.total_steps,
            "nstep": args.nstep,
            "discount": args.discount,
            "datasets": dataset_names,
            "num_datasets": len(dataset_names),
            "dataset_config": args.dataset_config,
            "use_curl": not args.no_curl,
            "use_reward": not args.no_reward,
            "resumed_from_checkpoint": args.resume_checkpoint,
            "max_episodes_per_dataset": args.max_episodes_per_dataset,
            "max_size": args.max_size,
            "homogeneous": args.homogeneous,
            "train_ratio": args.train_ratio,
            "feature_extractor": args.feature_extractor,
            "pretrained_path": pretrained_path
        }

        # Resume wandb run if ID is provided
        if args.resume_wandb_run:
            print(f"Resuming wandb run: {args.resume_wandb_run}")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                id=args.resume_wandb_run,
                resume="must",
                config=wandb_config
            )
        else:
            print("Starting new wandb run")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=wandb_config
            )

    # Get a batch from training data to initialize the agent
    batch = next(iter(train_dataloader))
    obs_shape = batch[0].shape[1:]
    action_shape = batch[1].shape[1:]

    # Initialize agent with appropriate parameters based on feature extractor
    if args.feature_extractor == "conv":
        taco_agent = TACOAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=args.device,
            lr=args.lr,
            encoder_lr=args.lr,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            critic_target_tau=None,
            num_expl_steps=None,
            update_every_steps=1,
            stddev_schedule=None,
            stddev_clip=None,
            use_tb=True,
            reward=not args.no_reward,
            multistep=args.multistep,
            latent_a_dim='none',
            curl=not args.no_curl,
            pretrained_path=args.resume_checkpoint,
            optimizer_type=args.optimizer
        )
    else:
        taco_agent = TACOAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=args.device,
            lr=args.lr,
            encoder_lr=args.lr,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            critic_target_tau=None,
            num_expl_steps=None,
            update_every_steps=1,
            stddev_schedule=None,
            stddev_clip=None,
            use_tb=True,
            reward=not args.no_reward,
            multistep=args.multistep,
            latent_a_dim='none',
            curl=not args.no_curl,
            height=args.height,
            width=args.width,
            pretrained_path=pretrained_path,
            freeze_encoder=False,
            no_taco=False
        )

    # Now that the agent is initialized with the loaded checkpoint, we're ready to continue training
    valid_iterator = iter(valid_dataloader)
    test_iterator = iter(test_dataloader) if test_dataloader is not None else None
    
    batch_count = 0
    while steps < args.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{args.total_steps}")
        
        for batch in train_dataloader:
            batch_count += 1
            
            # *** Validation evaluation step ***
            if (steps//args.batch_size) % args.eval_frequency == 0:
                taco_agent.train(False)  # Set to eval mode
                
                eval_metrics_sum = {
                    'eval/reward_loss': 0,
                    'eval/curl_loss': 0,
                    'eval/taco_loss': 0,
                    'eval/total_loss': 0,
                    'eval/batch_reward': 0,
                    'eval/avg_rew_pred_error_percentage': 0,
                    'eval/log_cosh': 0,
                    'eval/rel_error_filtered': 0,
                    'eval/smape': 0,
                }
                num_eval_batches = 0
                
                # Evaluate on validation set
                for _ in range(args.eval_batches):
                    try:
                        eval_batch = next(valid_iterator)
                    except StopIteration:
                        valid_iterator = iter(valid_dataloader)
                        eval_batch = next(valid_iterator)
                    obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                        eval_batch, args.device)
                    
                    eval_metrics = taco_agent.evaluate_taco(obs, action, action_seq, r_next_obs, reward)
                    
                    # Add eval/ prefix to metrics
                    eval_metrics_sum['eval/reward_loss'] += eval_metrics['reward_loss']
                    eval_metrics_sum['eval/curl_loss'] += eval_metrics['curl_loss']
                    eval_metrics_sum['eval/taco_loss'] += eval_metrics['taco_loss']
                    eval_metrics_sum['eval/batch_reward'] += eval_metrics['batch_reward']
                    eval_metrics_sum['eval/avg_rew_pred_error_percentage'] += eval_metrics['avg_rew_pred_error_percentage']
                    eval_metrics_sum['eval/total_loss'] += eval_metrics['total_loss']
                    if 'log_cosh' in eval_metrics:
                        eval_metrics_sum['eval/log_cosh'] += eval_metrics['log_cosh']
                    if 'rel_error_filtered' in eval_metrics:
                        eval_metrics_sum['eval/rel_error_filtered'] += eval_metrics['rel_error_filtered']
                    if 'smape' in eval_metrics:
                        eval_metrics_sum['eval/smape'] += eval_metrics['smape']
                    num_eval_batches += 1
                
                # Average the validation metrics
                for key in eval_metrics_sum:
                    eval_metrics_sum[key] /= num_eval_batches
                
                # Check if this is the best model based on validation loss
                current_eval_loss = eval_metrics_sum['eval/total_loss']
                if current_eval_loss < best_eval_loss:
                    print(f"New best model found! eval/total_loss: {current_eval_loss:.6f} (previous best: {best_eval_loss:.6f})")
                    best_eval_loss = current_eval_loss
                    
                    # Delete previous best model if it exists
                    if best_model_path is not None and os.path.exists(best_model_path):
                        print(f"Deleting previous best model: {best_model_path}")
                        os.remove(best_model_path)
                    
                    # Save new best model
                    curl_str = "curl" if not args.no_curl else "nocurl"
                    reward_str = "rew" if not args.no_reward else "norew"
                    optimizer_str = f"_{args.optimizer}" if args.optimizer != "adam" else ""
                    extractor_str = f"_{args.feature_extractor}" if args.feature_extractor != "conv" else ""
                    best_model_path = f"{args.save_path}/taco_ST{extractor_str}_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}_best.pt"
                    print(f"Saving new best model to {best_model_path} at step {steps}")
                    os.makedirs(args.save_path, exist_ok=True)
                    torch.save({
                        'encoder': taco_agent.encoder.state_dict(),
                        'taco': taco_agent.TACO.state_dict(),
                        'act_tok': taco_agent.act_tok.state_dict(),
                        'args': vars(args),
                        'steps': steps,
                        'epoch': epoch,
                        'best_eval_loss': best_eval_loss,
                        'feature_extractor': args.feature_extractor,
                        'pretrained_path': pretrained_path,
                    }, best_model_path)
                
                print(f"Validation metrics: {eval_metrics_sum}")
                taco_agent.train(True)  # Set back to train mode
            
            # *** Test evaluation step (disabled for single task pretraining) ***
            # In single task pretraining, we don't have a separate test set
            # Validation data is used as the only evaluation source
            
            # *** Training step ***
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                batch, args.device)
            # Training update
            metrics = taco_agent.update_taco(obs, action, action_seq, r_next_obs, reward)
            steps += args.batch_size
            
            # *** Logging ***
            # Log training metrics to wandb
            if args.use_wandb:
                metrics['steps'] = steps
                metrics['epoch'] = epoch
                # Log validation metrics if we just computed them
                if (steps//args.batch_size - 1) % args.eval_frequency == 0 and 'eval_metrics_sum' in locals():
                    metrics.update(eval_metrics_sum)
                wandb.log(metrics)
            
            # Print metrics every n batches
            if batch_count % args.log_frequency == 0:
                print(f"Steps: {steps}/{args.total_steps}, Batch: {batch_count}, Metrics: {metrics}")

            # *** Save model checkpoint ***
            # Check if we need to save a checkpoint at this step
            if any(s <= steps < s + args.batch_size for s in checkpoint_steps):
                curl_str = "curl" if not args.no_curl else "nocurl"
                reward_str = "rew" if not args.no_reward else "norew"
                optimizer_str = f"_{args.optimizer}" if args.optimizer != "adam" else ""
                extractor_str = f"_{args.feature_extractor}" if args.feature_extractor != "conv" else ""
                checkpoint_path = f"{args.save_path}/taco_ST{extractor_str}_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}.pt"
                print(f"Saving checkpoint at step {steps} to {checkpoint_path}")
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'encoder': taco_agent.encoder.state_dict(),
                    'taco': taco_agent.TACO.state_dict(),
                    'act_tok': taco_agent.act_tok.state_dict(),
                    'args': vars(args),  # Save configuration for easier loading
                    'steps': steps,
                    'epoch': epoch,
                    'feature_extractor': args.feature_extractor,
                    'pretrained_path': pretrained_path,
                }, checkpoint_path)
            
            if steps >= args.total_steps:
                break
    
    # *** Save the trained TACO agent ***
    print(f"Saving model to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    curl_str = "curl" if not args.no_curl else "nocurl"
    reward_str = "rew" if not args.no_reward else "norew"
    optimizer_str = f"_{args.optimizer}" if args.optimizer != "adam" else ""
    extractor_str = f"_{args.feature_extractor}" if args.feature_extractor != "conv" else ""
    torch.save({
        'encoder': taco_agent.encoder.state_dict(),
        'taco': taco_agent.TACO.state_dict(),
        'act_tok': taco_agent.act_tok.state_dict(),
        'args': vars(args),  # Save configuration for easier loading
        'steps': steps,
        'epoch': epoch,
        'feature_extractor': args.feature_extractor,
        'pretrained_path': pretrained_path,
    }, f"{args.save_path}/taco_ST{extractor_str}_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={args.total_steps}_{curl_str}_{reward_str}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if args.use_wandb:
        wandb.finish()

