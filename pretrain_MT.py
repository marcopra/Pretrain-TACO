"""
python pretrain_taco_multi_task_episodes.py --dataset_config data_episodes/ST50/0.0 --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4
python pretrain_taco_multi_task_episodes.py --dataset_config exp_local/metaworld/taco_prova_push-v2/buffer --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4
python pretrain_taco_multi_task_episodes.py --dataset_config data_episodes/my_config --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4

Examples for resuming a run:
python pretrain_taco_multi_task_episodes_from_checkpoint.py --dataset_config data_episodes/MT50/0.33 --save_path models/ --total_steps 300_000_000 --checkpoint "250_000_000,300_000_000" --lr 5e-4 --resume_checkpoint models/taco_MT_MT50_OOD_Push_0.33_lr=0.0005_ts=200000512_curl_rew.pt --resume_wandb_run ze7o8oaq
"""
import os
import torch
import utils
import wandb
import argparse
import json
from pathlib import Path
from agents.taco import TACOAgent
from pretraining_utils import load_unified_dataset


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
    parser.add_argument('--wandb_project', type=str, default='taco-mt-pretrain', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--total_steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--checkpoint', type=str, default="500000,1000000", 
                        help='Comma-separated list of steps at which to save checkpoints (e.g., "100000,500000,1000000")')
    parser.add_argument('--nstep', type=int, default=3, help='N-step returns')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained model')
    parser.add_argument('--eval_frequency', type=int, default=50, help='Frequency of evaluation steps')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--eval_batches', type=int, default=10, help='Number of batches to use for evaluation')
    parser.add_argument('--no_curl', action='store_true', help='Disable CURL loss (enabled by default)')
    parser.add_argument('--no_reward', action='store_true', help='Disable reward loss (enabled by default)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use for training (default: adam)')
    # New arguments for resuming training
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to the checkpoint to resume training from')
    parser.add_argument('--resume_wandb_run', type=str, default=None, help='ID of wandb run to resume')
    parser.add_argument('--continue_steps', action='store_true', default=True, help='Continue step counter from checkpoint (instead of starting from 0)')
    parser.add_argument('--max_episodes_per_dataset', type=int, default=8, help='Maximum episodes to load per dataset')
    parser.add_argument('--max_size', type=int, default=None, help='Maximum size of replay buffer for training')
    parser.add_argument('--homogeneous', action='store_true', help='Load transitions evenly across datasets')
    parser.add_argument('--log_frequency', type=int, default=100, help='Log metrics every n batches')
    parser.add_argument('--validation_source', type=str, default='test', choices=['test', 'split'], 
                        help='Source for validation data: "test" uses test_dataset, "split" splits pretraining data')
    parser.add_argument('--validation_split_ratio', type=float, default=0.8, 
                        help='Ratio of pretraining data to use for training when using validation_source=split')
    parser.add_argument('--test_eval_frequency', type=int, default=200, 
                        help='Frequency of test evaluation when using validation_source=split (0 to disable)')
    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in args.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    # Handle dataset config paths based on type (JSON vs folder)
    config_path = Path(args.dataset_config)
    
    if config_path.suffix == '.json':
        # For JSON config, we don't need the old path logic
        pretraining_config = args.dataset_config
        test_config = args.dataset_config
        
        # For JSON, we'll get dataset info from the config itself
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        dataset_dirs = config_data.get('pretraining_datasets', [])
        dataset_names = [extract_task_name_from_path(d) for d in dataset_dirs]
        print(f"Found {len(dataset_dirs)} pretraining datasets in config: {dataset_names}")
    else:
        # Original folder-based logic
        pretraining_dataset_path = args.dataset_config + "/pretraining_datasets"
        valid_dataset_path = args.dataset_config + "/test_dataset"
        
        pretraining_config = args.dataset_config
        test_config = args.dataset_config
        
        # Check for overlapping subdirectories between training and validation datasets
        if os.path.exists(pretraining_dataset_path) and os.path.exists(valid_dataset_path):
            train_dirs = {d.name for d in Path(pretraining_dataset_path).iterdir() if d.is_dir()}
            valid_dirs = {d.name for d in Path(valid_dataset_path).iterdir() if d.is_dir()}
            common_dirs = train_dirs.intersection(valid_dirs)
            assert len(common_dirs) == 0, f"Found overlapping directories in training and validation sets: {common_dirs}"
        
        # Get dataset directories for folder-based config
        if os.path.exists(pretraining_dataset_path):
            dataset_dirs = [d for d in Path(pretraining_dataset_path).iterdir() if d.is_dir()]
            dataset_names = [extract_task_name_from_path(d) for d in dataset_dirs]
            print(f"Found {len(dataset_dirs)} dataset subdirectories in {pretraining_dataset_path}")
        else:
            dataset_dirs = []
            dataset_names = []
            print(f"Pretraining dataset path not found: {pretraining_dataset_path}")
    
    # Load datasets based on validation source
    if args.validation_source == 'test':
        # Original behavior: use test_dataset as validation
        train_dataloader = load_unified_dataset(
            config_or_path=pretraining_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_episodes_per_dataset=args.max_episodes_per_dataset,
            max_size=args.max_size,
            homogeneous=args.homogeneous,
            is_test=False
        )

        valid_dataloader = load_unified_dataset(
            config_or_path=test_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_episodes_per_dataset=args.max_episodes_per_dataset,
            max_size=None,  # Always use all test data
            homogeneous=args.homogeneous,
            is_test=True
        )
        
        test_dataloader = None  # No separate test set
        
    else:  # validation_source == 'split'
        # When splitting, we need to respect max_size and max_episodes for training set
        # First load the full dataset to understand its size, then split appropriately
        print(f"Using split validation with ratio {args.validation_split_ratio}")
        print(f"Training set will respect max_episodes_per_dataset={args.max_episodes_per_dataset}, max_size={args.max_size}")
        
        # Load training data with the specified constraints
        train_dataloader = load_unified_dataset(
            config_or_path=pretraining_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_episodes_per_dataset=args.max_episodes_per_dataset,
            max_size=args.max_size,
            homogeneous=args.homogeneous,
            is_test=False,
            validation_split_ratio=args.validation_split_ratio,
            use_training_split=True  # Only get the training portion
        )
        
        # Load validation data from the remaining data (no size constraints)
        valid_dataloader = load_unified_dataset(
            config_or_path=pretraining_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_episodes_per_dataset=None,  # Use all remaining episodes for validation
            max_size=None,  # Use all remaining data for validation
            homogeneous=args.homogeneous,
            is_test=False,
            validation_split_ratio=args.validation_split_ratio,
            use_training_split=False  # Only get the validation portion
        )
        
        # Load test set separately if test evaluation is enabled
        if args.test_eval_frequency > 0:
            test_dataloader = load_unified_dataset(
                config_or_path=test_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_episodes_per_dataset=args.max_episodes_per_dataset,
                max_size=None,
                homogeneous=args.homogeneous,
                is_test=True
            )
        else:
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
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        saved_args = checkpoint.get('args', {})
        
        # Extract dataset config from checkpoint path
        checkpoint_basename = os.path.basename(args.resume_checkpoint)
        if "taco_MT_" in checkpoint_basename:
            # Extract dataset config from path format like "taco_MT_ST50_OOD_Push_0.66_lr=0.0005_ts=100352_curl_rew.pt"
            parts = checkpoint_basename.split("_")
            # Find dataset and ratio parts (e.g., "ST50" and "0.66")
            if len(parts) >= 4:
                dataset_type = "_".join(parts[2:4])  # e.g., "ST50" or "MT50" #TODO to test
                ratio = parts[5]  # e.g., "0.33" or "0.66"
                expected_config = f"data_episodes/{dataset_type}/{ratio}"
                
                # Check if the dataset config matches
                if not args.dataset_config.endswith(expected_config):
                    raise AssertionError(
                        f"Dataset config mismatch! Provided: {args.dataset_config}, "
                        f"Expected (based on checkpoint): {expected_config}"
                    )
                print(f"Verified dataset config: {args.dataset_config} matches checkpoint")
        
        # Set steps and epoch from checkpoint if requested
        if args.continue_steps and 'steps' in checkpoint:
            steps = checkpoint['steps']
            print(f"Resuming from step: {steps}")
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            print(f"Resuming from epoch: {epoch}")

    # Initialize wandb if enabled
    if args.use_wandb:
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
            "datasets": dataset_names if config_path.suffix == '.json' else [d.name for d in dataset_dirs],
            "num_datasets": len(dataset_dirs),
            "dataset_config": args.dataset_config,
            "use_curl": not args.no_curl,
            "use_reward": not args.no_reward,
            "resumed_from_checkpoint": args.resume_checkpoint,
            "max_episodes_per_dataset": args.max_episodes_per_dataset,
            "max_size": args.max_size,
            "homogeneous": args.homogeneous,
            "validation_source": args.validation_source,
            "validation_split_ratio": args.validation_split_ratio if args.validation_source == 'split' else None,
            "test_eval_frequency": args.test_eval_frequency if args.validation_source == 'split' else None
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
                    best_model_path = f"{args.save_path}/taco_MT_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}_best.pt"
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
                    }, best_model_path)
                
                print(f"Validation metrics: {eval_metrics_sum}")
                taco_agent.train(True)  # Set back to train mode
            
            # *** Test evaluation step (only when using split validation) ***
            if (args.validation_source == 'split' and test_dataloader is not None and 
                args.test_eval_frequency > 0 and (steps//args.batch_size) % args.test_eval_frequency == 0):
                
                taco_agent.train(False)  # Set to eval mode
                
                test_metrics_sum = {
                    'test/reward_loss': 0,
                    'test/curl_loss': 0,
                    'test/taco_loss': 0,
                    'test/total_loss': 0,
                    'test/batch_reward': 0,
                    'test/avg_rew_pred_error_percentage': 0,
                    'test/log_cosh': 0,
                    'test/rel_error_filtered': 0,
                    'test/smape': 0,
                }
                num_test_batches = 0
                
                # Evaluate on test set
                for _ in range(args.eval_batches):
                    try:
                        test_batch = next(test_iterator)
                    except StopIteration:
                        test_iterator = iter(test_dataloader)
                        test_batch = next(test_iterator)
                    obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                        test_batch, args.device)
                    
                    test_metrics = taco_agent.evaluate_taco(obs, action, action_seq, r_next_obs, reward)
                    
                    # Add test/ prefix to metrics
                    test_metrics_sum['test/reward_loss'] += test_metrics['reward_loss']
                    test_metrics_sum['test/curl_loss'] += test_metrics['curl_loss']
                    test_metrics_sum['test/taco_loss'] += test_metrics['taco_loss']
                    test_metrics_sum['test/batch_reward'] += test_metrics['batch_reward']
                    test_metrics_sum['test/avg_rew_pred_error_percentage'] += test_metrics['avg_rew_pred_error_percentage']
                    test_metrics_sum['test/total_loss'] += test_metrics['total_loss']
                    if 'log_cosh' in test_metrics:
                        test_metrics_sum['test/log_cosh'] += test_metrics['log_cosh']
                    if 'rel_error_filtered' in test_metrics:
                        test_metrics_sum['test/rel_error_filtered'] += test_metrics['rel_error_filtered']
                    if 'smape' in test_metrics:
                        test_metrics_sum['test/smape'] += test_metrics['smape']
                    num_test_batches += 1
                
                # Average the test metrics
                for key in test_metrics_sum:
                    test_metrics_sum[key] /= num_test_batches
                
                # Log test metrics to wandb
                if args.use_wandb:
                    test_metrics_sum['steps'] = steps
                    test_metrics_sum['epoch'] = epoch
                    wandb.log(test_metrics_sum)
                
                print(f"Test metrics: {test_metrics_sum}")
                taco_agent.train(True)  # Set back to train mode
            
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
                checkpoint_path = f"{args.save_path}/taco_MT_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}.pt"
                print(f"Saving checkpoint at step {steps} to {checkpoint_path}")
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'encoder': taco_agent.encoder.state_dict(),
                    'taco': taco_agent.TACO.state_dict(),
                    'act_tok': taco_agent.act_tok.state_dict(),
                    'args': vars(args),  # Save configuration for easier loading
                    'steps': steps,
                    'epoch': epoch,
                }, checkpoint_path)
            
            if steps >= args.total_steps:
                break
    
    # *** Save the trained TACO agent ***
    print(f"Saving model to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    curl_str = "curl" if not args.no_curl else "nocurl"
    reward_str = "rew" if not args.no_reward else "norew"
    optimizer_str = f"_{args.optimizer}" if args.optimizer != "adam" else ""
    torch.save({
        'encoder': taco_agent.encoder.state_dict(),
        'taco': taco_agent.TACO.state_dict(),
        'act_tok': taco_agent.act_tok.state_dict(),
        'args': vars(args),  # Save configuration for easier loading
        'steps': steps,
        'epoch': epoch,
    }, f"{args.save_path}/taco_MT_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}{optimizer_str}_ts={args.total_steps}_{curl_str}_{reward_str}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if args.use_wandb:
        wandb.finish()

