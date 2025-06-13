"""
python pretrain_taco_multi_task_episodes.py --dataset_config data_episodes/MT50/0.99 --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4
python pretrain_taco_multi_task_episodes.py --dataset_config exp_local/metaworld/taco_prova_push-v2/buffer --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4
python pretrain_taco_multi_task_episodes.py --dataset_config data_episodes/my_config --save_path models/debug/ --total_steps 300_000_000 --checkpoint "1_000_000, 2_000_000, 20_000_000,200_000_000" --lr 5e-4
"""
import os
import torch
import utils
import wandb
import argparse
from pathlib import Path
from agents.taco import TACOAgent
from pretraining_utils import load_unified_dataset





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
    parser.add_argument('--fastwork', action='store_true', help='Prepend /home/mprattico/fastwork/ to dataset paths')
    parser.add_argument('--no_curl', action='store_true', help='Disable CURL loss (enabled by default)')
    parser.add_argument('--no_reward', action='store_true', help='Disable reward loss (enabled by default)')
    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in args.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    pretraining_dataset_path = args.dataset_config + "/pretraining_datasets"
    valid_datset_path = args.dataset_config + "/test_dataset"

    if args.fastwork:
        pretraining_dataset_path = "/fastwork/mprattico/" + pretraining_dataset_path
        valid_datset_path = "/fastwork/mprattico/" + valid_datset_path
    
    # Check for overlapping subdirectories between training and validation datasets
    if os.path.exists(pretraining_dataset_path) and os.path.exists(valid_datset_path):
        train_dirs = {d.name for d in Path(pretraining_dataset_path).iterdir() if d.is_dir()}
        valid_dirs = {d.name for d in Path(valid_datset_path).iterdir() if d.is_dir()}
        common_dirs = train_dirs.intersection(valid_dirs)
        assert len(common_dirs) == 0, f"Found overlapping directories in training and validation sets: {common_dirs}"
    
    # load
    train_dataloader = load_unified_dataset(
        root_dir=pretraining_dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    valid_dataloader = load_unified_dataset(
        root_dir=valid_datset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # check the subdirectories
    dataset_dirs = [d for d in Path(pretraining_dataset_path).iterdir() if d.is_dir()]
    print(f"Found {len(dataset_dirs)} dataset subdirectories in {pretraining_dataset_path}")

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
            "datasets": dataset_dirs,
            "num_datasets": len(args.dataset_config),
            "dataset_config": args.dataset_config,
            "fastwork": args.fastwork,
            "use_curl": not args.no_curl,
            "use_reward": not args.no_reward,
        }
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
        curl=not args.no_curl
    )

    steps = 0
    epoch = 0
    valid_iterator = iter(valid_dataloader)
    
    while steps < args.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{args.total_steps}")
        
        for batch in train_dataloader:
            
            # *** Evaluation step ***
            if (steps//args.batch_size) % args.eval_frequency == 0:
                taco_agent.train(False)  # Set to eval mode
                
                eval_metrics_sum = {
                    'eval/reward_loss': 0,
                    'eval/curl_loss': 0,
                    'eval/taco_loss': 0,
                    'eval/batch_reward': 0,
                    'eval/avg_rew_pred_error_percentage': 0,
                    'eval/log_cosh': 0,
                    'eval/rel_error_filtered': 0,
                    'eval/smape': 0,
                }
                num_eval_batches = 0
                
                # Evaluate on multiple batches
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
                    if 'log_cosh' in eval_metrics:
                        eval_metrics_sum['eval/log_cosh'] += eval_metrics['log_cosh']
                    if 'rel_error_filtered' in eval_metrics:
                        eval_metrics_sum['eval/rel_error_filtered'] += eval_metrics['rel_error_filtered']
                    if 'smape' in eval_metrics:
                        eval_metrics_sum['eval/smape'] += eval_metrics['smape']
                    num_eval_batches += 1
                
                # Average the metrics
                for key in eval_metrics_sum:
                    eval_metrics_sum[key] /= num_eval_batches
                
                # Log eval metrics to wandb
                if args.use_wandb:
                    eval_metrics_sum['steps'] = steps
                    eval_metrics_sum['epoch'] = epoch
                    wandb.log(eval_metrics_sum)
                
                print(f"Validation metrics: {eval_metrics_sum}")
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
                wandb.log(metrics)
                
            print(f"Steps: {steps}/{args.total_steps}, Metrics: {metrics}")

            # *** Save model checkpoint ***
            # Check if we need to save a checkpoint at this step
            if any(s <= steps < s + args.batch_size for s in checkpoint_steps):
                curl_str = "curl" if not args.no_curl else "nocurl"
                reward_str = "rew" if not args.no_reward else "norew"
                checkpoint_path = f"{args.save_path}/taco_MT_{'_'.join(args.dataset_config.split('/')[1:])}_lr={args.lr}_ts={steps}_{curl_str}_{reward_str}.pt"
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
    torch.save({
        'encoder': taco_agent.encoder.state_dict(),
        'taco': taco_agent.TACO.state_dict(),
        'act_tok': taco_agent.act_tok.state_dict(),
        'args': vars(args),  # Save configuration for easier loading
    }, f"{args.save_path}/taco_MT_{args.dataset_config.split('/')[-1].split('.')[0]}_lr={args.lr}_ts={args.total_steps}_{curl_str}_{reward_str}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if args.use_wandb:
        wandb.finish()

