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


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        
class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class TACO(nn.Module):
    """
    TACO Constrastive loss
    """

    def __init__(self, repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, act_tok, encoder, multistep, device):
        super(TACO, self).__init__()

        self.multistep = multistep
        self.encoder = encoder
        self.device = device
        
        a_dim = action_shape[0]

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.act_tok = act_tok
        
        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.reward = nn.Sequential(
            nn.Linear(feature_dim+latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(utils.weight_init)
    
    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder(x))
        else:
            z_out = self.proj_s(self.encoder(x))
        return z_out
    
    def project_sa(self, s, a):
        x = torch.concat([s,a], dim=-1)
        return self.proj_sa(x)
        
    def compute_logits(self, z_a, z_pos):
        """
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    

class TACOAgent:
    def __init__(self, obs_shape, action_shape, device, encoder_lr, feature_dim,
                 hidden_dim, update_every_steps, stddev_clip, use_tb, use_wandb,
                 reward, multistep, latent_a_dim, curl):
        self.device = device
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip
        
        self.reward = reward
        self.multistep = multistep
        self.curl = curl

        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0]*1.25)+1
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep)
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        

        self.TACO = TACO(self.encoder.repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, multistep, device).to(device)
        
        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )
        self.encoder_opt = torch.optim.Adam(parameters, lr=encoder_lr)
        self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=encoder_lr)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.TACO.train()

   
    
    def update_taco(self, obs, action, action_seq, next_obs, reward):
        metrics = dict()
        
        obs_anchor = self.aug(obs.float())
        obs_pos = self.aug(obs.float())
        z_a = self.TACO.encode(obs_anchor)
        z_pos = self.TACO.encode(obs_pos, ema=True)   
        ### Compute CURL loss
        if self.curl:
            logits = self.TACO.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)
        
        ### Compute action encodings
        action_en = self.TACO.act_tok(action, seq=False) 
        action_seq_en = self.TACO.act_tok(action_seq, seq=True)
        
        ### Compute reward prediction loss
        if self.reward:
            reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), reward)
            # Average percentage of reward prediction error
            with torch.no_grad():
                # Average percentage of reward prediction error
                metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred.squeeze(-1) - reward) / (reward + 1e-6)).item() 
                error = reward_pred.squeeze(-1) - reward
                metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
                threshold = 1e-3  # puoi settarlo in base al tuo dominio
                mask = reward.abs() > threshold
                metrics['rel_error_filtered'] = torch.mean(
                    torch.abs(reward_pred.squeeze(-1)[mask] - reward[mask]) / (reward[mask] + 1e-6)
                ).item()
                numerator = torch.abs(reward_pred.squeeze(-1) - reward)
                denominator = torch.abs(reward_pred.squeeze(-1)) + torch.abs(reward) + 1e-6
                metrics['smape'] = torch.mean(2.0 * numerator / denominator).item()

        else:
            reward_loss = torch.tensor(0.)
        
        ### Compute TACO loss
        next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
        curr_za = self.TACO.project_sa(z_a, action_seq_en) 
        logits = self.TACO.compute_logits(curr_za, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        taco_loss = self.cross_entropy_loss(logits, labels)
            
        self.taco_opt.zero_grad()
        (taco_loss + curl_loss + reward_loss).backward()
        self.taco_opt.step()
        
        metrics['reward_loss']  = reward_loss.item()
        metrics['curl_loss'] = curl_loss.item()
        metrics['taco_loss']  = taco_loss.item()
        
        return metrics
        
        
    
    def update(self, batch):
        obs, action, action_seq, reward, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)
        metrics = dict()

        metrics['batch_reward'] = reward.mean().item()
        
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward))

        return metrics
    
    def evaluate(self, batch):
        """
        Compute losses without updating the network (for validation)
        """
        with torch.no_grad():
            obs, action, action_seq, reward, next_obs, r_next_obs = utils.to_torch(
                batch, self.device)
            metrics = dict()
            
            metrics['batch_reward'] = reward.mean().item()
            
            # Compute TACO losses similar to update_taco but without gradients
            obs_anchor = self.aug(obs.float())
            obs_pos = self.aug(obs.float())
            z_a = self.TACO.encode(obs_anchor)
            z_pos = self.TACO.encode(obs_pos, ema=True)
            
            if self.curl:
                logits = self.TACO.compute_logits(z_a, z_pos)
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                curl_loss = self.cross_entropy_loss(logits, labels)
            else:
                curl_loss = torch.tensor(0.)
            
            action_en = self.TACO.act_tok(action, seq=False) 
            action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
            if self.reward:
                reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
                reward_loss = F.mse_loss(reward_pred.squeeze(-1), reward)
                # Average percentage of reward prediction error
                metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred.squeeze(-1) - reward) / (reward + 1e-6)).item() 
                error = reward_pred.squeeze(-1) - reward
                metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
                threshold = 1e-3  # puoi settarlo in base al tuo dominio
                mask = reward.abs() > threshold
                metrics['rel_error_filtered'] = torch.mean(
                    torch.abs(reward_pred.squeeze(-1)[mask] - reward[mask]) / (reward[mask] + 1e-6)
                ).item()
                numerator = torch.abs(reward_pred.squeeze(-1) - reward)
                denominator = torch.abs(reward_pred.squeeze(-1)) + torch.abs(reward) + 1e-6
                metrics['smape'] = torch.mean(2.0 * numerator / denominator).item()


            else:
                reward_loss = torch.tensor(0.)
            
            next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
            curr_za = self.TACO.project_sa(z_a, action_seq_en) 
            logits = self.TACO.compute_logits(curr_za, next_z)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            taco_loss = self.cross_entropy_loss(logits, labels)
            
            metrics['reward_loss'] = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['taco_loss'] = taco_loss.item()
            
        return metrics
    
    def load_pretrained(self, model_path, map_location=None):
        """
        Load a pretrained TACO model from a saved checkpoint.
        
        Args:
            model_path: Path to the saved model checkpoint
            map_location: Optional device mapping for torch.load
        
        Returns:
            dict: The original training arguments
        """
        if map_location is None:
            map_location = self.device
            
        checkpoint = torch.load(model_path, map_location=map_location)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.TACO.load_state_dict(checkpoint['taco'])
        self.act_tok.load_state_dict(checkpoint['act_tok'])
        
        print(f"Loaded pretrained model from {model_path}")
        
        return checkpoint.get('args', {})  # Return the saved args for reference


def load_dataset_config(config_path):
    """
    Load dataset configuration from JSON file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    pretraining_datasets = config['pretraining_datasets']
    test_dataset = config['test_dataset']
    
    # Remove test dataset from pretraining datasets if it's in there
    if test_dataset in pretraining_datasets:
        pretraining_datasets.remove(test_dataset)
    
    return pretraining_datasets, test_dataset


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, dataset_paths, multistep=1, nstep=1, discount=0.99, split='train', train_ratio=0.8, fastwork=False):
        """
        Initialize replay buffer with multiple datasets
        
        Args:
            dataset_paths: List of paths to dataset files
            multistep: Number of steps for multistep learning
            nstep: Number of steps for n-step returns
            discount: Discount factor
            split: 'train' or 'valid'
            train_ratio: Ratio of data to use for training
            fastwork: Whether to prepend /home/mprattico/fastwork/ to dataset paths
        """
        self.multistep = multistep
        self.nstep = nstep
        self.discount = discount
        self.split = split
        self.fastwork = fastwork
        
        # Initialize storage for dataset-specific information
        self.datasets = []
        self.dataset_episode_starts = []
        self.dataset_episode_ends = []
        self.dataset_num_episodes = []
        self.total_episodes = 0
        self.dataset_indices = []  # Maps global episode idx to (dataset_idx, local_episode_idx)
        
        # Load each dataset and track episode boundaries
        for dataset_idx, path in enumerate(dataset_paths):
            # Prepend fastwork path if enabled
            if fastwork:
                load_path = os.path.join('/home/mprattico/fastwork/', path)
            else:
                load_path = path
                
            print(f"Loading dataset from {load_path}")
            with open(load_path, 'rb') as f:
                dataset = pickle.load(f)
            
            dataset_info = {
                'observations': dataset['observations'],
                'actions': dataset['actions'],
                'next_observations': dataset['next_observations'],
                'rewards': dataset['rewards'],
                'terminals': dataset['terminals'],
            }
            
            if 'proprio_states' in dataset:
                dataset_info['proprio_states'] = dataset['proprio_states']
            
            self.datasets.append(dataset_info)
            
            # Identify episode boundaries for this dataset
            episode_starts = [0]
            episode_ends = []
            for i, done in enumerate(dataset_info['terminals']):
                if done:
                    episode_ends.append(i)
                    if i + 1 < len(dataset_info['terminals']):
                        episode_starts.append(i + 1)
            
            # Handle case where the last episode doesn't end with done=True
            if len(episode_ends) < len(episode_starts):
                episode_ends.append(len(dataset_info['terminals']) - 1)
            
            num_episodes = len(episode_starts)
            self.dataset_episode_starts.append(episode_starts)
            self.dataset_episode_ends.append(episode_ends)
            self.dataset_num_episodes.append(num_episodes)
            
            # Track global episode indices
            for local_ep_idx in range(num_episodes):
                self.dataset_indices.append((dataset_idx, local_ep_idx))
            
            self.total_episodes += num_episodes
            print(f"Loaded {num_episodes} episodes from dataset {path}")
        # Split episodes for train/validation across all datasets
        # First we shuffle the global episode indices
        all_indices = np.arange(self.total_episodes)
        np.random.shuffle(all_indices)
        
        split_idx = int(self.total_episodes * train_ratio)
        if split == 'train':
            self.usable_indices = all_indices[:split_idx]
        else:  # 'valid'
            self.usable_indices = all_indices[split_idx:]
        
        self.num_episodes = len(self.usable_indices)
        print(f"Loaded {self.num_episodes} episodes for {split} split across {len(dataset_paths)} datasets")
    
    def _sample(self):
        # Sample a random episode from our usable indices
        global_episode_idx = np.random.choice(self.usable_indices)
        dataset_idx, local_episode_idx = self.dataset_indices[global_episode_idx]
        
        # Get the dataset and episode boundaries
        dataset = self.datasets[dataset_idx]
        start_idx = self.dataset_episode_starts[dataset_idx][local_episode_idx]
        end_idx = self.dataset_episode_ends[dataset_idx][local_episode_idx]
        
        # Ensure we have enough steps for multistep and nstep
        n_step = max(self.nstep, self.multistep)
        if end_idx - start_idx < n_step:
            # If episode is too short, try another one
            return self._sample()
        
        # Sample a random index within the episode that has enough future steps
        idx = np.random.randint(start_idx, end_idx - n_step + 1)
        
        # Get observation, action and reward
        obs = dataset['observations'][idx]
        r_next_obs = dataset['observations'][idx + self.multistep - 1]
        action = dataset['actions'][idx]
        
        # Create action_seq by concatenating multiple actions
        action_seq_indices = [min(idx + i, end_idx) for i in range(self.multistep)]
        action_seq = np.concatenate([dataset['actions'][i:i+1] for i in action_seq_indices])
        
        next_obs = dataset['observations'][min(idx + self.nstep - 1, end_idx)]
        
        # Calculate cumulative reward and discount
        reward = np.zeros_like(dataset['rewards'][idx])
        discount = np.ones_like(reward)
        
        for i in range(self.nstep):
            if idx + i <= end_idx:
                step_reward = dataset['rewards'][idx + i]
                reward += discount * step_reward
                # Use terminal flag to determine discount
                if idx + i < end_idx:  # Not the last step in the episode
                    discount_factor = 0.0 if dataset['terminals'][idx + i] else self.discount
                    discount *= discount_factor
        
        return (obs, action, action_seq, reward, next_obs, r_next_obs)
    
    def __iter__(self):
        while True:
            yield self._sample()


def make_offline_replay_loader(dataset_paths, batch_size, multistep=1, nstep=1, discount=0.99, 
                              num_workers=0, split='train', train_ratio=0.8, fastwork=False):
    """
    Create a data loader for multiple datasets
    """
    iterable = OfflineReplayBuffer(
        dataset_paths, 
        multistep, 
        nstep, 
        discount, 
        split=split, 
        train_ratio=train_ratio,
        fastwork=fastwork
    )
    
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
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
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained model')
    parser.add_argument('--eval_frequency', type=int, default=1000, help='Frequency of evaluation steps')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--eval_batches', type=int, default=10, help='Number of batches to use for evaluation')
    parser.add_argument('--fastwork', action='store_true', help='Prepend /home/mprattico/fastwork/ to dataset paths')
    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in args.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    # Load dataset configuration
    pretraining_datasets, test_dataset = load_dataset_config(args.dataset_config)
    print(f"Loaded configuration with {len(pretraining_datasets)} pretraining datasets")
    print(f"Test dataset: {test_dataset}")

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
            "datasets": pretraining_datasets,
            "num_datasets": len(pretraining_datasets),
            "dataset_config": args.dataset_config,
            "fastwork": args.fastwork,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )

    # Create train and validation dataloaders with multiple datasets
    train_dataloader = make_offline_replay_loader(
        pretraining_datasets, 
        args.batch_size, 
        multistep=args.multistep, 
        nstep=args.nstep,
        discount=args.discount,
        num_workers=args.num_workers,
        split='train',
        train_ratio=args.train_ratio,
        fastwork=args.fastwork
    )

    valid_dataloader = make_offline_replay_loader(
        pretraining_datasets, 
        args.batch_size, 
        multistep=args.multistep, 
        nstep=args.nstep,
        discount=args.discount,
        num_workers=args.num_workers,
        split='valid',
        train_ratio=args.train_ratio,
        fastwork=args.fastwork
    )

    # Get a batch from training data to initialize the agent
    batch = next(iter(train_dataloader))
    obs_shape = batch[0].shape[1:]
    action_shape = batch[1].shape[1:]

    taco_agent = TACOAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=args.device,
        encoder_lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        update_every_steps=1,
        stddev_clip=0.3,
        use_tb=False,
        use_wandb=args.use_wandb,
        reward=True,
        multistep=args.multistep,
        latent_a_dim='none',
        curl=True
    )

    steps = 0
    epoch = 0
    valid_iterator = iter(valid_dataloader)
    
    while steps < args.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{args.total_steps}")
        
        for batch in train_dataloader:
            # Training update
            metrics = taco_agent.update(batch)
            steps += args.batch_size
            
            # Log training metrics to wandb
            if args.use_wandb:
                metrics['steps'] = steps
                metrics['epoch'] = epoch
                wandb.log(metrics)
            
            # Periodically evaluate on validation set
            if steps % args.eval_frequency == 0:
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
                    
                    eval_metrics = taco_agent.evaluate(eval_batch)
                    
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
            
            print(f"Steps: {steps}/{args.total_steps}, Metrics: {metrics}")
            # Check if we need to save a checkpoint at this step
            if any(s <= steps < s + args.batch_size for s in checkpoint_steps):
                checkpoint_path = f"{args.save_path}/taco_MT_{args.dataset_config.split('/')[-1].split('.')[0]}_ts={steps}.pt"
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
    
    # Save the trained TACO agent
    print(f"Saving model to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
        'encoder': taco_agent.encoder.state_dict(),
        'taco': taco_agent.TACO.state_dict(),
        'act_tok': taco_agent.act_tok.state_dict(),
        'args': vars(args),  # Save configuration for easier loading
    }, f"{args.save_path}/taco_MT_{args.dataset_config.split('/')[-1].split('.')[0]}_ts={args.total_steps}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if args.use_wandb:
        wandb.finish()

