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
            reward_loss = F.mse_loss(reward_pred, reward)
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


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, multistep=1, nstep=1, discount=0.99):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.next_observations = dataset['next_observations']
        self.rewards = dataset['rewards']
        self.terminals = dataset['terminals']
        
        if 'proprio_states' in dataset:
            self.proprio_states = dataset['proprio_states']
        
        self.multistep = multistep
        self.nstep = nstep
        self.discount = discount
        
        # Identify episode boundaries
        self.episode_starts = [0]
        self.episode_ends = []
        for i, done in enumerate(self.terminals):
            if done:
                self.episode_ends.append(i)
                if i + 1 < len(self.terminals):
                    self.episode_starts.append(i + 1)
        
        # Handle case where the last episode doesn't end with done=True
        if len(self.episode_ends) < len(self.episode_starts):
            self.episode_ends.append(len(self.terminals) - 1)
            
        self.num_episodes = len(self.episode_starts)
        print(f"Loaded {self.num_episodes} episodes from dataset")
    
    def _sample(self):
        # Sample a random episode
        episode_idx = np.random.randint(0, self.num_episodes)
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        
        # Ensure we have enough steps for multistep and nstep
        n_step = max(self.nstep, self.multistep)
        if end_idx - start_idx < n_step:
            # If episode is too short, try another one
            return self._sample()
        
        # Sample a random index within the episode that has enough future steps
        idx = np.random.randint(start_idx, end_idx - n_step + 1) # +1 TODO original code put a +1 why??
        
        # Get observation, action and reward like in the original _sample method
        obs = self.observations[idx]
        r_next_obs = self.observations[idx + self.multistep - 1]
        action = self.actions[idx]
        
        # Create action_seq by concatenating multiple actions
        action_seq_indices = [min(idx + i, end_idx) for i in range(self.multistep)]
        action_seq = np.concatenate([self.actions[i:i+1] for i in action_seq_indices])
        
        next_obs = self.observations[min(idx + self.nstep - 1, end_idx)]
        
        # Calculate cumulative reward and discount
        reward = np.zeros_like(self.rewards[idx])
        discount = np.ones_like(reward)
        
        for i in range(self.nstep):
            if idx + i <= end_idx:
                step_reward = self.rewards[idx + i]
                reward += discount * step_reward
                # Use terminal flag to determine discount
                if idx + i < end_idx:  # Not the last step in the episode
                    discount_factor = 0.0 if self.terminals[idx + i] else self.discount
                    discount *= discount_factor
        
        return (obs, action, action_seq, reward, next_obs, r_next_obs)
    
    def __iter__(self):
        while True:
            yield self._sample()


def make_offline_replay_loader(dataset_path, batch_size, multistep=1, nstep=1, discount=0.99, num_workers=0):
    iterable = OfflineReplayBuffer(dataset_path, multistep, nstep, discount)
    
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
    parser.add_argument('--multistep', type=int, default=1, help='Multistep (set to 1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='taco-pretrain', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--total_steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--nstep', type=int, default=1, help='N-step returns')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained model')
    args = parser.parse_args()

    assert args.multistep == args.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

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
            "dataset_path": args.dataset_path,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )

    # Use the new replay buffer instead of directly loading the dataset
    dataloader = make_offline_replay_loader(
        args.dataset_path, 
        args.batch_size, 
        multistep=args.multistep, 
        nstep=args.nstep,
        discount=args.discount,
        num_workers=args.num_workers
    )

    batch = next(iter(dataloader))
    obs_shape = batch[0].shape[1:]
    action_shape = batch[1].shape[1:]
    # # Sample a batch to get the observation and action shapes
    # with open(args.dataset_path, 'rb') as f:
    #     dataset = pickle.load(f)
    #     obs_shape = dataset['observations'][0].shape
    #     action_shape = dataset['actions'][0].shape

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
    
    while steps < args.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{args.total_steps}")
        
        for batch in dataloader:
            metrics = taco_agent.update(batch)
            steps += args.batch_size
            
            # Log metrics to wandb
            if args.use_wandb:
                metrics['steps'] = steps
                metrics['epoch'] = epoch
                wandb.log(metrics)
            
            print(f"Steps: {steps}/{args.total_steps}, Metrics: {metrics}")
            
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
    }, f"{args.save_path}/taco_{args.dataset_path.split('/')[-1]}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if args.use_wandb:
        wandb.finish()

