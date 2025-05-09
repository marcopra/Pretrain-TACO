"""
python eval_taco_multi_task_episodes.py --pretrained_path "models/taco_MT_ST50_0.66_lr=0.0005_ts=50000896.pt" --dataset_config "data_episodes/my_config"
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
                 reward, multistep, latent_a_dim, curl, pretrained_path=None, freeze_encoder=True):
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
        self.freeze_encoder = freeze_encoder
        
        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )
        self.encoder_opt = torch.optim.Adam(parameters, lr=encoder_lr)
        self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=encoder_lr)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        if pretrained_path is None or pretrained_path.lower() == 'none':
            print("No pretrained model provided, initializing from scratch.")
        else:
            print(f"Loading pretrained model from {pretrained_path}, freeze encoder: {freeze_encoder}")
            self.load_pretrained(pretrained_path, None, self.freeze_encoder)
            
        self.pretrained_path = pretrained_path
        self.train()

    def load_pretrained(self, model_path, map_location=None, freeze_encoder=False):
        if map_location is None:
            map_location = self.device
            
        checkpoint = torch.load(model_path, map_location=map_location)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.TACO.load_state_dict(checkpoint['taco'])
        self.act_tok.load_state_dict(checkpoint['act_tok'])
        
        # Importante: se non vuoi congelare l'encoder e mantenerlo in modalità train
        if not freeze_encoder:
            self.encoder.train()
            self.TACO.train()
            self.act_tok.train()
        else:
            # Altrimenti congela i parametri come facevi prima
            self._frozen_fingerprints = {
                'encoder': self._get_model_fingerprint(self.encoder),
                'taco': self._get_model_fingerprint(self.TACO),
                'act_tok': self._get_model_fingerprint(self.act_tok)
            }
            
            self.encoder.eval()
            self.TACO.eval()
            self.act_tok.eval()
            
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.TACO.parameters():
                param.requires_grad = False
            for param in self.act_tok.parameters():
                param.requires_grad = False
        
        print(f"Loaded pretrained model from {model_path}")
        
        return checkpoint.get('args', {})
    
    def _get_model_fingerprint(self, model):
        """Generate a unique fingerprint for model parameters"""
        return {name: param.data.clone() for name, param in model.named_parameters()}
        
    def _check_frozen_models(self):
        """Check if frozen models have been modified"""
        if not hasattr(self, '_frozen_fingerprints'):
            return
            
        for model_name, fingerprint in self._frozen_fingerprints.items():
            model = getattr(self, model_name.upper() if model_name == 'taco' else model_name)
            current_fingerprint = self._get_model_fingerprint(model)
            
            for param_name, stored_param in fingerprint.items():
                current_param = current_fingerprint[param_name]
                assert torch.all(torch.eq(current_param, stored_param)), f"Parameter {param_name} in {model_name} has changed when it should be frozen!"

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
            # Average percentage of reward prediction error
            with torch.no_grad():
                # Average percentage of reward prediction error
                metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred - reward) / (reward + 1e-6)).item() 
                error = reward_pred - reward
                metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
                threshold = 1e-3  # puoi settarlo in base al tuo dominio
                mask = reward.abs() > threshold
                metrics['rel_error_filtered'] = torch.mean(
                    torch.abs(reward_pred[mask] - reward[mask]) / (reward[mask] + 1e-6)
                ).item()
                numerator = torch.abs(reward_pred - reward)
                denominator = torch.abs(reward_pred) + torch.abs(reward) + 1e-6
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
     
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
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
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
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
                reward_loss = F.mse_loss(reward_pred, reward)
                # Average percentage of reward prediction error
                metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred - reward) / (reward + 1e-6)).item() 
                error = reward_pred - reward
                metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
                threshold = 1e-3  # puoi settarlo in base al tuo dominio
                mask = reward.abs() > threshold
                metrics['rel_error_filtered'] = torch.mean(
                    torch.abs(reward_pred[mask] - reward[mask]) / (reward[mask] + 1e-6)
                ).item()
                numerator = torch.abs(reward_pred - reward)
                denominator = torch.abs(reward_pred) + torch.abs(reward) + 1e-6
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
            # Training update
            metrics = taco_agent.evaluate(batch)
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

