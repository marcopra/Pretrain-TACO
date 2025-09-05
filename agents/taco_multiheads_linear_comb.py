import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools
import re
import warnings


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

    def __init__(self, repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, act_tok, encoder, multistep, device, num_tasks=1):
        super(TACO, self).__init__()

        self.multistep = multistep
        self.encoder = encoder
        self.device = device
        self.num_tasks = num_tasks
        
        a_dim = action_shape[0]

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.act_tok = act_tok
        
        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        # Create multiple reward networks, one for each task
        self.reward_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim+latent_a_dim*multistep, hidden_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_tasks)
        ])
        
        # Linear combination network for pretrained reward networks (simplified)
        self.reward_combiner = None
        self.use_reward_combiner = False
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(utils.weight_init)
    
    def add_reward_combiner(self):
        """Add a simple linear combination layer for existing reward networks"""
        self.reward_combiner = nn.Parameter(torch.ones(self.num_tasks) / self.num_tasks)
        self.use_reward_combiner = True
        print("Added simple linear combination weights for frozen pretrained reward networks")
    
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
    
    def predict_reward(self, features, task_ids, mode='train'):
        """
        Predict rewards using task-specific networks or reward combiner.
        
        Args:
            features: Input features (batch_size, feature_dim)
            task_ids: Task IDs for each sample (batch_size,)
            mode: 'train' for using specific task networks, 'combiner' for learned linear combination, 'test' for averaging
        
        Returns:
            reward_pred: Predicted rewards (batch_size, 1)
        """
        
        if mode == 'combiner':
            assert self.use_reward_combiner, "Reward combiner not set up. Use add_reward_combiner() to initialize."
            # Use learned linear combination of frozen reward networks
            combination_weights = F.softmax(self.reward_combiner, dim=0).to(self.device)  # Ensure weights sum to 1
            
            # Get predictions from all frozen reward networks
            all_predictions = []
            for task_network in self.reward_networks:
                pred = task_network(features)  # (batch_size, 1)
                all_predictions.append(pred)
            
            # Stack predictions: (batch_size, num_tasks)
            stacked_predictions = torch.stack(all_predictions, dim=-1).squeeze(-1).to(self.device)
            
            # Apply learned combination weights: (batch_size,)
            reward_pred = torch.matmul(stacked_predictions, combination_weights)
            
            return reward_pred # (batch_size, 1)
            
        elif mode == 'train':
            assert not self.use_reward_combiner, "Reward combiner is not used in training mode. Use 'combiner' mode for pretrained models."
            # Use task-specific networks
            batch_size = features.shape[0]
            reward_pred = torch.zeros(batch_size, 1, device=self.device)
            
            for task_id in range(self.num_tasks):
                mask = (task_ids == task_id)
                if mask.any():
                    task_features = features[mask]
                    task_rewards = self.reward_networks[task_id](task_features)
                    reward_pred[mask] = task_rewards
                    
            return reward_pred
        elif mode == 'test':
            # Average predictions from all task networks for test evaluation
            all_predictions = []
            for task_network in self.reward_networks:
                pred = task_network(features)
                all_predictions.append(pred)
            
            # Average all predictions
            reward_pred = torch.stack(all_predictions, dim=0).mean(dim=0)
            return reward_pred
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'train', 'combiner', or 'test'.")

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, latent_a_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, act_tok=None):
        if act_tok is not None:
            action = act_tok(action)
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    

class TACOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, encoder_lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reward, multistep, latent_a_dim, curl, pretrained_path=None, freeze_encoder=False, no_taco=False, optimizer_type="adam", num_tasks=1):
    
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        self.reward = reward
        self.multistep = multistep
        self.curl = curl
        self.optimizer_type = optimizer_type.lower()
        
        # Warning for deprecated freeze_encoder parameter
        if freeze_encoder:
            warnings.warn(
                "\033[93mWarning: freeze_encoder parameter is deprecated and no longer available. "
                "When loading a pretrained model, reward networks will be automatically frozen and "
                "a linear combination will be used.\033[0m",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Extract num_tasks from pretrained path if provided 
        # because we don't know how many tasks we had in the pretraining session and we have to extract them from the model path.
        if pretrained_path is not None and pretrained_path.lower() != 'none':
            self.num_tasks = self._extract_num_tasks_from_path(pretrained_path)
            print(f"Extracted {self.num_tasks} tasks from pretrained path: {pretrained_path}")
            # Track if we're using pretrained model
            self.using_pretrained = True
        else:
            self.using_pretrained = False
            # Default to 1 task if no pretrained path is provided 
            self.num_tasks = num_tasks

        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0]*1.25)+1
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep)
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, latent_a_dim, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, latent_a_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.TACO = TACO(self.encoder.repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, multistep, device, self.num_tasks).to(device)
        # No Taco: it means to not use TACO losses but encoders are still trained with q losses.
        self.no_taco = no_taco 
        
        # Store fingerprints of pretrained parameters for verification TODO: remove
        self._pretrained_fingerprints = {}
        # Store fingerprints of pretrained reward networks only for verification
        self._pretrained_reward_fingerprints = {}
        
        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )
        
        # Selezione dell'optimizer
        if self.optimizer_type == "adam":
            optimizer_class = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer type '{self.optimizer_type}' not supported. Use 'adam' or 'sgd'.")
        
        self.encoder_opt = optimizer_class(parameters, lr=encoder_lr)
        if not self.no_taco:
            self.taco_opt = optimizer_class(self.TACO.parameters(), lr=encoder_lr)
        self.actor_opt = optimizer_class(self.actor.parameters(), lr=lr)
        self.critic_opt = optimizer_class(self.critic.parameters(), lr=lr)
        
        # Initialize reward combiner optimizer (will be set up after loading pretrained model)
        self.reward_combiner_opt = None
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        if pretrained_path is None or pretrained_path.lower() == 'none':
            print("No pretrained model provided, initializing from scratch.")
        else:
            print(f"Loading pretrained model from {pretrained_path}")
            self.load_pretrained(pretrained_path, None)
            
            # Add reward combiner and its optimizer if we loaded a pretrained model
            if self.reward and self.TACO.use_reward_combiner:
                self.reward_combiner_opt = optimizer_class([self.TACO.reward_combiner], lr=encoder_lr)
            
        self.pretrained_path = pretrained_path
        self.train()
        self.critic_target.train()

    def _extract_num_tasks_from_path(self, path):
        """Extract number of tasks from pretrained model path (e.g., MT49 -> 49)"""
        import os
        filename = os.path.basename(path)
        
        # Look for pattern MT followed by digits
        match = re.search(r'MT(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not extract number of tasks from path {path}, using default value 1")
            return 1

    def _get_model_fingerprint(self, model, prefix=""):
        """Generate a fingerprint for all model parameters"""
        fingerprint = {}
        for name, param in model.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            fingerprint[full_name] = param.data.clone()
        return fingerprint
    
    def _store_pretrained_fingerprints(self):
        """Store fingerprints of all pretrained model components"""
        self._pretrained_fingerprints = {}
        
        # Store encoder fingerprint
        encoder_fp = self._get_model_fingerprint(self.encoder, "encoder")
        self._pretrained_fingerprints.update(encoder_fp)
        
        # Store act_tok fingerprint  
        act_tok_fp = self._get_model_fingerprint(self.act_tok, "act_tok")
        self._pretrained_fingerprints.update(act_tok_fp)
        
        # Store TACO fingerprint (excluding reward_combiner)
        for name, param in self.TACO.named_parameters():
            if 'reward_combiner' not in name:
                full_name = f"taco.{name}"
                self._pretrained_fingerprints[full_name] = param.data.clone()
    
    def _store_pretrained_reward_fingerprints(self):
        """Store fingerprints of only the reward networks parameters"""
        self._pretrained_reward_fingerprints = {}
        
        # Store only reward networks fingerprints
        for i, reward_network in enumerate(self.TACO.reward_networks):
            for name, param in reward_network.named_parameters():
                full_name = f"reward_networks.{i}.{name}"
                self._pretrained_reward_fingerprints[full_name] = param.data.clone()

    def _verify_pretrained_parameters_frozen(self):
        """Verify that all pretrained parameters remain unchanged"""
        if not self.using_pretrained or not hasattr(self, '_pretrained_fingerprints'):
            return
        
        current_fingerprints = {}
        
        # Get current encoder fingerprint
        encoder_fp = self._get_model_fingerprint(self.encoder, "encoder")
        current_fingerprints.update(encoder_fp)
        
        # Get current act_tok fingerprint
        act_tok_fp = self._get_model_fingerprint(self.act_tok, "act_tok")
        current_fingerprints.update(act_tok_fp)
        
        # Get current TACO fingerprint (excluding reward_combiner)
        for name, param in self.TACO.named_parameters():
            if 'reward_combiner' not in name:
                full_name = f"taco.{name}"
                current_fingerprints[full_name] = param.data.clone()
        
        # Compare with stored fingerprints
        for param_name, stored_param in self._pretrained_fingerprints.items():
            if param_name in current_fingerprints:
                current_param = current_fingerprints[param_name]
                if not torch.allclose(current_param, stored_param, atol=1e-8):
                    raise AssertionError(f"Pretrained parameter '{param_name}' has changed when it should be frozen!")
            else:
                raise AssertionError(f"Pretrained parameter '{param_name}' not found in current model!")

    def _verify_pretrained_reward_parameters_frozen(self):
        """Verify that only the pretrained reward network parameters remain unchanged"""
        if not self.using_pretrained or not hasattr(self, '_pretrained_reward_fingerprints'):
            return
        
        # Get current reward networks fingerprints
        current_fingerprints = {}
        for i, reward_network in enumerate(self.TACO.reward_networks):
            for name, param in reward_network.named_parameters():
                full_name = f"reward_networks.{i}.{name}"
                current_fingerprints[full_name] = param.data.clone()
        
        # Compare with stored fingerprints
        for param_name, stored_param in self._pretrained_reward_fingerprints.items():
            if param_name in current_fingerprints:
                current_param = current_fingerprints[param_name]
                if not torch.allclose(current_param, stored_param, atol=1e-8):
                    raise AssertionError(f"Pretrained reward network parameter '{param_name}' has changed when it should be frozen!")
            else:
                raise AssertionError(f"Pretrained reward network parameter '{param_name}' not found in current model!")

    def load_pretrained(self, model_path, map_location=None):
        """
        Load a pretrained TACO model from a saved checkpoint.
        Automatically freezes reward networks and sets up linear combination.
        
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
        
        # Set using_pretrained flag
        self.using_pretrained = True
        
        # Add reward combiner for learned combination of frozen reward networks
        if self.reward:
            self.TACO.add_reward_combiner()
            
            # Freeze only the reward networks
            for reward_network in self.TACO.reward_networks:
                for param in reward_network.parameters():
                    param.requires_grad = False
                reward_network.eval()
            
            print("Frozen pretrained reward networks and added linear combination weights")
        
        # Store fingerprints of only the reward networks for verification
        self._store_pretrained_reward_fingerprints()
        
        print(f"Loaded pretrained model from {model_path}")
        
        return checkpoint.get('args', {})  # Return the saved args for reference
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        if not self.using_pretrained:
            self.TACO.train(training)
        else:
            # Keep reward networks frozen but allow other parts to train
            for name, module in self.TACO.named_children():
                if name == 'reward_networks':
                    for reward_network in module:
                        reward_network.eval()
                else:
                    module.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, self.act_tok)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, self.act_tok)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, self.act_tok)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update_taco(self, obs, action, action_seq, next_obs, reward, task_id=None):
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
        reward_loss = torch.tensor(0., device=self.device)
        if self.reward:
            features = torch.concat([z_a, action_seq_en], dim=-1)
            
            # Use reward combiner if available (for pretrained models)
            if self.using_pretrained and self.TACO.use_reward_combiner:
                reward_pred = self.TACO.predict_reward(features, None, mode='combiner')
                reward_loss = F.mse_loss(reward_pred, reward)
            else:
                raise ValueError("This script has been only tested with TACO with reward combiner with $$$pretrained models$$$, please use the TACO or TACO multihead script to train TACO without reward combiner.")
                reward_pred = self.TACO.predict_reward(features, task_id.squeeze(), mode='train')
                reward_loss = F.mse_loss(reward_pred, reward)
        
        ### Compute TACO loss
        next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
        curr_za = self.TACO.project_sa(z_a, action_seq_en) 
        logits = self.TACO.compute_logits(curr_za, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        taco_loss = self.cross_entropy_loss(logits, labels)
        
        if self.using_pretrained and self.TACO.use_reward_combiner:
            # Only update reward combiner when using pretrained model
            if self.reward_combiner_opt is not None and reward_loss.requires_grad:
                self.reward_combiner_opt.zero_grad()
                reward_loss.backward(retain_graph=True)
                self.reward_combiner_opt.step()
            
            # Update other TACO components (excluding frozen reward networks)
            if not self.no_taco:
                self.taco_opt.zero_grad()
                (taco_loss + curl_loss).backward()
                self.taco_opt.step()
        else:
            # Normal training without pretrained model
            if not self.no_taco:
                self.taco_opt.zero_grad()
                (taco_loss + curl_loss + reward_loss).backward()
                self.taco_opt.step()
        
        # Verify that only the reward network parameters remain frozen
        self._verify_pretrained_reward_parameters_frozen()
                
        if self.use_tb:
            metrics['reward_loss']  = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['taco_loss']  = taco_loss.item()
            metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
        return metrics
        
        
    
    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        if self.TACO.use_reward_combiner:
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                batch, self.device)
            task_id = None
        else:
            raise ValueError("this script has been only tested with TACO with reward combiner, please use the TACO or TACO multohead script to train TACO without reward combiner.")
            obs, action, action_seq, reward, discount, next_obs, r_next_obs, task_id = utils.to_torch(
                batch, self.device)

        # augment
        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())
        # encode
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs_en)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step))

        # update actor
        metrics.update(self.update_actor(obs_en.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        if self.no_taco:
            metrics['reward_loss']  = torch.tensor(0.)
            metrics['curl_loss'] = torch.tensor(0.)
            metrics['taco_loss']  = torch.tensor(0.)
            return metrics
            
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward, task_id))       

        return metrics
    
    def save(self, save_path, step):
        """
        Save the model state to a file.
        
        Args:
            save_path: Path to save the model state
            step: Current training step (for naming purposes)
        """
        state = {
            'encoder': self.encoder.state_dict(),
            'taco': self.TACO.state_dict(),
            'act_tok': self.act_tok.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'args': {
                'step': step,
                'freeze_encoder': self.freeze_encoder,
                'no_taco': self.no_taco
            }
        }
        torch.save(state, save_path)
        print(f"Model saved to {save_path}")
        
    # def evaluate_taco(self, obs, action, action_seq, next_obs, reward, task_id=None, use_test_mode=False):
    #     """
    #     Evaluate TACO with different reward prediction modes.
        
    #     Args:
    #         obs, action, action_seq, next_obs, reward: Standard inputs
    #         task_id: Task IDs for each sample
    #         use_test_mode: If True, use averaged reward predictions (for test set evaluation)
    #                       If False, use task-specific reward predictions (for validation split)
    #     """
    #     with torch.no_grad():
    #         metrics = dict()
    #         metrics['batch_reward'] = reward.mean().item()

    #         obs_anchor = self.aug(obs.float())
    #         obs_pos = self.aug(obs.float())
    #         z_a = self.TACO.encode(obs_anchor)
    #         z_pos = self.TACO.encode(obs_pos, ema=True)
    #         ### Compute CURL loss
    #         if self.curl:
    #             logits = self.TACO.compute_logits(z_a, z_pos)
    #             labels = torch.arange(logits.shape[0]).long().to(self.device)
    #             curl_loss = self.cross_entropy_loss(logits, labels)
    #             metrics['curl_loss'] = curl_loss.item()
    #         else:
    #             curl_loss = torch.tensor(0.)
            
    #         ### Compute action encodings
    #         action_en = self.TACO.act_tok(action, seq=False) 
    #         action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
    #         ### Compute reward prediction loss
    #         if self.reward:
    #             features = torch.concat([z_a, action_seq_en], dim=-1)
                
    #             # Determine validation mode based on use_test_mode flag and model type
    #             if use_test_mode:
    #                 mode = 'test'
    #             elif self.using_pretrained and self.TACO.use_reward_combiner:
    #                 mode = 'combiner'
    #             else:
    #                 mode = 'train'

    #             reward_pred = self.TACO.predict_reward(features, task_id.squeeze(), mode=mode)
    #             reward_loss = F.mse_loss(reward_pred, reward)

    #             # Average percentage of reward prediction error
    #             metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred - reward) / (reward + 1e-6)).item() 
    #             error = reward_pred - reward
    #             metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
    #             threshold = 1e-3  # puoi settarlo in base al tuo dominio
    #             mask = reward.abs() > threshold
    #             if mask.any():
    #                 metrics['rel_error_filtered'] = torch.mean(
    #                     torch.abs(reward_pred[mask] - reward[mask]) / (reward[mask] + 1e-6)
    #                 ).item()
    #             else:
    #                 metrics['rel_error_filtered'] = 0.0
    #             numerator = torch.abs(reward_pred - reward)
    #             denominator = torch.abs(reward_pred) + torch.abs(reward) + 1e-6
    #             metrics['smape'] = torch.mean(2.0 * numerator / denominator).item()

    #         else:
    #             reward_loss = torch.tensor(0.)
            
    #         ### Compute TACO loss
    #         next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
    #         curr_za = self.TACO.project_sa(z_a, action_seq_en) 
    #         logits = self.TACO.compute_logits(curr_za, next_z)
    #         labels = torch.arange(logits.shape[0]).long().to(self.device)

    #         taco_loss = self.cross_entropy_loss(logits, labels)
            
    #         if self.use_tb:
    #             metrics['reward_loss']  = reward_loss.item()
    #             metrics['curl_loss'] = curl_loss.item()
    #             metrics['taco_loss']  = taco_loss.item()
    #             metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
    #         return metrics
