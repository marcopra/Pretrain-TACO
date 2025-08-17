import hydra
import utils
import torch
import mvp
from r3m import load_r3m
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from agents.resnet_models import resnet_conv3_compressed, resnet_conv4_compressed, resnet_conv5
from agents.moco_models import moco_conv5, moco_conv3_compressed, moco_conv4_compressed
import re
import os


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
    def __init__(self, obs_shape, feature_dim, pretrained_path=None, device="cuda"):
        super().__init__()
        assert len(obs_shape) == 3
        # obs_shape is (N*C, H, W) where N is number of stacked frames, C=3 for RGB
        total_channels = obs_shape[0]  # N*C
        self.height = obs_shape[1]     # H  
        self.width = obs_shape[2]      # W
        
        # Assuming RGB images (C=3), calculate number of stacked frames
        self.channels = 3  # RGB
        self.num_stack = total_channels // self.channels
        
        self.range = None
        assert total_channels % self.channels == 0, f"Total channels {total_channels} not divisible by {self.channels}"
        
        # Parse pretrained_path to determine model configuration
        self.resnet, self.normalize, self.resize = self._create_resnet(pretrained_path)
        self.resnet = self.resnet
        
        # Move resnet to device before calculating dimensions
        self.resnet = self.resnet.to(device)
        
        # Calculate representation dimension based on the model architecture
        self.repr_dim = self._calculate_repr_dim(device) * self.num_stack
    
    def _create_resnet(self, pretrained_path):
        """Create ResNet model based on pretrained_path configuration"""
        normalize = None
        resize = None
        
        print(os.path.exists(pretrained_path), 'moco' in pretrained_path, "aoaooa")
        if pretrained_path is None or pretrained_path.lower() == 'none':
            # Default: ResNet18 without pretrained weights
            resnet = models.resnet18(weights=None)
            resnet = self._modify_resnet_for_input_size(resnet)
            resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
            
        elif os.path.exists(pretrained_path) and 'moco' in pretrained_path:
            print(f"Loading MoCo model from {pretrained_path}")
            if 'l3' in pretrained_path:
                resnet = moco_conv3_compressed(pretrained_path)
            elif 'l4' in pretrained_path:
                resnet = moco_conv4_compressed(pretrained_path)
            else:
                resnet = moco_conv5(pretrained_path)
             # Apply standard ResNet transforms for pretrained checkpoints
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # Apply resize and center crop as per ResNet standard
            resize = transforms.Compose([
                transforms.Resize(224),
            ])
            print(f"MoCo model loaded") 
        elif 'mvp' in pretrained_path:
            resnet = mvp.load("vits-mae-hoi")
             # Apply standard ResNet transforms for pretrained checkpoints
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # Apply resize and center crop as per ResNet standard
            resize = transforms.Compose([
                transforms.Resize(224),
            ])
        elif 'r3m' in pretrained_path:
            resnet = load_r3m("resnet50")
             # Apply standard ResNet transforms for pretrained checkpoints
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # Apply resize and center crop as per ResNet standard
            resize = transforms.Compose([
                transforms.Resize(224),
            ])

        elif os.path.exists(pretrained_path) or  'resnet50_l5' in pretrained_path:
            print(f"Loading ResNet model from {pretrained_path}")
            # Load from checkpoint file - these are pretrained models that need standard transforms
            if 'resnet50_l3' in pretrained_path:
                resnet = resnet_conv3_compressed(pretrained_path)
            elif 'resnet50_l4' in pretrained_path:
                resnet = resnet_conv4_compressed(pretrained_path)
            elif 'resnet50_l5' in pretrained_path:
                resnet = resnet_conv5(pretrained_path)
            else:
                raise ValueError(f"Unknown checkpoint format: {pretrained_path}")
            print(f"ResNet model loaded: {resnet}")
            # Apply standard ResNet transforms for pretrained checkpoints
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # Apply resize and center crop as per ResNet standard
            resize = transforms.Compose([
                transforms.Resize(224),
            ])
                
        else:
            print(f"Instantiating ResNet based on pretrained_path: {pretrained_path}")
            # Parse format: resnet<k>_l<n>_<initialization>
            match = re.match(r'resnet(\d+)_l(\d+)_(\w+)', pretrained_path)
            if not match:
                raise ValueError(f"Invalid pretrained_path format: {pretrained_path}. Expected format: resnet<k>_l<n>_<initialization>")
            
            k, n, initialization = match.groups()
            k, n = int(k), int(n)
            
            # Create base ResNet
            if k == 18:
                if initialization == 'pretrained':
                    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                    # Setup ImageNet normalization and resize
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    resize = transforms.Compose([
                        transforms.Resize(224), # USE DIRECTLY 224
                        # transforms.CenterCrop(224)
                    ])
                else:
                    resnet = models.resnet18(weights=None)
                    resnet = self._modify_resnet_for_input_size(resnet)
            elif k == 50:
                if initialization == 'pretrained':
                    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                    # Setup ImageNet normalization and resize
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    resize = transforms.Compose([
                        transforms.Resize(224),
                        # transforms.CenterCrop(224)
                    ])
                else:
                    resnet = models.resnet50(weights=None)
                    resnet = self._modify_resnet_for_input_size(resnet)
            else:
                raise ValueError(f"Unsupported ResNet variant: ResNet{k}")
            
            # Apply layer cutting based on n
            resnet = self._cut_resnet_at_layer(resnet, n)
            print(f"ResNet model created: {resnet} with layer cut at l{n} and initialization {initialization}")
        
        self.normalize = normalize
        self.resize = resize
        
        return resnet, normalize, resize
    
    def _modify_resnet_for_input_size(self, resnet):
        """Modify ResNet for non-224x224 input sizes"""
        if self.height != 224 or self.width != 224:
            kernel_size = min(7, self.height // 4, self.width // 4)
            stride = max(1, min(2, self.height // 112, self.width // 112))
            
            resnet.conv1 = nn.Conv2d(
                3, 64, kernel_size=kernel_size, stride=stride, 
                padding=kernel_size//2, bias=False
            )
        return resnet
    
    def _cut_resnet_at_layer(self, resnet, layer_num):
        """Cut ResNet at specified layer"""
        children = list(resnet.children())
        
        if layer_num == 5:
            # Remove only fc layer
            return nn.Sequential(*children[:-1])
        elif layer_num == 4:
            # Remove fc and avgpool
            return nn.Sequential(*children[:-2])
        elif layer_num == 3:
            # Remove fc, avgpool, and layer4
            return nn.Sequential(*children[:-3])
        else:
            raise ValueError(f"Unsupported layer cut: l{layer_num}")
    
    def _calculate_repr_dim(self, device="cuda"):
        """Calculate representation dimension based on model architecture"""
        # Determine the actual device of the model
        if hasattr(self.resnet, 'module'):
            # For DataParallel models (like r3m)
            model_device = next(self.resnet.module.parameters()).device
        else:
            # For regular models
            model_device = next(self.resnet.parameters()).device
        
        # Test with a dummy input to get output dimensions
        dummy_input = torch.randn(1, 3, self.height, self.width, device=model_device)
        
        # Apply transforms in the same order as forward pass
        if self.resize is not None:
            dummy_input = self.resize(dummy_input)
        if self.normalize is not None:
            dummy_input = self.normalize(dummy_input)
            
        with torch.no_grad():
            output = self.resnet(dummy_input)
            return output.view(output.size(0), -1).size(1)
    
    def forward(self, obs):
        # obs shape: (batch_size, N*C, H, W) = (batch_size, 9, 84, 84)
        batch_size = obs.shape[0]
        if self.range is None:
            assert obs.max() > 1, "Input observations should be in [0, 255] range"
            self.range = True
        if self.range is True:
            obs = obs/255.0
        
        # Reshape corretto per preservare la sequenzialità
        # Da (batch_size, 9, 84, 84) a (batch_size, 3, 3, 84, 84)
        obs_reshaped = obs.view(batch_size, self.num_stack, self.channels, self.height, self.width)
        
        # Flatten per processare ogni immagine separatamente: (batch_size * 3, 3, 84, 84)
        obs_flat = obs_reshaped.view(batch_size * self.num_stack, self.channels, self.height, self.width)

        # Apply preprocessing in the correct order: resize first, then normalize
        if self.resize is not None:
            obs_flat = self.resize(obs_flat)
        
        if self.normalize is not None:
            obs_normalized = self.normalize(obs_flat)
        else:
            obs_normalized = obs_flat
        
        # Extract features using ResNet
        features = self.resnet(obs_normalized)
        
        # Flatten features
        features = features.view(batch_size * self.num_stack, -1)
        
        # Reshape back to separate each frame's features
        features_per_frame = features.view(batch_size, self.num_stack, -1)
        
        # Concatenate features from all stacked images
        features_concat = features_per_frame.view(batch_size, -1)
        
        return features_concat
    
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
                 reward, multistep, latent_a_dim, curl, pretrained_path=None, 
                 freeze_encoder=False, no_taco=False):
    
    
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
        self.freeze_encoder = freeze_encoder
        self.no_taco = no_taco

        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0]*1.25)+1
        
        ### Create action embeddings - use Identity if freezing encoder
        if freeze_encoder:
            self.act_tok = nn.Identity()
            # When using Identity, latent_a_dim should match action_shape[0]
            latent_a_dim = action_shape[0]
        else:
            self.act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep)
        
        self.encoder = Encoder(obs_shape, feature_dim, pretrained_path).to(device)
        
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, latent_a_dim, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, latent_a_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.TACO = TACO(self.encoder.repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, multistep, device).to(device)
        
        ### State & Action Encoders - exclude from optimization if frozen
        if freeze_encoder:
            
            # Freeze encoder and TACO parameters
            if 'mvp' in pretrained_path:
                self.encoder.resnet.freeze()
            elif 'r3m' in pretrained_path:
                self.encoder.resnet.eval()
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.TACO.parameters():
                    param.requires_grad = False
                if hasattr(self.act_tok, 'parameters'):
                    for param in self.act_tok.parameters():
                        param.requires_grad = False
            
            # Only optimize non-frozen parameters
            parameters = []
            self.encoder_opt = None
            self.taco_opt = None

        else:
            parameters = itertools.chain(self.encoder.parameters(),
                                         self.act_tok.parameters(),
            )
            self.encoder_opt = torch.optim.Adam(parameters, lr=encoder_lr)
            self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=encoder_lr)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        if pretrained_path is not None:
            print(f"Using ResNet configuration: {pretrained_path}")
        else:
            print("Using default ResNet18 without pretrained weights")
        
        if freeze_encoder:
            print("Encoder is frozen - no updates will be performed on encoder, TACO, and action tokenizer")
            # Set frozen models to eval mode
            self.encoder.eval()
            self.TACO.eval()
            if hasattr(self.act_tok, 'eval'):
                self.act_tok.eval()
        
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if not self.freeze_encoder:
            self.encoder.train(training)
            self.TACO.train(training)
            if hasattr(self.act_tok, 'train'):
                self.act_tok.train(training)

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

        # optimize encoder and critic - only if encoder not frozen
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if not self.freeze_encoder:
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
        if isinstance(self.TACO.act_tok, nn.Identity):
            # When using Identity, pass actions directly
            action_en = action
            action_seq_en = action_seq.view(action_seq.size(0), -1)  # Flatten multistep actions
        else:
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
        
        # Only update if not frozen and optimizer exists
        if not self.freeze_encoder and self.taco_opt is not None:
            self.taco_opt.zero_grad()
            (taco_loss + curl_loss + reward_loss).backward()
            self.taco_opt.step()
        
        if self.use_tb:
            metrics['reward_loss']  = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['taco_loss']  = taco_loss.item()
        
        return metrics
        
    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
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
        
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward))       

        return metrics
    
    def evaluate_taco(self, obs, action, action_seq, next_obs, reward):
        with torch.no_grad():
            metrics = dict()
            metrics['batch_reward'] = reward.mean().item()

            obs_anchor = self.aug(obs.float())
            obs_pos = self.aug(obs.float())
            z_a = self.TACO.encode(obs_anchor)
            z_pos = self.TACO.encode(obs_pos, ema=True)
            ### Compute CURL loss
            if self.curl:
                logits = self.TACO.compute_logits(z_a, z_pos)
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                curl_loss = self.cross_entropy_loss(logits, labels)
                print(f"curl_loss: {curl_loss.item()}")
            else:
                curl_loss = torch.tensor(0.)
            
            ### Compute action encodings
            if isinstance(self.TACO.act_tok, nn.Identity):
                # When using Identity, pass actions directly
                action_en = action
                action_seq_en = action_seq.view(action_seq.size(0), -1)  # Flatten multistep actions
            else:
                action_en = self.TACO.act_tok(action, seq=False) 
                action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
            ### Compute reward prediction loss
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
            
            ### Compute TACO loss
            next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
            curr_za = self.TACO.project_sa(z_a, action_seq_en) 
            logits = self.TACO.compute_logits(curr_za, next_z)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            taco_loss = self.cross_entropy_loss(logits, labels)
            
            if self.use_tb:
                metrics['reward_loss']  = reward_loss.item()
                metrics['curl_loss'] = curl_loss.item()
                metrics['taco_loss']  = taco_loss.item()
            return metrics

