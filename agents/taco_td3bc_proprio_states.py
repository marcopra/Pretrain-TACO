import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from agents.taco_proprio_states import TACOAgent

import utils
from dm_control.utils import rewards


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)

        return q1, q2


class TD3BCAgent(TACOAgent):
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 encoder_lr,
                 repr_dim,
                 encoder_hidden_dim,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 use_tb,
                 reward,
                 multistep,
                 latent_a_dim,
                 alpha,
                 pretrained_path=None,
                 freeze_encoder=False,
                 no_taco=False, 
                 optimizer_type="adam",
                 has_next_action=False):
       
        super().__init__( obs_shape, action_shape, device, lr, encoder_lr, repr_dim, encoder_hidden_dim, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reward, multistep, latent_a_dim, pretrained_path, freeze_encoder, no_taco, optimizer_type)
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.alpha = alpha
 

    # def update_critic(self, obs, action, reward, discount, next_obs, step):
    #     metrics = dict()

    #     with torch.no_grad():
    #         stddev = utils.schedule(self.stddev_schedule, step)
    #         dist = self.actor(next_obs, stddev)
    #         next_action = dist.sample(clip=self.stddev_clip)
    #         target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    #         target_V = torch.min(target_Q1, target_Q2)
    #         target_Q = reward + (discount * target_V)

    #     Q1, Q2 = self.critic(obs, action)
    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     if self.use_tb:
    #         metrics['critic_target_q'] = target_Q.mean().item()
    #         metrics['critic_q1'] = Q1.mean().item()
    #         metrics['critic_q2'] = Q2.mean().item()
    #         metrics['critic_loss'] = critic_loss.item()

    #     # optimize critic
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.critic_opt.step()
    #     return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)

        Q1, Q2 = self.critic(obs, policy.sample(clip=self.stddev_clip), self.act_tok)
        Q = torch.min(Q1, Q2)

        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + F.mse_loss(policy.mean, action)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs_en = obs.float()
        next_obs_en = next_obs.float()
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
        metrics.update(self.update_actor(obs_en.detach(), action, step))

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
