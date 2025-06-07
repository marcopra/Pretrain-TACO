import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
if 'MUJOCO_GL' not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import numpy as np
from dm_env import specs
from datetime import datetime

import metaworld_env
import wandb
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

def setup_distributed(rank, world_size, port="12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def calculate_batch_distribution(total_batch_size, num_gpus, max_batch_per_gpu):
    """
    Calculate how to distribute the total batch size across GPUs
    """
    ideal_batch_per_gpu = total_batch_size // num_gpus
    
    if ideal_batch_per_gpu <= max_batch_per_gpu:
        # Can fit the batch directly
        batch_size_per_gpu = ideal_batch_per_gpu
        gradient_accumulation_steps = 1
    else:
        # Need gradient accumulation
        batch_size_per_gpu = max_batch_per_gpu
        gradient_accumulation_steps = ideal_batch_per_gpu // max_batch_per_gpu
        
    effective_batch_size = batch_size_per_gpu * gradient_accumulation_steps * num_gpus
    
    print(f"Batch size configuration:")
    print(f"  Total desired batch size: {total_batch_size}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {batch_size_per_gpu}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective total batch size: {effective_batch_size}")
    
    return batch_size_per_gpu, gradient_accumulation_steps

def make_agent(obs_spec, action_spec, cfg, gradient_accumulation_steps=1):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.gradient_accumulation_steps = gradient_accumulation_steps
    return hydra.utils.instantiate(cfg)

class DistributedWorkspace:
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.work_dir = Path.cwd()
        
        # Calculate batch distribution
        total_batch_size = getattr(cfg, 'total_batch_size', cfg.batch_size * world_size)
        max_batch_per_gpu = getattr(cfg, 'max_batch_size_per_gpu', 256)
        
        self.batch_size_per_gpu, self.gradient_accumulation_steps = calculate_batch_distribution(
            total_batch_size, world_size, max_batch_per_gpu
        )
        
        # Update config with calculated values
        cfg.batch_size = self.batch_size_per_gpu
        cfg.gradient_accumulation_steps = self.gradient_accumulation_steps
        
        if rank == 0:
            print(f'workspace: {self.work_dir}')
            print(f'world_size: {world_size}')

        self.cfg = cfg
        if cfg.seed == 1:
            cfg.seed = np.random.randint(0, 10000)
        utils.set_seed_everywhere(cfg.seed + rank)
        
        self.device = torch.device(f'cuda:{rank}')
        self.setup()

        # Get observation and action specs for the agent
        obs_spec = metaworld_env.observation_spec(self.train_env)
        action_spec = metaworld_env.action_spec(self.train_env)
        
        self.agent = make_agent(obs_spec, action_spec, self.cfg.agent, self.gradient_accumulation_steps)
        
        # Wrap models with DDP
        if hasattr(self.agent, 'encoder'):
            self.agent.encoder = DDP(self.agent.encoder, device_ids=[rank])
        if hasattr(self.agent, 'TACO'):
            self.agent.TACO = DDP(self.agent.TACO, device_ids=[rank])
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.saved_medium_policy = False

        # Only rank 0 handles wandb logging
        if cfg.use_wandb and rank == 0:
            # Use resolved config values with fallbacks
            wandb_id = getattr(cfg, 'wandb_id', None)
            if wandb_id and wandb_id != "none":
                wandb.init(
                    id=wandb_id,
                    resume='must',
                    project=getattr(cfg, 'wandb_project', 'taco_metaworld'),
                    name=getattr(cfg, 'wandb_run_name', f'distributed_run_{cfg.seed}'),
                    tags=getattr(cfg, 'wandb_tag', 'none').split('_') if getattr(cfg, 'wandb_tag', 'none') != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=getattr(cfg, 'wandb_project', 'taco_metaworld'),
                    name=getattr(cfg, 'wandb_run_name', f'distributed_run_{cfg.seed}'),
                    tags=getattr(cfg, 'wandb_tag', 'none').split('_') if getattr(cfg, 'wandb_tag', 'none') != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
            wandb.run.save()

    def setup(self):
        # Create logger (only rank 0)
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb) if self.rank == 0 else None
        
        # Create environments
        self.train_env = metaworld_env.make(
            self.cfg.env_name, 
            self.cfg.task,
            self.cfg.frame_stack,
            self.cfg.action_repeat, 
            self.cfg.seed + self.rank,
            resolution=self.cfg.resolution,
            camera=self.cfg.camera,
            random_init=self.cfg.random_init,
            randomize_goal_and_object_pos=self.cfg.random_goal
        )
        
        # Only rank 0 needs eval env
        if self.rank == 0:
            self.eval_env = metaworld_env.make(
                self.cfg.env_name, 
                self.cfg.task,
                self.cfg.frame_stack,
                self.cfg.action_repeat, 
                self.cfg.seed,
                resolution=self.cfg.resolution,
                camera=self.cfg.camera,
                random_init=self.cfg.random_init,
                randomize_goal_and_object_pos=self.cfg.random_goal
            )
        
        # Create replay buffer specs
        data_specs = (
            metaworld_env.observation_spec(self.train_env),
            metaworld_env.action_spec(self.train_env),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )

        # Create separate replay buffer for each process
        buffer_dir = self.work_dir / f'buffer_rank_{self.rank}'
        self.replay_storage = ReplayBufferStorage(data_specs, buffer_dir)

        # Use calculated batch size per GPU
        self.replay_loader = make_replay_loader(
            buffer_dir, 
            self.cfg.replay_buffer_size // self.world_size,
            self.batch_size_per_gpu,  # Use calculated batch size
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, 
            self.cfg.nstep, 
            self.cfg.multistep, 
            self.cfg.discount
        )
        
        self._replay_iter = None

        # Setup video recorders (only rank 0)
        if self.rank == 0:
            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None,
                metaworld=True
            )
            
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if self.cfg.save_train_video else None,
                metaworld=True
            )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def train(self):
        # Similar to original train method but with distributed considerations
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        
        if self.rank == 0:
            self.train_video_recorder.init(time_step.observation)
        
        metrics = None
        
        while train_until_step(self.global_step):
            # Synchronize all processes
            dist.barrier()
            
            if time_step.last():
                self._global_episode += 1
                if self.rank == 0:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                
                # Log metrics (only rank 0)
                if metrics is not None and self.rank == 0:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    
                    if self.logger is not None:
                        with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)
                    
                    if self.cfg.use_wandb:
                        wandb.log({
                            'fps': episode_frame / elapsed_time,
                            'total_time': total_time,
                            'episode_reward': episode_reward,
                            'episode_length': episode_frame,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'global_frame': self.global_frame,
                            'step': self.global_step
                        })

                # Reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                if self.rank == 0:
                    self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # Try to evaluate (only rank 0)
            if eval_every_step(self.global_step) and self.rank == 0:
                if self.logger is not None:
                    self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # Sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)

            # Try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                
                # Only rank 0 logs
                if self.rank == 0 and self.logger is not None and metrics:
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    
                    if self.cfg.use_wandb:
                        metrics.update({
                            'global_step': self.global_step,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'global_frame': self.global_frame
                        })
                        wandb.log(metrics)

            # Take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            
            if self.rank == 0:
                self.train_video_recorder.record(time_step.observation)
            
            episode_step += 1
            self._global_step += 1

    def eval(self):
        # ...existing code from original eval method...
        step, episode, total_reward, success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=True)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                if time_step.last():
                    success += time_step.success * 1
                    break
            episode += 1
            self.video_recorder.save(f'{self.global_frame}-{episode}.mp4')

        if self.logger is not None:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('success_rate', success / episode)

        if self.cfg.use_wandb:
            wandb.log({
                'eval/episode_reward': total_reward / episode,
                'eval/episode_length': step * self.cfg.action_repeat / episode,
                'eval/episode': self.global_episode,
                'eval/step': self.global_step,
                'eval/success_rate': success / episode,
                'buffer_size': len(self.replay_storage),
                'global_frame': self.global_frame
            })

def resolve_time_interpolations(cfg):
    """Resolve time-based interpolations in the config before multiprocessing"""
    current_time = datetime.now()
    
    # Create a resolved copy of the config
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    
    # Manually resolve time-based fields
    if hasattr(resolved_cfg, 'wandb_run_name'):
        wandb_run_name = str(resolved_cfg.wandb_run_name)
        # Replace ${now:%Y.%m.%d} and ${now:%H%M} patterns
        wandb_run_name = wandb_run_name.replace('${now:%Y.%m.%d}', current_time.strftime('%Y.%m.%d'))
        wandb_run_name = wandb_run_name.replace('${now:%H%M}', current_time.strftime('%H%M'))
        resolved_cfg.wandb_run_name = wandb_run_name
    
    # Resolve hydra sweep dir if present
    if hasattr(resolved_cfg, 'hydra') and hasattr(resolved_cfg.hydra, 'sweep') and hasattr(resolved_cfg.hydra.sweep, 'dir'):
        sweep_dir = str(resolved_cfg.hydra.sweep.dir)
        sweep_dir = sweep_dir.replace('${now:%Y.%m.%d}', current_time.strftime('%Y.%m.%d'))
        sweep_dir = sweep_dir.replace('${now:%H%M}', current_time.strftime('%H%M'))
        resolved_cfg.hydra.sweep.dir = sweep_dir
    
    # Resolve submitit_folder if present
    if (hasattr(resolved_cfg, 'hydra') and hasattr(resolved_cfg.hydra, 'launcher') and 
        hasattr(resolved_cfg.hydra.launcher, 'submitit_folder')):
        submitit_folder = str(resolved_cfg.hydra.launcher.submitit_folder)
        submitit_folder = submitit_folder.replace('${now:%Y.%m.%d}', current_time.strftime('%Y.%m.%d'))
        submitit_folder = submitit_folder.replace('${now:%H%M%S}', current_time.strftime('%H%M%S'))
        resolved_cfg.hydra.launcher.submitit_folder = submitit_folder
    
    return resolved_cfg

def train_distributed(rank, world_size, cfg):
    """Training function for each process"""
    setup_distributed(rank, world_size)
    
    workspace = DistributedWorkspace(cfg, rank, world_size)
    
    # Load snapshot if exists (only rank 0)
    if rank == 0:
        snapshot = workspace.work_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            workspace.load_snapshot()
    
    workspace.train()
    cleanup_distributed()

@hydra.main(config_path='cfgs', config_name='config_metaworld')  
def main(cfg):
    # Resolve time interpolations before multiprocessing
    resolved_cfg = resolve_time_interpolations(cfg)
    
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Warning: Less than 2 GPUs available. Using single GPU training.")
        # Fall back to single GPU training from train_metaworld_big.py
        from train_metaworld import Workspace
        workspace = Workspace(resolved_cfg)
        workspace.train()
    else:
        print(f"Starting distributed training on {world_size} GPUs")
        mp.spawn(train_distributed, args=(world_size, resolved_cfg), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
