import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
if 'MUJOCO_GL' not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
print(os.environ["MUJOCO_GL"])
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dm_env import specs

import metaworld_env
import wandb
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def aggregate_metrics_across_gpus(metrics, world_size):
    """Aggregate metrics across all GPUs"""
    if world_size == 1:
        return metrics
    
    aggregated = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, torch.Tensor)):
            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(float(value), device=torch.cuda.current_device())
            elif value.device.type != 'cuda':
                value = value.cuda()
            
            # Aggregate across GPUs (sum for counters, mean for rates/losses)
            if 'step' in key or 'episode' in key or 'frame' in key or 'buffer_size' in key:
                # Sum counters
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            else:
                # Average other metrics
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / world_size
            
            aggregated[key] = value.item() if isinstance(value, torch.Tensor) else value
        else:
            aggregated[key] = value
    
    return aggregated


def sync_global_counters(workspace):
    """Synchronize global counters across all processes"""
    if workspace.world_size == 1:
        return
    
    # Create tensors for global counters
    counters = torch.tensor([
        workspace._global_step,
        workspace._global_episode,
        len(workspace.replay_storage)
    ], dtype=torch.long, device=workspace.device)
    
    # Sum counters across all processes
    dist.all_reduce(counters, op=dist.ReduceOp.SUM)
    
    # Update global counters (only on rank 0 for logging purposes)
    if workspace.rank == 0:
        workspace._global_step_total = counters[0].item()
        workspace._global_episode_total = counters[1].item()
        workspace._global_buffer_size = counters[2].item()
    else:
        workspace._global_step_total = 0
        workspace._global_episode_total = 0
        workspace._global_buffer_size = 0


def clean_master_port(port_str):
    """Clean and validate MASTER_PORT environment variable"""
    if not port_str:
        return '29500'  # default port
    
    # Extract only the numeric part from the string
    import re
    numeric_part = re.search(r'\d+', port_str)
    if numeric_part:
        port = numeric_part.group()
        # Validate port range
        port_int = int(port)
        if 1024 <= port_int <= 65535:
            return port
    
    print(f"Warning: Invalid MASTER_PORT '{port_str}', using default 29500")
    return '29500'


def setup_ddp(rank, world_size):
    """Initialize the process group for DDP"""
    # Set NCCL environment variables for InfiniBand/Leonardo cluster
    nccl_env_vars = {
        'NCCL_IB_DISABLE': '0',  # Enable InfiniBand
        'NCCL_NET_GDR_LEVEL': '2',  # GPU Direct RDMA level
        'NCCL_IB_GID_INDEX': '3',  # InfiniBand GID index
        'NCCL_DEBUG': 'INFO',  # Set debug level
        'NCCL_SOCKET_IFNAME': '^lo,docker',  # Exclude loopback and docker interfaces
        'NCCL_IB_HCA': 'mlx5',  # Mellanox adapter
        'NCCL_IB_TIMEOUT': '22',  # Increase timeout for slow networks
        'NCCL_IB_RETRY_CNT': '7',  # Increase retry count
    }
    
    # Only set environment variables if they're not already set
    for key, value in nccl_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Rank {rank}: Set {key}={value}")
    
    # Use environment variables set by torchrun or set defaults
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    # Clean the MASTER_PORT variable to ensure it's a valid port number
    raw_master_port = os.environ.get('MASTER_PORT', '29500')
    clean_port = clean_master_port(raw_master_port)
    os.environ['MASTER_PORT'] = clean_port
    
    print(f"Rank {rank}: Raw MASTER_PORT='{raw_master_port}'")
    print(f"Rank {rank}: Cleaned MASTER_PORT='{clean_port}'")
    print(f"Rank {rank}: Initializing DDP with MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Print additional debugging info for SLURM environment
    slurm_vars = ['SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_JOB_NUM_NODES', 'SLURM_NODELIST']
    for var in slurm_vars:
        if var in os.environ:
            print(f"Rank {rank}: {var}={os.environ[var]}")
    
    # Print network interface information
    try:
        import socket
        hostname = socket.gethostname()
        print(f"Rank {rank}: Running on hostname: {hostname}")
    except:
        pass
    
    # Initialize the process group
    try:
        # For InfiniBand networks like Leonardo, use NCCL backend
        backend = 'nccl'
        
        # Set longer timeout for initialization (useful for slow networks)
        timeout = torch.distributed.default_pg_timeout * 2
        
        print(f"Rank {rank}: Attempting to initialize process group with backend={backend}, timeout={timeout}")
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        
        # For multi-node setup, use LOCAL_RANK for CUDA device
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cuda_device = local_rank if local_rank < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_device)
        print(f"Rank {rank}: DDP initialization successful with backend={backend}, CUDA device={cuda_device}")
        
        # Simple synchronization test without tensor operations that might fail
        if world_size > 1:
            print(f"Rank {rank}: Testing basic synchronization...")
            dist.barrier()
            print(f"Rank {rank}: Basic synchronization test passed")
        
    except Exception as e:
        print(f"Rank {rank}: NCCL DDP initialization failed: {e}")
        print(f"Rank {rank}: Attempting fallback to Gloo backend...")
        
        try:
            # Fallback to Gloo backend for CPU-based communication
            dist.init_process_group(
                backend='gloo',
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout
            )
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            cuda_device = local_rank if local_rank < torch.cuda.device_count() else 0
            torch.cuda.set_device(cuda_device)
            print(f"Rank {rank}: DDP initialization successful with Gloo backend, CUDA device={cuda_device}")
            
            if world_size > 1:
                print(f"Rank {rank}: Testing Gloo synchronization...")
                dist.barrier()
                print(f"Rank {rank}: Gloo synchronization test passed")
                
        except Exception as e2:
            print(f"Rank {rank}: Both NCCL and Gloo initialization failed")
            print(f"Rank {rank}: NCCL error: {e}")
            print(f"Rank {rank}: Gloo error: {e2}")
            print(f"Rank {rank}: Environment variables:")
            for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
                print(f"Rank {rank}: {key}={os.environ.get(key, 'NOT_SET')}")
            raise RuntimeError(f"Failed to initialize distributed training: NCCL={e}, Gloo={e2}")


def cleanup_ddp():
    """Clean up the process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.work_dir = Path.cwd()
        
        # Initialize global counters for distributed tracking
        self._global_step_total = 0
        self._global_episode_total = 0
        self._global_buffer_size = 0
        
        # Only print from rank 0
        if rank == 0:
            print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if cfg.seed == 1:
            cfg.seed = np.random.randint(0, 10000)
        
        # Set different seed for each process
        utils.set_seed_everywhere(cfg.seed + rank)
        
        # For multi-node setup, use LOCAL_RANK for CUDA device, not global rank
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cuda_device = local_rank if local_rank < torch.cuda.device_count() else 0
        self.device = torch.device(f'cuda:{cuda_device}')
        print(f'Rank {rank}: Using device {self.device} (LOCAL_RANK={local_rank})')
        
        self.setup()

        # Get observation and action specs for the agent
        obs_spec = metaworld_env.observation_spec(self.train_env)
        action_spec = metaworld_env.action_spec(self.train_env)
        
        self.agent = make_agent(obs_spec, action_spec, self.cfg.agent)
        
        # Wrap agent components with DDP (only the ones that need gradient synchronization)
        if world_size > 1 and hasattr(self.agent, 'actor'):
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            cuda_device = local_rank if local_rank < torch.cuda.device_count() else 0
            self.agent.actor = torch.nn.parallel.DistributedDataParallel(
                self.agent.actor, device_ids=[cuda_device], output_device=cuda_device
            )
            self.agent.critic = torch.nn.parallel.DistributedDataParallel(
                self.agent.critic, device_ids=[cuda_device], output_device=cuda_device
            )
            # Don't wrap encoder and TACO with DDP since we handle their gradients manually with ED4CT
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.saved_medium_policy = False

        # Only initialize wandb from rank 0
        if cfg.use_wandb and rank == 0:
            tmp_cfg = OmegaConf.to_container(cfg, resolve=True)
            tmp_cfg = OmegaConf.create(tmp_cfg)
            tmp_cfg.batch_size = cfg.batch_size*world_size if hasattr(cfg, 'batch_size') else None
            tmp_cfg.local_batch_size = cfg.batch_size if hasattr(cfg, 'batch_size') else None
            tmp_cfg.world_size = world_size
            tmp_cfg.rank = rank

            if cfg.wandb_id is not None and cfg.wandb_id != "none":
                wandb.init(
                    id=cfg.wandb_id,
                    config=OmegaConf.to_container(tmp_cfg, resolve=True),
                    resume='must',
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode=cfg.wandb_mode if cfg.wandb_mode else 'online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(tmp_cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode=cfg.wandb_mode if cfg.wandb_mode else 'online')
            wandb.run.save(self.work_dir)

    def setup(self):
        # Create logger only on rank 0
        if self.rank == 0:
            self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        else:
            self.logger = None
        
        # Create environments
        self.train_env = metaworld_env.make(
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

        # Create replay buffer
        self.replay_storage = ReplayBufferStorage(
            data_specs,
            self.work_dir / 'buffer'
        )

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', 
            self.cfg.replay_buffer_size,
            self.cfg.batch_size, 
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, 
            self.cfg.nstep, 
            self.cfg.multistep, 
            self.cfg.discount
        )
        
        self._replay_iter = None

        # Setup video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            metaworld = True
        )
        
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,
            metaworld = True
        )
    
    def save_policy(self, policy_type):
        checkpoint_path = self.work_dir / f'{policy_type}_policy.pt'
        payload = {
            'agent': self.agent,
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'timer': self.timer,
            'cfg': self.cfg
        }
        torch.save(payload, checkpoint_path)
        print(f'Policy checkpoint "{policy_type}" saved: {checkpoint_path}')


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

    @property
    def global_step_total(self):
        """Total steps across all GPUs"""
        return self._global_step_total

    @property
    def global_episode_total(self):
        """Total episodes across all GPUs"""
        return self._global_episode_total

    @property
    def global_frame_total(self):
        """Total frames across all GPUs"""
        return self._global_step_total * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward, success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled= True) #(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                if time_step.last():
                    success += time_step.success*1
                    break
            episode += 1
            self.video_recorder.save(f'{self.global_frame}-{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('success_rate', success / episode)
            # Add global counters
            log('global_step_total', self.global_step_total)
            log('global_episode_total', self.global_episode_total)
            log('global_frame_total', self.global_frame_total)

        if self.cfg.use_wandb:
            wandb.log({
                'eval/episode_reward': total_reward / episode,
                'eval/episode_length': step * self.cfg.action_repeat / episode,
                'eval/episode': self.global_episode,
                'eval/step': self.global_step,
                'eval/success_rate': success / episode,
                'buffer_size': len(self.replay_storage),
                'global_frame': self.global_frame,
                # Add global distributed metrics
                'eval/global_step_total': self.global_step_total,
                'eval/global_episode_total': self.global_episode_total,
                'eval/global_frame_total': self.global_frame_total,
                'eval/global_buffer_size': self._global_buffer_size,
                'distributed/world_size': self.world_size,
                'distributed/rank': self.rank
            })

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        synced_before_training = False
        # self.save_policy('random')
        
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                
                # Sync global counters
                sync_global_counters(self)
                
                if self.rank == 0:  # Only save video from rank 0
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                
                # wait until all the metrics schema is populated
                if metrics is not None and self.rank == 0:  # Only log from rank 0
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        # Add global counters
                        log('global_step_total', self.global_step_total)
                        log('global_episode_total', self.global_episode_total)
                        log('global_frame_total', self.global_frame_total)
                        log('global_buffer_size', self._global_buffer_size)
                    
                    # Log also on wandb:
                    if self.cfg.use_wandb:
                        # Add distributed training info
                        wandb_metrics = {
                            'fps': episode_frame / elapsed_time,
                            'total_time': total_time,
                            'episode_reward': episode_reward,
                            'episode_length': episode_frame,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'global_frame': self.global_frame,
                            'step': self.global_step,
                            # Global distributed metrics
                            'distributed/global_step_total': self.global_step_total,
                            'distributed/global_episode_total': self.global_episode_total,
                            'distributed/global_frame_total': self.global_frame_total,
                            'distributed/global_buffer_size': self._global_buffer_size,
                            'distributed/world_size': self.world_size,
                            'distributed/rank': self.rank,
                            'distributed/effective_fps': (episode_frame / elapsed_time) * self.world_size
                        }
                        wandb.log(wandb_metrics)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate (only from rank 0)
            if eval_every_step(self.global_step) and self.rank == 0:
                if self.logger:
                    self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if not synced_before_training:
                    # Synchronize all processes before starting training
                    if self.world_size > 1:
                        print(f"Rank {self.rank}: Waiting for all processes to sync before training...")
                        dist.barrier()
                        print(f"Rank {self.rank}: All processes synced")
                    synced_before_training = True
                
                metrics = self.agent.update(self.replay_iter, self.global_step)
                
                # Aggregate metrics across GPUs for logging
                if self.world_size > 1:
                    aggregated_metrics = aggregate_metrics_across_gpus(metrics.copy(), self.world_size)
                else:
                    aggregated_metrics = metrics
                
                if self.rank == 0 and self.logger:  # Only log from rank 0
                    self.logger.log_metrics(aggregated_metrics, self.global_frame, ty='train')
                
                # Sync global counters for consistent reporting
                sync_global_counters(self)

                if self.cfg.use_wandb and self.rank == 0:
                    wandb_metrics = aggregated_metrics.copy()
                    wandb_metrics.update({
                        'global_step': self.global_step,
                        'episode': self.global_episode,
                        'buffer_size': len(self.replay_storage),
                        'step': self.global_step,
                        'global_frame': self.global_frame,
                        # Global distributed counters
                        'distributed/global_step_total': self.global_step_total,
                        'distributed/global_episode_total': self.global_episode_total,
                        'distributed/global_frame_total': self.global_frame_total,
                        'distributed/global_buffer_size': self._global_buffer_size,
                        'distributed/world_size': self.world_size,
                        'distributed/rank': self.rank
                    })
                    wandb.log(wandb_metrics)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            
        if self.rank == 0:  # Only save from rank 0
            self.save_policy('expert')

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            print("loading snapshot: ", f)
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v


def run_training(rank, world_size, cfg):
    """Run training on a single process"""
    try:
        # Setup DDP
        if world_size > 1:
            setup_ddp(rank, world_size)
        
        # Create workspace
        workspace = Workspace(cfg, rank, world_size)
        
        # Load snapshot if exists (only check from rank 0)
        if rank == 0:
            snapshot = workspace.work_dir / 'snapshot.pt'
            if snapshot.exists():
                print(f'resuming: {snapshot}')
                workspace.load_snapshot()
        
        # Synchronize all processes
        if world_size > 1:
            print(f"Rank {rank}: Waiting for all processes to sync...")
            dist.barrier()
            print(f"Rank {rank}: All processes synced")
        
        # Start training
        workspace.train()
        
    except Exception as e:
        print(f"Rank {rank}: Training failed with error: {e}")
        raise
    finally:
        # Clean up
        if world_size > 1:
            cleanup_ddp()


@hydra.main(config_path='cfgs', config_name='config_metaworld_ed4c')
def main(cfg):
    from pathlib import Path
    
    # Print environment variables for debugging
    print("=== Environment Variables Debug ===")
    env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'SLURM_PROCID', 'SLURM_NTASKS', 
                'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_JOB_NUM_NODES', 'SLURM_NODELIST',
                'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'NOT_SET')}")
    print("=====================================")
    
    # Determine if we're running in distributed mode
    distributed = False
    rank = 0
    world_size = 1
    
    # Check SLURM environment variables first (srun)
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        distributed = True
        print(f"Detected SLURM/srun environment: rank={rank}, world_size={world_size}")
        
        # For multi-node setup with 1 task per node, LOCAL_RANK should be 0 for each node
        # but we need to use the global rank for CUDA device selection
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        
        # Set torchrun-like environment variables for compatibility
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        print(f"SLURM setup: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    # Check torchrun environment variables
    elif 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        distributed = True
        print(f"Detected torchrun environment: rank={rank}, world_size={world_size}")
    
    # Fallback to spawn method for multi-GPU on single node
    elif cfg.get('use_ed4ct', False):
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            world_size = available_gpus
            print(f"No distributed environment detected, using spawn method with {world_size} GPUs")
            mp.spawn(run_training, args=(world_size, cfg), nprocs=world_size, join=True)
            return
        else:
            print("Single GPU available, running in single-GPU mode")
    
    if distributed:
        print(f"Running distributed training: rank={rank}, world_size={world_size}")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        
        if cfg.use_wandb and rank == 0:
            wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
        
        run_training(rank, world_size, cfg)
    else:
        print("Starting single-GPU training")
        if cfg.use_wandb:
            wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
        run_training(0, 1, cfg)


if __name__ == '__main__':
    main()
