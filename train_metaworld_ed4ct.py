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


def setup_ddp(rank, world_size):
    """Initialize the process group for DDP"""
    # Use environment variables set by torchrun
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    print(f"Rank {rank}: Initializing DDP with MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Initialize the process group
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: DDP initialization successful")
    except Exception as e:
        print(f"Rank {rank}: DDP initialization failed: {e}")
        raise


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
        
        # Only print from rank 0
        if rank == 0:
            print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if cfg.seed == 1:
            cfg.seed = np.random.randint(0, 10000)
        
        # Set different seed for each process
        utils.set_seed_everywhere(cfg.seed + rank)
        self.device = torch.device(f'cuda:{rank}')
        self.setup()

        # Get observation and action specs for the agent
        obs_spec = metaworld_env.observation_spec(self.train_env)
        action_spec = metaworld_env.action_spec(self.train_env)
        
        self.agent = make_agent(obs_spec, action_spec, self.cfg.agent)
        
        # Wrap agent components with DDP (only the ones that need gradient synchronization)
        if world_size > 1 and hasattr(self.agent, 'actor'):
            self.agent.actor = torch.nn.parallel.DistributedDataParallel(
                self.agent.actor, device_ids=[rank], output_device=rank
            )
            self.agent.critic = torch.nn.parallel.DistributedDataParallel(
                self.agent.critic, device_ids=[rank], output_device=rank
            )
            # Don't wrap encoder and TACO with DDP since we handle their gradients manually with ED4CT
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.saved_medium_policy = False

        # Only initialize wandb from rank 0
        if cfg.use_wandb and rank == 0:
            if cfg.wandb_id is not None and cfg.wandb_id != "none":
                wandb.init(
                    id=cfg.wandb_id,
                    resume='must',
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
            wandb.run.save()

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
        # self.save_policy('random')
        
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
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
                            'world_size': self.world_size,
                            'rank': self.rank
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
                metrics = self.agent.update(self.replay_iter, self.global_step)
                if self.rank == 0 and self.logger:  # Only log from rank 0
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                if self.cfg.use_wandb and self.rank == 0:
                    metrics.update({
                        'global_step': self.global_step,
                        'episode': self.global_episode,
                        'buffer_size': len(self.replay_storage),
                        'step': self.global_step,
                        'global_frame': self.global_frame,
                        'world_size': self.world_size,
                        'rank': self.rank
                    })
                    wandb.log(metrics)

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


@hydra.main(config_path='cfgs', config_name='config_metaworld')
def main(cfg):
    from pathlib import Path
    
    # Check if running with torchrun (distributed)
    if 'LOCAL_RANK' in os.environ:
        # Running with torchrun - use environment variables
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Detected torchrun environment: rank={rank}, world_size={world_size}")
        
        if cfg.use_wandb and rank == 0:
            wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
        
        run_training(rank, world_size, cfg)
    else:
        # Determine number of GPUs to use for spawn method
        world_size = torch.cuda.device_count() if cfg.get('use_ed4ct', False) else 1
        
        if world_size > 1:
            print(f"Starting distributed training with spawn method using {world_size} GPUs")
            # Use spawn method for multi-GPU training
            mp.spawn(run_training, args=(world_size, cfg), nprocs=world_size, join=True)
        else:
            print("Starting single-GPU training")
            if cfg.use_wandb:
                wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
            run_training(0, 1, cfg)


if __name__ == '__main__':
    main()
