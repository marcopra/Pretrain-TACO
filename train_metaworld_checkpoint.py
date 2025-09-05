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
from dm_env import specs

import metaworld_env
import wandb
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

MAX_REWARD = 4200

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if cfg.seed==1: # TODO put none instead of 1
            cfg.seed = np.random.randint(0, 10000)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        # Get observation and action specs for the agent
        obs_spec = metaworld_env.observation_spec(self.train_env)
        action_spec = metaworld_env.action_spec(self.train_env)
        
        self.agent = make_agent(obs_spec, action_spec, self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.saved_medium_policy = False

        # Initialize encoder saving flags and thresholds
        self.encoder_save_thresholds = [0.0, 0.2, 0.4, 0.6, 1.0]  # 0%, 20%, 40%, 60%, 100% of MAX_REWARD
        self.encoder_saved_flags = {threshold: False for threshold in self.encoder_save_thresholds}
        self.max_eval_reward = 0  # Track maximum evaluation reward achieved

        if cfg.use_wandb:
           
            
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
        # Create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        
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


    def save_encoder_checkpoint(self, reward_percentage, reward_value):
        """Save encoder checkpoint with informative naming"""
        # Create encoder state dict similar to pretrain_MT_multiheads.py
        encoder_state = {
            'encoder': self.agent.encoder.state_dict(),
            'taco': self.agent.TACO.state_dict(),
            'act_tok': self.agent.act_tok.state_dict(),
            'args': {
                'env_name': self.cfg.env_name,
                'task': self.cfg.task,
                'seed': self.cfg.seed,
                'global_step': self.global_step,
                'global_episode': self.global_episode,
                'reward_percentage': reward_percentage,
                'reward_value': reward_value,
                'max_reward_threshold': MAX_REWARD
            },
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'reward_percentage': reward_percentage,
            'reward_value': reward_value
        }
        
        # Create informative filename
        percentage_str = f"{int(reward_percentage * 100)}pct"
        reward_str = f"rew{reward_value:.0f}"
        step_str = f"step{self.global_step}"
        episode_str = f"ep{self.global_episode}"
        
        filename = f"encoder_{self.cfg.env_name}_{self.cfg.task}_{percentage_str}_{reward_str}_{step_str}_{episode_str}.pt"
        checkpoint_path = self.work_dir / filename
        
        torch.save(encoder_state, checkpoint_path)
        print(f'Encoder checkpoint saved: {checkpoint_path} (Reward: {reward_value:.2f}, {percentage_str} of MAX_REWARD)')
        
        if self.cfg.use_wandb:
            wandb.log({
                f'encoder_checkpoint_saved_{percentage_str}': 1,
                f'encoder_checkpoint_reward_{percentage_str}': reward_value,
                f'encoder_checkpoint_step_{percentage_str}': self.global_step,
                'global_frame': self.global_frame
            })

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

        avg_episode_reward = total_reward / episode
        
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_episode_reward)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('success_rate', success / episode)

        if self.cfg.use_wandb:
            wandb.log({
                'eval/episode_reward': avg_episode_reward,
                'eval/episode_length': step * self.cfg.action_repeat / episode,
                'eval/episode': self.global_episode,
                'eval/step': self.global_step,
                'eval/success_rate': success / episode,
                'buffer_size': len(self.replay_storage),
                'global_frame': self.global_frame
            })

        # Check if we need to save encoder based on reward thresholds
        self.max_eval_reward = max(self.max_eval_reward, avg_episode_reward)
        
        for threshold in self.encoder_save_thresholds:
            threshold_reward = threshold * MAX_REWARD
            
            # Check if we've reached this threshold and haven't saved yet
            if (avg_episode_reward >= threshold_reward and 
                not self.encoder_saved_flags[threshold]):
                
                self.encoder_saved_flags[threshold] = True
                self.save_encoder_checkpoint(threshold, avg_episode_reward)

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
        
        print('Start training...')
        # Save initial encoder checkpoint (0% threshold)
        if not self.encoder_saved_flags[0.0]:
            print('Saving initial encoder checkpoint...')
            self.encoder_saved_flags[0.0] = True
            self.save_encoder_checkpoint(0.0, 0.0)
        print('Initial encoder checkpoint saved.')
        
        
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
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
                        
                    

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

                if self.cfg.use_wandb:
                    metrics.update({'global_step': self.global_step,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'step': self.global_step,
                            'global_frame': self.global_frame
                            })
                    wandb.log(metrics)


            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            


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


@hydra.main(config_path='cfgs', config_name='config_metaworld')
def main(cfg):
    from pathlib import Path
    if cfg.use_wandb:
        wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
    from train_metaworld_checkpoint import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

