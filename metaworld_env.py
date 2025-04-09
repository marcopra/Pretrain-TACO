from collections import deque
from typing import Any, NamedTuple

import numpy as np
import gym
import metaworld
import torch
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, num_repeats):
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            obs, reward_step, done, info = self.env.step(action)
            reward += reward_step * discount
            discount *= 0.99  # Standard discount factor
            if done:
                break
                
        # Convert gym step to dm_env format for compatibility
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
            
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=obs['image'],  # Use only image observations
            action=action
        )

    def reset(self):
        obs = self.env.reset()
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=obs['image'],  # Use only image observations
            action=np.zeros(self.env.action_space.shape, dtype=np.float32)
        )


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        # Update observation space to include stacked frames
        obs = env.reset()

        self.orig_obs_shape = obs.observation.shape
        
        # Create a new stacked observation space
        channels = self.orig_obs_shape[2] * num_frames
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(channels, self.orig_obs_shape[0], self.orig_obs_shape[1]),
            dtype=np.uint8
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, obs):
        # Transform HWC to CHW format
        return obs.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self.env.reset()
        pixels = self._extract_pixels(time_step.observation)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step.observation)
        self._frames.append(pixels)
        return self._transform_observation(time_step)


class ActionDTypeWrapper(gym.Wrapper):
    def __init__(self, env, dtype=np.float32):
        super().__init__(env)
        original_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=original_space.low.astype(dtype),
            high=original_space.high.astype(dtype),
            shape=original_space.shape,
            dtype=dtype
        )

    def step(self, action):
        action = action.astype(self.env.action_space.dtype)
        return self.env.step(action)


class ExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        time_step = self.env.reset()
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        return time_step


def make(env_name, frame_stack, action_repeat, seed, resolution=84, camera='corner'):
    """
    Create a MetaWorld environment with image observations, frame stacking, and action repeat.
    
    Args:
        env_name: Name of the MetaWorld environment (e.g., 'push-v2')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat the action
        seed: Random seed
        resolution: Image resolution (height and width)
        camera: Camera angle to use
    
    Returns:
        A wrapped MetaWorld environment
    """
    # Create MetaWorld environment with image observations
    env = metaworld.mw_gym_make(
        env_name,
        goal_cost_reward=True,
        stop_at_goal=False,
        cam_height=resolution,
        cam_width=resolution,
        depth=False,
        cam_name=camera,    
        )
    
    # Apply wrappers to match dm_control setup
    env = ActionDTypeWrapper(env, dtype=np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    
    
    # Apply frame stacking
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    
    return env


def observation_spec(env):
    """Get observation spec of the environment for agent initialization."""
    shape = env.observation_space.shape
    return specs.Array(shape, np.uint8, 'observation')


def action_spec(env):
    """Get action spec of the environment for agent initialization."""
    shape = env.action_space.shape
    min_action = env.action_space.low[0]
    max_action = env.action_space.high[0]
    return specs.BoundedArray(shape, np.float32, min_action, max_action, 'action')
