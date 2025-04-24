from collections import deque
from typing import Any, NamedTuple

import numpy as np
import gymnasium as gym
import metaworld
import torch
from dm_env import StepType, specs

from PIL import Image

class RandomizeInitialPositionWrapper(gym.Wrapper):
    """A wrapper that randomizes the initial position and orientation of the hand and object in MetaWorld environments."""
    
    def __init__(self, env, randomize_hand_pos = True, randomize_goal_and_object_pos = True):
        super().__init__(env)
        self.randomize_hand_pos = randomize_hand_pos
        self.randomize_goal_and_object_pos = randomize_goal_and_object_pos
        # Get the actual SawyerXYZEnv instance
        if hasattr(self.env, 'env'):
            self.sawyer_env = self.env.env
        else:
            self.sawyer_env = self.env

        
    def reset(self, seed=None, options=None):
        """Reset the environment and randomize hand position."""
        if self.randomize_goal_and_object_pos:
            if hasattr(self.sawyer_env, '_freeze_rand_vec'):
                original_freeze = self.sawyer_env._freeze_rand_vec
                original_seeded = self.sawyer_env.seeded_rand_vec
                
                self.sawyer_env._freeze_rand_vec = False  # Allow randomization
                self.sawyer_env.seeded_rand_vec = True    # Use seeded randomization

        # Reset environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Now let's randomize hand position
        if hasattr(self.sawyer_env, 'hand_low') and hasattr(self.sawyer_env, 'hand_high') and self.randomize_hand_pos:
            # Get hand position bounds
            hand_low = self.sawyer_env.hand_low
            hand_high = self.sawyer_env.hand_high
            
            # Generate random hand position within bounds
            random_hand_pos = np.random.uniform(hand_low, hand_high)
            
            # Direct method to set hand position through mocap
            mocap_id = self.sawyer_env.model.body_mocapid[self.sawyer_env.data.body("mocap").id]
            self.sawyer_env.data.mocap_pos[mocap_id] = random_hand_pos
            self.sawyer_env.data.mocap_quat[mocap_id] = np.array([1, 0, 1, 0])
            
            # Run simulation steps to apply the changes
            for _ in range(10):
                self.sawyer_env.do_simulation([-1, 1], self.sawyer_env.frame_skip)
            
            # Update the observation to reflect new hand position
            obs = self.sawyer_env._get_obs()
            
        return obs, info
    
    def set_task(self, task):
        """Set the task for the environment."""
        # Set the task in the base environment
        self.env.set_task(task)
    

class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=84):
        super().__init__(env)
        self.resolution = resolution

    def render(self):
        img =  super().render()

        # Convert numpy array to PIL Image
        img = Image.fromarray(img.astype(np.uint8))
        
        # Resize the image
        img_resized = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert back to numpy array
        return np.array(img_resized)
    
    def set_task(self, task):
        """Set the task for the environment."""
        # Set the task in the base environment
        self.env.set_task(task)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any = None

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
        done = False
        info = {}
        
        for i in range(self._num_repeats):
            obs, reward_step, terminated, truncated, info = self.env.step(action)
            # Handle success as a termination condition in MetaWorld
            
            done = terminated or truncated or int(info['success']) == 1
            
            reward += reward_step * discount
            discount *= 0.99  # Standard discount factor
            
            if done:
                break
                
        # Convert gym step to dm_env format for compatibility
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        image_obs = self.env.render()
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=image_obs,  # Use image observations
            action=action,
            success= (int(info['success']) == 1)
        )

    def reset(self):
        obs, info = self.env.reset()
        image_obs = self.env.render()
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=image_obs,  # Use image observations
            action=np.zeros(self.env.action_space.shape, dtype=np.float32)
        )


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        # Update observation space to include stacked frames
        obs = env.reset()

        # Get the shape from the observation
        if isinstance(obs.observation, np.ndarray):
            self.orig_obs_shape = obs.observation.shape
        else:
            # Handle case where observation might be a different structure
            raise ValueError("Expected observation to be a numpy array")
        
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
        # Stack frames along the channel dimension (axis 0 after transpose)
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, obs):
        # Transform HWC to CHW format
        if isinstance(obs, np.ndarray):
            return obs.transpose(2, 0, 1).copy()
        else:
            raise ValueError("Expected observation to be a numpy array")

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


def make(env_name, frame_stack, action_repeat, seed, resolution=84, camera='corner', rondom_init=True):
    """
    Create a MetaWorld environment with image observations, frame stacking, and action repeat.
    
    Args:
        env_name: Name of the MetaWorld environment (e.g., 'push-v2')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat the action
        seed: Random seed
        resolution: Image resolution (height and width)
        camera: Camera angle to use
        rondom_init: Whether to randomize the initial position of the hand and object
    
    Returns:
        A wrapped MetaWorld environment
    """
    # Create MetaWorld environment with image observations
    mt10 = metaworld.MT10(seed=seed)  # Use the provided seed instead of hardcoded 42
    task_number = 0
    env_task_indices = [i for i, task in enumerate(mt10.train_tasks) if task.env_name == env_name]
    if not env_task_indices:
        raise ValueError(f"Environment {env_name} not found in MT10 tasks")
    if task_number >= len(env_task_indices):
        print(f"Task number {task_number} out of range. Available tasks: 0-{len(env_task_indices)-1}")
        return
    
    # Get the task index
    task_idx = env_task_indices[task_number]
    # Create environment with image observations using PLEX-MetaWorld
    env = mt10.train_classes[env_name](render_mode="rgb_array", camera_name=camera)
    if rondom_init is True:
        env = RandomizeInitialPositionWrapper(env)
    env = ResizeRendering(env, resolution=resolution)
    env.set_task(mt10.train_tasks[task_idx])

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
