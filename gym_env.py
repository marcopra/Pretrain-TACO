from collections import deque
from typing import Any, NamedTuple
import os

import gym
import gymnasium
import numpy as np
from gymnasium import spaces
import gymnasium_robotics
gymnasium.register_envs(gymnasium_robotics)
from dm_env import StepType, specs
from PIL import Image

class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=84):
        super().__init__(env)
        self.resolution = resolution

    def render(self):
        img = super().render()

        # # Flip verticale per correggere l'orientamento (MuJoCo restituisce immagini capovolte)
        # img = np.flipud(img)

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
    proprio_observation: Any
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
    def __init__(self, env, num_repeats, data_collection=False):
        super().__init__(env)
        self._num_repeats = num_repeats
        self.data_collection = data_collection

    def _process_proprio_obs(self, obs):
        """Process proprioceptive observation, concatenating dict values if needed."""
        if isinstance(obs, dict):
            # Concatenate all values in the dictionary
            arrays = []
            for key in sorted(obs.keys(), reverse=True):  # Sort for consistent ordering
                arrays.append(obs[key].flatten())
            return np.concatenate(arrays)
        else:
            return obs

    def step(self, action):
        reward = 0.0
        discount = 1.0
        done = False
        info = {}
        
        for i in range(self._num_repeats):
            obs, reward_step, terminated, truncated, info = self.env.step(action)
            # Handle success as a termination condition in MetaWorld
            done = terminated or truncated
            
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
        proprio_obs = self._process_proprio_obs(obs)
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=image_obs,  # Use image observations
            proprio_observation=proprio_obs,
            action=action,
            success= terminated,
        )

    def reset(self):
        obs, info = self.env.reset()
        image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=image_obs,  # Use image observations
            proprio_observation=proprio_obs,
            action=np.zeros(self.env.action_space.shape, dtype=np.float32),
            success=False
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



def make(name, frame_stack=1, action_repeat=1, seed=None, resolution=224):
    """
    Create a Gymnasium environment with wrappers.
    
    Args:
        name: Environment name (e.g., 'PointMaze_Medium-v3')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat each action
        seed: Random seed
    
    Returns:
        Wrapped environment
    """
    # Create the environment
    env = gymnasium.make(name, render_mode='rgb_array')
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Add wrappers
    # env = ResizeRendering(env, resolution=resolution)   
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    
    # Add frame stacking if requested
  
    env = FrameStackWrapper(env, frame_stack)
    
    env = ExtendedTimeStepWrapper(env)
    
    return env


# Tests
if __name__ == "__main__":
    print("Testing gym_envs.py...")
    
    # Test 1: Basic environment creation (continuous action space)
    try:
        env = make('PointMaze_Medium-v3', seed=42, frame_stack=1, action_repeat=2)
        print("✓ PointMaze environment creation successful")
    except Exception as e:
        print(f"✗ PointMaze environment creation failed: {e}")
        # Fallback to another continuous environment
        try:
            env = make('PointMaze_Medium-v3', seed=42)
            print("✓ Fallback environment (MountainCarContinuous) creation successful")
        except Exception as e2:
            print(f"✗ Fallback environment creation failed: {e2}")
            exit(1)
    
    # Test 2: Reset functionality
    try:
        time_step = env.reset()
        assert hasattr(time_step, 'observation')
        assert hasattr(time_step, 'step_type')
        assert hasattr(time_step, 'action')
        assert hasattr(time_step, 'reward')
        assert hasattr(time_step, 'discount')
        assert hasattr(time_step, 'proprio_observation')
        assert hasattr(time_step, 'success')
        assert time_step.first()
        print(f"✓ Reset functionality works correctly")
        print(f"  - Observation shape: {time_step.observation.shape}")
        print(f"  - Proprio observation shape: {time_step.proprio_observation.shape}")
        print(f"  - Action shape: {time_step.action.shape}")
    except Exception as e:
        print(f"✗ Reset functionality failed: {e}")
    
    # Test 3: Step functionality
    try:
        action = env.action_space.sample()
        time_step = env.step(action)
        assert hasattr(time_step, 'observation')
        assert hasattr(time_step, 'step_type')
        assert hasattr(time_step, 'action')
        assert hasattr(time_step, 'reward')
        assert hasattr(time_step, 'discount')
        assert hasattr(time_step, 'proprio_observation')
        assert hasattr(time_step, 'success')
        print("✓ Step functionality works correctly")
        print(f"  - Reward: {time_step.reward}")
        print(f"  - Discount: {time_step.discount}")
    except Exception as e:
        print(f"✗ Step functionality failed: {e}")
    
    # Test 4: Frame stacking (only with continuous environments)
    try:
        env_stacked = make('PointMaze_Medium-v3', frame_stack=3, seed=42)
        time_step = env_stacked.reset()
        print(f"✓ Frame stacking works correctly")
        print(f"  - Stacked observation shape: {time_step.observation.shape}")
    except Exception as e:
        print(f"✗ Frame stacking failed: {e}")
    
    # Test 5: Action repeat
    try:
        env_repeat = make('PointMaze_Medium-v3', action_repeat=2, seed=42)
        time_step = env_repeat.reset()
        action = env_repeat.action_space.sample()
        time_step = env_repeat.step(action)
        print("✓ Action repeat wrapper works correctly")
    except Exception as e:
        print(f"✗ Action repeat failed: {e}")
    
    # Test 6: Full episode
    # try:
    try:
        env_test = make('PointMaze_Medium-v3', seed=42)
        time_step = env_test.reset()
        total_reward = 0
        steps = 0
        
        while not time_step.last() and steps < 200:
            action = env_test.action_space.sample()
            time_step = env_test.step(action)
            total_reward += time_step.reward
            steps += 1
        
        print(f"✓ Full episode completed: {steps} steps, total reward: {total_reward:.2f}")
    except Exception as e:
        print(f"✗ Full episode test failed: {e}")
    
    # Test 7: Action space compatibility
    try:
        print(f"✓ Environment details:")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Action space dtype: {env.action_space.dtype}")
        print(f"  - Action space shape: {env.action_space.shape}")
        action_sample = env.action_space.sample()
        print(f"  - Sample action: {action_sample}")
        print(f"  - Sample action dtype: {action_sample.dtype}")
    except Exception as e:
        print(f"✗ Action space test failed: {e}")
    
    print("All tests completed!")

     # L'osservazione è già in formato CHW (channels first) dopo il FrameStackWrapper
    observation = time_step.observation  # Shape: (channels*frames, height, width)
    
    # Prendi solo i primi 3 canali per visualizzare un singolo frame RGB
    single_frame = observation[:3, :, :].transpose(1, 2, 0)  # Converti da CHW a HWC
    
    # Salvando immagine d'esempio...
    # Immagine salvata come 'esempio_metaworld.png'
    # Forma dell'osservazione: (9, 84, 84)
    # Forma dell'osservazione proprietà: (39,)
    # Range valori immagine: [0, 255]
    # Forma dell'osservazione: (9, 84, 84)
    # Forma dell'osservazione proprietà: (39,)
    # Range valori immagine: [0, 255]
    # Converti in PIL Image e salva (flip già applicato nel render)
    img = Image.fromarray(single_frame.astype(np.uint8))
    img.save('/home/mprattico/Pretrain-TACO/esempio_gym.png')

    print("Immagine salvata come 'esempio_gym.png'")
    print(f"Forma dell'osservazione: {observation.shape}")
    print(f"Forma dell'osservazione proprietà: {time_step.proprio_observation}")
    print(f"Range valori immagine: [{observation.min()}, {observation.max()}]")
    print(f"Forma dell'osservazione: {observation.shape}")
    print(f"Forma dell'osservazione proprietà: {time_step.proprio_observation.shape}")
    print(f"Range valori immagine: [{observation.min()}, {observation.max()}]")