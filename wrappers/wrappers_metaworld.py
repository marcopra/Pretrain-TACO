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