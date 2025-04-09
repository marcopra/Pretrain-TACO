import numpy as np
import os
import pickle
import time
import importlib
import argparse
import metaworld
from metaworld.policies import *
from PIL import Image
from collections import deque
# python metaworld_image_dataset.py --env_names push-v2,door-open-v2 --expert_probs 0.8,0.0 --dataset_size 100000 --resolution 84 --camera corner

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/examples', exist_ok=True)  # Create directory for example images

# Helper function to get the policy for a given environment
def get_policy(env_name):
    # Map environment names to their corresponding policy classes
    env_to_policy = {
        'reach-v2': SawyerReachV2Policy,
        'push-v2': SawyerPushV2Policy,
        'pick-place-v2': SawyerPickPlaceV2Policy,
        'door-open-v2': SawyerDoorOpenV2Policy,
        'drawer-close-v2': SawyerDrawerCloseV2Policy,
        'drawer-open-v2': SawyerDrawerOpenV2Policy,
        'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
        'window-open-v2': SawyerWindowOpenV2Policy,
        'window-close-v2': SawyerWindowCloseV2Policy,
        'peg-insert-side-v2': SawyerPegInsertionSideV2Policy
    }
    
    if env_name not in env_to_policy:
        raise ValueError(f"No policy found for environment {env_name}")
    
    # Return the policy instance
    return env_to_policy[env_name]()

def save_example_image(image, env_name, expert_prob, resolution, camera):
    """
    Save a single example image as PNG.
    
    Args:
        image: RGB array of shape (height, width, 3)
        env_name: Environment name
        expert_prob: Expert probability value
        resolution: Image resolution
        camera: Camera angle used
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert numpy array to PIL Image
        img = Image.fromarray(image.astype(np.uint8))
        
        # Create filename and save
        filename = f"data/examples/{env_name}_img{resolution}_exp={int(expert_prob*100)}_{camera}.png"
        img.save(filename)
        print(f"Saved example image to {filename}")
    else:
        print(f"Could not save image, unexpected shape: {image.shape}")

def collect_dataset(env_names, expert_probs, dataset_size=int(1e6), resolution=84, camera='corner', 
                   include_depth=False, save_example=False, frame_stack=1, action_repeat=1):
    """
    Collects a dataset of trajectories from MetaWorld environments with image observations.
    
    Args:
        env_names: List of environment names to collect data from
        expert_probs: List of expert probabilities to use
        dataset_size: Number of transitions to collect per environment and expert probability
        resolution: Image resolution (height and width)
        camera: Camera angle to use
        include_depth: Whether to include depth images
        save_example: Whether to save an example image
        frame_stack: Number of consecutive frames to stack together
        action_repeat: Number of times to repeat the same action
    """
    
    for expert_prob in expert_probs:
        for env_name in env_names:
            print(f"\n===== Collecting data for {env_name} with expert_prob={expert_prob} =====")
            print(f"Frame stack: {frame_stack}, Action repeat: {action_repeat}")
            
            # Initialize the dataset dictionary
            dataset = {
                'observations': [],     # Image observations (stacked)
                'proprio_states': [],   # Proprioception states
                'actions': [],
                'next_observations': [], # Next image observations (stacked)
                'next_proprio_states': [],
                'rewards': [],
                'terminals': []
            }
            
            # Create environment with image observations using PLEX-MetaWorld
            env = metaworld.mw_gym_make(
                env_name,
                goal_cost_reward=False, 
                stop_at_goal=True,
                cam_height=resolution,
                cam_width=resolution,
                depth=include_depth,
                cam_name=camera
            )
            
            # Get the expert policy
            policy = get_policy(env_name)
            
            # Action space information for noise scaling
            action_space_ptp = env.action_space.high - env.action_space.low
            
            print(f"Collecting {dataset_size} transitions...")
            
            # For collecting transitions
            transitions_collected = 0
            episode_reward = 0
            episode_count = 0
            successful_trajectories = 0
            old_dataset_length = 0
            example_saved = False
            
            # Reset the environment
            state = env.reset()
            
            # Initialize frame stack with the first observation
            frame_buffer = deque(maxlen=frame_stack)
            for _ in range(frame_stack):
                frame_buffer.append(state['image'])
            
            while transitions_collected < dataset_size:
                # Choose between expert or random action based on probability
                if np.random.random() < expert_prob:
                    # Expert policy action
                    a = policy.get_action(state['full_state'])
                else:
                    # Random action
                    a = env.action_space.sample()
                
                # Store the current stacked observation before taking action
                stacked_obs = np.concatenate(list(frame_buffer), axis=-1) if frame_stack > 1 else frame_buffer[0]
                curr_state = state.copy()
                
                # Initialize variables for action repeat
                total_reward = 0
                discount = 1.0
                done = False
                
                # Apply action repeat
                for _ in range(action_repeat):
                    if done:
                        break
                    
                    next_state, reward, done, info = env.step(a)
                    
                    # Save example image if requested and not already saved
                    if save_example and not example_saved and transitions_collected > 10:
                        if frame_stack > 1:
                            # Save the stacked image as separate files
                            for i, img in enumerate(frame_buffer):
                                save_example_image(img, f"{env_name}_frame{i}", expert_prob, resolution, camera)
                        else:
                            save_example_image(next_state['image'], env_name, expert_prob, resolution, camera)
                        example_saved = True
                    
                    # Update frame buffer with new observation
                    frame_buffer.append(next_state['image'])
                    
                    # Accumulate reward with discount
                    total_reward += reward * discount
                    discount *= 0.99  # Use a discount factor for repeated actions
                    
                    # If done, break out of the loop
                    if done:
                        break
                
                # Create the stacked next observation
                next_stacked_obs = np.concatenate(list(frame_buffer), axis=-1) if frame_stack > 1 else frame_buffer[-1]
                
                # Store the transition in the dataset
                dataset['observations'].append(stacked_obs)
                dataset['proprio_states'].append(curr_state['proprio_state'])
                dataset['actions'].append(a)
                dataset['next_observations'].append(next_stacked_obs)
                dataset['next_proprio_states'].append(next_state['proprio_state'])
                dataset['rewards'].append(total_reward)
                dataset['terminals'].append(done)
                
                transitions_collected += 1
                episode_reward += total_reward
                
                if done:
                    episode_count += 1
                    
                    # Check if the trajectory was successful
                    if 'task_accomplished' in info and info['task_accomplished'] or \
                       'success' in info and int(info['success']) == 1:
                        successful_trajectories += 1
                    
                    print(f"Episode {episode_count} ended with total reward: {episode_reward:.4f}, length: {transitions_collected - old_dataset_length}")
                    episode_reward = 0
                    old_dataset_length = transitions_collected
                    
                    # Reset the environment and frame buffer
                    state = env.reset()
                    frame_buffer.clear()
                    for _ in range(frame_stack):
                        frame_buffer.append(state['image'])
                else:
                    state = next_state
                
                # Print progress
                if transitions_collected % 10000 == 0:
                    print(f"Collected {transitions_collected}/{dataset_size} transitions")
            
            # Convert lists to numpy arrays
            for key in dataset:
                dataset[key] = np.array(dataset[key])
            
            # Save the dataset
            dataset_path = f"data/{env_name}_img{resolution}_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}"
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"Dataset saved to {dataset_path}")
            print(f"Dataset size: {transitions_collected} transitions")
            print(f"Dataset shape - observations: {dataset['observations'].shape}")
            print(f"Successful trajectories: {successful_trajectories} out of {episode_count}")
            
            # Write successful trajectories information to a text file
            success_file_path = "data/successful_image_trajectories.txt"
            with open(success_file_path, 'a') as f:
                f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
            
            print(f"Success information saved to {success_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect image-based dataset from MetaWorld environments")
    parser.add_argument("--env_names", type=str, default="push-v2,window-open-v2,door-open-v2,reach-v2,drawer-close-v2,pick-place-v2,drawer-open-v2,button-press-topdown-v2,window-close-v2,peg-insert-side-v2", 
                        help="Comma-separated list of environment names")
    parser.add_argument("--expert_probs", type=str, default="0.8", 
                        help="Comma-separated list of expert probabilities")
    parser.add_argument("--dataset_size", type=int, default=1000000, 
                        help="Number of transitions to collect per environment and expert probability")
    parser.add_argument("--resolution", type=int, default=84, 
                        help="Image resolution (height and width)")
    parser.add_argument("--camera", type=str, default="corner", 
                        choices=["corner", "topview", "corner2", "corner3", "behindGripper", "gripperPOV"],
                        help="Camera angle to use")
    parser.add_argument("--include_depth", action="store_true", 
                        help="Include depth images in the dataset")
    parser.add_argument("--save_example", action="store_true",
                        help="Save an example image as PNG")
    parser.add_argument("--frame_stack", type=int, default=3,
                        help="Number of consecutive frames to stack")
    parser.add_argument("--action_repeat", type=int, default=1,
                        help="Number of times to repeat the same action")
    
    args = parser.parse_args()
    
    env_names = args.env_names.split(",")
    expert_probs = [float(p) for p in args.expert_probs.split(",")]
    
    print(f"Collecting data for environments: {env_names}")
    print(f"Expert probabilities: {expert_probs}")
    print(f"Dataset size per env and probability: {args.dataset_size}")
    print(f"Image resolution: {args.resolution}x{args.resolution}")
    print(f"Camera angle: {args.camera}")
    print(f"Include depth: {args.include_depth}")
    print(f"Save example image: {args.save_example}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Action repeat: {args.action_repeat}")
    
    collect_dataset(
        env_names=env_names,
        expert_probs=expert_probs,
        dataset_size=args.dataset_size,
        resolution=args.resolution,
        camera=args.camera,
        include_depth=args.include_depth,
        save_example=args.save_example,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat
    )
