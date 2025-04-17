import numpy as np
import os
os.environ['MUJOVCO_GL'] = 'osmesa'
import pickle
import argparse
from metaworld_env2 import *
from metaworld.policies import *

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

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

def collect_dataset(env_names, expert_probs, dataset_size=int(1e6), checkpoint=[3e5],  resolution=84, camera='corner', 
                   include_depth=False, save_example=False, frame_stack=1, action_repeat=1):
    """
    Collects a dataset of trajectories from MetaWorld environments with image observations.
    
    Args:
        env_names: List of environment names to collect data from
        expert_probs: List of expert probabilities to use
        dataset_size: Number of transitions to collect per environment and expert probability
        checkpoint: Checkpoint to save datsets
        resolution: Image resolution (height and width)
        camera: Camera angle to use
        include_depth: Whether to include depth images
        save_example: Whether to save an example image
        frame_stack: Number of consecutive frames to stack together
        action_repeat: Number of times to repeat the same action
    """

    for expert_prob in expert_probs:
        os.makedirs(f'data/exp={int(expert_prob*100)}', exist_ok=True)
        for env_name in env_names:
            print(f"\n===== Collecting data for {env_name} with expert_prob={expert_prob} =====")
            print(f"Frame stack: {frame_stack}, Action repeat: {action_repeat}")
            
            # Create environment with image observations using DMC
            env = make(env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=42, resolution=resolution, camera=camera)
            
            # Get initial timestep
            time_step = env.reset()
            state = time_step.observation  # This is already stacked and processed
            proprio_state = time_step.proprio_observation
            
            # Pre-determine dataset shapes based on first observation
            obs_shape = (dataset_size,) + state.shape
            action_shape = (dataset_size,) + env.action_space.shape
            
            # Initialize the dataset dictionary with pre-allocated numpy arrays
            dataset = {
                'observations': np.zeros(obs_shape, dtype=np.uint8),
                'actions': np.zeros(action_shape, dtype=np.float32),
                'next_observations': np.zeros(obs_shape, dtype=np.uint8),
                'rewards': np.zeros(dataset_size, dtype=np.float32),
                'terminals': np.zeros(dataset_size, dtype=bool)
            }
            
            print(f"Collecting {dataset_size} transitions...")
            
            # Get the expert policy
            policy = get_policy(env_name)


            # For collecting transitions
            transitions_collected = 0
            episode_reward = 0
            episode_count = 0
            successful_trajectories = 0
            old_dataset_length = 0
            
            while transitions_collected < dataset_size:
                # Choose between expert or random action based on probability
                if np.random.random() < expert_prob:
                    # Expert policy action
                    a = policy.get_action(proprio_state)
                else:
                    # Random action
                    a = env.action_space.sample()
                
                # Store current observation
                curr_obs = state.copy()
                
                # Take a step in the environment
                time_step = env.step(a)
                next_obs = time_step.observation
                reward = time_step.reward
                done = time_step.last()
                proprio_state = time_step.proprio_observation
                
                # Store the transition in the dataset using indexing
                dataset['observations'][transitions_collected] = curr_obs
                dataset['actions'][transitions_collected] = a
                dataset['next_observations'][transitions_collected] = next_obs
                dataset['rewards'][transitions_collected] = reward
                dataset['terminals'][transitions_collected] = done
                
                transitions_collected += 1
                episode_reward += reward
                
                if done:
                    episode_count += 1
                    
                    # For DMC, we consider a task successful if reward is above a threshold
                    if episode_reward > 0.8:  # Adjust this threshold based on your environment
                        successful_trajectories += 1
                    
                    print(f"Episode {episode_count} ended with total reward: {episode_reward:.4f}, length: {transitions_collected - old_dataset_length}")
                    episode_reward = 0
                    old_dataset_length = transitions_collected
                    
                    # Reset the environment
                    time_step = env.reset()
                    state = time_step.observation
                    proprio_state = time_step.proprio_observation
                else:
                    state = next_obs
                
                # Print progress
                if transitions_collected % 10000 == 0:
                    print(f"Collected {transitions_collected}/{dataset_size} transitions")
                
                if len(checkpoint) > 0 and any(s <= transitions_collected for s in checkpoint):
                    # Save only the filled portion of the dataset
                    dataset_slice = {}
                    for key in dataset:
                        dataset_slice[key] = dataset[key][:transitions_collected]
                    
                    dataset_path = f"data/exp={int(expert_prob*100)}/{env_name}_mod2_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}_ds={transitions_collected}"
                    with open(dataset_path, 'wb') as f:
                        pickle.dump(dataset_slice, f)
                    print(f"Dataset saved to {dataset_path}")
                    
                    # Write successful trajectories information to a text file
                    success_file_path = f"data/exp={int(expert_prob*100)}/successful_image_trajectories.txt"
                    with open(success_file_path, 'a') as f:
                        f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
                    
                    checkpoint.pop(0)  # Remove the checkpoint that was just reached
            
            # Save the final dataset with only the filled portion
            final_dataset = {}
            for key in dataset:
                final_dataset[key] = dataset[key][:transitions_collected]
            
            # Save the dataset
            dataset_path = f"data/exp={int(expert_prob*100)}/{env_name}_mod2_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}__ds={transitions_collected}"
            with open(dataset_path, 'wb') as f:
                pickle.dump(final_dataset, f)
            
            print(f"Dataset saved to {dataset_path}")
            print(f"Dataset size: {transitions_collected} transitions")
            print(f"Dataset shape - observations: {final_dataset['observations'].shape}")
            print(f"Successful trajectories: {successful_trajectories} out of {episode_count}")
            
            # Write successful trajectories information to a text file
            success_file_path = f"data/exp={int(expert_prob*100)}/successful_image_trajectories.txt"
            with open(success_file_path, 'a') as f:
                f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
            
            print(f"Success information saved to {success_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect image-based dataset from DMC environments")
    parser.add_argument("--env_names", type=str, default="quadruped_walk", 
                        help="Comma-separated list of environment names (format: domain_task)")
    parser.add_argument("--expert_probs", type=str, default="0.0", 
                        help="Comma-separated list of expert probabilities (use 0.0 for random policy)")
    parser.add_argument("--dataset_size", type=int, default=200000, 
                        help="Number of transitions to collect per environment and expert probability")
    parser.add_argument("--frame_stack", type=int, default=3,
                        help="Number of consecutive frames to stack")
    parser.add_argument("--action_repeat", type=int, default=2,
                        help="Number of times to repeat the same action")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Comma-separated list of checkpoints to save datasets at")
    
    args = parser.parse_args()
    
    env_names = args.env_names.split(",")

    expert_probs = [float(p) for p in args.expert_probs.split(",")]
    checkpoints = [int(c) for c in args.checkpoint.split(",")] if args.checkpoint else []
    
    print(f"Collecting data for environments: {env_names}")
    print(f"Expert probabilities: {expert_probs}")
    print(f"Dataset size per env and probability: {args.dataset_size}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Action repeat: {args.action_repeat}")
    
    collect_dataset(
        env_names=env_names,
        expert_probs=expert_probs,
        dataset_size=args.dataset_size,
        checkpoint=checkpoints,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat
    )
