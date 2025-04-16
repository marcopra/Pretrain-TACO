import numpy as np
import os
os.environ['MUJOVCO_GL'] = 'osmesa'
import pickle
import argparse
import metaworld
from metaworld_env import *
from metaworld.policies import *
from PIL import Image
from collections import deque
# python metaworld_image_dataset.py --env_names push-v2,door-open-v2 --expert_probs 0.8,0.0 --dataset_size 100000 --resolution 84 --camera corner

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
# os.makedirs('data/examples', exist_ok=True)  # Create directory for example images

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

# def save_example_image(image, env_name, expert_prob, resolution, camera):
#     """
#     Save a single example image as PNG.
    
#     Args:
#         image: RGB array of shape (height, width, 3)
#         env_name: Environment name
#         expert_prob: Expert probability value
#         resolution: Image resolution
#         camera: Camera angle used
#     """
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         # Convert numpy array to PIL Image
#         image = np.flipud(image)
#         img = Image.fromarray(image.astype(np.uint8))
        
#         # Create filename and save
#         filename = f"data/exp={int(expert_prob*100)}/examples/{env_name}_img{resolution}_exp={int(expert_prob*100)}_{camera}.png"
#         img.save(filename)
#         print(f"Saved example image to {filename}")
#     else:
#         print(f"Could not save image, unexpected shape: {image.shape}")

def collect_dataset(env_names, expert_probs, dataset_size=int(1e6), checkpoint = [3e5], resolution=84, camera='corner', 
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
    if action_repeat > 1:
        raise NotImplementedError("Action repeat > 1 is implemented yet, but I m not sure it is necessary ")
    
    mt10 = metaworld.MT10(seed=42)
    for expert_prob in expert_probs:
        os.makedirs(f'data/exp={int(expert_prob*100)}', exist_ok=True)
        for env_name in env_names:
            print(f"\n===== Collecting data for {env_name} with expert_prob={expert_prob} =====")
            print(f"Frame stack: {frame_stack}, Action repeat: {action_repeat}")
            
            task_number = 0
            env_task_indices = [i for i, task in enumerate(mt10.train_tasks) if task.env_name == env_name]
            if task_number >= len(env_task_indices):
                print(f"Task number {task_number} out of range. Available tasks: 0-{len(env_task_indices)-1}")
                return
            
            # Get the task index
            task_idx = env_task_indices[task_number]
            # Create environment with image observations using PLEX-MetaWorld
            env = mt10.train_classes[env_name](render_mode="rgb_array", camera_name="corner")
            env = RandomizeInitialPositionWrapper(env)
            env = ResizeRendering(env, resolution=resolution)
            env.set_task(mt10.train_tasks[task_idx])
            state, info = env.reset()
            
            # Get the expert policy
            policy = get_policy(env_name)
            
            # Pre-render a single frame to determine shapes
            image = env.render()
            obs_shape = (dataset_size, 3 * frame_stack, resolution, resolution)  # Update shape to (c, h, w)
            state_shape = (dataset_size,) + np.array(state).shape
            action_shape = (dataset_size,) + env.action_space.shape
            
            # Initialize the dataset dictionary with pre-allocated numpy arrays
            dataset = {
                'observations': np.zeros(obs_shape, dtype=np.uint8),
                'proprio_states': np.zeros(state_shape, dtype=np.float32),
                'actions': np.zeros(action_shape, dtype=np.float32),
                'next_observations': np.zeros(obs_shape, dtype=np.uint8),
                'next_proprio_states': np.zeros(state_shape, dtype=np.float32),
                'rewards': np.zeros(dataset_size, dtype=np.float32),
                'terminals': np.zeros(dataset_size, dtype=bool)
            }
            
            print(f"Collecting {dataset_size} transitions...")
            
            # For collecting transitions
            transitions_collected = 0
            episode_reward = 0
            episode_count = 0
            successful_trajectories = 0
            old_dataset_length = 0
            example_saved = False
            episode_video = []
            
            # Initialize frame stack with the first observation
            frame_buffer = deque(maxlen=frame_stack)
            image = env.render()
            episode_video.append(np.flipud(image))
            for _ in range(frame_stack):
                frame_buffer.append(image)
            
            while transitions_collected < dataset_size:
                # Choose between expert or random action based on probability
                if np.random.random() < expert_prob:
                    # Expert policy action
                    a = policy.get_action(state)
                else:
                    # Random action
                    a = env.action_space.sample()
                
                # Store the current stacked observation before taking action
                stacked_obs = np.concatenate(list(frame_buffer), axis=-1) if frame_stack > 1 else frame_buffer[0]
                stacked_obs = np.transpose(stacked_obs, (2, 0, 1))  # Transpose to (c, h, w)
                curr_state = state.copy()
                
                # Initialize variables for action repeat
                total_reward = 0
                discount = 1.0
                
                # Apply action repeat
                for _ in range(action_repeat):
                    
                    next_state, reward, truncation, termination, info = env.step(a)
                    episode_video.append(np.flipud(env.render()))
                    
                    # # Save example image if requested and not already saved
                    # if save_example and not example_saved and transitions_collected > 10:
                    #     if frame_stack > 1:
                    #         # Save the stacked image as separate files
                    #         for i, img in enumerate(frame_buffer):
                    #             save_example_image(img, f"{env_name}_frame{i}", expert_prob, resolution, camera)
                    #     else:
                    #         save_example_image(env.render(), env_name, expert_prob, resolution, camera)
                    #     example_saved = True
                    
                    # Update frame buffer with new observation
                    frame_buffer.append(env.render())
                    
                    # Accumulate reward with discount
                    total_reward += reward * discount
                    discount *= 0.99  # Use a discount factor for repeated actions
                    
                    done = truncation or termination or int(info['success']) == 1
                    # If done, break out of the loop
                    if done:
                        break
                
                # Create the stacked next observation
                next_stacked_obs = np.concatenate(list(frame_buffer), axis=-1) if frame_stack > 1 else frame_buffer[-1]
                next_stacked_obs = np.transpose(next_stacked_obs, (2, 0, 1))  # Transpose to (c, h, w)
                
                # Store the transition in the dataset using indexing
                dataset['observations'][transitions_collected] = stacked_obs
                dataset['proprio_states'][transitions_collected] = curr_state
                dataset['actions'][transitions_collected] = a
                dataset['next_observations'][transitions_collected] = next_stacked_obs
                dataset['next_proprio_states'][transitions_collected] = next_state
                dataset['rewards'][transitions_collected] = total_reward
                dataset['terminals'][transitions_collected] = done
                
                transitions_collected += 1
                episode_reward += total_reward
                
                if done:
                    episode_count += 1
                    path = f"data/exp={int(expert_prob*100)}/{env_name}_img{resolution}_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}_video_{episode_count}.mp4"
                    # imageio.mimsave(path, episode_video, fps=30)
                    
                    # Check if the trajectory was successful
                    if 'task_accomplished' in info and info['task_accomplished'] or \
                       'success' in info and int(info['success']) == 1:
                        successful_trajectories += 1
                    
                    print(f"Episode {episode_count} ended with total reward: {episode_reward:.4f}, length: {transitions_collected - old_dataset_length}")
                    episode_reward = 0
                    old_dataset_length = transitions_collected
                    
                    # Reset the environment and frame buffer
                    episode_video = []
                    state, _ = env.reset()
                    frame_buffer.clear()
                    image = env.render()
                    episode_video.append(np.flipud(image))
                    for _ in range(frame_stack):
                        frame_buffer.append(image)
                else:
                    state = next_state
                
                # Print progress
                if transitions_collected % 10000 == 0:
                    print(f"Collected {transitions_collected}/{dataset_size} transitions")
                
                if len(checkpoint) > 0 and any(s <= transitions_collected for s in checkpoint):
                    # Save only the filled portion of the dataset
                    dataset_slice = {}
                    for key in dataset:
                        dataset_slice[key] = dataset[key][:transitions_collected]
                    
                    dataset_path = f"data/exp={int(expert_prob*100)}/{env_name}_img{resolution}_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}_ds={transitions_collected}"
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
            dataset_path = f"data/exp={int(expert_prob*100)}/{env_name}_img{resolution}_fs{frame_stack}_ar{action_repeat}_exp={int(expert_prob*100)}__ds={transitions_collected}"
            with open(dataset_path, 'wb') as f:
                pickle.dump(final_dataset, f)
            
            print(f"Dataset saved to {dataset_path}")
            print(f"Dataset size: {transitions_collected} transitions")
            print(f"Dataset shape - observations: {final_dataset['observations'].shape}")
            print(f"Successful trajectories: {successful_trajectories} out of {episode_count}")
            
            # Write successful trajectories information to a text file
            success_file_path = "data/exp={int(expert_prob*100)}/successful_image_trajectories.txt"
            with open(success_file_path, 'a') as f:
                f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
            
            print(f"Success information saved to {success_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect image-based dataset from MetaWorld environments")
    parser.add_argument("--env_names", type=str, default="push-v2", #,window-open-v2,door-open-v2,reach-v2,drawer-close-v2,pick-place-v2,drawer-open-v2,button-press-topdown-v2,window-close-v2,peg-insert-side-v2", 
                        help="Comma-separated list of environment names")
    parser.add_argument("--expert_probs", type=str, default="0.8", 
                        help="Comma-separated list of expert probabilities")
    parser.add_argument("--dataset_size", type=int, default=200000, 
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
        checkpoint=checkpoints,
        resolution=args.resolution,
        camera=args.camera,
        include_depth=args.include_depth,
        save_example=args.save_example,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat
    )
