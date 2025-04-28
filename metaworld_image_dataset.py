import numpy as np
import os
if 'MUJOCO_GL' not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
print(os.environ["MUJOCO_GL"])
import pickle
import argparse
import cv2
import logging
from pathlib import Path
from PIL import Image
from metaworld_env import *
from metaworld.policies import *
from video import VideoRecorder

# Set up logging
logger = logging.getLogger("metaworld_dataset")

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Helper function to get the policy for a given environment
def get_policy(env_name):
    # Map environment names to their corresponding policy classes
    env_to_policy = {
        'assembly-v2': SawyerAssemblyV2Policy,
        'basketball-v2': SawyerBasketballV2Policy,
        'bin-picking-v2': SawyerBinPickingV2Policy,
        'box-close-v2': SawyerBoxCloseV2Policy,
        'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
        'button-press-topdown-wall-v2': SawyerButtonPressTopdownWallV2Policy,
        'button-press-v2': SawyerButtonPressV2Policy,
        'button-press-wall-v2': SawyerButtonPressWallV2Policy,
        'coffee-button-v2': SawyerCoffeeButtonV2Policy,
        'coffee-pull-v2': SawyerCoffeePullV2Policy,
        'coffee-push-v2': SawyerCoffeePushV2Policy,
        'dial-turn-v2': SawyerDialTurnV2Policy,
        'disassemble-v2': SawyerDisassembleV2Policy,
        'door-close-v2': SawyerDoorCloseV2Policy,
        'door-lock-v2': SawyerDoorLockV2Policy,
        'door-open-v2': SawyerDoorOpenV2Policy,
        'door-unlock-v2': SawyerDoorUnlockV2Policy,
        'drawer-close-v2': SawyerDrawerCloseV2Policy,
        'drawer-open-v2': SawyerDrawerOpenV2Policy,
        'faucet-close-v2': SawyerFaucetCloseV2Policy,
        'faucet-open-v2': SawyerFaucetOpenV2Policy,
        'hammer-v2': SawyerHammerV2Policy,
        'hand-insert-v2': SawyerHandInsertV2Policy,
        'handle-press-side-v2': SawyerHandlePressSideV2Policy,
        'handle-press-v2': SawyerHandlePressV2Policy,
        'handle-pull-side-v2': SawyerHandlePullSideV2Policy,
        'handle-pull-v2': SawyerHandlePullV2Policy,
        'lever-pull-v2': SawyerLeverPullV2Policy,
        'peg-insert-side-v2': SawyerPegInsertionSideV2Policy,
        'peg-unplug-side-v2': SawyerPegUnplugSideV2Policy,
        'pick-out-of-hole-v2': SawyerPickOutOfHoleV2Policy,
        'pick-place-v2': SawyerPickPlaceV2Policy,
        'pick-place-wall-v2': SawyerPickPlaceWallV2Policy,
        'plate-slide-v2': SawyerPlateSlideV2Policy,
        'plate-slide-back-v2': SawyerPlateSlideBackV2Policy,
        'plate-slide-back-side-v2': SawyerPlateSlideBackSideV2Policy,
        'plate-slide-side-v2': SawyerPlateSlideSideV2Policy,
        'push-back-v2': SawyerPushBackV2Policy,
        'push-v2': SawyerPushV2Policy,
        'push-wall-v2': SawyerPushWallV2Policy,
        'reach-v2': SawyerReachV2Policy,
        'reach-wall-v2': SawyerReachWallV2Policy,
        'shelf-place-v2': SawyerShelfPlaceV2Policy,
        'soccer-v2': SawyerSoccerV2Policy,
        'stick-pull-v2': SawyerStickPullV2Policy,
        'stick-push-v2': SawyerStickPushV2Policy,
        'sweep-into-v2': SawyerSweepIntoV2Policy,
        'sweep-v2': SawyerSweepV2Policy,
        'window-close-v2': SawyerWindowCloseV2Policy,
        'window-open-v2': SawyerWindowOpenV2Policy
    }
    
    if env_name not in env_to_policy:
        raise ValueError(f"No policy found for environment {env_name}")
    
    # Return the policy instance
    return env_to_policy[env_name]()

def save_episode_video(frames, env_name, expert_prob, episode_num, save_path):
    """
    Save a video of an episode.
    
    Args:
        frames: List of frames (numpy arrays)
        env_name: Environment name
        expert_prob: Expert probability value
        episode_num: Episode number
        save_path: Path to save the video
    """
    if not frames:
        print("No frames to save")
        return
        
    video_dir = f"{save_path}exp={int(expert_prob*100)}/videos/"
    os.makedirs(video_dir, exist_ok=True)
    
    video_path = f"{video_dir}{env_name}_exp={int(expert_prob*100)}_episode_{episode_num}.mp4"
    
    # Get height and width of the frames
    height, width, _ = frames[0].shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    # Write frames to video
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    video.release()
    print(f"Saved video to {video_path}")

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
        image = np.flipud(image)
        img = Image.fromarray(image.astype(np.uint8))
        
        # Create filename and save
        filename = f"data/exp={int(expert_prob*100)}/examples/{env_name}_img{resolution}_exp={int(expert_prob*100)}_{camera}.png"
        img.save(filename)
        print(f"Saved example image to {filename}")
    else:
        print(f"Could not save image, unexpected shape: {image.shape}")

def collect_dataset(env_names, expert_probs, dataset_size=int(1e6), checkpoint=[3e5], resolution=84, render_resolution=1024, camera='corner', 
                   include_depth=False, save_example=False, frame_stack=1, action_repeat=1, save_path='data/',
                   random_init=True, randomize_goal_and_object_pos=True):
    """
    Collects a dataset of trajectories from MetaWorld environments with image observations.
    
    Args:
        env_names: List of environment names to collect data from
        expert_probs: List of expert probabilities to use
        dataset_size: Number of transitions to collect per environment and expert probability
        checkpoint: Checkpoint to save datsets
        resolution: Image resolution (height and width) for observations
        render_resolution: Resolution for video rendering (for debugging)
        camera: Camera angle to use
        include_depth: Whether to include depth images
        save_example: Whether to save an example image
        frame_stack: Number of consecutive frames to stack together
        action_repeat: Number of times to repeat the same action
        random_init: Whether to randomize the initial hand position
        randomize_goal_and_object_pos: Whether to randomize the goal and object positions
    """

    for expert_prob in expert_probs:
        os.makedirs(f'{save_path}exp={int(expert_prob*100)}', exist_ok=True)
        os.makedirs(f'{save_path}exp={int(expert_prob*100)}/videos', exist_ok=True)  # Create videos directory
        
        for env_name in env_names:
            logger.info(f"\n===== Collecting data for {env_name} with expert_prob={expert_prob} =====")
            logger.info(f"Frame stack: {frame_stack}, Action repeat: {action_repeat}")
            logger.info(f"Random init: {random_init}, Random goal/object pos: {randomize_goal_and_object_pos}")
            logger.info(f"Observation resolution: {resolution}, Render resolution: {render_resolution}")
            
            # Create environment with image observations using DMC
            env = make(env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=42, resolution=resolution, 
                       camera=camera, random_init=random_init, randomize_goal_and_object_pos=randomize_goal_and_object_pos)
            
            
            
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
            
            logger.info(f"Collecting {dataset_size} transitions...")
            
            # Get the expert policy
            policy = get_policy(env_name)

            # For collecting transitions
            transitions_collected = 0
            episode_reward = 0
            successful_trajectories = 0
            old_dataset_length = 0
            episode_count = 0
            
            # Create video recorder for first 3 episodes (only if logging level is DEBUG)
            should_record_video = logger.isEnabledFor(logging.DEBUG)
            if should_record_video:
                video_dir = Path(f"{save_path}exp={int(expert_prob*100)}/videos")
                video_recorder = VideoRecorder(
                    root_dir=video_dir,
                    render_size=render_resolution,
                    fps=30,
                    metaworld=True
                )
            # Initialize video recorder if we're in DEBUG mode
            if should_record_video:
                video_recorder.init(env, enabled=True)
            
            while transitions_collected < dataset_size:
                # Record frame if in debug mode and within first 3 episodes
                if logger.isEnabledFor(logging.DEBUG) and episode_count < 3:
                    video_recorder.record(env)
                
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
                    # Save the video if we're in debug mode and within first 3 episodes
                    if logger.isEnabledFor(logging.DEBUG) and episode_count < 3:
                        video_filename = f"{env_name}_exp={int(expert_prob*100)}_episode_{episode_count}.mp4"
                        video_recorder.save(video_filename)
                        logger.debug(f"Saved video for episode {episode_count}")
                    
                    episode_count += 1
                    
                    # For DMC, we consider a task successful if reward is above a threshold
                    if episode_reward > 0.8:  # Adjust this threshold based on your environment
                        successful_trajectories += 1
                    
                    logger.info(f"Episode {episode_count} ended with total reward: {episode_reward:.4f}, length: {transitions_collected - old_dataset_length}")
                    episode_reward = 0
                    old_dataset_length = transitions_collected
                    
                    # Reset the environment
                    time_step = env.reset()
                    state = time_step.observation
                    proprio_state = time_step.proprio_observation
                    
                    # Initialize video recorder for next episode if needed
                    if logger.isEnabledFor(logging.DEBUG) and episode_count < 3:
                        video_recorder.init(env, enabled=True)
                else:
                    state = next_obs
                
                # Print progress
                if transitions_collected % 10000 == 0:
                    logger.info(f"Collected {transitions_collected}/{dataset_size} transitions")
                
                if len(checkpoint) > 0 and any(s <= transitions_collected for s in checkpoint):
                    # Save only the filled portion of the dataset
                    dataset_slice = {}
                    for key in dataset:
                        dataset_slice[key] = dataset[key][:transitions_collected]
                    
                    dataset_path = f"{save_path}exp={int(expert_prob*100)}/{env_name}_mod2_fs{frame_stack}_ar{action_repeat}_ri{int(random_init)}_rg{int(randomize_goal_and_object_pos)}_exp={int(expert_prob*100)}_ds={transitions_collected}"
                    with open(dataset_path, 'wb') as f:
                        pickle.dump(dataset_slice, f)
                    logger.info(f"Dataset saved to {dataset_path}")
                    
                    # Write successful trajectories information to a text file
                    success_file_path = f"{save_path}exp={int(expert_prob*100)}/successful_image_trajectories.txt"
                    with open(success_file_path, 'a') as f:
                        f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
                    
                    checkpoint.pop(0)  # Remove the checkpoint that was just reached
            
            # Save the final dataset with only the filled portion
            final_dataset = {}
            for key in dataset:
                final_dataset[key] = dataset[key][:transitions_collected]
            
            # Save the dataset
            dataset_path = f"{save_path}exp={int(expert_prob*100)}/{env_name}_mod2_fs{frame_stack}_ar{action_repeat}_ri{int(random_init)}_rg{int(randomize_goal_and_object_pos)}_exp={int(expert_prob*100)}_ds={transitions_collected}"
            with open(dataset_path, 'wb') as f:
                pickle.dump(final_dataset, f)
            
            logger.info(f"Dataset saved to {dataset_path}")
            logger.info(f"Dataset size: {transitions_collected} transitions")
            logger.info(f"Dataset shape - observations: {final_dataset['observations'].shape}")
            logger.info(f"Successful trajectories: {successful_trajectories} out of {episode_count}")
            
            # Write successful trajectories information to a text file
            success_file_path = f"{save_path}exp={int(expert_prob*100)}/successful_image_trajectories.txt"
            with open(success_file_path, 'a') as f:
                f.write(f"{dataset_path}: {successful_trajectories} successful trajectories out of {episode_count}\n")
            
            logger.info(f"Success information saved to {success_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect image-based dataset from DMC environments")
    parser.add_argument("--env_names", type=str, default="push-v2", 
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
    parser.add_argument("--save_path", type=str, default="data/",
                        help="Path to save the dataset")
    parser.add_argument("--random_init", action="store_true", default=True,
                        help="Whether to randomize the initial hand position")
    parser.add_argument("--no_random_init", action="store_false", dest="random_init",
                        help="Disable randomization of the initial hand position")
    parser.add_argument("--randomize_goal", action="store_true", default=True,
                        help="Whether to randomize the goal and object positions")
    parser.add_argument("--no_randomize_goal", action="store_false", dest="randomize_goal",
                        help="Disable randomization of the goal and object positions")
    parser.add_argument("--resolution", type=int, default=84,
                        help="Resolution for observations")
    parser.add_argument("--render_resolution", type=int, default=4096,
                        help="Resolution for video rendering")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    args = parser.parse_args()
    
    # Configure logging based on command line argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    env_names = args.env_names.split(",")

    expert_probs = [float(p) for p in args.expert_probs.split(",")]
    checkpoints = [int(c) for c in args.checkpoint.split(",")] if args.checkpoint else []

    # Create data directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    logger.info(f"Collecting data for environments: {env_names}")
    logger.info(f"Expert probabilities: {expert_probs}")
    logger.info(f"Dataset size per env and probability: {args.dataset_size}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Frame stack: {args.frame_stack}")
    logger.info(f"Action repeat: {args.action_repeat}")
    logger.info(f"Random init: {args.random_init}")
    logger.info(f"Randomize goal: {args.randomize_goal}")
    logger.info(f"Logging level: {args.log_level}")
    
    collect_dataset(
        env_names=env_names,
        expert_probs=expert_probs,
        dataset_size=args.dataset_size,
        checkpoint=checkpoints,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        save_path=args.save_path,
        random_init=args.random_init,
        randomize_goal_and_object_pos=args.randomize_goal,
        resolution=args.resolution,
        render_resolution=args.render_resolution
    )
