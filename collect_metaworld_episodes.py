"""
Collects MetaWorld episodes and saves them directly in the format compatible with ReplayBuffer.
python collect_metaworld_episodes.py --config_name "my_config/10" --env_names "push-v2" --expert_probs "0.1" --num_episodes 10 --no_randomize_goal
"""
import numpy as np
import os
if 'MUJOCO_GL' not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
print(os.environ["MUJOCO_GL"])
import pickle
import argparse
import datetime
import cv2
import logging
import torch  # Add PyTorch import
torch.set_default_dtype(torch.float32)  # Set default tensor type to float32
np.set_printoptions(precision=6, suppress=True)  # Configure NumPy display options
from pathlib import Path
from PIL import Image
from metaworld_env import *
from video import VideoRecorder
import traceback
from metaworld.policies import *

# Set up logging
logger = logging.getLogger("metaworld_episodes")

def check_metaworld_version():
    """Check if v2 environments are available, otherwise use v3."""
    import metaworld
    try:
        # Try to create a v2 environment to check availability
        mt50 = metaworld.MT50(seed=42)
        test_env_name = 'push-v2'
        if test_env_name in mt50.train_classes:
            logger.info("MetaWorld v2 environments detected")
            return 'v2'
    except:
        pass
    
    # If v2 fails, assume v3
    logger.info("MetaWorld v2 not available, using v3 environments")
    return 'v3'

def convert_env_name_to_v3(env_name):
    """Convert v2 environment name to v3."""
    return env_name.replace('-v2', '-v3')

def get_policy(env_name):
    """Load policy based on available MetaWorld version."""
    import metaworld.policies as policies_module
    
    # Extract base environment name (remove version suffix)
    base_name = env_name.replace('-v2', '').replace('-v3', '')
    
    # Create policy class names for both versions
    v2_policy_name = f"Sawyer{base_name.replace('-', '').title()}V2Policy"
    v3_policy_name = f"Sawyer{base_name.replace('-', '').title()}V3Policy"
    
    # Determine which version to try first based on environment name
    if '-v3' in env_name or check_metaworld_version() == 'v3':
        primary_policy = v3_policy_name
        fallback_policy = v2_policy_name
    else:
        primary_policy = v2_policy_name
        fallback_policy = v3_policy_name
    
    # Try primary policy first
    if hasattr(policies_module, primary_policy):
        try:
            policy_class = getattr(policies_module, primary_policy)
            logger.info(f"Using policy: {primary_policy}")
            return policy_class()
        except Exception as e:
            logger.warning(f"Failed to instantiate {primary_policy}: {e}")
    
    # Try fallback policy
    if hasattr(policies_module, fallback_policy):
        try:
            policy_class = getattr(policies_module, fallback_policy)
            logger.warning(f"Using fallback policy: {fallback_policy}")
            return policy_class()
        except Exception as e:
            logger.warning(f"Failed to instantiate fallback {fallback_policy}: {e}")
    
    # If automatic naming fails, try manual mapping for special cases
    manual_mapping = {
        'bin-picking': 'BinPicking',
        'button-press-topdown': 'ButtonPressTopdown', 
        'button-press-topdown-wall': 'ButtonPressTopdownWall',
        'button-press-wall': 'ButtonPressWall',
        'coffee-button': 'CoffeeButton',
        'coffee-pull': 'CoffeePull',
        'coffee-push': 'CoffeePush',
        'dial-turn': 'DialTurn',
        'door-close': 'DoorClose',
        'door-lock': 'DoorLock',
        'door-open': 'DoorOpen',
        'door-unlock': 'DoorUnlock',
        'drawer-close': 'DrawerClose',
        'drawer-open': 'DrawerOpen',
        'faucet-close': 'FaucetClose',
        'faucet-open': 'FaucetOpen',
        'hand-insert': 'HandInsert',
        'handle-press-side': 'HandlePressSide',
        'handle-press': 'HandlePress',
        'handle-pull-side': 'HandlePullSide',
        'handle-pull': 'HandlePull',
        'lever-pull': 'LeverPull',
        'peg-insert-side': 'PegInsertionSide',
        'peg-unplug-side': 'PegUnplugSide',
        'pick-out-of-hole': 'PickOutOfHole',
        'pick-place': 'PickPlace',
        'pick-place-wall': 'PickPlaceWall',
        'plate-slide': 'PlateSlide',
        'plate-slide-back': 'PlateSlideBack',
        'plate-slide-back-side': 'PlateSlideBackSide',
        'plate-slide-side': 'PlateSlideSide',
        'push-back': 'PushBack',
        'push-wall': 'PushWall',
        'reach-wall': 'ReachWall',
        'shelf-place': 'ShelfPlace',
        'stick-pull': 'StickPull',
        'stick-push': 'StickPush',
        'sweep-into': 'SweepInto',
        'window-close': 'WindowClose',
        'window-open': 'WindowOpen'
    }
    
    if base_name in manual_mapping:
        corrected_name = manual_mapping[base_name]
        v2_manual = f"Sawyer{corrected_name}V2Policy"
        v3_manual = f"Sawyer{corrected_name}V3Policy"
        
        for policy_name in [v3_manual, v2_manual]:
            if hasattr(policies_module, policy_name):
                try:
                    policy_class = getattr(policies_module, policy_name)
                    logger.info(f"Using manual mapping policy: {policy_name}")
                    return policy_class()
                except Exception as e:
                    logger.warning(f"Failed to instantiate manual policy {policy_name}: {e}")
    
    raise ValueError(f"Could not find or instantiate any policy for environment {env_name}")

def save_episode_video(frames, video_dir, env_name, expert_prob, episode_num):
    """
    Save a video of an episode.
    
    Args:
        frames: List of frames (numpy arrays)
        video_dir: Directory to save the video
        env_name: Environment name
        expert_prob: Expert probability value
        episode_num: Episode number
    """
    if not frames:
        print("No frames to save")
        return
    
    video_path = f"{video_dir}/{env_name}_exp={int(expert_prob*100)}_episode_{episode_num}.mp4"
    
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
    logger.debug(f"Saved video to {video_path}")

def collect_episodes(config_name, env_names, expert_probs, tasks=[0], num_episodes=1000, 
                    resolution=84, render_resolution=1024, camera='corner', 
                    include_depth=False, save_example=False, frame_stack=1, action_repeat=1,
                    random_init=True, randomize_goal_and_object_pos=True, stop_on_success = False, render_episodes=1):
    """
    Collects episodes from MetaWorld environments with image observations and saves them
    directly in the format compatible with ReplayBuffer.
    
    Args:
        config_name: Name of the configuration used to organize episodes
        env_names: List of environment names to collect data from
        expert_probs: List of expert probabilities to use
        tasks: List of task indices to collect data from
        num_episodes: Number of episodes to collect per environment and expert probability
        resolution: Image resolution (height and width) for observations
        render_resolution: Resolution for video rendering (for debugging)
        camera: Camera angle to use
        include_depth: Whether to include depth images
        save_example: Whether to save an example image
        frame_stack: Number of consecutive frames to stack together
        action_repeat: Number of times to repeat the same action
        random_init: Whether to randomize the initial hand position
        randomize_goal_and_object_pos: Whether to randomize the goal and object positions
        stop_on_success: Whether to stop the episode when success is achieved
        render_episodes: Number of episodes to render for video
    """
    for expert_prob in expert_probs:
        for env_name in env_names:
            for task in tasks:
                logger.info(f"\n===== Collecting episodes for {env_name} and task {task} with expert_prob={expert_prob} =====")
                logger.info(f"Frame stack: {frame_stack}, Action repeat: {action_repeat}")
                logger.info(f"Random init: {random_init}, Random goal/object pos: {randomize_goal_and_object_pos}, Stop on success: {stop_on_success}")
                
                # Create dataset name from parameters
                dataset_name = f"{env_name}_task{task}_fs{frame_stack}_ar{action_repeat}_ri{int(random_init)}_rg{int(randomize_goal_and_object_pos)}_exp={int(expert_prob*100)}"
                
                # Create output directory structure
                output_dir = Path(f"data_episodes/{config_name}/pretraining_datasets/{dataset_name}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if episodes already exist in this directory
                existing_episodes = list(output_dir.glob('*.npz'))
                if existing_episodes:
                    episode_count = len(existing_episodes)
                    logger.info(f"Found {episode_count} existing episodes in {output_dir}")
                    logger.info(f"Skipping collection for {env_name}, task {task}, expert_prob={expert_prob}")
                    
                    # Add a note to the successful trajectories file about the skipped run
                    success_file_path = output_dir / "successful_trajectories.txt"
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(success_file_path, 'a') as f:
                        f.write(f"\n[{timestamp}] Collection skipped - {episode_count} episodes already exist in directory\n")
                    
                    logger.info(f"Added skip note to {success_file_path}")
                    continue
                
                # Create videos directory if needed
                videos_dir = output_dir / "videos"
                videos_dir.mkdir(exist_ok=True)
                
                # Create environment with image observations 
                env = make(env_name, task, frame_stack=frame_stack, action_repeat=action_repeat, seed=42, resolution=resolution, 
                        camera=camera, random_init=random_init, randomize_goal_and_object_pos=randomize_goal_and_object_pos, data_collection=stop_on_success)
                
                # Get the expert policy
                policy = get_policy(env_name)
                
                # For tracking progress
                episodes_collected = 0
                successful_trajectories = 0
                
                # Create video recorder for first 3 episodes (only if logging level is DEBUG)
                should_record_video = logger.isEnabledFor(logging.DEBUG)
                if should_record_video:
                    video_recorder = VideoRecorder(
                        root_dir=videos_dir,
                        render_size=render_resolution,
                        fps=30,
                        metaworld=True
                    )
                
                while episodes_collected < num_episodes:
                    # Reset environment
                    time_step = env.reset()
                    state = time_step.observation
                    proprio_state = time_step.proprio_observation
                    
                    # Initialize episode data structures
                    episode = {
                        'observation': [],
                        'action': [],
                        'reward': [],
                        'discount': [],
                        'terminal': []
                    }
                    
                    # Track episode information
                    episode_frames = []
                    episode_reward = 0
                    episode_success = False
                    episode_length = 0
                    
                    # Initialize video recorder if we're in DEBUG mode
                    if should_record_video and episodes_collected < render_episodes:
                        video_recorder.init(env, enabled=True)
                    # add dummy transition to the episode   
                    episode['observation'].append(state)
                    episode['action'].append(np.zeros(env.action_space.shape, dtype=np.float32))
                    episode['reward'].append(torch.tensor(0.0, dtype=torch.float32))
                    episode['discount'].append(torch.tensor(1.0, dtype=torch.float32))
                    episode['terminal'].append(False)
            
                    
                    # Continue until episode ends or reaches a maximum length (if the env doesn't have a built-in limit)
                    max_episode_length = 1000  # Safety limit
                    done = False
                    
                    while not done and episode_length < max_episode_length:
                        # Record frame if in debug mode and within first 3 episodes
                        if should_record_video and episodes_collected < render_episodes:
                            video_recorder.record(env)
                            
                        # Choose between expert or random action based on probability
                        if np.random.random() < expert_prob:
                            # Expert policy action
                            a = policy.get_action(proprio_state)
                            # Convert action to float32 if it's not already
                            if not isinstance(a, np.ndarray) or a.dtype != np.float32:
                                a = np.array(a, dtype=np.float32)
                        else:
                            # Random action
                            a = env.action_space.sample().astype(np.float32)
                            
                        # Take a step in the environment
                        time_step = env.step(a)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        discount = time_step.discount
                        done = time_step.last()
                        proprio_state = time_step.proprio_observation
                        
                        # Store transition in episode
                        episode['observation'].append(next_obs)
                        episode['action'].append(a)
                        episode['reward'].append(torch.tensor(reward, dtype=torch.float32))
                        episode['discount'].append(torch.tensor(discount, dtype=torch.float32))
                        episode['terminal'].append(done)
                        
                        episode_reward += reward
                        episode_length += 1
                        if time_step.success == 1:
                            episode_success = True

                        if done:
                            if episode_success:
                                successful_trajectories += 1
                            break
                    
                    # Convert episode data to numpy arrays
                    for key in episode:
                        if key in ['reward', 'discount']:
                            # Convert torch tensors to numpy float32 arrays
                            episode[key] = np.array([r.numpy() if isinstance(r, torch.Tensor) else r 
                                                for r in episode[key]], dtype=np.float32)
                        else:
                            # Ensure other arrays are float32 if they contain floating point values
                            temp_array = np.array(episode[key])
                            if np.issubdtype(temp_array.dtype, np.floating):
                                temp_array = temp_array.astype(np.float32)
                            episode[key] = temp_array
                    
                    # Reshape reward and discount to have shape [n, 1]
                    episode['reward'] = episode['reward'].reshape(-1, 1)
                    episode['discount'] = episode['discount'].reshape(-1, 1)
                    
        
                    # Save the episode only if it's not empty
                    if episode_length > 10:
                        # Generate filename with timestamp
                        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                        filename = f'{ts}_{episodes_collected}_{episode_length}.npz'
                        
                        # Save episode
                        np_file_path = output_dir / filename
                        with open(np_file_path, 'wb') as f:
                            np.savez_compressed(f, **episode)
                        
                        logger.info(f"Episode {episodes_collected}: Reward: {episode_reward:.4f}, Length: {episode_length}, Success: {episode_success}")
                        episodes_collected += 1
                        
                        # Save video for first 3 episodes if in debug mode
                        if should_record_video and episodes_collected <= render_episodes:
                            video_filename = f"{env_name}_exp={int(expert_prob*100)}_episode_{episodes_collected-1}.mp4"
                            video_recorder.save(video_filename)
                    else:
                        logger.warning("Skipping empty episode")
                
                env.close()
                try:
                    import mujoco
                    mujoco.MjRenderContext.release_all_contexts()
                    logger.info("Released all MuJoCo rendering contexts")
                except (ImportError, AttributeError):
                    logger.warning("Could not explicitly release MuJoCo rendering contexts")
                
                logger.info(f"Collected {episodes_collected} episodes in {output_dir}")
                logger.info(f"Successful trajectories: {successful_trajectories} out of {episodes_collected}")
                
                # Write successful trajectories information to a text file
                success_file_path = output_dir / "successful_trajectories.txt"
                with open(success_file_path, 'w') as f:
                    f.write(f"Successful trajectories: {successful_trajectories} out of {episodes_collected}\n")
                
                logger.info(f"Success information saved to {success_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect image-based episodes from MetaWorld environments")
    parser.add_argument("--config_name", type=str, required=True, 
                        help="Configuration name for organizing episodes")
    parser.add_argument("--env_names", type=str, default="push-v2", 
                        help="Comma-separated list of environment names")
    parser.add_argument("--tasks", type=str, default="0", 
                        help="Comma-separated list of tasks for environment names (format: 0,1,2,...,n)")
    parser.add_argument("--expert_probs", type=str, default="0.0", 
                        help="Comma-separated list of expert probabilities (use 0.0 for random policy)")
    parser.add_argument("--num_episodes", type=int, default=15, 
                        help="Number of episodes to collect per environment and expert probability")
    parser.add_argument("--frame_stack", type=int, default=3,
                        help="Number of consecutive frames to stack")
    parser.add_argument("--action_repeat", type=int, default=2,
                        help="Number of times to repeat the same action")
    parser.add_argument("--random_init", action="store_true", default=True,
                        help="Whether to randomize the initial hand position")
    parser.add_argument("--no_random_init", action="store_false", dest="random_init",
                        help="Disable randomization of the initial hand position")
    parser.add_argument("--randomize_goal", action="store_true", default=True,
                        help="Whether to randomize the goal and object positions")
    parser.add_argument("--no_randomize_goal", action="store_false", dest="randomize_goal",
                        help="Disable randomization of the goal and object positions")
    parser.add_argument("--stop_on_success", action="store_true",
                        help="Whether to stop the episode when success")
    parser.add_argument("--resolution", type=int, default=84,
                        help="Resolution for observations")
    parser.add_argument("--render_resolution", type=int, default=1024,
                        help="Resolution for video rendering")
    parser.add_argument("--render_episodes", type=int, default=1,
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
    tasks = [int(p) for p in args.tasks.split(",")]
    
    assert len(env_names) == 1 and len(expert_probs) == 1 and len(tasks) == 1, "Only one environment, expert probability, and task should be specified for this script. MAJOR BUF FOR MORE TASKS, EXPERT PROBABILITIES, AND ENVIRONMENTS"
    logger.info(f"Config name: {args.config_name}")
    logger.info(f"Collecting data for environments: {env_names}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Expert probabilities: {expert_probs}")
    logger.info(f"Number of episodes per env and probability: {args.num_episodes}")
    logger.info(f"Frame stack: {args.frame_stack}")
    logger.info(f"Action repeat: {args.action_repeat}")
    logger.info(f"Random init: {args.random_init}")
    logger.info(f"Randomize goal: {args.randomize_goal}")
    logger.info(f"Stop on success: {args.stop_on_success}")
    logger.info(f"Logging level: {args.log_level}")
    
    collect_episodes(
        config_name=args.config_name,
        env_names=env_names,
        expert_probs=expert_probs,
        tasks=tasks,
        num_episodes=args.num_episodes,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        random_init=args.random_init,
        randomize_goal_and_object_pos=args.randomize_goal,
        resolution=args.resolution,
        render_resolution=args.render_resolution,
        stop_on_success=args.stop_on_success,
        render_episodes=args.render_episodes
    )
