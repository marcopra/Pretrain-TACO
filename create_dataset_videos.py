import os
import numpy as np
import pickle
import cv2
import argparse
import re
from pathlib import Path

def extract_episodes(dataset, max_episode_length=None):
    """
    Extract episodes from the dataset based on terminal flags.
    
    Args:
        dataset: Dataset dictionary with observations and terminals
        max_episode_length: Maximum length of non-terminal episodes (None for no limit)
        
    Returns:
        List of episodes, where each episode is a list of observations
    """
    episodes = []
    current_episode = []
    
    for i in range(len(dataset['terminals'])):
        obs = dataset['observations'][i]
        current_episode.append(obs)
        
        if dataset['terminals'][i]:
            # Terminal state reached, end of episode
            episodes.append(current_episode)
            current_episode = []
        elif max_episode_length and len(current_episode) >= max_episode_length:
            # Episode exceeded maximum length, split it
            episodes.append(current_episode)
            current_episode = []
    
    # Add the last episode if it's not empty and doesn't end with terminal
    if current_episode:
        episodes.append(current_episode)
    
    return episodes

def create_episode_video(observations, output_path, episode_idx, resolution=256, frame_stack=1, save_as_png=False):
    """
    Create a video or save frames as PNG from an episode's observations.
    
    Args:
        observations: List of observation frames for the episode
        output_path: Base path for the output video or frames
        episode_idx: Episode index
        resolution: Video resolution
        frame_stack: Number of frames stacked in each observation
        save_as_png: If True, save individual frames as PNG instead of creating a video
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    frames = []
    
    # Process frames based on frame stacking
    sample_obs = observations[0]
    print(f"Observation shape: {sample_obs.shape}, dtype: {sample_obs.dtype}")
    
    for obs in observations:
        print(obs)
        # Handle channels-first format (C, H, W)
        if len(obs.shape) == 3 and obs.shape[0] == 3 * frame_stack:
            print(f"Detected channels-first format with stacked frames: {obs.shape}")
            # For frame_stack=3, shape would be (9, H, W)
            # Extract most recent frame (last 3 channels) and convert to channels-last
            frame = obs[-3:].transpose(1, 2, 0)  # Convert from (3, H, W) to (H, W, 3)
        elif len(obs.shape) == 3 and obs.shape[0] > 3 and obs.shape[0] % 3 == 0:
            # Generic case for channels-first with multiple stacked frames
            print(f"Handling generic channels-first format: {obs.shape}")
            # Extract most recent frame and convert to channels-last
            frame = obs[-3:].transpose(1, 2, 0)
        # Check if this is a stacked observation in channels-last format
        elif frame_stack > 1 and len(obs.shape) == 3 and obs.shape[2] == 3 * frame_stack:
            # Extract the most recent frame (last 3 channels)
            frame = obs[:, :, -3:]
        # If the shape is (H, W) - grayscale image
        elif len(obs.shape) == 2:
            # Convert grayscale to RGB
            frame = np.stack([obs, obs, obs], axis=2)
        # If the observation has an unusual shape that might need transposition
        elif len(obs.shape) == 3 and (obs.shape[0] == resolution or obs.shape[0] == 84):
            # This might be a transposed observation with shape (84, 84, 9) or similar
            print(f"Handling transposed observation: {obs.shape}")
            if obs.shape[2] >= 3:
                # If depth/channels is 3 or more, it might be a misaligned RGB image
                frame = obs
            else:
                # If first dimension is resolution, this might be a channels-first grayscale
                # Transpose and convert to RGB
                frame = np.stack([obs[:, :, 0], obs[:, :, 0], obs[:, :, 0]], axis=2)
        # If the resolution equals the number of channels - transposed observation
        elif len(obs.shape) == 3 and (obs.shape[2] == resolution or obs.shape[2] > 10):
            # This might be a transposed observation
            print(f"Unusual observation shape detected: {obs.shape}. Attempting to reshape...")
            # Attempt to reshape to a proper image format
            if obs.shape[2] == resolution:
                # Might be a square grayscale image in wrong orientation
                frame = obs.reshape(resolution, resolution, 1)
                # Convert grayscale to RGB
                frame = np.concatenate([frame, frame, frame], axis=2)
            else:
                # If not sure, convert to a displayable format
                # Use only first 3 channels or average across channels to create RGB
                if obs.shape[2] >= 3:
                    frame = obs[:, :, :3]  # Take first 3 channels
                else:
                    # Average across channels to get a single channel, then duplicate for RGB
                    avg_channel = np.mean(obs, axis=2, keepdims=True)
                    frame = np.concatenate([avg_channel, avg_channel, avg_channel], axis=2)
        else:
            # Standard RGB image
            frame = obs
            
        print(f"Processed frame shape: {frame.shape}, dtype: {frame.dtype}, frame_stack: {frame}")
        # Make sure frame is 3-channel for video
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # Convert grayscale to RGB
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]  # Extract the single channel
            frame = np.stack([frame, frame, frame], axis=2)
            
        # Ensure we have 3 channels for video
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Warning: Unusual frame shape after processing: {frame.shape}. Attempting to fix.")
            # Handle various cases
            if len(frame.shape) == 3 and frame.shape[2] > 3:
                # Too many channels, take first 3
                frame = frame[:, :, :3]
            else:
                # Create placeholder frame with correct shape
                placeholder = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                # Try to embed original data in some way
                if len(frame.shape) >= 2:
                    h, w = min(resolution, frame.shape[0]), min(resolution, frame.shape[1])
                    if len(frame.shape) == 3:
                        c = min(3, frame.shape[2])
                        placeholder[:h, :w, :c] = frame[:h, :w, :c]
                    else:
                        placeholder[:h, :w, 0] = frame[:h, :w]
                        placeholder[:h, :w, 1] = frame[:h, :w]
                        placeholder[:h, :w, 2] = frame[:h, :w]
                frame = placeholder
        
        # Resize the frame if necessary
        if frame.shape[0] < resolution or frame.shape[1] < resolution:
            frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        
        # Ensure frame is uint8 for video encoding
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        frames.append(frame)
    
    # Define the output path
    if save_as_png:
        # Create a directory for frame images
        frames_dir = f"{output_path}_ep{episode_idx}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save each frame as PNG
        for i, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
        
        print(f"Saved {len(frames)} frames to {frames_dir}")
    else:
        # Define the output video path
        video_path = f"{output_path}_ep{episode_idx}.mp4"
        
        # Create video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        
        # Write frames to video
        for frame in frames:
            # Convert RGB to BGR for OpenCV if needed
            if frame.shape[2] == 3:  # Make sure it's a 3-channel image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                print(f"Warning: Frame has unexpected number of channels: {frame.shape[2]}. Writing as is.")
            video.write(frame)
        
        video.release()
        print(f"Saved video to {video_path}")

def parse_dataset_path(dataset_path):
    """
    Parse the dataset path to extract metadata.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary with parsed metadata
    """
    # Extract meaningful parts from the path
    basename = os.path.basename(dataset_path)
    
    # Extract parameters using regex
    env_match = re.search(r'([a-z-]+)_task(\d+)', basename)
    env_name = env_match.group(1) if env_match else "unknown"
    task_id = env_match.group(2) if env_match else "0"
    
    frame_stack = int(re.search(r'fs(\d+)', basename).group(1)) if re.search(r'fs(\d+)', basename) else 1
    action_repeat = int(re.search(r'ar(\d+)', basename).group(1)) if re.search(r'ar(\d+)', basename) else 1
    
    return {
        'env_name': env_name,
        'task_id': task_id,
        'frame_stack': frame_stack,
        'action_repeat': action_repeat,
        'basename': basename
    }

def create_videos(dataset_path, resolution=256, max_episodes=None, max_episode_length=None, save_as_png=False):
    """
    Create videos for episodes in the dataset.
    
    Args:
        dataset_path: Path to the dataset file
        resolution: Resolution of the output videos
        max_episodes: Maximum number of episodes to process (None for all)
        max_episode_length: Maximum length of non-terminal episodes (None for no limit)
        save_as_png: If True, save individual frames as PNG instead of creating a video
    """
    print(f"Loading dataset from {dataset_path}...")
    # Load the dataset
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get metadata from dataset path
    metadata = parse_dataset_path(dataset_path)
    print(f"Processing dataset with metadata: {metadata}")
    print(f"Frame stack: {metadata['frame_stack']}, Action repeat: {metadata['action_repeat']}")
    
    # Print observation shape to understand the data structure
    if 'observations' in dataset and len(dataset['observations']) > 0:
        sample_obs = dataset['observations'][0]
        print(f"Sample observation shape: {sample_obs.shape}, dtype: {sample_obs.dtype}")
    
    # Extract episodes
    episodes = extract_episodes(dataset, max_episode_length)
    print(f"Found {len(episodes)} episodes in the dataset")
    
    # Create output directory based on dataset path
    output_dir = os.path.join(os.path.dirname(dataset_path), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # Base output path for videos
    filename_base = os.path.splitext(os.path.basename(dataset_path))[0]
    base_output_path = os.path.join(output_dir, filename_base)
    
    # Process at most max_episodes episodes
    episodes_to_process = episodes[:max_episodes] if max_episodes else episodes
    
    # Create a video for each episode
    for i, episode in enumerate(episodes_to_process):
        print(f"Creating video for episode {i+1} of {len(episodes_to_process)} with length {len(episode)}")
        create_episode_video(episode, base_output_path, i+1, resolution, metadata['frame_stack'], save_as_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create videos from a dataset of trajectories")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset file")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution for the output videos (minimum 256)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to process (None for all)")
    parser.add_argument("--max_episode_length", type=int, default=None,
                        help="Maximum length of non-terminal episodes (None for no limit)")
    parser.add_argument("--save_as_png", action="store_true",
                        help="Save individual frames as PNG files instead of creating a video")
    
    args = parser.parse_args()
    
    # Ensure resolution is at least 256
    resolution = args.resolution
    
    create_videos(args.dataset_path, resolution, args.max_episodes, args.max_episode_length, args.save_as_png)
