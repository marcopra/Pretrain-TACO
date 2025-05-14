import os
import numpy as np
import pickle
import cv2
import argparse
import re
import glob
import subprocess
from pathlib import Path

def extract_episodes_from_npz(npz_path):
    """
    Extract episodes from a .npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        Dictionary with episode data including observations
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            episode = {key: data[key] for key in data.files}
        return episode
    except Exception as e:
        print(f"Error loading episode from {npz_path}: {e}")
        return None

def parse_dataset_path(dataset_path):
    """
    Parse the dataset path to extract metadata.
    
    Args:
        dataset_path: Path to the dataset file or directory
        
    Returns:
        Dictionary with parsed metadata
    """
    # Convert to Path object for easier path manipulation
    path = Path(dataset_path)
    
    # Try to extract metadata from the folder name
    # Format: env-name_task0_fs3_ar2_ri1_rg0_exp=99
    folder_name = path.name if path.is_dir() else path.parent.name
    
    # Extract environment name and task ID
    env_match = re.search(r'([a-z-]+)_task(\d+)', folder_name)
    env_name = env_match.group(1) if env_match else "unknown"
    task_id = env_match.group(2) if env_match else "0"
    
    # Extract frame stack
    fs_match = re.search(r'fs(\d+)', folder_name)
    frame_stack = int(fs_match.group(1)) if fs_match else 1
    
    # Extract action repeat
    ar_match = re.search(r'ar(\d+)', folder_name)
    action_repeat = int(ar_match.group(1)) if ar_match else 1
    
    # Extract expert probability
    exp_match = re.search(r'exp=(\d+)', folder_name)
    expert_prob = int(exp_match.group(1)) / 100.0 if exp_match else 0.0
    
    return {
        'env_name': env_name,
        'task_id': task_id,
        'frame_stack': frame_stack,
        'action_repeat': action_repeat,
        'expert_prob': expert_prob,
        'basename': os.path.basename(dataset_path)
    }

def create_episode_video(observations, output_path, resolution=256, frame_stack=1, save_as_png=False):
    """
    Create a video or save frames as PNG from an episode's observations.
    
    Args:
        observations: List of observation frames for the episode
        output_path: Path for the output video or frames
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
        # Handle channels-first format (C, H, W)
        if len(obs.shape) == 3 and obs.shape[0] == 3 * frame_stack:
            print(f"Detected channels-first format with stacked frames: {obs.shape}")
            frame = obs[-3:].transpose(1, 2, 0)  # Convert from (3, H, W) to (H, W, 3)
        elif len(obs.shape) == 3 and obs.shape[0] > 3 and obs.shape[0] % 3 == 0:
            print(f"Handling generic channels-first format: {obs.shape}")
            frame = obs[-3:].transpose(1, 2, 0)
        elif frame_stack > 1 and len(obs.shape) == 3 and obs.shape[2] == 3 * frame_stack:
            frame = obs[:, :, -3:]
        elif len(obs.shape) == 2:
            frame = np.stack([obs, obs, obs], axis=2)
        elif len(obs.shape) == 3 and (obs.shape[0] == resolution or obs.shape[0] == 84):
            print(f"Handling transposed observation: {obs.shape}")
            if obs.shape[2] >= 3:
                frame = obs
            else:
                frame = np.stack([obs[:, :, 0], obs[:, :, 0], obs[:, :, 0]], axis=2)
        elif len(obs.shape) == 3 and (obs.shape[2] == resolution or obs.shape[2] > 10):
            print(f"Unusual observation shape detected: {obs.shape}. Attempting to reshape...")
            if obs.shape[2] == resolution:
                frame = obs.reshape(resolution, resolution, 1)
                frame = np.concatenate([frame, frame, frame], axis=2)
            else:
                if obs.shape[2] >= 3:
                    frame = obs[:, :, :3]
                else:
                    avg_channel = np.mean(obs, axis=2, keepdims=True)
                    frame = np.concatenate([avg_channel, avg_channel, avg_channel], axis=2)
        else:
            frame = obs
            
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            frame = np.stack([frame, frame, frame], axis=2)
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Warning: Unusual frame shape after processing: {frame.shape}. Attempting to fix.")
            if len(frame.shape) == 3 and frame.shape[2] > 3:
                frame = frame[:, :, :3]
            else:
                placeholder = np.zeros((resolution, resolution, 3), dtype=np.uint8)
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
        
        if frame.shape[0] < resolution or frame.shape[1] < resolution:
            frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Flip the frame vertically to correct orientation
        frame = cv2.flip(frame, 0)  # 0 means flipping around x-axis (vertical flip)
        
        frames.append(frame)
    
    if save_as_png:
        frames_dir = f"{output_path}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
        
        print(f"Saved {len(frames)} frames to {frames_dir}")
    else:
        try:
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frame_bgr)
            
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-preset", "medium", 
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                output_path
            ]
            
            print(f"Running ffmpeg command to create video: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Successfully created video: {output_path} ({os.path.getsize(output_path)} bytes)")
            else:
                print(f"Video creation failed or file is empty: {output_path}")
                
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
                
        except Exception as e:
            print(f"Error creating video with ffmpeg: {e}")
            try:
                height, width = frames[0].shape[:2]
                
                if width % 2 == 1:
                    width -= 1
                if height % 2 == 1:
                    height -= 1
                
                for codec in ['avc1', 'H264', 'XVID', 'MJPG', 'mp4v']:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
                        
                        if video.isOpened():
                            for frame in frames:
                                if frame.shape[0] != height or frame.shape[1] != width:
                                    frame = cv2.resize(frame, (width, height))
                                
                                if frame.shape[2] == 3:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                
                                video.write(frame)
                            
                            video.release()
                            print(f"Saved video to {output_path} using codec {codec}")
                            
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                break
                    except Exception as codec_error:
                        print(f"Error with codec {codec}: {codec_error}")
            except Exception as cv_error:
                print(f"OpenCV video writing failed: {cv_error}")

def process_npz_episodes(directory_path, resolution=256, max_episodes=None, save_as_png=False):
    metadata = parse_dataset_path(directory_path)
    print(f"Processing directory with metadata: {metadata}")
    print(f"Frame stack: {metadata['frame_stack']}, Action repeat: {metadata['action_repeat']}")
    
    npz_files = sorted(glob.glob(os.path.join(directory_path, "*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {directory_path}")
        return
    
    print(f"Found {len(npz_files)} .npz files in {directory_path}")
    
    videos_dir = os.path.join(directory_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    files_to_process = npz_files[:max_episodes] if max_episodes else npz_files
    
    for i, npz_path in enumerate(files_to_process):
        npz_basename = os.path.basename(npz_path)
        video_basename = os.path.splitext(npz_basename)[0] + ".mp4"
        video_path = os.path.join(videos_dir, video_basename)
        
        if os.path.exists(video_path) and not save_as_png:
            print(f"Video already exists: {video_path}, skipping")
            continue
        
        print(f"Processing episode {i+1} of {len(files_to_process)}: {npz_path}")
        
        episode_data = extract_episodes_from_npz(npz_path)
        if episode_data is None:
            continue
            
        observations = episode_data.get('observation', [])
        
        if len(observations) == 0:
            print(f"No observations found in {npz_path}")
            continue
            
        print(f"Episode length: {len(observations)}")
        
        create_episode_video(
            observations=observations,
            output_path=video_path,
            resolution=resolution,
            frame_stack=metadata['frame_stack'],
            save_as_png=save_as_png
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create videos from dataset episodes")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset directory containing .npz files")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution for the output videos (minimum 256)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to process (None for all)")
    parser.add_argument("--save_as_png", action="store_true",
                        help="Save individual frames as PNG files instead of creating a video")
    
    args = parser.parse_args()
    
    resolution = max(256, args.resolution)
    
    process_npz_episodes(args.dataset_path, resolution, args.max_episodes, args.save_as_png)
