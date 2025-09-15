#!/usr/bin/env python3
"""Alternative test script using imageio for better video compatibility."""

import argparse
import os
import json
import numpy as np
import imageio
from collections import defaultdict
import cdmc

def save_video_imageio(frames, video_path, fps=20):
    """Save frames as MP4 video using imageio (better VSCode compatibility)."""
    if not frames:
        print("No frames to save")
        return
    
    try:
        # Ensure frames are uint8
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            processed_frames.append(frame)
        
        # Save using imageio (better compatibility)
        imageio.mimsave(video_path, processed_frames, fps=fps)
        print(f"Video saved with imageio: {video_path}")
        
    except Exception as e:
        print(f"Error saving video with imageio: {e}")
        # Fallback to saving frames as images
        base_path = video_path.rsplit('.', 1)[0]
        for i, frame in enumerate(frames):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            imageio.imwrite(f"{base_path}_frame_{i:04d}.png", frame)
        print(f"Saved {len(frames)} frames as images: {base_path}_frame_*.png")

def record_episode(env, episode_idx, save_dir, record_video=True):
    """Record a single episode and return episode data."""
    episode_data = {
        'episode': episode_idx,
        'states': [],
        'actions': [],
        'rewards': [],
        'timesteps': [],
        'physics': []
    }
    
    frames = [] if record_video else None
    time_step = env.reset()
    timestep = 0
    
    # Record initial state
    physics_state = time_step.physics
    position = physics_state[:2]
    velocity = physics_state[2:4] if len(physics_state) >= 4 else [0.0, 0.0]
    
    episode_data['states'].append({
        'position': position.tolist(),
        'velocity': velocity.tolist()
    })
    episode_data['actions'].append(time_step.action.tolist())
    episode_data['rewards'].append(float(time_step.reward))
    episode_data['timesteps'].append(timestep)
    episode_data['physics'].append(physics_state.tolist())
    
    # Render initial frame
    if record_video:
        try:
            frame = env.physics.render(height=480, width=480, camera_id=0)
            frames.append(frame)
        except Exception as e:
            print(f"Warning: Failed to render frame: {e}")
            record_video = False
            frames = None
    
    while not time_step.last():
        # Random action for testing
        action = np.random.uniform(-1, 1, size=env.action_spec().shape)
        time_step = env.step(action)
        timestep += 1
        
        # Record data
        physics_state = time_step.physics
        position = physics_state[:2]
        velocity = physics_state[2:4] if len(physics_state) >= 4 else [0.0, 0.0]
        
        episode_data['states'].append({
            'position': position.tolist(),
            'velocity': velocity.tolist()
        })
        episode_data['actions'].append(action.tolist())
        episode_data['rewards'].append(float(time_step.reward))
        episode_data['timesteps'].append(timestep)
        episode_data['physics'].append(physics_state.tolist())
        
        # Render frame
        if record_video and frames is not None:
            try:
                frame = env.physics.render(height=480, width=480, camera_id=0)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to render frame: {e}")
                record_video = False
                frames = None
    
    # Save video using imageio
    if record_video and frames:
        video_path = os.path.join(save_dir, f'episode_{episode_idx:03d}.mp4')
        save_video_imageio(frames, video_path)
    
    return episode_data

def main():
    parser = argparse.ArgumentParser(description='Test custom DMC tasks with imageio')
    parser.add_argument('--task', type=str, default='continuous_reach_top_right',
                       help='Task name (e.g., continuous_reach_top_right)')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Number of episodes to record')
    parser.add_argument('--save_dir', type=str, default='./test_results_imageio',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video recording (only save data)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    print(f"Creating environment for task: {args.task}")
    task_name = f"point_mass_maze_{args.task}"
    env = cdmc.make(task_name, obs_type='states', frame_stack=1, action_repeat=1, seed=args.seed)
    
    # Record episodes
    all_episodes_data = []
    summary_stats = defaultdict(list)
    
    print(f"Recording {args.n_episodes} episodes with imageio...")
    for episode_idx in range(args.n_episodes):
        print(f"Episode {episode_idx + 1}/{args.n_episodes}")
        
        episode_data = record_episode(env, episode_idx, args.save_dir, record_video=not args.no_video)
        all_episodes_data.append(episode_data)
        
        # Collect summary statistics
        total_reward = sum(episode_data['rewards'])
        episode_length = len(episode_data['rewards'])
        final_position = episode_data['states'][-1]['position']
        
        summary_stats['total_rewards'].append(total_reward)
        summary_stats['episode_lengths'].append(episode_length)
        summary_stats['final_positions'].append(final_position)
        
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Final position: [{final_position[0]:.3f}, {final_position[1]:.3f}]")
    
    # Save episode data
    data_path = os.path.join(args.save_dir, 'episodes_data.json')
    with open(data_path, 'w') as f:
        json.dump(all_episodes_data, f, indent=2)
    print(f"Episode data saved: {data_path}")
    
    # Save summary statistics
    summary = {
        'task': args.task,
        'n_episodes': args.n_episodes,
        'seed': args.seed,
        'stats': {
            'mean_total_reward': float(np.mean(summary_stats['total_rewards'])),
            'std_total_reward': float(np.std(summary_stats['total_rewards'])),
            'mean_episode_length': float(np.mean(summary_stats['episode_lengths'])),
            'std_episode_length': float(np.std(summary_stats['episode_lengths'])),
            'total_rewards': summary_stats['total_rewards'],
            'episode_lengths': summary_stats['episode_lengths'],
            'final_positions': summary_stats['final_positions']
        }
    }
    
    summary_path = os.path.join(args.save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    
    print("\nSummary Statistics:")
    print(f"Mean total reward: {summary['stats']['mean_total_reward']:.3f} ± {summary['stats']['std_total_reward']:.3f}")
    print(f"Mean episode length: {summary['stats']['mean_episode_length']:.1f} ± {summary['stats']['std_episode_length']:.1f}")

if __name__ == '__main__':
    main()