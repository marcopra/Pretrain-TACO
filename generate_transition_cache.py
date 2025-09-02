#!/usr/bin/env python3
"""
Script to pre-generate transition count cache files for datasets.

Usage:
    python generate_transition_cache.py --dataset_path data_episodes/FullDataset/0.0
    python generate_transition_cache.py --config_file configs_data_MT/MT49OODBasketball.json
"""

import argparse
import json
from pathlib import Path
import numpy as np

class ColorPrint:
    @staticmethod
    def blue(text):
        print(f"\033[94m{text}\033[0m")
    
    @staticmethod
    def green(text):
        print(f"\033[92m{text}\033[0m")
    
    @staticmethod
    def yellow(text):
        print(f"\033[93m{text}\033[0m")
    
    @staticmethod
    def red(text):
        print(f"\033[91m{text}\033[0m")

def count_transitions_in_episodes(episode_files):
    """Count total transitions in a list of episode files"""
    total_transitions = 0
    for episode_file in episode_files:
        try:
            episode_file = Path(episode_file)
            with episode_file.open('rb') as f:
                episode = np.load(f)
                obs_key = next(iter(episode.keys()))
                episode_len = episode[obs_key].shape[0] - 1
                total_transitions += episode_len
        except Exception as e:
            ColorPrint.red(f"Warning: Could not count transitions in {episode_file}: {e}")
            continue
    return total_transitions

def generate_cache_for_dataset_path(dataset_base_path):
    """Generate transition cache for a dataset base path"""
    dataset_base_path = Path(dataset_base_path)
    
    if not dataset_base_path.exists():
        ColorPrint.red(f"Dataset path does not exist: {dataset_base_path}")
        return False
    
    ColorPrint.blue(f"Generating transition cache for: {dataset_base_path}")
    
    # Find all task directories
    task_dirs = [d for d in dataset_base_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        ColorPrint.red(f"No task directories found in: {dataset_base_path}")
        return False
    
    ColorPrint.yellow(f"Found {len(task_dirs)} task directories")
    
    transition_counts = {}
    episode_lengths = {}
    
    for task_dir in task_dirs:
        ColorPrint.blue(f"Processing {task_dir.name}...")
        episode_files = list(task_dir.glob('*.npz'))
        
        if episode_files:
            total_transitions = 0
            task_episode_lengths = {}
            
            # Count transitions and collect individual episode lengths
            for episode_file in episode_files:
                try:
                    with episode_file.open('rb') as f:
                        episode = np.load(f)
                        obs_key = next(iter(episode.keys()))
                        episode_len = episode[obs_key].shape[0] - 1
                        total_transitions += episode_len
                        task_episode_lengths[episode_file.name] = episode_len
                except Exception as e:
                    ColorPrint.red(f"Warning: Could not process {episode_file.name}: {e}")
                    continue
            
            transition_counts[task_dir.name] = total_transitions
            episode_lengths[task_dir.name] = task_episode_lengths
            print(f"  {task_dir.name}: {len(episode_files)} episodes, {total_transitions} transitions")
        else:
            ColorPrint.yellow(f"  No episode files found in {task_dir.name}")
    
    # Save cache file with episode lengths
    cache_file = dataset_base_path / "transition_counts.txt"
    try:
        with open(cache_file, 'w') as f:
            f.write("# Dataset transition counts cache\n")
            f.write("# Format: dataset_name: total_transitions\n")
            f.write("#   episode_file: episode_length\n")
            f.write(f"# Generated for: {dataset_base_path}\n")
            f.write(f"# Total tasks: {len(transition_counts)}\n")
            f.write("#\n")
            
            total_transitions = 0
            for dataset_name in sorted(transition_counts.keys()):
                count = transition_counts[dataset_name]
                f.write(f"{dataset_name}: {count}\n")
                
                # Write individual episode lengths
                if dataset_name in episode_lengths:
                    for episode_name, length in sorted(episode_lengths[dataset_name].items()):
                        f.write(f"  {episode_name}: {length}\n")
                
                total_transitions += count
            
            f.write(f"#\n")
            f.write(f"# Total transitions across all tasks: {total_transitions}\n")
        
        ColorPrint.green(f"Successfully saved transition cache to: {cache_file}")
        ColorPrint.green(f"Total: {len(transition_counts)} tasks, {total_transitions} transitions")
        return True
        
    except Exception as e:
        ColorPrint.red(f"Error saving cache file: {e}")
        return False

def generate_cache_from_config(config_file):
    """Generate cache from a config file by extracting the base dataset path"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        ColorPrint.red(f"Config file does not exist: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get both pretraining and test datasets
        all_datasets = []
        all_datasets.extend(config.get('pretraining_datasets', []))
        all_datasets.extend(config.get('test_datasets', []))
        
        if not all_datasets:
            ColorPrint.red("No datasets found in config file")
            return False
        
        # Extract base path from first dataset
        first_dataset = all_datasets[0]
        dataset_base_path = Path(first_dataset).parent
        
        ColorPrint.blue(f"Extracted base dataset path from config: {dataset_base_path}")
        
        return generate_cache_for_dataset_path(dataset_base_path)
        
    except Exception as e:
        ColorPrint.red(f"Error reading config file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate transition count cache for datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str, 
                      help='Path to dataset base directory (e.g., data_episodes/FullDataset/0.0)')
    group.add_argument('--config_file', type=str,
                      help='Path to config JSON file (e.g., configs_data_MT/MT49OODBasketball.json)')
    
    args = parser.parse_args()
    
    if args.dataset_path:
        success = generate_cache_for_dataset_path(args.dataset_path)
    else:
        success = generate_cache_from_config(args.config_file)
    
    if success:
        ColorPrint.green("Cache generation completed successfully!")
    else:
        ColorPrint.red("Cache generation failed!")
        exit(1)

if __name__ == "__main__":
    main()
