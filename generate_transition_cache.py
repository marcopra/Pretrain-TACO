#!/usr/bin/env python3
"""
Script to pre-generate transition count cache files for datasets.

Usage:
    python generate_transition_cache.py --dataset_path data_episodes/FullDataset/0.0
    python generate_transition_cache.py --config_file configs_data_MT/MT49OODBasketball.json
    python generate_transition_cache.py --preprocessed_dataset data_episodes_preprocessed/MT49OODBasketball
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tqdm
from utils import ColorPrint

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

def is_preprocessed_dataset(dataset_path):
    """Check if the dataset is already preprocessed with task IDs"""
    dataset_path = Path(dataset_path)
    
    # Check if it's a preprocessed dataset directory
    if (dataset_path.is_dir() and 
        (dataset_path / "pretraining_datasets").exists() and
        (dataset_path / "preprocessing_info.json").exists()):
        return True
    return False

def generate_cache_for_preprocessed_dataset(preprocessed_path):
    """Generate transition cache for a preprocessed dataset"""
    preprocessed_path = Path(preprocessed_path)
    
    if not preprocessed_path.exists():
        ColorPrint.red(f"Preprocessed dataset path does not exist: {preprocessed_path}")
        return False
    
    if not is_preprocessed_dataset(preprocessed_path):
        ColorPrint.red(f"Path is not a valid preprocessed dataset: {preprocessed_path}")
        return False
    
    ColorPrint.blue(f"Generating transition cache for preprocessed dataset: {preprocessed_path}")
    
    # Process both pretraining and test datasets
    success = True
    
    # Process pretraining datasets
    pretraining_dir = preprocessed_path / "pretraining_datasets"
    if pretraining_dir.exists():
        ColorPrint.yellow("Processing pretraining datasets...")
        success &= generate_cache_for_dataset_path(pretraining_dir)
    
    # Process test datasets  
    test_dir = preprocessed_path / "test_dataset"
    if test_dir.exists():
        ColorPrint.yellow("Processing test datasets...")
        success &= generate_cache_for_dataset_path(test_dir)
    
    return success

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
    
    ColorPrint.blue(f"Processing {len(task_dirs)} task directories...")
    
    transition_counts = {}
    episode_lengths = {}
    
    for task_dir in tqdm.tqdm(task_dirs, desc="Processing tasks"):
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
            ColorPrint.green(f"  {task_dir.name}: {len(episode_files)} episodes, {total_transitions} transitions")
        else:
            ColorPrint.yellow(f"  No episode files found in {task_dir.name}")
    
    # Save cache file in optimized format
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
                
                # Write individual episode lengths for optimization
                if dataset_name in episode_lengths:
                    for episode_name, length in sorted(episode_lengths[dataset_name].items()):
                        f.write(f"  {episode_name}: {length}\n")
                
                total_transitions += count
            
            f.write(f"#\n")
            f.write(f"# Total transitions across all tasks: {total_transitions}\n")
        
        ColorPrint.green(f"Successfully saved optimized transition cache to: {cache_file}")
        ColorPrint.green(f"Cache contains: {len(transition_counts)} tasks, {total_transitions} total transitions")
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
    group.add_argument('--preprocessed_dataset', type=str,
                      help='Path to preprocessed dataset directory (e.g., data_episodes_preprocessed/MT49OODBasketball)')
    
    args = parser.parse_args()
    
    if args.dataset_path:
        success = generate_cache_for_dataset_path(args.dataset_path)
    elif args.config_file:
        success = generate_cache_from_config(args.config_file)
    else:  # args.preprocessed_dataset
        success = generate_cache_for_preprocessed_dataset(args.preprocessed_dataset)
    
    if success:
        ColorPrint.green("Cache generation completed successfully!")
    else:
        ColorPrint.red("Cache generation failed!")
        exit(1)

if __name__ == "__main__":
    main()
