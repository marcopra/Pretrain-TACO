"""
Preprocessing script to add task_ids to episodes and save them in a structured format.
This avoids the slow preprocessing step during training.

Usage:
python preprocess_dataset_with_task_ids.py --config configs_data_MT/MT49OODBasketball.json --output_dir data_episodes_preprocessed
"""

import argparse
import json
import os
import shutil
import numpy as np
from pathlib import Path
from pretraining_utils_multiheads import extract_task_name_from_path, create_task_mapping, ColorPrint
import tempfile
import datetime
from tqdm import tqdm


def save_episode_with_task_id(episode_data, output_path, task_id):
    """Save episode data with task_id added"""
    # Add task_id to the episode data
    episode_data['task_id'] = np.full(episode_data[list(episode_data.keys())[0]].shape[0], task_id, dtype=np.int32)
    
    # Save to file
    np.savez_compressed(output_path, **episode_data)


def process_episodes_batch(episode_files, output_dir, task_id, batch_size=50):
    """Process a batch of episodes efficiently"""
    counter = 0
    
    for i in tqdm(range(0, len(episode_files), batch_size), desc=f"Processing batches for task_id {task_id}"):
        batch_files = episode_files[i:i + batch_size]
        
        for episode_file in batch_files:
            try:
                # Load episode
                with open(episode_file, 'rb') as f:
                    episode = np.load(f)
                    # Convert to dict while file is open
                    episode_dict = {k: episode[k] for k in episode.keys()}
                
                # Check if task_id already exists
                if 'task_id' not in episode_dict:
                    # Create proper filename
                    episode_len = episode_dict[list(episode_dict.keys())[0]].shape[0] - 1
                    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                    proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
                    output_path = output_dir / proper_name
                    
                    # Save with task_id
                    save_episode_with_task_id(episode_dict, output_path, task_id)
                    counter += 1
                else:
                    # Episode already has task_id, just copy
                    episode_len = episode_dict[list(episode_dict.keys())[0]].shape[0] - 1
                    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                    proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
                    output_path = output_dir / proper_name
                    shutil.copy2(episode_file, output_path)
                    counter += 1
                    
            except Exception as e:
                ColorPrint.red(f"Error processing {episode_file}: {e}")
                continue
    
    return counter


def preprocess_dataset(config_path, output_base_dir, batch_size=50):
    """Main preprocessing function"""
    config_path = Path(config_path)
    output_base_dir = Path(output_base_dir)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract config name for output directory
    config_name = config_path.stem  # e.g., "MT49OODBasketball"
    output_dir = output_base_dir / config_name
    
    # Create output directories
    pretraining_output_dir = output_dir / "pretraining_datasets"
    test_output_dir = output_dir / "test_dataset"
    pretraining_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    ColorPrint.blue(f"Preprocessing dataset from {config_path}")
    ColorPrint.blue(f"Output directory: {output_dir}")
    
    # Get dataset paths
    pretraining_datasets = config.get('pretraining_datasets', [])
    test_datasets = config.get('test_datasets', [])
    
    # Create separate task mappings for pretraining and test
    pretraining_task_to_id, pretraining_id_to_task = create_task_mapping(pretraining_datasets)
    test_task_to_id, test_id_to_task = create_task_mapping(test_datasets)
    
    ColorPrint.green(f"Pretraining task mapping: {pretraining_task_to_id}")
    ColorPrint.green(f"Test task mapping: {test_task_to_id}")
    
    # Save task mappings for reference
    task_mapping_file = output_dir / "task_mapping.json"
    with open(task_mapping_file, 'w') as f:
        json.dump({
            'pretraining_task_to_id': pretraining_task_to_id,
            'pretraining_id_to_task': pretraining_id_to_task,
            'test_task_to_id': test_task_to_id,
            'test_id_to_task': test_id_to_task,
            'num_pretraining_tasks': len(pretraining_task_to_id),
            'num_test_tasks': len(test_task_to_id)
        }, f, indent=2)
    
    # Process pretraining datasets
    ColorPrint.yellow("Processing pretraining datasets...")
    total_pretraining_episodes = 0
    
    for dataset_path in tqdm(pretraining_datasets, desc="Pretraining datasets"):
        dataset_path = Path(dataset_path)
        task_name = extract_task_name_from_path(dataset_path)
        task_id = pretraining_task_to_id[task_name]
        
        # Create output subdirectory for this dataset
        dataset_output_dir = pretraining_output_dir / dataset_path.name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Get episode files
        episode_files = list(dataset_path.glob('*.npz'))
        ColorPrint.blue(f"Processing {len(episode_files)} episodes for {task_name} (pretraining task_id: {task_id})")
        
        # Process episodes in batches
        processed_count = process_episodes_batch(episode_files, dataset_output_dir, task_id, batch_size)
        total_pretraining_episodes += processed_count
        
        ColorPrint.green(f"Processed {processed_count} episodes for {task_name}")
    
    # Process test datasets
    ColorPrint.yellow("Processing test datasets...")
    total_test_episodes = 0
    
    for dataset_path in tqdm(test_datasets, desc="Test datasets"):
        dataset_path = Path(dataset_path)
        task_name = extract_task_name_from_path(dataset_path)
        task_id = test_task_to_id[task_name]
        
        # Create output subdirectory for this dataset
        dataset_output_dir = test_output_dir / dataset_path.name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Get episode files
        episode_files = list(dataset_path.glob('*.npz'))
        ColorPrint.blue(f"Processing {len(episode_files)} episodes for {task_name} (test task_id: {task_id})")
        
        # Process episodes in batches
        processed_count = process_episodes_batch(episode_files, dataset_output_dir, task_id, batch_size)
        total_test_episodes += processed_count
        
        ColorPrint.green(f"Processed {processed_count} episodes for {task_name}")
    
    # Save preprocessing info
    info_file = output_dir / "preprocessing_info.json"
    preprocessing_info = {
        'source_config': str(config_path),
        'total_pretraining_episodes': total_pretraining_episodes,
        'total_test_episodes': total_test_episodes,
        'pretraining_task_mapping': pretraining_task_to_id,
        'test_task_mapping': test_task_to_id,
        'preprocessing_date': datetime.datetime.now().isoformat(),
        'batch_size_used': batch_size
    }
    
    with open(info_file, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    ColorPrint.green(f"Preprocessing completed!")
    ColorPrint.green(f"Total pretraining episodes: {total_pretraining_episodes}")
    ColorPrint.green(f"Total test episodes: {total_test_episodes}")
    ColorPrint.green(f"Pretraining tasks: {len(pretraining_task_to_id)}")
    ColorPrint.green(f"Test tasks: {len(test_task_to_id)}")
    ColorPrint.green(f"Output directory: {output_dir}")
    ColorPrint.green(f"Use this path for training: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with task IDs")
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the dataset configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='data_episodes_preprocessed',
                        help='Base output directory for preprocessed data')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for processing episodes (memory efficiency)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing preprocessed data')
    
    args = parser.parse_args()
    
    # Check if output already exists
    config_path = Path(args.config)
    config_name = config_path.stem
    output_dir = Path(args.output_dir) / config_name
    
    if output_dir.exists() and not args.force:
        ColorPrint.yellow(f"Output directory {output_dir} already exists.")
        ColorPrint.yellow("Use --force to overwrite or use the existing preprocessed data.")
        ColorPrint.blue(f"Existing preprocessed data can be used with: --dataset_config {output_dir}")
        exit(0)
    
    preprocess_dataset(args.config, args.output_dir, args.batch_size)
