"""
python convert_datasets_to_episodes.py --config "mt_config/debug.json"
"""
import os
import json
import pickle
import argparse
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_config_name(config_path):
    """Extract config name from config path including directory structure."""
    # Convert to Path object for easier manipulation
    path = Path(config_path)
    
    # Get the directory structure after 'mt_config/'
    parts = path.parts
    if 'mt_config' in parts:
        mt_config_index = parts.index('mt_config')
        # Include all directories after mt_config and the filename without extension
        relevant_parts = parts[mt_config_index+1:-1] + (path.stem,)
        config_name = '/'.join(relevant_parts)
    else:
        # If mt_config is not in path, use parent directory + filename
        config_name = str(path.parent.name) + '/' + path.stem
    
    return config_name

def load_config(config_path):
    """Load dataset paths from config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get pretraining datasets
    pretraining_datasets = config.get('pretraining_datasets', [])
    
    # Get test datasets - handle both single dataset and array
    test_datasets = config.get('test_dataset', [])
    if not isinstance(test_datasets, list):
        test_datasets = [test_datasets]
    
    return {
        'pretraining_datasets': pretraining_datasets,
        'test_dataset': test_datasets
    }

def load_dataset(dataset_path):
    """Load a dataset from pickle file."""
    print(f"Loading dataset: {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}")
        try:
            with open(dataset_path + ".pickle", 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        except FileNotFoundError:
            print(f"Dataset file also not found with .pickle extension: {dataset_path}.pickle")
            return None

def create_output_dir(config_name, dataset_type, dataset_path):
    """Create output directory for episodes."""
    # Extract dataset name from path
    dataset_name = dataset_path.split('/')[-1]
    
    # Create output directory with the required structure
    output_dir = Path(f"data_episodes/{config_name}/{dataset_type}/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def convert_dataset_to_episodes(config_name, dataset_type, dataset_path):
    """Convert a dataset to episodes and save them."""
    # Create output directory
    output_dir = create_output_dir(config_name, dataset_type, dataset_path)
    
    # Check if directory already has episodes, if so, skip conversion
    if list(output_dir.glob('*.npz')):
        print(f"Episodes already exist in {output_dir}, skipping conversion")
        return True, True  # Return an additional True to indicate it was skipped
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    if dataset is None:
        return False, False
    
    # Check if dataset contains expected fields
    required_fields = ['observations', 'actions', 'rewards', 'terminals']
    for field in required_fields:
        if field not in dataset:
            print(f"Warning: Dataset {dataset_path} missing required field '{field}'")
            return False, False
    
    # Identify episode boundaries based on terminals
    episode_starts = [0]
    episode_ends = []
    
    for i, done in enumerate(dataset['terminals']):
        if done:
            episode_ends.append(i)
            if i + 1 < len(dataset['terminals']):
                episode_starts.append(i + 1)
    
    # Handle case where last episode doesn't end with terminal=True
    if len(episode_ends) < len(episode_starts):
        episode_ends.append(len(dataset['terminals']) - 1)
    
    print(f"Dataset contains {len(episode_starts)} episodes")
    
    # Save each episode as a separate file
    for ep_idx, (start, end) in enumerate(tqdm(zip(episode_starts, episode_ends), 
                                          total=len(episode_starts), 
                                          desc=f"Converting {dataset_path}")):
        # Create episode dictionary
        episode = {
            'observation': dataset['observations'][start:end+1],
            'action': dataset['actions'][start:end+1],
            'reward': dataset['rewards'][start:end+1],
            'discount': np.ones_like(dataset['rewards'][start:end+1])*0.99,
            'terminal': dataset['terminals'][start:end+1],
        }
        
        # Add a dummy first transition as ReplayBuffer expects
        for key in episode:
            if key == 'observation':
                episode[key] = np.vstack([episode[key][0:1], episode[key]])
            else:
                episode[key] = np.concatenate([episode[key][0:1], episode[key]])
        
        # Calculate episode length
        eps_len = len(episode['observation']) - 1  # Subtract 1 for the dummy transition
        
        # Generate filename with timestamp
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        filename = f'{ts}_{ep_idx}_{eps_len}.npz'
        
        # Save episode
        np_file_path = output_dir / filename
        with open(np_file_path, 'wb') as f:
            np.savez_compressed(f, **episode)
    
    print(f"Saved {len(episode_starts)} episodes to {output_dir}")
    return True, False  # Return additional False to indicate it wasn't skipped

def verify_episodes(config_name, dataset_type, dataset_path):
    """Verify that episodes were saved correctly."""
    from replay_buffer import load_episode
    
    # Get output directory
    dataset_name = dataset_path.split('/')[-1]
    output_dir = Path(f"data_episodes/{config_name}/{dataset_type}/{dataset_name}")
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} not found")
        return False
    
    # List all episode files
    episode_files = list(output_dir.glob('*.npz'))
    if not episode_files:
        print(f"Error: No episode files found in {output_dir}")
        return False
    
    # Load a random episode to verify
    import random
    test_file = random.choice(episode_files)
    try:
        episode = load_episode(test_file)
        keys = list(episode.keys())
        print(f"Successfully loaded episode {test_file.name}")
        print(f"Episode contains keys: {keys}")
        print(f"Episode length: {len(episode['observation']) - 1}")
        return True
    except Exception as e:
        print(f"Error loading episode {test_file}: {e}")
        return False

def main(config_path):
    """Main function to convert all datasets to episodes."""
    # Get config name
    config_name = get_config_name(config_path)
    print(f"Config name: {config_name}")
    
    # Load dataset paths from config
    datasets = load_config(config_path)
    
    # Process pretraining datasets
    pretraining_count = len(datasets['pretraining_datasets'])
    print(f"Found {pretraining_count} pretraining datasets in config")
    
    pretraining_successful = 0
    pretraining_skipped = 0
    
    for dataset_path in datasets['pretraining_datasets']:
        print(f"\nProcessing pretraining dataset: {dataset_path}")
        result, skipped = convert_dataset_to_episodes(config_name, 'pretraining_datasets', dataset_path)
        if result:
            pretraining_successful += 1
            if skipped:
                pretraining_skipped += 1
            else:
                # Verify conversion only if we actually performed conversion
                print("\nVerifying conversion...")
                verify_episodes(config_name, 'pretraining_datasets', dataset_path)
    
    # Process test datasets
    test_count = len(datasets['test_dataset'])
    print(f"\nFound {test_count} test datasets in config")
    
    test_successful = 0
    test_skipped = 0
    
    for dataset_path in datasets['test_dataset']:
        print(f"\nProcessing test dataset: {dataset_path}")
        result, skipped = convert_dataset_to_episodes(config_name, 'test_dataset', dataset_path)
        if result:
            test_successful += 1
            if skipped:
                test_skipped += 1
            else:
                # Verify conversion only if we actually performed conversion
                print("\nVerifying conversion...")
                verify_episodes(config_name, 'test_dataset', dataset_path)
    
    # Summary with skipped count
    print(f"\nConverted {pretraining_successful}/{pretraining_count} pretraining datasets successfully")
    print(f"Skipped {pretraining_skipped}/{pretraining_count} pretraining datasets (already existed)")
    print(f"Actually converted {pretraining_successful - pretraining_skipped}/{pretraining_count} pretraining datasets")
    
    print(f"\nConverted {test_successful}/{test_count} test datasets successfully")
    print(f"Skipped {test_skipped}/{test_count} test datasets (already existed)")
    print(f"Actually converted {test_successful - test_skipped}/{test_count} test datasets")
    
    total_successful = pretraining_successful + test_successful
    total_count = pretraining_count + test_count
    total_skipped = pretraining_skipped + test_skipped
    
    print(f"\nTotal: {total_successful}/{total_count} datasets converted")
    print(f"Total: {total_skipped}/{total_count} datasets skipped (already existed)")
    print(f"Total: {total_successful - total_skipped}/{total_count} datasets actually converted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert datasets to episode format for ReplayBuffer")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    
    main(args.config)
