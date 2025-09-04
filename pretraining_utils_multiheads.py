from replay_buffer_multi_task import make_replay_loader
from pathlib import Path
import json
import random
import datetime
import numpy as np
import tempfile
import os
import tqdm
from replay_buffer_multi_task import save_episode_with_task_id
from utils import ColorPrint


def load_transition_cache(dataset_base_path):
    """Load transition counts from cache file if it exists"""
    cache_file = Path(dataset_base_path) / "transition_counts.txt"
    if cache_file.exists():
        ColorPrint.blue(f"Loading transition counts from cache: {cache_file}")
        transition_counts = {}
        episode_lengths = {}  # New: cache individual episode lengths
        try:
            with open(cache_file, 'r') as f:
                current_dataset = None
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ':' in line and not line.startswith('  '):
                            # Dataset line: "dataset_name: total_count"
                            dataset_name, count = line.split(':', 1)
                            transition_counts[dataset_name.strip()] = int(count.strip())
                            current_dataset = dataset_name.strip()
                            episode_lengths[current_dataset] = {}
                        elif line.startswith('  ') and current_dataset:
                            # Episode line: "  episode_file: length"
                            episode_line = line.strip()
                            if ':' in episode_line:
                                episode_name, length = episode_line.split(':', 1)
                                episode_lengths[current_dataset][episode_name.strip()] = int(length.strip())
            return transition_counts, episode_lengths
        except Exception as e:
            ColorPrint.red(f"Error reading transition cache: {e}")
            return None, None
    else:
        ColorPrint.red(f"No transition cache file found at: {cache_file}")
        return None, None

def save_transition_cache(dataset_base_path, transition_counts, episode_lengths=None):
    """Save transition counts and episode lengths to cache file"""
    cache_file = Path(dataset_base_path) / "transition_counts.txt"
    try:
        with open(cache_file, 'w') as f:
            f.write("# Dataset transition counts cache\n")
            f.write("# Format: dataset_name: total_transitions\n")
            f.write("#   episode_file: episode_length\n")
            f.write("#\n")
            
            total_transitions = 0
            for dataset_name in sorted(transition_counts.keys()):
                count = transition_counts[dataset_name]
                f.write(f"{dataset_name}: {count}\n")
                
                # Write individual episode lengths if available
                if episode_lengths and dataset_name in episode_lengths:
                    for episode_name, length in sorted(episode_lengths[dataset_name].items()):
                        f.write(f"  {episode_name}: {length}\n")
                
                total_transitions += count
            
            f.write(f"#\n")
            f.write(f"# Total transitions across all datasets: {total_transitions}\n")
        
        ColorPrint.green(f"Saved transition counts cache to: {cache_file}")
    except Exception as e:
        ColorPrint.red(f"Error saving transition cache: {e}")

def get_dataset_base_path(dataset_paths):
    """Extract base dataset path from list of dataset paths"""
    if not dataset_paths:
        return None
    
    # For paths like "data_episodes/FullDataset/0.0/task-name_...", 
    # we want "data_episodes/FullDataset/0.0"
    first_path = Path(dataset_paths[0])
    return first_path.parent

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
            print(f"Warning: Could not count transitions in {episode_file}: {e}")
            continue
    return total_transitions

def count_transitions_with_cache(dataset_paths):
    """Count transitions using cache if available, otherwise count and save cache"""
    # Get base dataset path for cache file
    dataset_base_path = get_dataset_base_path(dataset_paths)
    transition_counts = {}
    episode_lengths = {}
    
    if dataset_base_path:
        # Try to load from cache
        cached_counts, cached_episode_lengths = load_transition_cache(dataset_base_path)
        if cached_counts:
            need_update = False
            
            # Use cached data for available datasets
            for dataset_path in dataset_paths:
                dataset_name = Path(dataset_path).name
                if dataset_name in cached_counts:
                    transition_counts[dataset_name] = cached_counts[dataset_name]
                    if cached_episode_lengths and dataset_name in cached_episode_lengths:
                        episode_lengths[dataset_name] = cached_episode_lengths[dataset_name]
                else:
                    ColorPrint.yellow(f"Dataset {dataset_name} not found in cache, counting manually")
                    dataset_path_obj = Path(dataset_path)
                    episode_files = list(dataset_path_obj.glob('*.npz'))
                    
                    # Count transitions and collect episode lengths
                    count, dataset_episode_lengths = count_transitions_and_lengths_in_episodes(episode_files)
                    transition_counts[dataset_name] = count
                    episode_lengths[dataset_name] = dataset_episode_lengths
                    need_update = True
            
            # If we had to count some datasets manually, update the cache
            if need_update:
                ColorPrint.yellow("Updating transition cache with new datasets")
                # Merge with existing cache
                if cached_counts:
                    for dataset_name, count in cached_counts.items():
                        if dataset_name not in transition_counts:
                            transition_counts[dataset_name] = count
                if cached_episode_lengths:
                    for dataset_name, episodes in cached_episode_lengths.items():
                        if dataset_name not in episode_lengths:
                            episode_lengths[dataset_name] = episodes
                
                save_transition_cache(dataset_base_path, transition_counts, episode_lengths)
            
            return transition_counts, episode_lengths
    
    # No cache available, count all datasets manually
    ColorPrint.yellow("No cache available, counting transitions for all datasets")
    for dataset_path in dataset_paths:
        dataset_path_obj = Path(dataset_path)
        dataset_name = dataset_path_obj.name
        episode_files = list(dataset_path_obj.glob('*.npz'))
        
        count, dataset_episode_lengths = count_transitions_and_lengths_in_episodes(episode_files)
        transition_counts[dataset_name] = count
        episode_lengths[dataset_name] = dataset_episode_lengths
    
    # Save cache for future use
    if dataset_base_path:
        save_transition_cache(dataset_base_path, transition_counts, episode_lengths)
    
    return transition_counts, episode_lengths

def count_transitions_and_lengths_in_episodes(episode_files):
    """Count total transitions and get individual episode lengths"""
    total_transitions = 0
    episode_lengths = {}
    
    for episode_file in episode_files:
        try:
            episode_file = Path(episode_file)
            with episode_file.open('rb') as f:
                episode = np.load(f)
                obs_key = next(iter(episode.keys()))
                episode_len = episode[obs_key].shape[0] - 1
                total_transitions += episode_len
                episode_lengths[episode_file.name] = episode_len
        except Exception as e:
            print(f"Warning: Could not count transitions in {episode_file}: {e}")
            continue
    
    return total_transitions, episode_lengths

def extract_task_name_from_path(dataset_path):
    """Extract task name from dataset path"""
    dataset_path = Path(dataset_path)
    folder_name = dataset_path.name
    
    # Extract task name from folder name 
    parts = folder_name.split('_')
    if parts:
        # Take the first part which should contain the task name
        task_part = parts[0]
        # Remove version suffixes like -v2, -v3
        task_name = task_part.replace('-v2', '').replace('-v3', '')
        return task_name
    
    # Fallback to using the full folder name if parsing fails
    return folder_name

def create_task_mapping(dataset_paths):
    """Create a mapping from task names to unique IDs"""
    task_names = []
    for dataset_path in dataset_paths:
        task_name = extract_task_name_from_path(dataset_path)
        if task_name not in task_names:
            task_names.append(task_name)
    
    task_to_id = {task_name: idx for idx, task_name in enumerate(sorted(task_names))}
    id_to_task = {idx: task_name for task_name, idx in task_to_id.items()}
    
    ColorPrint.blue(f"Created task mapping: {task_to_id}")
    return task_to_id, id_to_task

def create_symlink_with_proper_naming(episode_file, dest_dir, counter, task_id=None, episode_len=None):
    """Create symlink to episode file with proper naming format for replay_buffer
    
    Args:
        episode_file: Path to episode file
        dest_dir: Destination directory
        counter: Counter for naming
        task_id: Task ID to add if needed
        episode_len: Pre-computed episode length (if None, will be computed)
    """
    episode_file = Path(episode_file)
    
    # If episode_len is not provided, we need to load the file to get it
    if episode_len is None:
        try:
            with episode_file.open('rb') as f:
                episode = np.load(f)
                obs_key = next(iter(episode.keys()))
                episode_len = episode[obs_key].shape[0] - 1
                has_task_id = 'task_id' in episode
                
                # If task_id is provided and episode doesn't have task_id, convert to dict and save with task_id
                if task_id is not None and not has_task_id:
                    # Convert episode to regular dict while file is still open
                    episode_dict = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Warning: Could not process episode {episode_file}: {e}")
            return None
    else:
        # We have pre-computed episode_len, but still need to check if we need to add task_id
        has_task_id = None
        episode_dict = None
        
        if task_id is not None:
            # Only load the file if we need to check/add task_id
            try:
                with episode_file.open('rb') as f:
                    episode = np.load(f)
                    has_task_id = 'task_id' in episode
                    
                    if not has_task_id:
                        # Convert episode to regular dict while file is still open
                        episode_dict = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Warning: Could not process episode {episode_file}: {e}")
                return None
        else:
            # No task_id needed, we can skip loading the file entirely
            has_task_id = True  # Assume it's fine to create symlink
    
    # If task_id is provided and episode doesn't have task_id, we need to create a new file
    if task_id is not None and has_task_id is False:
        # Create a temporary file with task_id added
        
        
        # Create temporary file with task_id
        temp_file = dest_dir / f"temp_{counter:06d}.npz"
        save_episode_with_task_id(episode_dict, temp_file, task_id)
        
        # Create proper filename
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
        dest_path = dest_dir / proper_name
        
        # Move temp file to proper name
        temp_file.rename(dest_path)
        return dest_path
    else:
        # Create proper filename format that replay_buffer expects
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
        dest_path = dest_dir / proper_name
        
        # Create symlink instead of copying
        os.symlink(episode_file.absolute(), dest_path)
        return dest_path

def get_episode_length_from_cache_or_file(episode_file, transition_counts, episode_lengths=None):
    """Get episode length from cache or by loading the file"""
    episode_file = Path(episode_file)
    dataset_name = episode_file.parent.name
    episode_name = episode_file.name
    
    # Check if we have cached episode length
    if (episode_lengths and dataset_name in episode_lengths and 
        episode_name in episode_lengths[dataset_name]):
        return episode_lengths[dataset_name][episode_name]
    
    # Fallback: load the file to get episode length
    try:
        with episode_file.open('rb') as f:
            episode = np.load(f)
            obs_key = next(iter(episode.keys()))
            episode_len = episode[obs_key].shape[0] - 1
            return episode_len
    except Exception as e:
        ColorPrint.red(f"Warning: Could not get episode length from {episode_file}: {e}")
        return None

def select_episodes_homogeneous(dataset_paths, max_episodes_per_dataset, max_size=None, is_test=False):
    """Select episodes from datasets to ensure homogeneous distribution (always enabled)"""
    all_episodes = []
    dataset_info = []
    
    # Get transition counts (with caching)
    transition_counts, episode_lengths = count_transitions_with_cache(dataset_paths)
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.name
        episode_files = list(dataset_path.glob('*.npz'))
        
        if max_episodes_per_dataset is not None:
            episode_files = episode_files[:max_episodes_per_dataset]
        
        # Use cached count if available, otherwise count manually
        if dataset_name in transition_counts:
            transitions_count = transition_counts[dataset_name]
            # Adjust count if we're limiting episodes
            if max_episodes_per_dataset is not None and len(episode_files) < len(list(dataset_path.glob('*.npz'))):
                transitions_count = count_transitions_in_episodes(episode_files)
        else:
            transitions_count = count_transitions_in_episodes(episode_files)
        
        dataset_info.append({
            'path': dataset_path,
            'episodes': episode_files,
            'transitions': transitions_count
        })
        all_episodes.extend(episode_files)
    
    total_transitions = sum(info['transitions'] for info in dataset_info)
    if max_size is None:
        max_size = total_transitions
    
    if not is_test:
        # For training data, always select at least 1 episode per dataset for homogeneous distribution
        transitions_per_dataset = max(1, max_size // len(dataset_paths))  # At least 1 transition per dataset
        ColorPrint.green(f"Homogeneous loading: ~{transitions_per_dataset} transitions per dataset")
        
        selected_episodes = []
        actual_transitions_loaded = 0
        
        for info in dataset_info:
            dataset_episodes = info['episodes'].copy()
            random.shuffle(dataset_episodes)
            
            current_transitions = 0
            dataset_selected = []
            
            for episode_file in dataset_episodes:
                # Use cached episode length if available
                episode_len = get_episode_length_from_cache_or_file(
                    episode_file, transition_counts, episode_lengths
                )
                if episode_len is None:
                    continue
                
                # Always select at least one episode per dataset
                if len(dataset_selected) == 0:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_len
                elif current_transitions + episode_len <= transitions_per_dataset:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_len
                else:
                    break
            
            selected_episodes.extend(dataset_selected)
            actual_transitions_loaded += current_transitions
            print(f"Dataset {info['path'].name}: selected {len(dataset_selected)} episodes with {current_transitions} transitions")
        
        print(f"Total selected: {len(selected_episodes)} episodes with {actual_transitions_loaded} transitions")
        return selected_episodes, episode_lengths
    else:
        ColorPrint.green(f"Using all {total_transitions} transitions from {len(dataset_paths)} datasets")
        return all_episodes, episode_lengths

def is_preprocessed_dataset(config_or_path):
    """Check if the dataset is already preprocessed with task IDs"""
    config_path = Path(config_or_path)
    
    # Check if it's a preprocessed dataset directory
    if (config_path.is_dir() and 
        (config_path / "pretraining_datasets").exists() and
        (config_path / "preprocessing_info.json").exists()):
        return True
    return False

def load_preprocessed_task_mapping(dataset_dir, is_test=False):
    """Load task mapping from preprocessed dataset"""
    task_mapping_file = Path(dataset_dir) / "task_mapping.json"
    if task_mapping_file.exists():
        with open(task_mapping_file, 'r') as f:
            data = json.load(f)
        
        if is_test:
            return data['test_task_to_id'], data['test_id_to_task']
        else:
            return data['pretraining_task_to_id'], data['pretraining_id_to_task']
    else:
        raise FileNotFoundError(f"Task mapping file not found: {task_mapping_file}")

def _load_from_symlink_directory(config_path, batch_size, num_workers, max_size, nstep, multistep, discount):
    """Handle Case 3: Symlink directory"""
    ColorPrint.green("Using pre-created symlink directory - no preprocessing needed")
    
    # Create loader directly
    loader = make_replay_loader(
        replay_dir=config_path,
        max_size=1000000 if max_size is None else max_size,
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    
    # Try to load task mapping if available
    task_mapping_file = config_path / "task_mapping.json"
    if task_mapping_file.exists():
        with open(task_mapping_file, 'r') as f:
            mapping_data = json.load(f)
        loader.task_to_id = mapping_data.get('task_to_id', {})
        loader.id_to_task = mapping_data.get('id_to_task', {})
        loader.num_tasks = len(loader.task_to_id)
    else:
        # Default single task
        loader.task_to_id = {'default': 0}
        loader.id_to_task = {0: 'default'}
        loader.num_tasks = 1
    
    ColorPrint.green(f"Loaded symlink dataset with {loader.num_tasks} tasks")
    return loader

def _load_from_preprocessed_dataset(config_path, batch_size, num_workers, nstep, multistep, discount,
                                  max_episodes_per_dataset, max_size, is_test,
                                  split_ratio, use_training_split, **kwargs):
    """Handle Case 2: Preprocessed dataset"""
    # Load task mapping from preprocessed dataset
    task_to_id, id_to_task = load_preprocessed_task_mapping(config_path, is_test)
    ColorPrint.blue(f"Loaded {'test' if is_test else 'pretraining'} task mapping: {task_to_id}")
    
    # Get dataset paths
    if is_test:
        datasets_dir = config_path / "test_dataset"
    else:
        datasets_dir = config_path / "pretraining_datasets"
    
    dataset_paths = [str(d) for d in datasets_dir.iterdir() if d.is_dir()]
    ColorPrint.green(f"Found {len(dataset_paths)} dataset directories")
    
    # Get transition counts for homogeneous loading
    transition_counts, episode_lengths = count_transitions_with_cache(dataset_paths)
    
    return _create_loader_with_split_handling(
        dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
        batch_size, num_workers, nstep, multistep, discount,
        max_episodes_per_dataset, max_size, is_test,
        split_ratio, use_training_split, preprocessed=True, **kwargs
    )

def _load_from_json_config(config_path, batch_size, num_workers, nstep, multistep, discount,
                          max_episodes_per_dataset, max_size, is_test,
                          split_ratio, use_training_split, **kwargs):
    """Handle Case 1: JSON config or raw dataset folder"""
    # Determine if it's a config file or folder
    if config_path.suffix == '.json':
        ColorPrint.blue(f"Loading from config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if is_test:
            dataset_paths = config.get('test_datasets', [])
        else:
            dataset_paths = config.get('pretraining_datasets', [])
    else:
        ColorPrint.blue(f"Loading from folder structure: {config_path}")
        if is_test:
            datasets_dir = config_path / "test_dataset"
        else:
            datasets_dir = config_path / "pretraining_datasets"
        
        dataset_paths = [str(d) for d in datasets_dir.iterdir() if d.is_dir()]
    
    # Create task mapping
    task_to_id, id_to_task = create_task_mapping(dataset_paths)
    ColorPrint.green(f"Found {len(dataset_paths)} dataset directories")
    
    # Check for cached transition counts to optimize loading
    dataset_base_path = get_dataset_base_path(dataset_paths)
    transition_counts, episode_lengths = None, None
    
    if dataset_base_path:
        cached_counts, cached_episode_lengths = load_transition_cache(dataset_base_path)
        if cached_counts:
            ColorPrint.green("Using cached transition counts for optimized homogeneous loading")
            transition_counts, episode_lengths = cached_counts, cached_episode_lengths
        else:
            ColorPrint.yellow("No transition cache found - will count transitions for homogeneous loading")
    
    # Get transition counts for homogeneous loading (using cache if available)
    if not transition_counts:
        transition_counts, episode_lengths = count_transitions_with_cache(dataset_paths)
    
    return _create_loader_with_split_handling(
        dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
        batch_size, num_workers, nstep, multistep, discount,
        max_episodes_per_dataset, max_size, is_test,
        split_ratio, use_training_split, preprocessed=False, **kwargs
    )

def _create_loader_with_split_handling(dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
                                     batch_size, num_workers, nstep, multistep, discount,
                                     max_episodes_per_dataset, max_size, is_test,
                                     split_ratio, use_training_split, preprocessed=False, **kwargs):
    """Create loader with proper train/validation split handling"""
    # Always use split validation for training data (not test data)
    using_split_validation = not is_test

    if using_split_validation:
        # Simplified approach: split episodes directly by indices
        all_episodes = []
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            episode_files = list(dataset_path.glob('*.npz'))
            all_episodes.extend(episode_files)
        
        # Apply homogeneous selection first to get the episodes we want to use
        selected_episodes, _ = select_episodes_homogeneous(
            dataset_paths, max_episodes_per_dataset, max_size, is_test
        )
        
        # Now split the selected episodes by dataset to maintain homogeneous distribution
        train_episodes = []
        val_episodes = []
        train_transitions = 0
        val_transitions = 0
        
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            dataset_name = dataset_path.name
            
            # Get episodes for this dataset from selected episodes
            dataset_episodes = [ep for ep in selected_episodes if ep.parent.name == dataset_name]
            
            if not dataset_episodes:
                continue
                
            # Calculate split for this dataset
            num_episodes = len(dataset_episodes)
            train_count = max(1, int(num_episodes * split_ratio))  # At least 1 episode for training
            
            # Ensure we have at least 1 episode for validation if possible
            if train_count >= num_episodes and num_episodes > 1:
                train_count = num_episodes - 1
            
            dataset_train = dataset_episodes[:train_count]
            dataset_val = dataset_episodes[train_count:]
            
            # Count transitions for this dataset split
            for ep in dataset_train:
                ep_len = get_episode_length_from_cache_or_file(ep, transition_counts, episode_lengths)
                if ep_len:
                    train_transitions += ep_len
            
            for ep in dataset_val:
                ep_len = get_episode_length_from_cache_or_file(ep, transition_counts, episode_lengths)
                if ep_len:
                    val_transitions += ep_len
            
            train_episodes.extend(dataset_train)
            val_episodes.extend(dataset_val)
            
            print(f"Dataset {dataset_name}: {len(dataset_train)} train, {len(dataset_val)} val episodes")
        
        # Calculate actual split ratio
        total_transitions = train_transitions + val_transitions
        actual_split_ratio = train_transitions / total_transitions if total_transitions > 0 else 0
        
        print(f"Actual split: {train_transitions} train / {val_transitions} val transitions (ratio: {actual_split_ratio:.3f})")
        
        if use_training_split:
            # Create training dataset
            temp_dir = Path(tempfile.mkdtemp())
            ColorPrint.green(f"Created temporary directory: {temp_dir}")
            
            _create_dataset_symlinks(train_episodes, temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed)
            loader_dir = temp_dir
            ColorPrint.green(f"Training split: {len(train_episodes)} episodes with {train_transitions} transitions")
            
        else:
            # Create validation dataset
            temp_dir = Path(tempfile.mkdtemp())
            ColorPrint.green(f"Created temporary directory: {temp_dir}")
            
            _create_dataset_symlinks(val_episodes, temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed)
            loader_dir = temp_dir
            ColorPrint.green(f"Validation split: {len(val_episodes)} episodes with {val_transitions} transitions")
    
    else:
        # Test data - no splitting, use all episodes
        selected_episodes, _ = select_episodes_homogeneous(
            dataset_paths, max_episodes_per_dataset, max_size, is_test
        )
        
        temp_dir = Path(tempfile.mkdtemp())
        ColorPrint.green(f"Created temporary directory: {temp_dir}")
        
        _create_dataset_symlinks(selected_episodes, temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed)
        loader_dir = temp_dir
        ColorPrint.green(f"Test dataset: {len(selected_episodes)} episodes")
    
    # Create loader
    loader = make_replay_loader(
        replay_dir=loader_dir,
        max_size=1000000 if max_size is None else max_size,
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    
    # Attach task mapping
    loader.task_to_id = task_to_id
    loader.id_to_task = id_to_task
    loader.num_tasks = len(task_to_id)
    
    return loader

def _create_dataset_symlinks(selected_episodes, temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed):
    """Create symlinks for dataset episodes"""
    counter = 0
    for episode_file in tqdm.tqdm(selected_episodes, desc="Creating episode symlinks"):
        if not preprocessed:
            task_name = extract_task_name_from_path(episode_file.parent)
            task_id = task_to_id[task_name]
            episode_len = get_episode_length_from_cache_or_file(
                episode_file, transition_counts, episode_lengths
            )
            processed_path = create_symlink_with_proper_naming(
                episode_file, temp_dir, counter, task_id, episode_len
            )
        else:
            episode_len = get_episode_length_from_cache_or_file(
                episode_file, transition_counts, episode_lengths
            )
            processed_path = create_symlink_with_proper_naming(
                episode_file, temp_dir, counter, None, episode_len
            )
        
        if processed_path is not None:
            counter += 1

def is_symlink_dataset(config_or_path):
    """Check if the dataset is a symlink directory (ready-to-use episodes)"""
    config_path = Path(config_or_path)
    
    # Check if it's a directory with .npz files that are symlinks
    if config_path.is_dir():
        npz_files = list(config_path.glob('*.npz'))
        if npz_files and all(f.is_symlink() for f in npz_files[:5]):  # Check first 5 files
            return True
    return False

def load_unified_dataset(config_or_path, batch_size=32, num_workers=4,
                         nstep=3, multistep=3, discount=0.99, 
                         max_episodes_per_dataset=8, max_size=None,
                         homogeneous=None, is_test=False, 
                         split_ratio=0.8, use_training_split=True, **kwargs):
    """
    Load episodes from datasets specified in config or folder structure
    
    Args:
        config_or_path: JSON config, preprocessed dataset directory, or symlink directory
        homogeneous: DEPRECATED - now always enabled for optimal dataset distribution
        split_ratio: Ratio of data to use for training (remaining goes to validation)
        use_training_split: Whether to use training split (True) or validation split (False)
        **kwargs: Additional arguments
    """
    # Monitor memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    ColorPrint.blue(f"Memory before dataset loading: {memory_before:.2f} MB")
    
    # Deprecation warning for homogeneous parameter
    if homogeneous is not None:
        ColorPrint.yellow("WARNING: 'homogeneous' parameter is deprecated and will be ignored. Homogeneous loading is now always enabled for optimal performance.")
    
    config_path = Path(config_or_path)
    
    # Determine dataset type and handle accordingly
    if is_symlink_dataset(config_path):
        # Case 3: Symlink directory - ready to use
        ColorPrint.green(f"Loading from symlink directory: {config_path}")
        loader = _load_from_symlink_directory(config_path, batch_size, num_workers, max_size, nstep, multistep, discount)
    
    elif is_preprocessed_dataset(config_path):
        # Case 2: Preprocessed dataset
        ColorPrint.green(f"Loading from preprocessed dataset: {config_path}")
        loader = _load_from_preprocessed_dataset(
            config_path, batch_size, num_workers, nstep, multistep, discount,
            max_episodes_per_dataset, max_size, is_test, 
            split_ratio, use_training_split, **kwargs
        )
    
    else:
        # Case 1: JSON config or raw dataset folder
        ColorPrint.blue(f"Loading from JSON config or raw dataset: {config_path}")
        loader = _load_from_json_config(
            config_path, batch_size, num_workers, nstep, multistep, discount,
            max_episodes_per_dataset, max_size, is_test,
            split_ratio, use_training_split, **kwargs
        )
    
    # Monitor memory usage after loading
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    ColorPrint.green(f"Memory after dataset loading: {memory_after:.2f} MB")
    ColorPrint.green(f"Memory used by dataset loading: {memory_used:.2f} MB")
    
    return loader

def load_train_val_datasets(config_or_path, batch_size=32, num_workers=4,
                           nstep=3, multistep=3, discount=0.99, 
                           max_episodes_per_dataset=8, max_size=None,
                           split_ratio=0.8, **kwargs):
    """
    Load both training and validation datasets with a single call
    
    Args:
        config_or_path: JSON config, preprocessed dataset directory, or symlink directory
        split_ratio: Ratio of data to use for training (remaining goes to validation)
        **kwargs: Additional arguments
        
    Returns:
        tuple: (train_dataloader, valid_dataloader)
    """
    ColorPrint.blue(f"=== LOADING TRAIN/VAL DATASETS (split_ratio={split_ratio}) ===")
    
    # Monitor memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    config_path = Path(config_or_path)
    
    # Determine dataset type and handle accordingly
    if is_symlink_dataset(config_path):
        raise NotImplementedError("Split validation not supported for symlink datasets")
    
    elif is_preprocessed_dataset(config_path):
        # Case 2: Preprocessed dataset
        ColorPrint.green(f"Loading from preprocessed dataset: {config_path}")
        train_loader, val_loader = _load_train_val_from_preprocessed_dataset(
            config_path, batch_size, num_workers, nstep, multistep, discount,
            max_episodes_per_dataset, max_size, split_ratio, **kwargs
        )
    
    else:
        # Case 1: JSON config or raw dataset folder
        ColorPrint.blue(f"Loading from JSON config or raw dataset: {config_path}")
        train_loader, val_loader = _load_train_val_from_json_config(
            config_path, batch_size, num_workers, nstep, multistep, discount,
            max_episodes_per_dataset, max_size, split_ratio, **kwargs
        )
    
    # Monitor memory usage after loading
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    ColorPrint.green(f"Memory used for train/val datasets: {memory_used:.2f} MB")
    
    return train_loader, val_loader

def _load_train_val_from_preprocessed_dataset(config_path, batch_size, num_workers, nstep, multistep, discount,
                                            max_episodes_per_dataset, max_size, split_ratio, **kwargs):
    """Handle Case 2: Preprocessed dataset - return both train and val loaders"""
    # Load task mapping from preprocessed dataset
    task_to_id, id_to_task = load_preprocessed_task_mapping(config_path, is_test=False)
    ColorPrint.blue(f"Loaded pretraining task mapping: {task_to_id}")
    
    # Get dataset paths
    datasets_dir = config_path / "pretraining_datasets"
    dataset_paths = [str(d) for d in datasets_dir.iterdir() if d.is_dir()]
    ColorPrint.green(f"Found {len(dataset_paths)} dataset directories")
    
    # Get transition counts for homogeneous loading
    transition_counts, episode_lengths = count_transitions_with_cache(dataset_paths)
    
    return _create_train_val_loaders(
        dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
        batch_size, num_workers, nstep, multistep, discount,
        max_episodes_per_dataset, max_size, split_ratio, preprocessed=True, **kwargs
    )

def _load_train_val_from_json_config(config_path, batch_size, num_workers, nstep, multistep, discount,
                                    max_episodes_per_dataset, max_size, split_ratio, **kwargs):
    """Handle Case 1: JSON config or raw dataset folder - return both train and val loaders"""
    # Determine if it's a config file or folder
    if config_path.suffix == '.json':
        ColorPrint.blue(f"Loading from config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        dataset_paths = config.get('pretraining_datasets', [])
    else:
        ColorPrint.blue(f"Loading from folder structure: {config_path}")
        datasets_dir = config_path / "pretraining_datasets"
        dataset_paths = [str(d) for d in datasets_dir.iterdir() if d.is_dir()]
    
    # Create task mapping
    task_to_id, id_to_task = create_task_mapping(dataset_paths)
    ColorPrint.green(f"Found {len(dataset_paths)} dataset directories")
    
    # Get transition counts for homogeneous loading (with caching)
    transition_counts, episode_lengths = count_transitions_with_cache(dataset_paths)
    
    return _create_train_val_loaders(
        dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
        batch_size, num_workers, nstep, multistep, discount,
        max_episodes_per_dataset, max_size, split_ratio, preprocessed=False, **kwargs
    )

def _create_train_val_loaders(dataset_paths, task_to_id, id_to_task, transition_counts, episode_lengths,
                             batch_size, num_workers, nstep, multistep, discount,
                             max_episodes_per_dataset, max_size, split_ratio, preprocessed=False, **kwargs):
    """Create both train and validation loaders from the same episode pool"""
    
    # Apply homogeneous selection first to get the episodes we want to use
    selected_episodes, _ = select_episodes_homogeneous(
        dataset_paths, max_episodes_per_dataset, max_size, is_test=False
    )
    
    # Split the selected episodes by dataset to maintain homogeneous distribution
    train_episodes = []
    val_episodes = []
    train_transitions = 0
    val_transitions = 0
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.name
        
        # Get episodes for this dataset from selected episodes
        dataset_episodes = [ep for ep in selected_episodes if ep.parent.name == dataset_name]
        
        if not dataset_episodes:
            continue
            
        # Calculate split for this dataset
        num_episodes = len(dataset_episodes)
        train_count = max(1, int(num_episodes * split_ratio))  # At least 1 episode for training
        
        # Ensure we have at least 1 episode for validation if possible
        if train_count >= num_episodes and num_episodes > 1:
            train_count = num_episodes - 1
        
        dataset_train = dataset_episodes[:train_count]
        dataset_val = dataset_episodes[train_count:]
        
        # Count transitions for this dataset split
        for ep in dataset_train:
            ep_len = get_episode_length_from_cache_or_file(ep, transition_counts, episode_lengths)
            if ep_len:
                train_transitions += ep_len
        
        for ep in dataset_val:
            ep_len = get_episode_length_from_cache_or_file(ep, transition_counts, episode_lengths)
            if ep_len:
                val_transitions += ep_len
        
        train_episodes.extend(dataset_train)
        val_episodes.extend(dataset_val)
        
        print(f"Dataset {dataset_name}: {len(dataset_train)} train, {len(dataset_val)} val episodes")
    
    # Calculate actual split ratio
    total_transitions = train_transitions + val_transitions
    actual_split_ratio = train_transitions / total_transitions if total_transitions > 0 else 0
    
    print(f"Actual split: {train_transitions} train / {val_transitions} val transitions (ratio: {actual_split_ratio:.3f})")
    
    # Create training dataset
    train_temp_dir = Path(tempfile.mkdtemp())
    ColorPrint.green(f"Created training temp directory: {train_temp_dir}")
    _create_dataset_symlinks(train_episodes, train_temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed)
    
    # Create validation dataset
    val_temp_dir = Path(tempfile.mkdtemp())
    ColorPrint.green(f"Created validation temp directory: {val_temp_dir}")
    _create_dataset_symlinks(val_episodes, val_temp_dir, task_to_id, transition_counts, episode_lengths, preprocessed)
    
    # Create both loaders
    train_loader = make_replay_loader(
        replay_dir=train_temp_dir,
        max_size=1000000 if max_size is None else max_size,
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    
    val_loader = make_replay_loader(
        replay_dir=val_temp_dir,
        max_size=1000000,  # No size limit for validation
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    
    # Attach task mapping to both loaders
    for loader in [train_loader, val_loader]:
        loader.task_to_id = task_to_id
        loader.id_to_task = id_to_task
        loader.num_tasks = len(task_to_id)
    
    ColorPrint.green(f"Training split: {len(train_episodes)} episodes with {train_transitions} transitions")
    ColorPrint.green(f"Validation split: {len(val_episodes)} episodes with {val_transitions} transitions")
    
    return train_loader, val_loader