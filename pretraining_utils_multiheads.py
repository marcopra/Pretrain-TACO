from replay_buffer_multi_task import make_replay_loader
from pathlib import Path
import json
import random
import numpy as np
import tempfile
import os
import threading

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

def select_episodes_homogeneous(dataset_paths, max_episodes_per_dataset, max_size=None, is_test=False):
    """Select episodes from datasets to ensure homogeneous distribution"""
    all_episodes = []
    dataset_info = []
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        episode_files = list(dataset_path.glob('*.npz'))
        
        if max_episodes_per_dataset is not None:
            episode_files = episode_files[:max_episodes_per_dataset]
        
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
        if total_transitions > max_size:       
            ColorPrint.yellow(f"Warning: Total transitions ({total_transitions}) exceed max_size ({max_size})")
        transitions_per_dataset = max_size // len(dataset_paths)
        ColorPrint.green(f"Homogeneous loading: {transitions_per_dataset} transitions per dataset")
        
        selected_episodes = []
        actual_transitions_loaded = 0
        
        for info in dataset_info:
            dataset_episodes = info['episodes'].copy()
            random.shuffle(dataset_episodes)
            
            current_transitions = 0
            dataset_selected = []
            
            for episode_file in dataset_episodes:
                episode_transitions = count_transitions_in_episodes([episode_file])
                
                if current_transitions + episode_transitions <= transitions_per_dataset:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_transitions
                elif len(dataset_selected) == 0:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_transitions
                    ColorPrint.yellow(f"Added episode with {episode_transitions} transitions (exceeds per-dataset limit of {transitions_per_dataset})")
                    break
                else:
                    break
            
            selected_episodes.extend(dataset_selected)
            actual_transitions_loaded += current_transitions
            print(f"Dataset {info['path'].name}: selected {len(dataset_selected)} episodes with {current_transitions} transitions")
        
        print(f"Total selected: {len(selected_episodes)} episodes with {actual_transitions_loaded} transitions")
        return selected_episodes
    else:
        if not is_test:
            ColorPrint.green(f"Using all {total_transitions} transitions from {len(dataset_paths)} datasets")
        return all_episodes

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

def create_symlink_with_proper_naming(episode_file, dest_dir, counter, task_id=None):
    """Create symlink to episode file with proper naming format for replay_buffer"""
    episode_file = Path(episode_file)
    
    # Load episode data first to get episode length and check for task_id
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
    
    # If task_id is provided and episode doesn't have task_id, we need to create a new file
    if task_id is not None and not has_task_id:
        # Create a temporary file with task_id added
        from replay_buffer_multi_task import save_episode_with_task_id
        
        # Create temporary file with task_id
        temp_file = dest_dir / f"temp_{counter:06d}.npz"
        save_episode_with_task_id(episode_dict, temp_file, task_id)
        
        # Create proper filename
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
        dest_path = dest_dir / proper_name
        
        # Move temp file to proper name
        temp_file.rename(dest_path)
        return dest_path
    else:
        # Create proper filename format that replay_buffer expects
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
        dest_path = dest_dir / proper_name
        
        # Create symlink instead of copying
        os.symlink(episode_file.absolute(), dest_path)
        return dest_path

def can_use_direct_loading(dataset_paths, selected_episodes, max_episodes_per_dataset, homogeneous):
    """Check if we can use direct loading without temp directory"""
    # If we're using all episodes from all datasets without filtering, we can load directly
    if not homogeneous and max_episodes_per_dataset is None:
        return True, dataset_paths[0] if len(dataset_paths) == 1 else None
    
    # If all selected episodes are from a single dataset and we're not filtering
    if len(dataset_paths) == 1 and not homogeneous:
        dataset_path = Path(dataset_paths[0])
        all_episodes_in_dataset = list(dataset_path.glob('*.npz'))
        if len(selected_episodes) == len(all_episodes_in_dataset):
            return True, dataset_path
    
    return False, None

class EpisodePool:
    """Manages a shared pool of episodes for split validation to prevent data leakage"""
    def __init__(self, temp_dir, all_episodes, task_to_id):
        self.temp_dir = Path(temp_dir)
        self.task_to_id = task_to_id
        self.lock = threading.Lock()
        self.used_episodes = set()
        
        # Create symlinks for all episodes
        self.episode_map = {}  # Maps original episode path to symlink path
        counter = 0
        
        for episode_file in all_episodes:
            task_name = extract_task_name_from_path(episode_file.parent)
            task_id = task_to_id[task_name]
            
            processed_path = create_symlink_with_proper_naming(
                episode_file, self.temp_dir, counter, task_id
            )
            if processed_path is not None:
                self.episode_map[str(episode_file)] = processed_path
                counter += 1
    
    def mark_episode_used(self, episode_path):
        """Mark an episode as used by removing its symlink"""
        with self.lock:
            episode_path_str = str(episode_path)
            if episode_path_str in self.episode_map:
                symlink_path = self.episode_map[episode_path_str]
                if symlink_path.exists():
                    symlink_path.unlink()
                self.used_episodes.add(episode_path_str)
    
    def get_available_episodes(self):
        """Get list of available (unused) episode symlinks"""
        with self.lock:
            available = []
            for orig_path, symlink_path in self.episode_map.items():
                if orig_path not in self.used_episodes and symlink_path.exists():
                    available.append(symlink_path)
            return available

# Global episode pool for split validation
_episode_pool = None

def set_episode_pool(pool):
    """Set the global episode pool"""
    global _episode_pool
    _episode_pool = pool

def get_episode_pool():
    """Get the global episode pool"""
    return _episode_pool

def load_unified_dataset(config_or_path, batch_size=32, num_workers=4,
                         nstep=3, multistep=3, discount=0.99, 
                         max_episodes_per_dataset=8, max_size=None,
                         homogeneous=False, is_test=False, 
                         validation_split_ratio=0.8, use_training_split=True, **kwargs):
    """
    Load episodes from datasets specified in config or folder structure
    
    Args:
        config_or_path: Either path to config.json or folder with pretraining_datasets/test_dataset
        max_episodes_per_dataset: Maximum episodes to load per dataset (for training only when splitting)
        max_size: Maximum size of replay buffer (for training only when splitting)
        homogeneous: Whether to load transitions evenly across datasets
        is_test: Whether this is test dataset (affects max_size behavior)
        validation_split_ratio: Ratio of data to use for training when splitting
        use_training_split: Whether to use training split (True) or validation split (False)
        **kwargs: Additional arguments
    """
    config_path = Path(config_or_path)
    
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
    
    print(f"Found {len(dataset_paths)} dataset directories")
    
    # Create task mapping
    task_to_id, id_to_task = create_task_mapping(dataset_paths)
    
    # Check if we're doing split validation
    using_split_validation = 'validation_split_ratio' in kwargs and not is_test
    
    # Handle split validation case
    if using_split_validation:
        global _episode_pool
        
        if use_training_split:
            # First call: create episode pool and load training data
            ColorPrint.blue("Creating episode pool for split validation")
            
            # Collect all episodes without any filtering first
            all_episodes = []
            for dataset_path in dataset_paths:
                dataset_path = Path(dataset_path)
                episode_files = list(dataset_path.glob('*.npz'))
                all_episodes.extend(episode_files)
            
            # Create temporary directory and episode pool
            temp_dir = Path(tempfile.mkdtemp())
            print(f"Temporary directory created at: {temp_dir}")
            
            _episode_pool = EpisodePool(temp_dir, all_episodes, task_to_id)
            
            # Now select episodes for training with constraints
            if homogeneous:
                selected_episodes = select_episodes_homogeneous(
                    dataset_paths, max_episodes_per_dataset, max_size, is_test
                )
            else:
                selected_episodes = []
                for dataset_path in dataset_paths:
                    dataset_path = Path(dataset_path)
                    episode_files = list(dataset_path.glob('*.npz'))
                    
                    if max_episodes_per_dataset is not None:
                        episode_files = episode_files[:max_episodes_per_dataset]
                    
                    selected_episodes.extend(episode_files)
            
            # Mark selected episodes as used in the pool
            for episode_file in selected_episodes:
                _episode_pool.mark_episode_used(episode_file)
            
            # Get the symlink paths for training
            training_symlinks = []
            for episode_file in selected_episodes:
                episode_path_str = str(episode_file)
                if episode_path_str in _episode_pool.episode_map:
                    symlink_path = _episode_pool.episode_map[episode_path_str]
                    if symlink_path.exists():  # Double check it wasn't removed
                        training_symlinks.append(symlink_path)
            
            print(f"Selected {len(training_symlinks)} episodes for training")
            
            # Create a temporary directory with only training symlinks
            train_temp_dir = Path(tempfile.mkdtemp())
            counter = 0
            for symlink_path in training_symlinks:
                # Create new symlink in training directory
                new_name = f"train_{counter:06d}_{symlink_path.name.split('_')[-1]}"
                new_path = train_temp_dir / new_name
                os.symlink(symlink_path.readlink(), new_path)
                counter += 1
            
            loader_dir = train_temp_dir
            
        else:
            # Second call: load validation data from remaining episodes
            ColorPrint.blue("Loading validation split from remaining episodes")
            
            if _episode_pool is None:
                raise RuntimeError("Episode pool not initialized. Training split must be loaded first.")
            
            # Get available (unused) episodes for validation
            available_episodes = _episode_pool.get_available_episodes()
            print(f"Found {len(available_episodes)} episodes available for validation")
            
            # Create validation temporary directory
            val_temp_dir = Path(tempfile.mkdtemp())
            counter = 0
            for symlink_path in available_episodes:
                # Create new symlink in validation directory
                new_name = f"val_{counter:06d}_{symlink_path.name.split('_')[-1]}"
                new_path = val_temp_dir / new_name
                if symlink_path.exists():  # Check if symlink still exists
                    os.symlink(symlink_path.readlink(), new_path)
                    counter += 1
            
            loader_dir = val_temp_dir
    
    else:
        # Original behavior for non-split cases
        if homogeneous:
            ColorPrint.green("Using homogeneous dataset loading")
            selected_episodes = select_episodes_homogeneous(
                dataset_paths, max_episodes_per_dataset, max_size, is_test
            )
        else:
            selected_episodes = []
            for dataset_path in dataset_paths:
                dataset_path = Path(dataset_path)
                episode_files = list(dataset_path.glob('*.npz'))
                
                if max_episodes_per_dataset is not None:
                    episode_files = episode_files[:max_episodes_per_dataset]
                
                selected_episodes.extend(episode_files)
        
        print(f"Selected {len(selected_episodes)} episodes total")
        
        # Create temporary directory with task IDs
        ColorPrint.yellow("Creating temporary directory with task IDs")
        temp_dir = Path(tempfile.mkdtemp())
        print(f"Temporary directory created at: {temp_dir}")
        
        counter = 0
        for episode_file in selected_episodes:
            # Get task ID for this episode
            task_name = extract_task_name_from_path(episode_file.parent)
            task_id = task_to_id[task_name]
            
            processed_path = create_symlink_with_proper_naming(
                episode_file, temp_dir, counter, task_id
            )
            if processed_path is not None:
                counter += 1
        
        loader_dir = temp_dir
    
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
    
    # Attach task mapping to loader for reference
    loader.task_to_id = task_to_id
    loader.id_to_task = id_to_task
    loader.num_tasks = len(task_to_id)
    
    return loader