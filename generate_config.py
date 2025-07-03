"""
Script to generate config.json files for TACO pretraining

Examples:
python generate_config.py --base_path data_episodes/MT50/0.33 --output config.json
python generate_config.py --base_path /path/to/datasets --n_pretrain 8 --n_test 2 --absolute_paths --output config.json
python generate_config.py --scan_path /path/to/unstructured --n_pretrain 10 --n_test 3 --output my_config.json
"""

import os
import json
import random
import argparse
from pathlib import Path


def find_datasets(path):
    """Find all directories containing .npz files (episode datasets)"""
    path = Path(path)
    datasets = []
    
    if not path.exists():
        print(f"Warning: Path {path} does not exist")
        return datasets
    
    # Look for directories containing .npz files
    for item in path.rglob('*'):
        if item.is_dir():
            npz_files = list(item.glob('*.npz'))
            if npz_files:
                datasets.append(item)
                print(f"Found dataset: {item} ({len(npz_files)} episodes)")
    
    return datasets


def has_structured_layout(base_path):
    """Check if the path has pretraining_datasets and test_dataset folders"""
    base_path = Path(base_path)
    pretraining_dir = base_path / "pretraining_datasets"
    test_dir = base_path / "test_dataset"
    
    return pretraining_dir.exists() and test_dir.exists()


def get_structured_datasets(base_path):
    """Get datasets from structured layout (pretraining_datasets and test_dataset folders)"""
    base_path = Path(base_path)
    
    pretraining_datasets = []
    test_datasets = []
    
    # Get pretraining datasets
    pretraining_dir = base_path / "pretraining_datasets"
    if pretraining_dir.exists():
        for dataset_dir in pretraining_dir.iterdir():
            if dataset_dir.is_dir() and list(dataset_dir.glob('*.npz')):
                pretraining_datasets.append(dataset_dir)
    
    # Get test datasets
    test_dir = base_path / "test_dataset"
    if test_dir.exists():
        for dataset_dir in test_dir.iterdir():
            if dataset_dir.is_dir() and list(dataset_dir.glob('*.npz')):
                test_datasets.append(dataset_dir)
    
    return pretraining_datasets, test_datasets


def split_datasets_randomly(datasets, n_pretrain, n_test):
    """Randomly split datasets into pretraining and test sets"""
    if len(datasets) < n_pretrain + n_test:
        print(f"Warning: Found only {len(datasets)} datasets, but requested {n_pretrain + n_test} total")
        print(f"Adjusting to use {min(len(datasets), n_pretrain)} for pretraining and the rest for test")
        n_pretrain = min(len(datasets), n_pretrain)
        n_test = max(0, len(datasets) - n_pretrain)
    
    # Shuffle and split
    datasets_copy = datasets.copy()
    random.shuffle(datasets_copy)
    
    pretraining_datasets = datasets_copy[:n_pretrain]
    test_datasets = datasets_copy[n_pretrain:n_pretrain + n_test]
    
    return pretraining_datasets, test_datasets


def to_path_strings(datasets, absolute_paths=False, base_path=None):
    """Convert Path objects to strings (relative or absolute)"""
    if absolute_paths:
        return [str(dataset.resolve()) for dataset in datasets]
    else:
        # Make paths relative to current working directory, not base_path
        cwd = Path.cwd()
        try:
            return [str(dataset.resolve().relative_to(cwd)) for dataset in datasets]
        except ValueError:
            # If relative path calculation fails, use absolute paths
            print("Warning: Could not compute relative paths, using absolute paths")
            return [str(dataset.resolve()) for dataset in datasets]


def count_episodes_in_dataset(dataset_path):
    """Count number of episodes (.npz files) in a dataset"""
    dataset_path = Path(dataset_path)
    npz_files = list(dataset_path.glob('*.npz'))
    return len(npz_files)


def generate_config(base_path=None, scan_path=None, n_pretrain=8, n_test=2, 
                   absolute_paths=False, output="config.json", seed=None):
    """Generate configuration file"""
    
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Determine which path to use
    if base_path and scan_path:
        raise ValueError("Cannot specify both --base_path and --scan_path")
    
    working_path = base_path or scan_path
    if not working_path:
        raise ValueError("Must specify either --base_path or --scan_path")
    
    working_path = Path(working_path)
    
    # Check if using structured layout
    if base_path and has_structured_layout(base_path):
        print(f"Found structured layout in {base_path}")
        pretraining_datasets, test_datasets = get_structured_datasets(base_path)
        print(f"Pretraining datasets: {len(pretraining_datasets)}")
        print(f"Test datasets: {len(test_datasets)}")
    else:
        # Scan for all datasets and split randomly
        if base_path:
            print(f"No structured layout found in {base_path}, scanning for datasets...")
        else:
            print(f"Scanning {scan_path} for datasets...")
        
        all_datasets = find_datasets(working_path)
        
        if not all_datasets:
            raise ValueError(f"No datasets found in {working_path}")
        
        print(f"Found {len(all_datasets)} total datasets")
        pretraining_datasets, test_datasets = split_datasets_randomly(all_datasets, n_pretrain, n_test)
        
        print(f"Randomly assigned {len(pretraining_datasets)} datasets to pretraining")
        print(f"Randomly assigned {len(test_datasets)} datasets to test")
    
    # Convert to string paths
    pretraining_paths = to_path_strings(pretraining_datasets, absolute_paths)
    test_paths = to_path_strings(test_datasets, absolute_paths)
    
    # Create config dictionary
    config = {
        "pretraining_datasets": pretraining_paths,
        "test_datasets": test_paths,
        "metadata": {
            "total_pretraining_datasets": len(pretraining_paths),
            "total_test_datasets": len(test_paths),
            "generated_from": str(working_path),
            "absolute_paths": absolute_paths,
            "random_seed": seed
        }
    }
    
    # Add episode counts - use the actual paths, not the dataset objects
    config["metadata"]["pretraining_episodes"] = {}
    config["metadata"]["test_episodes"] = {}
    
    total_pretrain_episodes = 0
    for path in pretraining_paths:
        # Resolve path properly for counting
        if absolute_paths:
            count_path = path
        else:
            count_path = Path.cwd() / path
        episodes = count_episodes_in_dataset(count_path)
        config["metadata"]["pretraining_episodes"][path] = episodes
        total_pretrain_episodes += episodes
    
    total_test_episodes = 0
    for path in test_paths:
        # Resolve path properly for counting
        if absolute_paths:
            count_path = path
        else:
            count_path = Path.cwd() / path
        episodes = count_episodes_in_dataset(count_path)
        config["metadata"]["test_episodes"][path] = episodes
        total_test_episodes += episodes
    
    config["metadata"]["total_pretraining_episodes"] = total_pretrain_episodes
    config["metadata"]["total_test_episodes"] = total_test_episodes
    
    # Save config
    output_path = Path(output)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGenerated config saved to: {output_path}")
    print(f"Pretraining datasets: {len(pretraining_paths)} ({total_pretrain_episodes} episodes)")
    print(f"Test datasets: {len(test_paths)} ({total_test_episodes} episodes)")
    
    # Print dataset lists
    print("\nPretraining datasets:")
    for i, path in enumerate(pretraining_paths, 1):
        episodes = config["metadata"]["pretraining_episodes"][path]
        print(f"  {i}. {path} ({episodes} episodes)")
    
    print("\nTest datasets:")
    for i, path in enumerate(test_paths, 1):
        episodes = config["metadata"]["test_episodes"][path]
        print(f"  {i}. {path} ({episodes} episodes)")
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config.json for TACO pretraining")
    
    # Path arguments (mutually exclusive)
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument('--base_path', type=str, 
                           help='Base path with pretraining_datasets/test_dataset structure (or will scan for datasets)')
    path_group.add_argument('--scan_path', type=str,
                           help='Path to scan for datasets (will randomly split)')
    
    # Dataset split arguments
    parser.add_argument('--n_pretrain', type=int, default=8,
                       help='Number of datasets for pretraining (default: 8)')
    parser.add_argument('--n_test', type=int, default=2,
                       help='Number of datasets for testing (default: 2)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='config.json',
                       help='Output config file path (default: config.json)')
    parser.add_argument('--absolute_paths', action='store_true',
                       help='Use absolute paths instead of relative paths')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for dataset splitting')
    
    args = parser.parse_args()
    
    try:
        generate_config(
            base_path=args.base_path,
            scan_path=args.scan_path,
            n_pretrain=args.n_pretrain,
            n_test=args.n_test,
            absolute_paths=args.absolute_paths,
            output=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
