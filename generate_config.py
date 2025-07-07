"""
Script to generate config.json files for TACO pretraining

Examples:
python generate_config.py --base_path data_episodes/MT50/0.33 --output config.json
python generate_config.py --base_path /path/to/datasets --n_pretrain 8 --n_test 2 --absolute_paths --output config.json
python generate_config.py --scan_path /path/to/unstructured --n_pretrain 10 --n_test 3 --output my_config.json
python generate_config.py --base_path data_episodes/MT50 --n_test 10 --permutations 5 --ood_constraints "basketball,bin-picking,button-press,push,shelf-place"
"""

import os
import json
import random
import argparse
import itertools
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
    # Removed - not needed anymore
    return 0


def extract_task_name(dataset_path):
    """Extract task name from dataset path"""
    dataset_path = Path(dataset_path)
    folder_name = dataset_path.name
    
    # Extract task name from folder name like "assembly-v2_task0_fs3_ar2_ri1_rg0_exp=0"
    # The task name is the part before the first underscore that's not "task"
    parts = folder_name.split('_')
    if parts:
        # Take the first part which should contain the task name
        task_part = parts[0]
        # Remove version suffixes like -v2, -v1
        task_name = task_part.replace('-v2', '').replace('-v1', '')
        return task_name
    
    # Fallback to using the full folder name if parsing fails
    return folder_name


def encode_task_name_camelcase(task_name):
    """Convert task name to CamelCase encoding"""
    # Remove common suffixes like -v2
    clean_name = task_name.replace('-v2', '').replace('-v1', '')
    
    # Split by hyphens and capitalize each part
    parts = clean_name.split('-')
    camel_case = ''.join(word.capitalize() for word in parts)
    
    return camel_case


def generate_encoded_task_string(task_names, max_length=200):
    """Generate encoded string from task names, choosing shortest representation"""
    # Encode all task names
    encoded_tasks = [encode_task_name_camelcase(task) for task in sorted(task_names)]
    
    # Try the full concatenated string first
    full_string = ''.join(encoded_tasks)
    
    if len(full_string) <= max_length:
        return full_string, 'ID' if len(task_names) > 1 else 'ID'
    
    # If too long, try first 3-4 characters of each task
    abbreviated = ''.join(task[:4] for task in encoded_tasks)
    
    if len(abbreviated) <= max_length:
        return abbreviated, 'ID' if len(task_names) > 1 else 'ID'
    
    # If still too long, try first 3 characters
    abbreviated_short = ''.join(task[:3] for task in encoded_tasks)
    
    if len(abbreviated_short) <= max_length:
        return abbreviated_short, 'ID' if len(task_names) > 1 else 'ID'
    
    # If still too long, take first N tasks and add "Plus" at the end
    if len(encoded_tasks) > 10:
        # Take first 8 tasks with 3 chars each + "Plus" = ~28 chars
        truncated = ''.join(task[:3] for task in encoded_tasks[:8]) + "Plus"
        return truncated, 'ID' if len(task_names) > 1 else 'ID'
    
    # Last resort: just use count (this should rarely happen now)
    return f"{len(task_names)}Tasks", 'ID' if len(task_names) > 1 else 'ID'


def check_ood_constraints(test_datasets, ood_constraints):
    """Check if OOD constraints are satisfied"""
    test_task_names = [extract_task_name(dataset) for dataset in test_datasets]
    
    for constraint in ood_constraints:
        if constraint not in test_task_names:
            return False
    return True


def generate_valid_permutation(all_datasets, n_test, ood_constraints):
    """Generate a single valid permutation that satisfies OOD constraints"""
    # Find datasets that match OOD constraints
    ood_datasets = []
    for constraint in ood_constraints:
        matching_datasets = [dataset for dataset in all_datasets 
                           if extract_task_name(dataset) == constraint]
        if not matching_datasets:
            raise ValueError(f"OOD constraint task '{constraint}' not found in datasets")
        ood_datasets.extend(matching_datasets)
    
    # Remove duplicates while preserving order
    ood_datasets = list(dict.fromkeys(ood_datasets))
    
    if len(ood_datasets) > n_test:
        raise ValueError(f"OOD constraints require {len(ood_datasets)} datasets but only {n_test} test datasets requested")
    
    # Get remaining datasets for selection
    remaining_datasets = [d for d in all_datasets if d not in ood_datasets]
    additional_test_needed = n_test - len(ood_datasets)
    
    if additional_test_needed == 0:
        # Only OOD datasets in test set
        test_datasets = ood_datasets
        pretrain_datasets = remaining_datasets
    else:
        # Randomly select additional test datasets
        if len(remaining_datasets) < additional_test_needed:
            raise ValueError(f"Not enough remaining datasets to satisfy n_test requirement")
        
        additional_test = random.sample(remaining_datasets, additional_test_needed)
        test_datasets = ood_datasets + additional_test
        pretrain_datasets = [d for d in all_datasets if d not in test_datasets]
    
    return pretrain_datasets, test_datasets


def generate_config(base_path=None, scan_path=None, n_pretrain=None, n_test=2, 
                   absolute_paths=False, output="config.json", seed=None,
                   permutations=None, ood_constraints=None):
    """Generate configuration file(s)"""
    
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Parse OOD constraints
    if ood_constraints:
        ood_constraints = [task.strip() for task in ood_constraints.split(',')]
        print(f"OOD constraints: {ood_constraints}")
    else:
        ood_constraints = []
    
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
        all_datasets = pretraining_datasets + test_datasets
    else:
        # Scan for all datasets
        if base_path:
            print(f"No structured layout found in {base_path}, scanning for datasets...")
        else:
            print(f"Scanning {scan_path} for datasets...")
        
        all_datasets = find_datasets(working_path)
        
        if not all_datasets:
            raise ValueError(f"No datasets found in {working_path}")
    
    print(f"Found {len(all_datasets)} total datasets")
    
    # Calculate n_pretrain if not specified
    if n_pretrain is None:
        n_pretrain = len(all_datasets) - n_test
        print(f"Calculated n_pretrain = {n_pretrain}")
    
    # Validate parameters
    if n_pretrain + n_test != len(all_datasets):
        raise ValueError(f"n_pretrain ({n_pretrain}) + n_test ({n_test}) != total datasets ({len(all_datasets)})")
    
    # Generate permutations
    if permutations is None or permutations == 1:
        # Single configuration
        if ood_constraints:
            pretraining_datasets, test_datasets = generate_valid_permutation(all_datasets, n_test, ood_constraints)
        else:
            # Random split
            pretraining_datasets, test_datasets = split_datasets_randomly(all_datasets, n_pretrain, n_test)
        
        configs_to_generate = [(pretraining_datasets, test_datasets, 0)]
    else:
        # Multiple permutations - generate exactly what's requested
        configs_to_generate = []
        seen_configs = set()
        attempts = 0
        max_attempts_per_config = 4
        
        for perm_id in range(permutations):
            config_attempts = 0
            found_unique = False
            
            while config_attempts < max_attempts_per_config and not found_unique:
                config_attempts += 1
                attempts += 1
                
                try:
                    if ood_constraints:
                        pretrain_datasets, test_datasets = generate_valid_permutation(all_datasets, n_test, ood_constraints)
                    else:
                        pretrain_datasets, test_datasets = split_datasets_randomly(all_datasets, n_pretrain, n_test)
                    
                    # Create a hashable representation to check for duplicates
                    pretrain_names = tuple(sorted([extract_task_name(d) for d in pretrain_datasets]))
                    test_names = tuple(sorted([extract_task_name(d) for d in test_datasets]))
                    config_signature = (pretrain_names, test_names)
                    
                    if config_signature not in seen_configs:
                        seen_configs.add(config_signature)
                        configs_to_generate.append((pretrain_datasets, test_datasets, perm_id))
                        found_unique = True
                        print(f"Generated permutation {perm_id + 1}/{permutations}")
                    
                except Exception as e:
                    print(f"Error generating permutation {perm_id}: {e}")
                    continue
            
            if not found_unique:
                raise RuntimeError(f"Failed to generate unique permutation {perm_id + 1} after {max_attempts_per_config} attempts. "
                                 f"Total attempts so far: {attempts}")
        
        print(f"Successfully generated {len(configs_to_generate)} unique configurations in {attempts} total attempts")
    
    # Generate configuration files
    generated_files = []
    
    for pretraining_datasets, test_datasets, perm_id in configs_to_generate:
        # Convert to string paths
        pretraining_paths = to_path_strings(pretraining_datasets, absolute_paths)
        test_paths = to_path_strings(test_datasets, absolute_paths)
        
        # Generate encoded strings for naming - ensure alphabetical order
        pretrain_task_names = sorted([extract_task_name(d) for d in pretraining_datasets])
        test_task_names = sorted([extract_task_name(d) for d in test_datasets])
        
        pretrain_encoded, _ = generate_encoded_task_string(pretrain_task_names)
        test_encoded, _ = generate_encoded_task_string(test_task_names)
        
        # Determine which set to use for filename and set type
        if ood_constraints:
            # If we have OOD constraints, use the shorter encoding between ID and OOD
            if len(pretrain_encoded) <= len(test_encoded):
                encoded_tasks = pretrain_encoded
                set_type = "ID"
            else:
                encoded_tasks = test_encoded
                set_type = "OOD"
        else:
            # No OOD constraints, use pretraining tasks (ID) by default
            # unless test encoding is significantly shorter
            if len(test_encoded) < len(pretrain_encoded):
                encoded_tasks = test_encoded
                set_type = "ID"  # Still ID since no OOD constraints
            else:
                encoded_tasks = pretrain_encoded
                set_type = "ID"
        
        # Generate filename
        filename = f"MT{n_pretrain}{set_type}{encoded_tasks}.json"
        
        # Use provided output path for directory
        output_dir = Path(output).parent if output != "config.json" else Path(".")
        output_path = output_dir / filename
        
        # Create config dictionary
        config = {
            "pretraining_datasets": pretraining_paths,
            "test_datasets": test_paths,
            "metadata": {
                "total_pretraining_datasets": len(pretraining_paths),
                "total_test_datasets": len(test_paths),
                "n_pretrain": n_pretrain,
                "n_test": n_test,
                "generated_from": str(working_path),
                "absolute_paths": absolute_paths,
                "random_seed": seed,
                "permutation_id": perm_id if permutations and permutations > 1 else None,
                "ood_constraints": ood_constraints,
                "pretrain_tasks": pretrain_task_names,
                "test_tasks": test_task_names,
                "encoded_tasks": encoded_tasks,
                "set_type": set_type
            }
        }
        
        # Remove episode counting - not needed
        
        # Save config
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        generated_files.append(str(output_path))
        
        print(f"\nGenerated config saved to: {output_path}")
        print(f"Pretraining datasets: {len(pretraining_paths)}")
        print(f"Test datasets: {len(test_paths)}")
        
        if len(configs_to_generate) == 1:  # Only print details for single config
            print("\nPretraining datasets:")
            for i, path in enumerate(pretraining_paths, 1):
                task_name = extract_task_name(path)
                print(f"  {i}. {task_name} -> {path}")
            
            print("\nTest datasets:")
            for i, path in enumerate(test_paths, 1):
                task_name = extract_task_name(path)
                print(f"  {i}. {task_name} -> {path}")
    
    print(f"\nTotal generated files: {len(generated_files)}")
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config.json for TACO pretraining")
    
    # Path arguments (mutually exclusive)
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument('--base_path', type=str, 
                           help='Base path with pretraining_datasets/test_dataset structure (or will scan for datasets)')
    path_group.add_argument('--scan_path', type=str,
                           help='Path to scan for datasets (will randomly split)')
    
    # Dataset split arguments
    parser.add_argument('--n_pretrain', type=int, default=None,
                       help='Number of datasets for pretraining (default: total - n_test)')
    parser.add_argument('--n_test', type=int, default=2,
                       help='Number of datasets for testing')
    
    # Permutation arguments
    parser.add_argument('--permutations', type=int, default=None,
                       help='Number of permutations to generate (default: None for single config)')
    parser.add_argument('--ood_constraints', type=str, default=None,
                       help='Comma-separated list of tasks that must be in test set (e.g., "basketball,bin-picking,button-press")')
    
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
            seed=args.seed,
            permutations=args.permutations,
            ood_constraints=args.ood_constraints
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
        generate_config(
            base_path=args.base_path,
            scan_path=args.scan_path,
            n_pretrain=args.n_pretrain,
            n_test=args.n_test,
            absolute_paths=args.absolute_paths,
            output=args.output,
            seed=args.seed,
            permutations=args.permutations,
            ood_constraints=args.ood_constraints
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
