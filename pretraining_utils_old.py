from replay_buffer import make_replay_loader
from pathlib import Path

def load_unified_dataset(root_dir, batch_size=32, num_workers=4,
                         nstep=3, multistep=3, discount=0.99):
    """
    Load all episodes from all dataset subdirectories as a single unified dataset
    
    Args:
        root_dir: Root directory containing dataset subdirectories with episodes
        batch_size: Batch size for sampling
        num_workers: Number of workers for parallel loading
        nstep: N-step returns parameter
        multistep: Multistep parameter 
        discount: Discount factor
    """
    replay_dir = Path(root_dir)
    
    # Count subdirectories to give feedback
    dataset_dirs = [d for d in replay_dir.iterdir() if d.is_dir()]
    print(f"Found {len(dataset_dirs)} dataset subdirectories")
    
    # Count total episodes
    episode_count = 0
    for dataset_dir in dataset_dirs:
        episode_count += len(list(dataset_dir.glob('*.npz')))
    print(f"Total episodes across all datasets: {episode_count}")
    
    # Create loader using the root directory (will now recursively find all episodes)
    loader = make_replay_loader(
        replay_dir=replay_dir,
        max_size=1000000,  # Adjust as needed
        batch_size=batch_size,
        num_workers=num_workers,
        save_snapshot=True,
        nstep=nstep,
        multistep=multistep,
        discount=discount
    )
    print(f"Total samples in loader: {len(loader)}")
    return loader