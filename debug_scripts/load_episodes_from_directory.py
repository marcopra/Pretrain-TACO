import argparse
import pathlib
from pathlib import Path
from replay_buffer import make_replay_loader

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
    
    return loader

def test_loader(loader, num_batches=3):
    """Test the replay buffer by sampling a few batches"""
    print("\nTesting unified dataset loader by sampling batches...")
    
    try:
        # Sample batches
        loader_iter = iter(loader)
        for i in range(num_batches):
            print(f"Sampling batch {i+1}/{num_batches}...")
            
            # Get next batch
            batch = next(loader_iter)
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = batch
            
            # Print batch info
            print(f"  Observations shape: {obs.shape}")
            print(f"  Actions shape: {action.shape}")
            print(f"  Action sequence shape: {action_seq.shape}")
            print(f"  Rewards shape: {reward.shape}")
    
        print("\nLoader test successful!")
        return True
    except Exception as e:
        print(f"\nError testing loader: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Load all episodes from subdirectories as a unified dataset")
    parser.add_argument("--root_dir", type=str, default="data_episodes",
                        help="Root directory containing dataset subdirectories")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for sampling")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--test", action="store_true",
                        help="Test the loader by sampling batches")
    args = parser.parse_args()
    
    print(f"Loading all episodes from {args.root_dir} as a unified dataset")
    
    loader = load_unified_dataset(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if args.test:
        test_loader(loader)
    
    return loader

if __name__ == "__main__":
    main()
