import os
import numpy as np
import torch
import pickle
import argparse
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Importa entrambi i tipi di buffer
from pretrain_taco_multi_task import OfflineReplayBuffer, make_offline_replay_loader
from replay_buffer import make_replay_loader

class DebugReplayBuffer:
    def __init__(self, dataset_path, temp_dir="temp_episodes", multistep=3, nstep=3, discount=0.99, 
                 batch_size=32, num_workers=0):
        """
        Debug tool to compare original ReplayBuffer and new OfflineReplayBuffer
        
        Args:
            dataset_path: Path to the dataset pickle file created by metaworld_image_dataset.py
            temp_dir: Directory to store temporary episode files for original ReplayBuffer
            multistep: Multistep parameter (should be same for both buffers)
            nstep: N-step returns parameter (should be same for both buffers)
            discount: Discount factor for rewards
            batch_size: Batch size for sampling
            num_workers: Number of worker processes
        """
        self.dataset_path = dataset_path
        self.temp_dir = Path(temp_dir)
        self.multistep = multistep
        self.nstep = nstep
        self.discount = discount
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load the dataset
        print(f"Caricamento dataset da {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
        # Print dataset info
        print(f"Dataset caricato: {len(self.dataset['observations'])} transizioni")
        print(f"Shape osservazioni: {self.dataset['observations'].shape}")
        print(f"Shape azioni: {self.dataset['actions'].shape}")
    
    def setup_offline_buffer(self):
        """Setup the OfflineReplayBuffer"""
        print("\nInitializzazione OfflineReplayBuffer...")
        offline_loader = make_offline_replay_loader(
            dataset_paths=[self.dataset_path],
            batch_size=self.batch_size,
            multistep=self.multistep,
            nstep=self.nstep,
            discount=self.discount,
            split='train',
            train_ratio=1.0  # Use all data for training for comparison
        )
        return offline_loader
    
    def convert_to_episodes(self):
        """Convert dataset to episodes format for original ReplayBuffer"""
        print("\nConversione dataset in episodi per ReplayBuffer originale...")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Identify episode boundaries based on terminals
        episode_starts = [0]
        episode_ends = []
        
        for i, done in enumerate(self.dataset['terminals']):
            if done:
                episode_ends.append(i)
                if i + 1 < len(self.dataset['terminals']):
                    episode_starts.append(i + 1)
        
        # Handle case where last episode doesn't end with terminal=True
        if len(episode_ends) < len(episode_starts):
            episode_ends.append(len(self.dataset['terminals']) - 1)
        
        print(f"Dataset contiene {len(episode_starts)} episodi")
        
        # Save each episode as a separate file
        for ep_idx, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            episode = {
                'observation': self.dataset['observations'][start:end+1],
                'action': self.dataset['actions'][start:end+1],
                'reward': self.dataset['rewards'][start:end+1],
                'discount': np.ones_like(self.dataset['rewards'][start:end+1]),
                'terminal': self.dataset['terminals'][start:end+1],
            }
            
            # Add a dummy first transition as ReplayBuffer expects
            for key in episode:
                if key == 'observation':
                    episode[key] = np.vstack([episode[key][0:1], episode[key]])
                else:
                    episode[key] = np.concatenate([episode[key][0:1], episode[key]])
            
            # Calculate episode length
            eps_len = len(episode['observation']) - 1  # Subtract 1 for the dummy transition
            
            # Generate filename in the format expected by ReplayBuffer
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            filename = f'{ts}_{ep_idx}_{eps_len}.npz'
            
            # Save episode
            with open(os.path.join(self.temp_dir, filename), 'wb') as f:
                np.savez_compressed(f, **episode)
                
        print(f"Episodi salvati in {self.temp_dir}")
    
    def setup_original_buffer(self):
        """Setup the original ReplayBuffer"""
        print("\nInitializzazione ReplayBuffer originale...")
        replay_loader = make_replay_loader(
            replay_dir=self.temp_dir,
            max_size=1000000,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            save_snapshot=True,
            nstep=self.nstep,
            multistep=self.multistep,
            discount=self.discount
        )
        return replay_loader
    
    def visualize_samples(self, samples, title, output_path):
        """Visualize samples from a replay buffer"""
        if len(samples) == 7:  # Original ReplayBuffer returns 7 elements
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = samples
        else:  # OfflineReplayBuffer returns 6 elements
            obs, action, action_seq, reward, next_obs, r_next_obs = samples
            
        # Convert to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
            action = action.numpy()
            action_seq = action_seq.numpy()
            reward = reward.numpy()
            next_obs = next_obs.numpy()
            r_next_obs = r_next_obs.numpy()
        
        # Create visualization grid
        num_samples = min(4, len(obs))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(num_samples):
            # Handle different channel layouts
            if obs.shape[-1] == 3:  # HWC format
                axes[i, 0].imshow(obs[i])
                axes[i, 1].imshow(r_next_obs[i])
                axes[i, 2].imshow(next_obs[i])
            else:  # CHW format
                # If frame-stacked, use only the first 3 channels or first frame
                if obs.shape[1] > 3:
                    # Take first 3 channels or combine channels to make RGB
                    axes[i, 0].imshow(np.transpose(obs[i][:3], (1, 2, 0)))
                    axes[i, 1].imshow(np.transpose(r_next_obs[i][:3], (1, 2, 0)))
                    axes[i, 2].imshow(np.transpose(next_obs[i][:3], (1, 2, 0)))
                else:
                    axes[i, 0].imshow(np.transpose(obs[i], (1, 2, 0)))
                    axes[i, 1].imshow(np.transpose(r_next_obs[i], (1, 2, 0)))
                    axes[i, 2].imshow(np.transpose(next_obs[i], (1, 2, 0)))
            
            axes[i, 0].set_title(f"Obs {i}")
            axes[i, 1].set_title(f"R-Next Obs {i}, Reward: {reward[i]:.4f}")
            axes[i, 2].set_title(f"Next Obs {i}")
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualizzazione salvata in {output_path}")
    
    def compare_statistics(self, original_samples, offline_samples):
        """Compare statistical properties of samples from both buffers"""
        # Unpack samples
        if len(original_samples) == 7:
            orig_obs, orig_action, orig_action_seq, orig_reward, orig_discount, orig_next_obs, orig_r_next_obs = original_samples
        else:
            raise ValueError("ReplayBuffer originale dovrebbe restituire 7 elementi")
            
        if len(offline_samples) == 6:
            off_obs, off_action, off_action_seq, off_reward, off_next_obs, off_r_next_obs = offline_samples
        else:
            raise ValueError("OfflineReplayBuffer dovrebbe restituire 6 elementi")
        
        # Convert to numpy if needed
        if isinstance(orig_obs, torch.Tensor):
            orig_obs = orig_obs.numpy()
            orig_action = orig_action.numpy()
            orig_reward = orig_reward.numpy()
            
        if isinstance(off_obs, torch.Tensor):
            off_obs = off_obs.numpy()
            off_action = off_action.numpy()
            off_reward = off_reward.numpy()
        
        # Compare statistics
        print("\n=== Confronto statistiche ===")
        print(f"Osservazioni shape: Original {orig_obs.shape}, Offline {off_obs.shape}")
        print(f"Azioni shape: Original {orig_action.shape}, Offline {off_action.shape}")
        print(f"Action seq shape: Original {orig_action_seq.shape}, Offline {off_action_seq.shape}")
        print(f"Rewards shape: Original {orig_reward.shape}, Offline {off_reward.shape}")
        
        print("\nOsservazioni:")
        print(f"Media: Original {np.mean(orig_obs):.4f}, Offline {np.mean(off_obs):.4f}")
        print(f"Std: Original {np.std(orig_obs):.4f}, Offline {np.std(off_obs):.4f}")
        print(f"Min: Original {np.min(orig_obs):.4f}, Offline {np.min(off_obs):.4f}")
        print(f"Max: Original {np.max(orig_obs):.4f}, Offline {np.max(off_obs):.4f}")
        
        print("\nAzioni:")
        print(f"Media: Original {np.mean(orig_action):.4f}, Offline {np.mean(off_action):.4f}")
        print(f"Std: Original {np.std(orig_action):.4f}, Offline {np.std(off_action):.4f}")
        print(f"Min: Original {np.min(orig_action):.4f}, Offline {np.min(off_action):.4f}")
        print(f"Max: Original {np.max(orig_action):.4f}, Offline {np.max(off_action):.4f}")
        
        print("\nRewards:")
        print(f"Media: Original {np.mean(orig_reward):.4f}, Offline {np.mean(off_reward):.4f}")
        print(f"Std: Original {np.std(orig_reward):.4f}, Offline {np.std(off_reward):.4f}")
        print(f"Min: Original {np.min(orig_reward):.4f}, Offline {np.min(off_reward):.4f}")
        print(f"Max: Original {np.max(orig_reward):.4f}, Offline {np.max(off_reward):.4f}")
        
        # Calculate difference metrics
        stats = {
            "obs_mean_diff": np.abs(np.mean(orig_obs) - np.mean(off_obs)),
            "action_mean_diff": np.abs(np.mean(orig_action) - np.mean(off_action)),
            "reward_mean_diff": np.abs(np.mean(orig_reward) - np.mean(off_reward)),
        }
        
        print("\nDifferenze:")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")
    
    def run_comparison(self, num_batches=10):
        """Run full comparison between the two replay buffers"""
        # Setup both replay buffers
        self.convert_to_episodes()
        original_loader = self.setup_original_buffer()
        offline_loader = self.setup_offline_buffer()
        
        # Get iterators for both loaders
        original_iter = iter(original_loader)
        offline_iter = iter(offline_loader)
        
        # Get samples from both buffers
        print("\nEstraendo campioni da entrambi i buffer...")
        
        # Sample and visualize from both buffers
        for i in range(num_batches):
            print(f"\nBatch {i+1}/{num_batches}:")
            
            original_batch = next(original_iter)
            offline_batch = next(offline_iter)
            
            # Visualize first batch only
            if i == 0:
                self.visualize_samples(
                    original_batch, 
                    "Campioni da ReplayBuffer originale", 
                    "original_samples.png"
                )
                
                self.visualize_samples(
                    offline_batch, 
                    "Campioni da OfflineReplayBuffer", 
                    "offline_samples.png"
                )
            
            # Compare statistics for all batches
            self.compare_statistics(original_batch, offline_batch)
        
        print("\nAnalisi completata! Controlla le immagini salvate per una verifica visiva.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug tool per confrontare replay buffers")
    parser.add_argument("--dataset_path", type=str, required=True, help="Percorso al file del dataset pickle")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch per il campionamento")
    parser.add_argument("--multistep", type=int, default=3, help="Parametro multistep")
    parser.add_argument("--nstep", type=int, default=3, help="Parametro n-step")
    parser.add_argument("--discount", type=float, default=0.99, help="Fattore di sconto")
    parser.add_argument("--num_batches", type=int, default=3, help="Numero di batch da analizzare")
    parser.add_argument("--temp_dir", type=str, default="temp_episodes", help="Directory per i file temporanei")
    args = parser.parse_args()
    
    # Esegui il debug
    debugger = DebugReplayBuffer(
        dataset_path=args.dataset_path,
        temp_dir=args.temp_dir,
        multistep=args.multistep,
        nstep=args.nstep,
        discount=args.discount,
        batch_size=args.batch_size
    )
    
    debugger.run_comparison(num_batches=args.num_batches)