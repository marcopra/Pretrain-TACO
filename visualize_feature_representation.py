import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict
import math
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import utils
from omegaconf import OmegaConf
from hydra.utils import instantiate

CFG_ROOT = Path(__file__).resolve().parent / "cfgs"

def detect_agent_type(checkpoint_path, device):
    """
    Detect the type of agent from a checkpoint or snapshot file.
    Returns the agent type ("taco" or "taco_proprio") and the checkpoint data.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"File not found: {checkpoint_path}")
    
    print(f"Loading checkpoint to detect agent type: {checkpoint_path}")
    # Set weights_only=False to allow loading pickled agent objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine if this is a snapshot (contains 'agent') or checkpoint (contains 'encoder')
    is_snapshot = 'agent' in checkpoint
    
    if is_snapshot:
        # For snapshots, check the agent's encoder type
        agent = checkpoint['agent']
        if hasattr(agent, 'encoder') and hasattr(agent.encoder, 'convnet'):
            return "taco", checkpoint, is_snapshot
        elif hasattr(agent, 'encoder') and hasattr(agent.encoder, 'encoder'):
            return "taco_proprio", checkpoint, is_snapshot
        else:
            raise ValueError("Could not determine agent type from snapshot")
    else:
        # For checkpoints, examine encoder architecture
        if 'encoder' in checkpoint:
            encoder_state = checkpoint['encoder']
            # Check if this is a convolutional encoder (image-based) or MLP (proprio-based)
            if any('convnet' in key for key in encoder_state.keys()):
                return "taco", checkpoint, is_snapshot
            else:
                return "taco_proprio", checkpoint, is_snapshot
        else:
            raise ValueError("Could not find encoder in checkpoint")

def extract_dataset_shapes(episodes, agent_type):
    for episode in episodes:
        if agent_type == "taco_proprio" and 'proprio_observation' in episode:
            states = episode['proprio_observation']
        elif agent_type == "taco" and 'observation' in episode:
            states = episode['observation']
        else:
            continue
        actions = episode.get('action')
        if states is None or actions is None:
            continue
        states = states[1:]
        actions = actions[1:]
        if len(states) == 0 or len(actions) == 0:
            continue
        obs_shape = states[0].shape
        action_shape = actions[0].shape
        return obs_shape, action_shape
    raise ValueError("Unable to infer shapes from dataset")

def build_agent_from_hydra(agent_type, device, obs_shape, action_shape):
    """
    Build agent using hydra configuration, setting the proper agent@_global_ value
    and resolving hyperparameter references.
    """
    # Load main config
    main_cfg_path = CFG_ROOT / "config_gym.yaml"
    print(f"Loading main config from: {main_cfg_path}")
    main_cfg = OmegaConf.load(main_cfg_path)
    
    # Set the appropriate agent@_global_ based on agent_type
    agent_name = "taco" if agent_type == "taco" else "taco_proprio_states"
    print(f"Setting agent type to: {agent_name}")
    
    # Load the agent config
    agent_cfg_path = CFG_ROOT / "agent" / f"{agent_name}.yaml"
    print(f"Loading agent config from: {agent_cfg_path}")
    agent_cfg = OmegaConf.load(agent_cfg_path)
    
    # Extract hyperparameters from main config needed for interpolation
    # Add all potential interpolation keys
    hydra_vars = {
        "lr": main_cfg.get("lr", 1e-4),
        "encoder_lr": main_cfg.get("encoder_lr", 1e-4),
        "feature_dim": main_cfg.get("feature_dim", 50),
        "device": str(device),
        "use_tb": main_cfg.get("use_tb", True),
        "stddev_schedule": main_cfg.get("stddev_schedule", "linear(1.0, 0.1, 100000)"),
        "curl": agent_cfg.get("curl", True),
        "reward": agent_cfg.get("reward", True),
        "multistep": agent_cfg.get("multistep", 3),
        "latent_a_dim": agent_cfg.get("latent_a_dim", "none"),
        "batch_size": agent_cfg.get("batch_size", 1024),
        "pretrained_path": agent_cfg.get("pretrained_path", "none"),
        "freeze_encoder": agent_cfg.get("freeze_encoder", False),
        "no_taco": agent_cfg.get("no_taco", False)
    }
    
    # Create a config with all variables to resolve interpolation
    resolver_cfg = OmegaConf.create(hydra_vars)
    
    # Merge with agent config to resolve interpolation
    agent_cfg = OmegaConf.merge(resolver_cfg, agent_cfg)
    
    # Set agent-specific parameters directly
    agent_cfg.agent.obs_shape = list(obs_shape)
    agent_cfg.agent.action_shape = list(action_shape)
    agent_cfg.agent.device = str(device)
    
    print(f"Instantiating agent with config: {OmegaConf.to_yaml(agent_cfg.agent)}")
    
    # Instantiate the agent (remove the .to(device) call)
    agent = instantiate(agent_cfg.agent)
    return agent

def load_agent(agent_type, checkpoint_path, device, is_snapshot=False, obs_shape=None, action_shape=None):
    """
    Load the appropriate agent type and use its load_pretrained method.
    """
    if is_snapshot:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent = checkpoint['agent']
        agent.change_device(device)
        agent.eval()
        print(f"Loaded agent directly from snapshot: {checkpoint_path}")
        return agent

    if obs_shape is None or action_shape is None:
        raise ValueError("obs_shape and action_shape are required when loading from checkpoint files.")

    agent = build_agent_from_hydra(agent_type, device, obs_shape, action_shape)
    agent.load_pretrained(checkpoint_path, map_location=device)
    print(f"Loaded pretrained weights from: {checkpoint_path}")
    return agent

def load_dataset(dataset_path):
    """
    Load a dataset of episodes from npz files.
    
    Returns:
        A list of episodes, where each episode is a dictionary of arrays
    """
    # Check if the path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # If the path is a directory, find all npz files
    if os.path.isdir(dataset_path):
        episode_files = sorted(glob.glob(os.path.join(dataset_path, '*.npz')))
        if not episode_files:
            raise FileNotFoundError(f"No episode files (.npz) found in {dataset_path}")
    else:
        # If the path is a file, use that directly
        if dataset_path.endswith('.npz'):
            episode_files = [dataset_path]
        else:
            raise ValueError(f"Dataset path should be a directory or a .npz file: {dataset_path}")
    
    episodes = []
    total_transitions = 0
    
    for episode_file in episode_files:
        episode_data = dict(np.load(episode_file, allow_pickle=True))
        episodes.append(episode_data)
        
        # Count transitions (excluding the first dummy transition)
        if 'observation' in episode_data:
            transitions = len(episode_data['observation']) - 1
            total_transitions += transitions
    
    print(f"Loaded {len(episodes)} episodes with {total_transitions} total transitions from {dataset_path}")
    return episodes

def encode_episodes(agent, episodes, agent_type, device, max_episodes=None):
    """
    Encode states from episodes using the provided agent's encoder.
    
    Returns:
        encoded_states: Array of encoded states
        time_indices: Array of normalized time indices (0-1) for each state
        episode_boundaries: List of indices where each episode ends
    """
    encoded_states = []
    time_indices = []
    episode_boundaries = []
    episode_rewards = []
    
    # Get encoder and TACO components from agent
    encoder = agent.encoder
    taco = agent.TACO
    
    # Limit number of episodes if specified
    if max_episodes and max_episodes > 0:
        episodes = episodes[:max_episodes]
    
    for episode_idx, episode in enumerate(episodes):
        print(f"Encoding episode {episode_idx+1}/{len(episodes)}")
        
        # Select the appropriate observation type
        if agent_type == "taco_proprio":
            if 'proprio_observation' in episode:
                states = episode['proprio_observation']
            else:
                raise KeyError(f"Episode doesn't contain proprio_observation, keys: {episode.keys()}")
        else:
            if 'observation' in episode:
                states = episode['observation']
            else:
                raise KeyError(f"Episode doesn't contain observation, keys: {episode.keys()}")
        
        # Skip first (dummy) transition
        states = states[1:]
        episode_length = len(states)
        
        if episode_length == 0:
            print(f"Episode {episode_idx} is empty, skipping")
            continue
            
        # Get rewards for this episode (skip first dummy)
        if 'reward' in episode:
            rewards = episode['reward'][1:]
            episode_rewards.append(rewards.sum())
        
        # Process in batches
        batch_size = 64
        all_encodings = []
        
        for i in range(0, episode_length, batch_size):
            batch = states[i:min(i+batch_size, episode_length)]
            
            # Convert to torch tensor
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            # Normalize images if needed
            if agent_type == "taco" and batch_tensor.max() > 1.0:
                batch_tensor = batch_tensor / 255.0 - 0.5
            
            # Encode batch
            with torch.no_grad():
                try:
                    # Try using TACO's encode method first
                    encodings = taco.encode(batch_tensor)
                except Exception as e:
                    print(f"Error using TACO encoder: {e}")
                    # Fall back to direct encoder
                    encodings = encoder(batch_tensor)
            
            all_encodings.append(encodings.cpu().numpy())
        
        if all_encodings:  # Check if we have any encodings
            episode_encodings = np.vstack(all_encodings)
            encoded_states.append(episode_encodings)
            
            # Create normalized time indices for this episode
            episode_times = np.linspace(0, 1, episode_length)
            time_indices.append(episode_times)
            
            # Track episode boundary
            if episode_boundaries:
                episode_boundaries.append(episode_boundaries[-1] + episode_length)
            else:
                episode_boundaries.append(episode_length)
    
    if not encoded_states:
        raise ValueError("No states could be encoded from the episodes")
    
    # Combine all encoded states and time indices
    encoded_states = np.vstack(encoded_states)
    time_indices = np.concatenate(time_indices)
    
    print(f"Encoded {encoded_states.shape[0]} states with {encoded_states.shape[1]} dimensions")
    return encoded_states, time_indices, episode_boundaries, episode_rewards

def create_pca_visualization(encoded_states, time_indices, episode_boundaries, episode_rewards,
                            output_dir, model_name, dataset_name, max_components=10):
    """
    Apply PCA to encoded states and create visualizations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output filename base
    model_name_short = Path(model_name).stem
    dataset_name_short = Path(dataset_name).stem if Path(dataset_name).is_file() else Path(dataset_name).name
    output_base = f"{output_dir}/{model_name_short}_{dataset_name_short}"
    
    # Calculate variance explained for all components
    max_components = min(max_components, encoded_states.shape[1], encoded_states.shape[0])
    pca = PCA(n_components=max_components)
    pca_result = pca.fit_transform(encoded_states)
    
    # Create main 2D PCA visualization
    plt.figure(figsize=(10, 8))
    
    # Create a colormap that transitions from cyan to magenta
    cmap = plt.cm.viridis
    
    # Create scatter plot with trajectory progress coloring
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=time_indices, cmap=cmap, 
                          alpha=0.7, s=30)
    
    # Mark episode transitions if available
    if episode_boundaries:
        for boundary in episode_boundaries[:-1]:  # Skip the last boundary
            plt.axvline(x=pca_result[boundary-1, 0], color='red', linestyle='--', alpha=0.3)
            plt.axhline(y=pca_result[boundary-1, 1], color='red', linestyle='--', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Trajectory Progress')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Start', '25%', '50%', '75%', 'End'])
    
    # Add title and labels
    variance_explained = pca.explained_variance_ratio_[:2].sum() * 100
    plt.title(f"PCA of Feature Embeddings\nModel: {model_name_short}\nDataset: {dataset_name_short}\n"
              f"Total Variance Explained: {variance_explained:.1f}%", fontsize=12)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_file = f"{output_base}_pca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved main PCA visualization to {output_file}")
    plt.close()
    
    # Create a grid of the first few components if we have enough
    if max_components >= 6:
        # Create grid showing relationships between first N components
        num_components = min(6, max_components)
        fig, axes = plt.subplots(num_components, num_components, figsize=(20, 20))
        
        # Add super title
        fig.suptitle(f"PCA Component Relationships\nModel: {model_name_short}, Dataset: {dataset_name_short}", 
                    fontsize=16)
        
        for i in range(num_components):
            for j in range(num_components):
                ax = axes[i, j]
                if i == j:
                    # Histogram on diagonal
                    ax.hist(pca_result[:, i], bins=30, color='skyblue', alpha=0.7)
                    ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
                else:
                    # Scatter plot on off-diagonal
                    sc = ax.scatter(pca_result[:, j], pca_result[:, i], c=time_indices, 
                                   cmap=cmap, s=5, alpha=0.5)
                    if i == num_components-1:
                        ax.set_xlabel(f"PC{j+1}")
                    if j == 0:
                        ax.set_ylabel(f"PC{i+1}")
                    ax.tick_params(axis='both', which='both', labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the grid figure
        output_file_grid = f"{output_base}_pca_grid.png"
        plt.savefig(output_file_grid, dpi=300, bbox_inches='tight')
        print(f"Saved PCA component grid to {output_file_grid}")
        plt.close()
    
    # Create scree plot showing explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components+1), pca.explained_variance_ratio_, 'o-', linewidth=2)
    plt.plot(range(1, max_components+1), np.cumsum(pca.explained_variance_ratio_), 's-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot: PCA Explained Variance by Component')
    plt.legend(['Individual', 'Cumulative'])
    plt.xticks(range(1, max_components+1))
    plt.grid(True)
    
    # Save scree plot
    output_file_scree = f"{output_base}_pca_scree.png"
    plt.savefig(output_file_scree, dpi=300, bbox_inches='tight')
    print(f"Saved scree plot to {output_file_scree}")
    plt.close()
    
    # Return paths to created visualizations
    return {
        'main_pca': output_file,
        'pca_grid': output_file_grid if max_components >= 6 else None,
        'scree_plot': output_file_scree
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize feature representations from encoders')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the encoder model (snapshot or checkpoint)')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset directory or episode file')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum number of episodes to process (None=all)')
    parser.add_argument('--max_components', type=int, default=10,
                       help='Maximum number of PCA components to analyze')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # First, load the dataset
    print(f"Loading dataset from {args.dataset_path}")
    episodes = load_dataset(args.dataset_path)
    
    # Second, detect the agent type from the model
    print(f"Detecting agent type from {args.model_path}")
    agent_type, _, is_snapshot = detect_agent_type(args.model_path, device)
    print(f"Detected agent type: {agent_type}")
    
    # Third, extract shapes from the dataset
    print("Extracting observation and action shapes from dataset")
    obs_shape, action_shape = extract_dataset_shapes(episodes, agent_type)
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")
    
    # Fourth, load the agent using the extracted shapes
    agent = load_agent(
        agent_type=agent_type,
        checkpoint_path=args.model_path,
        device=device,
        is_snapshot=is_snapshot,
        obs_shape=obs_shape,
        action_shape=action_shape
    )
    
    # Encode states from episodes using the loaded agent
    encoded_states, time_indices, episode_boundaries, episode_rewards = encode_episodes(
        agent=agent,
        episodes=episodes,
        agent_type=agent_type,
        device=device,
        max_episodes=args.max_episodes
    )
    
    # Create and save visualizations
    visualization_files = create_pca_visualization(
        encoded_states=encoded_states,
        time_indices=time_indices,
        episode_boundaries=episode_boundaries,
        episode_rewards=episode_rewards,
        output_dir=args.output_dir,
        model_name=args.model_path,
        dataset_name=args.dataset_path,
        max_components=args.max_components
    )
    
    print(f"Visualization complete! Files created:")
    for name, path in visualization_files.items():
        if path:
            print(f"- {name}: {path}")

if __name__ == '__main__':
    main()
    