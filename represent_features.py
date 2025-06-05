"""
Script per visualizzare le features estratte dall'encoder TACO usando t-SNE.
Confronta le prime e ultime osservazioni degli episodi per vedere se stati simili
sono rappresentati vicini nello spazio delle features.

Usage:
python represent_features.py --dataset_path data_episodes/repr_dataset/pretraining_datasets/ --pretrained_path /home/mprattico/Pretrain-TACO/models/taco_MT_MT50_0.99_lr=0.0005_ts=50000896_curl_rew.pt --n_obs 3 --k_episodes 450 --save_plot 3obs_450eps_encoder_direct --use_encoder_direct
python represent_features.py --dataset_path data_episodes/repr_dataset/pretraining_datasets/push-wall-v2_task0_fs3_ar2_ri1_rg0_exp=99 --pretrained_path /home/mprattico/Pretrain-TACO/models/taco_MT_MT50_0.99_lr=0.0005_ts=50000896_curl_rew.pt --n_obs 10 --k_episodes 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import sys
import os


# Import existing classes from the codebase
from agents.taco import Encoder, TACO
import utils

def load_episode(episode_path):
    """Carica un singolo episodio dal file .npz"""
    episode = np.load(episode_path)
    return {k: episode[k] for k in episode.keys()}

def load_episodes_from_dataset(dataset_path, k_episodes=None):
    """
    Carica episodi dal dataset specificato.
    
    Args:
        dataset_path: Path al dataset (directory contenente file .npz)
        k_episodes: Numero massimo di episodi da caricare (None = tutti)
    
    Returns:
        List di episodi caricati
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Trova tutti i file .npz nella directory e nelle sottocartelle
    episode_files = list(dataset_path.glob("**/*.npz"))
    if not episode_files:
        raise ValueError(f"No .npz files found in {dataset_path} or its subdirectories")
    
    # Ordina i file per nome
    episode_files.sort()
    
    # Limita il numero di episodi se specificato
    if k_episodes is not None:
        episode_files = episode_files[:k_episodes]
    
    print(f"Loading {len(episode_files)} episodes from {dataset_path} (including subdirectories)")
    
    episodes = []
    for episode_file in episode_files:
        try:
            episode = load_episode(episode_file)
            episodes.append(episode)
        except Exception as e:
            print(f"Error loading episode {episode_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(episodes)} episodes")
    return episodes

def extract_observations(episodes, n_obs):
    """
    Estrae le prime n e ultime n osservazioni da ogni episodio.
    
    Args:
        episodes: Lista di episodi
        n_obs: Numero di osservazioni da estrarre (prime e ultime)
    
    Returns:
        first_obs: Array delle prime osservazioni
        last_obs: Array delle ultime osservazioni
        episode_indices: Array degli indici degli episodi per ogni osservazione
    """
    first_obs = []
    last_obs = []
    first_episode_indices = []
    last_episode_indices = []
    
    for ep_idx, episode in enumerate(episodes):
        observations = episode['observation']
        episode_length = len(observations)
        
        # Salta il primo dummy transition (indice 0)
        actual_start = 1
        actual_end = episode_length
        actual_length = actual_end - actual_start
        
        if actual_length < 2 * n_obs:
            print(f"Episode {ep_idx} too short ({actual_length} observations), skipping")
            continue
        
        # Prime n osservazioni (dopo il dummy transition)
        first_n = observations[actual_start:actual_start + n_obs]
        first_obs.extend(first_n)
        first_episode_indices.extend([ep_idx] * len(first_n))
        
        # Ultime n osservazioni
        last_n = observations[actual_end - n_obs:actual_end]
        last_obs.extend(last_n)
        last_episode_indices.extend([ep_idx] * len(last_n))
    
    return (np.array(first_obs), np.array(last_obs), 
            np.array(first_episode_indices), np.array(last_episode_indices))

def load_pretrained_model(pretrained_path, obs_shape, feature_dim, hidden_dim, action_shape, multistep, device):
    """
    Carica il modello TACO preaddestrato dal checkpoint.
    
    Args:
        pretrained_path: Path al modello preaddestrato
        obs_shape: Shape delle osservazioni
        feature_dim: Dimensione delle features
        hidden_dim: Dimensione hidden
        action_shape: Shape delle azioni
        multistep: Numero di step per le azioni
        device: Device per il modello
    
    Returns:
        Tuple (encoder, taco, act_tok) con modelli preaddestrati in modalità eval
    """
    # Inizializza l'encoder
    encoder = Encoder(obs_shape, feature_dim).to(device)
    
    # Determina latent_a_dim come nel codice originale
    latent_a_dim = int(action_shape[0] * 1.25) + 1
    
    # Inizializza action tokenizer
    act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep).to(device)
    
    # Inizializza TACO con l'encoder
    taco = TACO(encoder.repr_dim, feature_dim, action_shape, latent_a_dim, 
                hidden_dim, act_tok, encoder, multistep, device).to(device)
    
    # Carica il checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Carica i pesi dei modelli
    encoder.load_state_dict(checkpoint['encoder'])
    taco.load_state_dict(checkpoint['taco'])
    act_tok.load_state_dict(checkpoint['act_tok'])
    
    # Imposta in modalità eval
    encoder.eval()
    taco.eval()
    act_tok.eval()
    
    print(f"Loaded pretrained model from {pretrained_path}")
    return encoder, taco, act_tok

def extract_features(model, observations, device, batch_size=32, use_taco_encode=True):
    """
    Estrae features dalle osservazioni usando l'encoder.
    
    Args:
        model: Tuple (encoder, taco) o solo encoder
        observations: Array di osservazioni
        device: Device per il calcolo
        batch_size: Dimensione del batch per l'inferenza
        use_taco_encode: Se True usa taco.encode(), altrimenti encoder diretto
    
    Returns:
        Array delle features estratte
    """
    features = []
    
    if use_taco_encode:
        encoder, taco, _ = model
        taco.eval()
    else:
        encoder, _, _ = model
        encoder.eval()
    
    with torch.no_grad():
        num_batches = (len(observations) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(observations))
            
            batch_obs = observations[start_idx:end_idx]
            
            # Converti in tensor e sposta su device
            batch_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
            
            # Estrai features
            if use_taco_encode:
                batch_features = taco.encode(batch_tensor)
            else:
                batch_features = encoder(batch_tensor)
            
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def visualize_features_tsne(first_features, last_features, 
                           first_episode_indices, last_episode_indices,
                           save_path=None, perplexity=30, random_state=42, 
                           representation_type="TACO encode"):
    """
    Visualizza le features usando t-SNE.
    
    Args:
        first_features: Features delle prime osservazioni
        last_features: Features delle ultime osservazioni
        first_episode_indices: Indici degli episodi per prime osservazioni
        last_episode_indices: Indici degli episodi per ultime osservazioni
        save_path: Path per salvare il plot (opzionale)
        perplexity: Parametro perplexity per t-SNE
        random_state: Seed per riproducibilità
        representation_type: Tipo di rappresentazione utilizzata per il titolo
    """
    # Combina tutte le features
    all_features = np.concatenate([first_features, last_features], axis=0)
    
    # Crea labels: 0 per prime osservazioni, 1 per ultime
    labels = np.concatenate([
        np.zeros(len(first_features)), 
        np.ones(len(last_features))
    ])
    
    # Crea indici episodi combinati
    all_episode_indices = np.concatenate([first_episode_indices, last_episode_indices])
    
    print(f"Running t-SNE on {len(all_features)} features...")
    print(f"Feature dimension: {all_features.shape[1]}")
    
    # Applica t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                verbose=1, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    
    # Crea il plot
    plt.figure(figsize=(12, 8))
    
    # Separa i punti per prime e ultime osservazioni
    first_mask = labels == 0
    last_mask = labels == 1
    
    # Plot prime osservazioni (rosso)
    plt.scatter(features_2d[first_mask, 0], features_2d[first_mask, 1], 
                c='red', alpha=0.6, s=30, label=f'First observations (n={len(first_features)})')
    
    # Plot ultime n osservazioni (blu)
    plt.scatter(features_2d[last_mask, 0], features_2d[last_mask, 1], 
                c='blue', alpha=0.6, s=30, label=f'Last observations (n={len(last_features)})')
    
    plt.title(f't-SNE Visualization of {representation_type} Features\n(Red: First observations, Blue: Last observations)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aggiungi informazioni sul plot
    unique_episodes = len(np.unique(all_episode_indices))
    plt.figtext(0.02, 0.02, f'Episodes: {unique_episodes}, Perplexity: {perplexity}, Representation: {representation_type}', 
                fontsize=8, ha='left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize encoder features using t-SNE")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory containing .npz files")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained TACO model")
    parser.add_argument("--n_obs", type=int, default=10,
                        help="Number of first and last observations to extract from each episode")
    parser.add_argument("--k_episodes", type=int, default=50,
                        help="Number of episodes to load (None for all)")
    parser.add_argument("--feature_dim", type=int, default=50,
                        help="Feature dimension of the encoder")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension for TACO")
    parser.add_argument("--multistep", type=int, default=3,
                        help="Multistep parameter for TACO")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexity parameter for t-SNE")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Path to save the t-SNE plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for t-SNE reproducibility")
    parser.add_argument("--use_encoder_direct", action="store_true", default=False,
                        help="Use encoder directly instead of taco.encode() method")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Parameters: n_obs={args.n_obs}, k_episodes={args.k_episodes}")
    
    representation_type = "Encoder direct" if args.use_encoder_direct else "TACO encode"
    print(f"Representation type: {representation_type}")
    
    # Carica episodi dal dataset
    episodes = load_episodes_from_dataset(args.dataset_path, args.k_episodes)
    
    if not episodes:
        print("No episodes loaded, exiting.")
        return
    
    # Estrai prime e ultime osservazioni
    first_obs, last_obs, first_ep_idx, last_ep_idx = extract_observations(episodes, args.n_obs)
    
    print(f"Extracted {len(first_obs)} first observations and {len(last_obs)} last observations")
    
    # Determina la shape delle osservazioni e azioni
    obs_shape = first_obs[0].shape
    
    # Ottieni action_shape dal primo episodio
    action_shape = episodes[0]['action'][1].shape  # Salta il dummy transition
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    # Carica il modello preaddestrato
    encoder, taco, act_tok = load_pretrained_model(
        args.pretrained_path, obs_shape, args.feature_dim, args.hidden_dim,
        action_shape, args.multistep, args.device
    )
    
    # Estrai features
    print("Extracting features from first observations...")
    first_features = extract_features(
        (encoder, taco, act_tok), first_obs, args.device, 
        args.batch_size, use_taco_encode=not args.use_encoder_direct
    )
    
    print("Extracting features from last observations...")
    last_features = extract_features(
        (encoder, taco, act_tok), last_obs, args.device, 
        args.batch_size, use_taco_encode=not args.use_encoder_direct
    )
    
    print(f"Feature shapes: first={first_features.shape}, last={last_features.shape}")
    
    # Visualizza con t-SNE
    visualize_features_tsne(
        first_features, last_features,
        first_ep_idx, last_ep_idx,
        save_path=args.save_plot,
        perplexity=args.perplexity,
        random_state=args.random_state,
        representation_type=representation_type
    )
    
    # Statistiche finali
    print("\n=== Summary ===")
    print(f"Total episodes processed: {len(episodes)}")
    print(f"First observations: {len(first_features)}")
    print(f"Last observations: {len(last_features)}")
    print(f"Feature dimension: {first_features.shape[1]}")
    print(f"Representation used: {representation_type}")
    
    # Calcola distanze medie tra prime e ultime osservazioni dello stesso episodio
    unique_episodes = np.unique(np.concatenate([first_ep_idx, last_ep_idx]))
    same_episode_distances = []
    
    for ep_id in unique_episodes:
        first_mask = first_ep_idx == ep_id
        last_mask = last_ep_idx == ep_id
        
        if np.sum(first_mask) > 0 and np.sum(last_mask) > 0:
            first_ep_features = first_features[first_mask]
            last_ep_features = last_features[last_mask]
            
            # Calcola distanza media tra prime e ultime osservazioni dello stesso episodio
            for first_feat in first_ep_features:
                for last_feat in last_ep_features:
                    dist = np.linalg.norm(first_feat - last_feat)
                    same_episode_distances.append(dist)
    
    if same_episode_distances:
        print(f"Average distance between first and last observations of same episode: {np.mean(same_episode_distances):.4f}")

if __name__ == "__main__":
    main()
