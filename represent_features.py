"""
Script per visualizzare le features estratte dall'encoder TACO usando t-SNE o PCA.
Supporta configurazioni JSON con dataset di pretraining e test.

Usage:
python represent_features.py --config configs_data_MT/MT49OODBasketball.json --pretrained_path models/model.pt --method tsne
python represent_features.py --config configs_data_MT/MT49OODBasketball.json --pretrained_path models/model.pt --method pca
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import os
import json


# Import existing classes from the codebase
from agents.taco import TACOAgent
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

def load_datasets_from_config(config_path, k_episodes_train=None, k_episodes_test=None):
    """
    Carica dataset di training e test dalla configurazione JSON.
    
    Args:
        config_path: Path al file di configurazione JSON
        k_episodes_train: Numero massimo di episodi per dataset di training
        k_episodes_test: Numero massimo di episodi per dataset di test
    
    Returns:
        tuple: (train_episodes, test_episodes, train_dataset_names, test_dataset_names)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_datasets = config.get('pretraining_datasets', [])
    test_datasets = config.get('test_datasets', [])
    
    print(f"Found {len(train_datasets)} training datasets and {len(test_datasets)} test datasets")
    
    # Carica episodi di training
    train_episodes = []
    train_dataset_names = []
    for dataset_path in train_datasets:
        print(f"Loading training dataset: {dataset_path}")
        episodes = load_episodes_from_dataset(dataset_path, k_episodes_train)
        train_episodes.extend(episodes)
        
        # Estrai nome del task dal path
        task_name = Path(dataset_path).name.split('_')[0].replace('-v2', '')
        train_dataset_names.extend([task_name] * len(episodes))
    
    # Carica episodi di test
    test_episodes = []
    test_dataset_names = []
    for dataset_path in test_datasets:
        print(f"Loading test dataset: {dataset_path}")
        episodes = load_episodes_from_dataset(dataset_path, k_episodes_test)
        test_episodes.extend(episodes)
        
        # Estrai nome del task dal path
        task_name = Path(dataset_path).name.split('_')[0].replace('-v2', '')
        test_dataset_names.extend([task_name] * len(episodes))
    
    return train_episodes, test_episodes, train_dataset_names, test_dataset_names

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
    Carica il modello TACO preaddestrato usando TACOAgent.
    
    Args:
        pretrained_path: Path al modello preaddestrato
        obs_shape: Shape delle osservazioni
        feature_dim: Dimensione delle features
        hidden_dim: Dimensione hidden
        action_shape: Shape delle azioni
        multistep: Numero di step per le azioni
        device: Device per il modello
    
    Returns:
        TACOAgent preaddestrato in modalità eval
    """
    # Inizializza TACOAgent con i parametri appropriati
    taco_agent = TACOAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,  # Non importante per l'inferenza
        encoder_lr=1e-4,  # Non importante per l'inferenza
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        critic_target_tau=0.005,  # Non importante per l'inferenza
        num_expl_steps=0,  # Non importante per l'inferenza
        update_every_steps=1,  # Non importante per l'inferenza
        stddev_schedule="linear(1.0,0.1,100000)",  # Non importante per l'inferenza
        stddev_clip=0.3,  # Non importante per l'inferenza
        use_tb=False,
        reward=True,
        multistep=multistep,
        latent_a_dim='none',
        curl=True,
        pretrained_path=pretrained_path,  # Carica automaticamente il checkpoint
        freeze_encoder=False,
        no_taco=False
    )
    
    # Imposta in modalità eval
    taco_agent.train(False)
    
    print(f"Loaded pretrained TACOAgent from {pretrained_path}")
    return taco_agent

def extract_features(taco_agent, observations, device, batch_size=32, use_taco_encode=True):
    """
    Estrae features dalle osservazioni usando TACOAgent.
    
    Args:
        taco_agent: TACOAgent preaddestrato
        observations: Array di osservazioni
        device: Device per il calcolo
        batch_size: Dimensione del batch per l'inferenza
        use_taco_encode: Se True usa taco.encode(), altrimenti encoder diretto
    
    Returns:
        Array delle features estratte
    """
    features = []
    
    taco_agent.train(False)  # Assicurati che sia in modalità eval
    
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
                batch_features = taco_agent.TACO.encode(batch_tensor)
            else:
                batch_features = taco_agent.encoder(batch_tensor)
            
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def apply_dimensionality_reduction(features, method='tsne', **kwargs):
    """
    Applica riduzione di dimensionalità alle features.
    
    Args:
        features: Array delle features
        method: 'tsne' o 'pca'
        **kwargs: Parametri aggiuntivi per il metodo
    
    Returns:
        features_2d: Features ridotte a 2D
        method_info: Informazioni aggiuntive sul metodo (es. autovalori per PCA)
    """
    print(f"Applying {method.upper()} to {features.shape[0]} features of dimension {features.shape[1]}")
    
    if method.lower() == 'pca':
        n_components = kwargs.get('n_components', 2)
        pca = PCA(n_components=n_components, random_state=kwargs.get('random_state', 42))
        features_2d = pca.fit_transform(features)
        
        # Stampa autovalori in ordine decrescente
        eigenvalues = pca.explained_variance_
        eigenvalues_ratio = pca.explained_variance_ratio_
        
        print("\n=== PCA Analysis ===")
        print("Autovalori (explained variance) in ordine decrescente:")
        for i, (eigenval, ratio) in enumerate(zip(eigenvalues, eigenvalues_ratio)):
            print(f"  Component {i+1}: {eigenval:.6f} ({ratio:.4f} of variance)")
        print(f"Total explained variance: {np.sum(eigenvalues_ratio):.4f}")
        
        method_info = {
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': eigenvalues_ratio,
            'total_variance': np.sum(eigenvalues_ratio)
        }
        
    elif method.lower() == 'tsne':
        perplexity = kwargs.get('perplexity', 30)
        random_state = kwargs.get('random_state', 42)
        n_iter = kwargs.get('n_iter', 1000)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                    verbose=1, n_iter=n_iter)
        features_2d = tsne.fit_transform(features)
        
        method_info = {
            'perplexity': perplexity,
            'n_iter': n_iter,
            'kl_divergence': tsne.kl_divergence_
        }
        
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'pca' or 'tsne'.")
    
    return features_2d, method_info

def visualize_features(train_features_2d, test_features_2d, method_info, 
                      method='tsne', representation_type="TACO encode", 
                      config_name="", model_name="", save_path=None):
    """
    Visualizza le features ridotte con colori diversi per training (blu) e test (rosso).
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training data (blu)
    plt.scatter(train_features_2d[:, 0], train_features_2d[:, 1], 
                c='blue', alpha=0.6, s=30, label=f'Training data (n={len(train_features_2d)})')
    
    # Plot test data (rosso)
    plt.scatter(test_features_2d[:, 0], test_features_2d[:, 1], 
                c='red', alpha=0.6, s=30, label=f'Test data (n={len(test_features_2d)})')
    
    # Titolo e labels
    if method.lower() == 'pca':
        title = f'PCA Visualization of {representation_type} Features'
        xlabel = f'PC1 ({method_info["explained_variance_ratio"][0]:.3f} variance)'
        ylabel = f'PC2 ({method_info["explained_variance_ratio"][1]:.3f} variance)'
    else:
        title = f't-SNE Visualization of {representation_type} Features'
        xlabel = 't-SNE Component 1'
        ylabel = 't-SNE Component 2'
    
    plt.title(f'{title}\n(Blue: Training, Red: Test)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Informazioni aggiuntive
    info_text = f'Method: {method.upper()}, Representation: {representation_type}'
    if method.lower() == 'pca':
        info_text += f', Total variance: {method_info["total_variance"]:.4f}'
    elif method.lower() == 'tsne':
        info_text += f', Perplexity: {method_info["perplexity"]}'
    
    plt.figtext(0.02, 0.02, info_text, fontsize=8, ha='left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def generate_save_filename(method, config_path, pretrained_path):
    """
    Genera il nome del file di salvataggio nel formato:
    <metodo>-<dataset_config>-<nome_modello>
    """
    # Estrai nome del config (senza estensione)
    config_name = Path(config_path).stem
    
    # Estrai nome del modello (senza estensione e path)
    model_name = Path(pretrained_path).stem
    
    # Formato: metodo-config-modello.png
    filename = f"{method}-{config_name}-{model_name}.png"
    return filename

def main():
    parser = argparse.ArgumentParser(description="Visualize encoder features using PCA or t-SNE")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON configuration file")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained TACO model")
    parser.add_argument("--method", type=str, default='tsne', choices=['pca', 'tsne'],
                        help="Dimensionality reduction method: 'pca' or 'tsne'")
    parser.add_argument("--n_obs", type=int, default=10,
                        help="Number of first and last observations to extract from each episode")
    parser.add_argument("--k_episodes_train", type=int, default=5,
                        help="Number of episodes to load per training dataset")
    parser.add_argument("--k_episodes_test", type=int, default=5,
                        help="Number of episodes to load per test dataset")
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
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components for PCA")
    parser.add_argument("--save_dir", type=str, default="./plots/",
                        help="Directory to save the plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--use_encoder_direct", action="store_true", default=False,
                        help="Use encoder directly instead of taco.encode() method")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Config file: {args.config}")
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Method: {args.method.upper()}")
    print(f"Parameters: n_obs={args.n_obs}, k_episodes_train={args.k_episodes_train}, k_episodes_test={args.k_episodes_test}")
    
    representation_type = "Encoder direct" if args.use_encoder_direct else "TACO encode"
    print(f"Representation type: {representation_type}")
    
    # Carica dataset dalla configurazione JSON
    train_episodes, test_episodes, train_names, test_names = load_datasets_from_config(
        args.config, args.k_episodes_train, args.k_episodes_test
    )
    
    if not train_episodes and not test_episodes:
        print("No episodes loaded, exiting.")
        return
    
    print(f"Loaded {len(train_episodes)} training episodes and {len(test_episodes)} test episodes")
    
    # Estrai osservazioni
    all_train_obs = []
    all_test_obs = []
    
    if train_episodes:
        train_first_obs, train_last_obs, _, _ = extract_observations(train_episodes, args.n_obs)
        all_train_obs = np.concatenate([train_first_obs, train_last_obs], axis=0)
    
    if test_episodes:
        test_first_obs, test_last_obs, _, _ = extract_observations(test_episodes, args.n_obs)
        all_test_obs = np.concatenate([test_first_obs, test_last_obs], axis=0)
    
    print(f"Extracted {len(all_train_obs)} training observations and {len(all_test_obs)} test observations")
    
    # Determina shapes
    if train_episodes:
        obs_shape = all_train_obs[0].shape
        action_shape = train_episodes[0]['action'][1].shape
    else:
        obs_shape = all_test_obs[0].shape
        action_shape = test_episodes[0]['action'][1].shape
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    # Carica il modello preaddestrato usando TACOAgent
    taco_agent = load_pretrained_model(
        args.pretrained_path, obs_shape, args.feature_dim, args.hidden_dim,
        action_shape, args.multistep, args.device
    )
    
    # Estrai features
    train_features = None
    test_features = None
    
    if len(all_train_obs) > 0:
        print("Extracting features from training observations...")
        train_features = extract_features(
            taco_agent, all_train_obs, args.device, 
            args.batch_size, use_taco_encode=not args.use_encoder_direct
        )
    
    if len(all_test_obs) > 0:
        print("Extracting features from test observations...")
        test_features = extract_features(
            taco_agent, all_test_obs, args.device, 
            args.batch_size, use_taco_encode=not args.use_encoder_direct
        )
    
    # Combina features per la riduzione di dimensionalità
    all_features = []
    train_indices = []
    test_indices = []
    
    if train_features is not None:
        all_features.append(train_features)
        train_indices = list(range(len(train_features)))
    
    if test_features is not None:
        start_idx = len(train_features) if train_features is not None else 0
        all_features.append(test_features)
        test_indices = list(range(start_idx, start_idx + len(test_features)))
    
    if not all_features:
        print("No features extracted, exiting.")
        return
    
    combined_features = np.concatenate(all_features, axis=0)
    
    # Applica riduzione di dimensionalità
    method_kwargs = {
        'random_state': args.random_state,
        'perplexity': args.perplexity,
        'n_components': args.n_components
    }
    
    features_2d, method_info = apply_dimensionality_reduction(
        combined_features, method=args.method, **method_kwargs
    )
    
    # Separa features ridotte per training e test
    train_features_2d = features_2d[train_indices] if train_indices else np.array([]).reshape(0, 2)
    test_features_2d = features_2d[test_indices] if test_indices else np.array([]).reshape(0, 2)
    
    # Genera nome file e salva plot
    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = generate_save_filename(args.method, args.config, args.pretrained_path)
    save_path = os.path.join(args.save_dir, save_filename)
    
    # Visualizza
    visualize_features(
        train_features_2d, test_features_2d, method_info,
        method=args.method, representation_type=representation_type,
        config_name=Path(args.config).stem,
        model_name=Path(args.pretrained_path).stem,
        save_path=save_path
    )
    
    # Statistiche finali
    print("\n=== Summary ===")
    print(f"Method: {args.method.upper()}")
    print(f"Training episodes: {len(train_episodes)}")
    print(f"Test episodes: {len(test_episodes)}")
    print(f"Training observations: {len(train_features_2d)}")
    print(f"Test observations: {len(test_features_2d)}")
    print(f"Feature dimension: {combined_features.shape[1]}")
    print(f"Representation used: {representation_type}")
    print(f"Plot saved as: {save_path}")

if __name__ == "__main__":
    main()
