import argparse
from pathlib import Path
import numpy as np
import torch
import dmc
import utils
from dm_env import specs
from video import VideoRecorder
from replay_buffer import ReplayBufferStorage, save_episode

def collect_episodes(env, num_eps, policy_fn, save_dir):
    episodes = []  # opzionale
    video_recorder = VideoRecorder(save_dir / 'videos', render_size=256, fps=20)
    for ep in range(num_eps):
        # Inizializza l'episodio con struttura replay buffer
        episode = {
            'observation': [],
            'action': [],
            'reward': [],
            'discount': []
        }
        time_step = env.reset()
        episode['observation'].append(time_step.observation)
        video_recorder.init(env, enabled=True)
        while not time_step.last():
            # Azione: se policy_fn è None usiamo una random action altrimenti il callable
            action = env.action_spec().generate_value() if policy_fn is None else policy_fn(time_step.observation)
            time_step = env.step(action)
            # Registra la transizione
            episode['action'].append(action)
            episode['reward'].append(time_step.reward)
            episode['discount'].append(time_step.discount)
            episode['observation'].append(time_step.observation)
            video_recorder.record(env)
        video_recorder.save(f'episode_{ep}.mp4')
        # Salva l'episodio nel formato replay buffer (.npz)
        save_episode(episode, save_dir / f'episode_{ep}.npz')
        print(f"Episode {ep} saved.")
        episodes.append(episode)
    return episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='medium')
    parser.add_argument('--policy_path', type=str, default='/home/mprattico/Pretrain-TACO/exp_local/default2/medium_policy.pt',
                        help='Percorso del checkpoint della policy da usare.')
    parser.add_argument('--num_eps', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='quadruped_run')
    parser.add_argument('--save_replay', action='store_true',
                        help='Salva il replay buffer (solo per medium policy)')
    args = parser.parse_args()
    
    # Configurazione ambiente
    env = dmc.make(args.env_name, frame_stack=3, action_repeat=2, seed=1)
    save_dir = Path('./dataset')
    save_dir.mkdir(exist_ok=True)
    (save_dir / 'videos').mkdir(parents=True, exist_ok=True)
    
    policy_type = args.policy_path.split('/')[-1].split('_')[0]
    if policy_type == 'medium':
        save_dir /= 'medium'
    elif policy_type == 'random':
        save_dir /= 'random'
    else:
        raise ValueError(f"Policy type '{policy_type}' non supportata.")
    save_dir.mkdir(parents=True, exist_ok=True) 

    # Definisci la policy da usare
    checkpoint = torch.load(args.policy_path, weights_only=False)
    agent = checkpoint['agent']
    def policy_fn(observation):
        with torch.no_grad():
            return agent.act(observation, step=0, eval_mode=True)
    
    episodes = collect_episodes(env, args.num_eps, policy_fn, save_dir)

if __name__ == '__main__':
    main()