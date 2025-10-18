import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from config import MAX_POINTS

def evaluate(dqn_agent, q_agent, render=False, episodes=50):
    env = PongEnv(render_mode=render, max_points=MAX_POINTS)
    right_wins = 0
    left_rally_lengths, right_rally_lengths = [], []
    left_reactions, right_reactions = [], []

    for _ in range(episodes):
        state_left, state_right = env.reset()
        done = False
        info = {}
        while not done:
            action_left, _ = q_agent.act(state_left, epsilon=0.0)
            action_right = dqn_agent.act(state_right, epsilon=0.0)
            state_left, state_right, _, _, done, info = env.step(action_left, action_right)
            if render:
                env.render()
            if 'rally_length_left' in info:
                left_rally_lengths.append(info['rally_length_left'])
                right_rally_lengths.append(info['rally_length_right'])
            if 'reaction_time_left' in info:
                left_reactions.append(info['reaction_time_left'])
            if 'reaction_time_right' in info:
                right_reactions.append(info['reaction_time_right'])

        if info.get('winner') == 'right':
            right_wins += 1

    env.close()
    metrics = {
        'avg_rally_left': np.mean(left_rally_lengths) if left_rally_lengths else 0.0,
        'avg_rally_right': np.mean(right_rally_lengths) if right_rally_lengths else 0.0,
        'avg_reaction_left': np.mean(left_reactions) if left_reactions else 0.0,
        'avg_reaction_right': np.mean(right_reactions) if right_reactions else 0.0
    }

    return right_wins / episodes, metrics
