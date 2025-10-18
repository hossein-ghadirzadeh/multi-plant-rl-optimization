import sys
from pathlib import Path

import numpy as np
import random

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import Q_ALPHA, DISCOUNT_FACTOR, EPSILON_START_Q, Q_DISCRETE_BINS, EPSILON_DECAY_Q

class QLearningAgent:
    """
    Tabular Q-learning over a discretized 5D state: (bx, by, sign(vx), sign(vy), py)
    Actions: 0=up, 1=stay, 2=down
    """
    def __init__(
            self, 
            action_size: int = 3, 
            alpha: float = Q_ALPHA, 
            gamma: float = DISCOUNT_FACTOR, 
            epsilon_start: float = EPSILON_START_Q, 
            epsilon_decay: float = EPSILON_DECAY_Q):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

        # bin edges (np.digitize expects ascending edges)
        self.bins = {
            'bx': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['bx'] - 1),
            'by': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['by'] - 1),
            'vx': np.array([-1.0, 0.0, 1.0]),
            'vy': np.array([-1.0, 0.0, 1.0]),
            'py': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['py'] - 1)
        }

    def discretize(self, state):
        bx, by, vx, vy, py = state
        bx_idx = int(np.digitize(bx, self.bins['bx']))
        by_idx = int(np.digitize(by, self.bins['by']))
        vx_sign = int(np.sign(vx)) + 1
        vy_sign = int(np.sign(vy)) + 1
        py_idx = int(np.digitize(py, self.bins['py']))
        return (bx_idx, by_idx, vx_sign, vy_sign, py_idx)

    def get_qs(self, s_disc):
        if s_disc not in self.q_table:
            self.q_table[s_disc] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[s_disc]

    def act(self, state, epsilon=None):
        s_disc = self.discretize(state)
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.randrange(self.action_size), s_disc
        qs = self.get_qs(s_disc)
        return int(np.argmax(qs)), s_disc

    def learn(self, s_disc, action, reward, next_state, done):
        next_disc = self.discretize(next_state)
        q_vals = self.get_qs(s_disc)
        q_next = self.get_qs(next_disc)
        target = reward if done else reward + self.gamma * np.max(q_next)
        q_vals[action] += self.alpha * (target - q_vals[action])
        self.q_table[s_disc] = q_vals