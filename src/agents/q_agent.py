import sys
from pathlib import Path

import numpy as np
import random

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import Q_ALPHA, DISCOUNT_FACTOR, EPSILON_START_Q, Q_DISCRETE_BINS


class QLearningAgent:
    def __init__(self, action_size=3, alpha=Q_ALPHA, gamma=DISCOUNT_FACTOR, epsilon_start=EPSILON_START_Q):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.q_table = {}  # dict: state_tuple -> np.array(action_size)

        # discretization bins
        self.bins = {
            'bx': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['bx'] - 1),
            'by': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['by'] - 1),
            'vx': np.array([-1.0, 0.0, 1.0]),  # treat as sign (-,0,+) thresholds
            'vy': np.array([-1.0, 0.0, 1.0]),
            'py': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['py'] - 1)
        }

    def discretize(self, state):
        # state: [bx, by, vx, vy, py] normalized as in env
        bx, by, vx, vy, py = state
        bx_idx = int(np.digitize(bx, self.bins['bx']))
        by_idx = int(np.digitize(by, self.bins['by']))
        # vx, vy are in [-1,1]; we convert to sign buckets: -1,0,1
        vx_sign = int(np.sign(vx)) + 1   # map -1->0, 0->1, 1->2
        vy_sign = int(np.sign(vy)) + 1
        py_idx = int(np.digitize(py, self.bins['py']))
        return (bx_idx, by_idx, vx_sign, vy_sign, py_idx)

    def get_qs(self, s_disc):
        if s_disc not in self.q_table:
            self.q_table[s_disc] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[s_disc]

    def act(self, state):
        s_disc = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size), s_disc
        qs = self.get_qs(s_disc)
        return int(np.argmax(qs)), s_disc

    def learn(self, s_disc, action, reward, next_state, done):
        next_disc = self.discretize(next_state)
        q_vals = self.get_qs(s_disc)
        q_next = self.get_qs(next_disc)
        target = reward
        if not done:
            target = reward + self.gamma * np.max(q_next)
        q_vals[action] += self.alpha * (target - q_vals[action])
        # store back
        self.q_table[s_disc] = q_vals
