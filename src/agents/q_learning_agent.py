import random

from config import (Q_LEARNING_ALPHA, Q_LEARNING_GAMMA, Q_LEARNING_EPSILON_START, 
                    Q_LEARNING_EPSILON_MIN, Q_LEARNING_EPSILON_DECAY)

class SimpleQLearningAgent:
    """
    A tabular Q-Learning agent for the multi-plant garden environment.
    
    This agent learns a policy by storing state-action values in a dictionary (Q-table).
    It uses an epsilon-greedy strategy for exploration.
    """
    def __init__(self, env, 
                 alpha=Q_LEARNING_ALPHA, 
                 gamma=Q_LEARNING_GAMMA,
                 epsilon=Q_LEARNING_EPSILON_START, 
                 epsilon_min=Q_LEARNING_EPSILON_MIN, 
                 epsilon_decay=Q_LEARNING_EPSILON_DECAY):
        """
        Initializes the Q-Learning agent.

        Args:
            env: The garden environment instance.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            epsilon (float): The initial exploration rate.
            epsilon_min (float): The minimum exploration rate.
            epsilon_decay (float): The rate at which epsilon decays.
        """
        self.env = env
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = [0, 1, 2] # available actions

    def _state_key(self, s):
        """Converts the state (a tuple) into a hashable key for the Q-table dictionary."""
        return tuple(s)

    def choose_action(self, s):
        """
        Chooses an action for each plant using an epsilon-greedy strategy.
        With probability epsilon, it chooses a random action (exploration).
        Otherwise, it chooses the best-known action from the Q-table (exploitation).
        """
        k = self._state_key(s)
        if random.random() < self.epsilon or k not in self.q_table:
            # Explore: choose a random action for each plant
            return [random.choice(self.actions) for _ in range(self.env.num_plants)]
        # Exploit: choose the best action found so far
        return list(self.q_table[k]["best_action"])

    def learn(self, s, acts, r, s_next):
        """
        Updates the Q-table based on the Bellman equation after a transition.
        Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]
        """
        k, k2 = self._state_key(s), self._state_key(s_next)
        a_tup = tuple(acts)

        # Initialize state in Q-table if it's new
        if k not in self.q_table:
            self.q_table[k] = {"Q": {}, "best_action": a_tup}

        # Get the current Q-value, or 0.0 if the action is new for this state
        Qsa = self.q_table[k]["Q"].get(a_tup, 0.0)

        # Find the maximum Q-value for the next state
        max_next = max(self.q_table[k2]["Q"].values()) if k2 in self.q_table and self.q_table[k2]["Q"] else 0

        # Apply the Bellman update rule
        new_q_value = Qsa + self.alpha * (r + self.gamma * max_next - Qsa)
        self.q_table[k]["Q"][a_tup] = new_q_value

        # Update the best known action for the current state for faster lookups later
        best_action_for_k = max(self.q_table[k]["Q"], key=self.q_table[k]["Q"].get)
        self.q_table[k]["best_action"] = best_action_for_k

    def decay_epsilon(self):
        """Reduces the exploration rate (epsilon) after each episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
