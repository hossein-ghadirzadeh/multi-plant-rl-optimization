import random

# simple q-learning agent
class SimpleQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = [0, 1, 2] # available actions

    def _state_key(self, s):
        return tuple(s) # convert state to tuple 

    def choose_action(self, s):
        # epsilon-greedy action selection
        k = self._state_key(s)
        if random.random() < self.epsilon or k not in self.q_table:
            return [random.choice(self.actions) for _ in range(self.env.num_plants)]
        return list(self.q_table[k]["best_action"])

    def learn(self, s, acts, r, s_next):
        # update q-values using q-learning rule
        k, k2 = self._state_key(s), self._state_key(s_next)
        a_tup = tuple(acts)
        if k not in self.q_table:
            self.q_table[k] = {"Q": {}, "best_action": a_tup}
        Qsa = self.q_table[k]["Q"].get(a_tup, 0.0)
        max_next = max(self.q_table[k2]["Q"].values(
        )) if k2 in self.q_table and self.q_table[k2]["Q"] else 0
        new = Qsa + self.alpha * (r + self.gamma * max_next - Qsa)
        self.q_table[k]["Q"][a_tup] = new
        best = max(self.q_table[k]["Q"], key=self.q_table[k]["Q"].get)
        self.q_table[k]["best_action"] = best

    def decay_epsilon(self):
        # gradually reduce exploration
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
