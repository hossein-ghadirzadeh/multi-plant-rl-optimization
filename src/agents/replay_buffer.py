import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0

    def add_events(self, events):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(events)
        else:
            self.buffer[self.position] = events
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
