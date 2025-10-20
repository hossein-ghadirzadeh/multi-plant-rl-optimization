import numpy as np
import random


class MultiPlantGardenEnv:
    def __init__(self, num_plants=5, max_height=10, max_water=12):
        self.num_plants = num_plants
        self.max_height = max_height
        self.max_water = max_water
        self.reset()

    def reset(self):
        self.heights = np.zeros(self.num_plants)
        self.done = False
        self.resources = {"water": self.max_water}
        return self._get_state()

    def _get_state(self):
        discrete_heights = tuple(np.round(self.heights / 2).astype(int))
        return discrete_heights + (self.resources["water"],)

    def step(self, actions):
        total_reward = 0
        for i, action in enumerate(actions):
            if action == 1 and self.resources["water"] > 0:
                self.heights[i] += random.choice([1, 2])
                self.resources["water"] -= 1
            elif action == 1:
                total_reward -= 1
            elif action == 0:
                self.heights[i] += 0.3 if self.resources["water"] > 0 else 0.05
            elif action == 2:
                total_reward += 10 if 7 <= self.heights[i] <= self.max_height else -2
                self.heights[i] = 0
            if 5 <= self.heights[i] <= 9:
                total_reward += 0.2
        self.heights = np.clip(self.heights, 0, self.max_height)
        if self.resources["water"] <= 0 and np.all(self.heights < 1):
            self.done = True
        return self._get_state(), total_reward, self.done
