import numpy as np
import random

from config import NUM_PLANTS, MAX_PLANT_HEIGHT, MAX_WATER_RESOURCES

# environment that simulates a garden with multiple plants
class MultiPlantGardenEnv:
    def __init__(self, num_plants=NUM_PLANTS, max_height=MAX_PLANT_HEIGHT, max_water=MAX_WATER_RESOURCES):
        self.num_plants = num_plants
        self.max_height = max_height
        self.max_water = max_water
        self.reset()

    def reset(self):
        # reset the environment to the initial state
        self.heights = np.zeros(self.num_plants)
        self.done = False
        self.resources = {"water": self.max_water}
        return self._get_state()

    def _get_state(self):
        # return the current state as discrete heights and remaining water
        discrete_heights = tuple(np.round(self.heights / 2).astype(int))
        return discrete_heights + (self.resources["water"],)

    def step(self, actions):
        # apply actions and return new state, reward, and done flag
        total_reward = 0
        for i, action in enumerate(actions):
            # action 1: water the plant
            if action == 1 and self.resources["water"] > 0: 
                self.heights[i] += random.choice([1, 2])
                self.resources["water"] -= 1
            elif action == 1:
                total_reward -= 1
            # action 0: do nothing 
            elif action == 0:
                self.heights[i] += 0.3 if self.resources["water"] > 0 else 0.05
             # action 2: harvest the plant
            elif action == 2:
                total_reward += 10 if 7 <= self.heights[i] <= self.max_height else -2
                self.heights[i] = 0
            if 5 <= self.heights[i] <= 9:
                total_reward += 0.2
        self.heights = np.clip(self.heights, 0, self.max_height)
        # episode ends if no water left and all plants are short
        if self.resources["water"] <= 0 and np.all(self.heights < 1):
            self.done = True
        return self._get_state(), total_reward, self.done
