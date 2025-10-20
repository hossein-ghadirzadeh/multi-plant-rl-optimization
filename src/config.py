# ===================================
# General Training Settings
# ===================================
EPISODES = 4500           # Total number of training episodes
MAX_STEPS_PER_EPISODE = 50 # Maximum steps allowed in one episode

# ===================================
# Environment Settings
# ===================================
# Note: NUM_PLANTS is the main variable you might change for experiments
NUM_PLANTS = 12           # Number of plants in the garden
MAX_PLANT_HEIGHT = 10     # Maximum height a plant can reach
MAX_WATER_RESOURCES = 12  # Initial amount of water available

# ===================================
# Q-Learning Agent Hyperparameters
# ===================================
Q_LEARNING_ALPHA = 0.1          # Learning rate
Q_LEARNING_GAMMA = 0.9          # Discount factor
Q_LEARNING_EPSILON_START = 1.0  # Starting exploration rate
Q_LEARNING_EPSILON_MIN = 0.05   # Minimum exploration rate
Q_LEARNING_EPSILON_DECAY = 0.995 # Decay rate for exploration

# ===================================
# Policy Gradient Agent Hyperparameters
# ===================================
PG_LEARNING_RATE = 1e-3   # Learning rate for the Adam optimizer
PG_GAMMA = 0.95           # Discount factor
PG_HIDDEN_LAYER_SIZE = 64 # Size of the neural network's hidden layer

# ===================================
# Analysis & Visualization Settings
# ===================================
ANALYSIS_WINDOW = 25      # Number of episodes for "early" and "late" analysis
PLOT_SMOOTHING_WINDOW = 50 # Window size for smoothing the reward curve
ANIMATION_FPS = 5         # Frames per second for the output video