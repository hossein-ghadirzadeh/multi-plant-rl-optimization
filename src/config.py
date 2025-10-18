from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# -------------------- Hyperparameters (defaults; overwritten by grid_search) --------------------
LEARNING_RATE = 5e-4
MINIBATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 1000
MAX_EPISODES = 50
MAX_TIMESTEPS = 10000
EVAL_EVERY_EPISODES = 50
RENDER_DURING_TRAIN = False
MAX_POINTS = 3

# Epsilon-greedy parameters (defaults; overwritten by grid_search)
EPSILON_START_DQN = 1.0
EPSILON_END_DQN = 0.01
EPSILON_DECAY_DQN = 0.995

EPSILON_START_Q = 1.0
EPSILON_END_Q = 0.01
EPSILON_DECAY_Q = 0.995

# Q-learning specific
Q_ALPHA = 0.1
Q_DISCRETE_BINS = {'bx': 6, 'by': 6, 'vx': 3, 'vy': 3, 'py': 6}

# Game geometry
WIDTH = 400
HEIGHT = 300
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 50
PADDLE_SPEED = 4.0
BALL_RADIUS = 5
FPS = 60
