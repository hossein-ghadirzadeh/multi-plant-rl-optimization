import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from train.train_multi_agent import grid_search

if __name__ == "__main__":
    print("Multi-agent Pong: LEFT = Q-learning | RIGHT = DQN")
    print("Grid-search over hyperparameters will run. This may take a long time (full training per combo).")
    grid_search()
