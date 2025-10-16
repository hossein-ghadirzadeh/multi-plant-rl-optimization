import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from train.train_multi_agent import train

if __name__ == "__main__":
    print("Multi-agent Pong: LEFT = Q-learning  |  RIGHT = DQN")
    train()
