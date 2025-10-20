import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import PLOT_SMOOTHING_WINDOW

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def smooth(x, n=PLOT_SMOOTHING_WINDOW):
    # smooth the rewards by using a moving average
    return np.convolve(x, np.ones(n)/n, mode="valid")


def plot_rewards(rewards_q, rewards_pg, num_plants=5, save_path=None):

    plt.figure(figsize=(8, 4))
   # plot smoothed reward curves
    plt.plot(smooth(rewards_q), label="Q-Learning", color="forestgreen")
    plt.plot(smooth(rewards_pg), label="Policy Gradient", color="royalblue")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Learning Progress — {num_plants} Plants")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = PLOTS_DIR / f"plot_{num_plants}plants.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300) # save plot to file
    print(f"Saved reward plot to: {save_path}")

    plt.show()
