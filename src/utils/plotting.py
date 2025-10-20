import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def smooth(x, n=50):
    """Apply a simple moving average to smooth reward curves."""
    return np.convolve(x, np.ones(n)/n, mode="valid")


def plot_rewards(rewards_q, rewards_pg, num_plants=5, save_path=None):
    """
    Plot and save the learning progress comparison.

    Parameters
    ----------
    rewards_q : list[float]
        Episode rewards for Q-learning.
    rewards_pg : list[float]
        Episode rewards for Policy Gradient.
    num_plants : int, optional
        Number of plants (used in title and filename).
    save_path : str or None, optional
        File path to save figure. If None, auto-generates one.
    """
    plt.figure(figsize=(8, 4))

    plt.plot(smooth(rewards_q), label="Q‑Learning", color="forestgreen")
    plt.plot(smooth(rewards_pg), label="Policy Gradient", color="royalblue")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Learning Progress — {num_plants} Plants")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = PLOTS_DIR / f"plot_{num_plants}plants.png"

    plt.savefig(save_path, dpi=300)
    print(f"Saved reward plot to: {save_path}")

    plt.show()
