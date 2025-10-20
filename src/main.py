"""
Main entry point for the project.

This script initializes the garden environment and the two reinforcement learning agents
(Q-Learning and Policy Gradient), runs the training process, analyzes the results,
and generates plots and animations to visualize the agents' performance.
"""

from pathlib import Path
from envs.garden_env import MultiPlantGardenEnv
from agents.q_learning_agent import SimpleQLearningAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from utils.plotting import plot_rewards
from utils.growth_analysis import avg_growth
from train.train_agents import train_agents
from visualization.animate_garden import animate_garden
from config import NUM_PLANTS


def main():
    """Main function to run the entire experiment."""
    print("Initializing environments and agents...")
    # Initialize two separate environments, one for each agent, to keep their states independent.
    env_q = MultiPlantGardenEnv(num_plants=NUM_PLANTS)
    env_pg = MultiPlantGardenEnv(num_plants=NUM_PLANTS)

    # Create the Q-Learning and Policy Gradient agents.
    q_agent = SimpleQLearningAgent(env_q)
    pg_agent = PolicyGradientAgent(env_pg)

    print("Starting agent training...")
    # Train both agents and collect rewards and growth data over all episodes.
    rewards_q, rewards_pg, growth = train_agents(
        env_q, env_pg, q_agent, pg_agent)

    print("Training complete. Generating plots and analysis...")
    # Generate and save a plot comparing the smoothed rewards of both agents.
    plot_rewards(
        rewards_q,
        rewards_pg,
        num_plants=env_q.num_plants
    )
    
    
    # Analyze the growth patterns from the first and last few episodes.
    # "Early" behavior shows the agent before learning; "Late" behavior shows the agent after training.
    q_early_stats = avg_growth(growth["Q_first"])
    q_late_stats = avg_growth(growth["Q_last"])
    pg_early_stats = avg_growth(growth["PG_first"])
    pg_late_stats = avg_growth(growth["PG_last"])

    print("Creating final animation...")
    # Create a 4-panel animation comparing the early vs. late strategies of both agents.
    animate_garden(
        env_q,
        q_early=q_early_stats[0],
        q_late=q_late_stats[0],
        pg_early=pg_early_stats[0],
        pg_late=pg_late_stats[0],
        q_early_stats=q_early_stats,
        q_late_stats=q_late_stats,
        pg_early_stats=pg_early_stats,
        pg_late_stats=pg_late_stats,
        save_path=f"garden_growth_{env_q.num_plants}.gif",
        fps=5
    )
    print("Project finished successfully!")


if __name__ == "__main__":
    main()
