from envs.garden_env import MultiPlantGardenEnv
from agents.q_learning_agent import SimpleQLearningAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from utils.plotting import plot_rewards
from utils.growth_analysis import avg_growth
from train.train_agents import train_agents
from visualization.animate_garden import animate_garden


def main():
    # initialize environments and agents
    env_q = MultiPlantGardenEnv(num_plants=12)
    env_pg = MultiPlantGardenEnv(num_plants=12)
    q_agent = SimpleQLearningAgent(env_q)
    pg_agent = PolicyGradientAgent(env_pg)

    # Train both agents
    rewards_q, rewards_pg, growth = train_agents(
        env_q, env_pg, q_agent, pg_agent)

    # Plot learning curve
    plot_rewards(
        rewards_q,
        rewards_pg,
        num_plants=env_q.num_plants
    )
    # Compute average growth patterns and stats for both agents
    q_early_stats = avg_growth(growth["Q_first"])
    q_late_stats = avg_growth(growth["Q_last"])
    pg_early_stats = avg_growth(growth["PG_first"])
    pg_late_stats = avg_growth(growth["PG_last"])

    # Animate the growth comparison
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
        save_path=f"garden_growth_{env_q.num_plants}.mp4",
        fps=5
    )


if __name__ == "__main__":
    main()
