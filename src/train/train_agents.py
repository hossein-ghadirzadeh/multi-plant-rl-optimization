from tqdm import trange
from config import EPISODES, MAX_STEPS_PER_EPISODE, ANALYSIS_WINDOW

def train_agents(env_q, env_pg, q_agent, pg_agent, 
                 episodes=EPISODES, 
                 max_steps=MAX_STEPS_PER_EPISODE):
    
    """
    Train both Q-learning and Policy Gradient agents in parallel over a set number of episodes.

    Args:
        env_q (MultiPlantGardenEnv): The environment instance for the Q-learning agent.
        env_pg (MultiPlantGardenEnv): The environment instance for the Policy Gradient agent.
        q_agent (SimpleQLearningAgent): The Q-learning agent to be trained.
        pg_agent (PolicyGradientAgent): The Policy Gradient agent to be trained.
        episodes (int): The total number of episodes to train for.
        max_steps (int): The maximum number of steps within each episode.

    Returns:
        tuple: A tuple containing:
            - rewards_q (list): A list of total rewards per episode for the Q-learning agent.
            - rewards_pg (list): A list of total rewards per episode for the Policy Gradient agent.
            - growth (dict): A dictionary containing plant growth histories from the first
                             and last `ANALYSIS_WINDOW` episodes for both agents.
    """
    
    
    # Lists to store the total reward for each episode
    rewards_q, rewards_pg = [], []

    # Dictionary to store the plant height trajectories for early and late stages of training
    growth = {"Q_first": [], "Q_last": [], "PG_first": [], "PG_last": []}

    # Use trange for a progress bar during training
    for ep in trange(episodes, desc="Training Agents"):

        # -----------------------------
        # 1. Train Q-learning Agent
        # -----------------------------
        s = env_q.reset()
        # track episode reward and heights
        total_r, ep_ht = 0, [] 
        for _ in range(max_steps):
            a = q_agent.choose_action(s)
            s2, r, d = env_q.step(a)

            # The agent learns from the transition (s, a, r, s2)
            q_agent.learn(s, a, r, s2)
            s, total_r = s2, total_r + r
            ep_ht.append(env_q.heights.copy())
            if d:
                break

        rewards_q.append(total_r)

        # Store growth data for the first few ("early") episodes
        if ep < ANALYSIS_WINDOW:
            growth["Q_first"].append(ep_ht)
        # Store growth data for the last few ("late") episodes    
        if ep >= episodes - ANALYSIS_WINDOW:
            growth["Q_last"].append(ep_ht)

        # Decay epsilon to reduce exploration over time    
        q_agent.decay_epsilon()

        
        # -----------------------------
        # 2. Train Policy Gradient Agent
        # -----------------------------
        s = env_pg.reset()
        total_r, ep_ht = 0, []
        for _ in range(max_steps):
            a = pg_agent.choose_action(s)
            s2, r, d = env_pg.step(a)

            # PG agent stores rewards for the end-of-episode update
            pg_agent.store_reward(r)


            s, total_r = s2, total_r + r
            ep_ht.append(env_pg.heights.copy())
            if d:
                break

        # At the end of the episode, the PG agent learns from all stored transitions    
        pg_agent.finish_episode()

        rewards_pg.append(total_r)

        # Store growth data for early and late episodes
        if ep < ANALYSIS_WINDOW:
            growth["PG_first"].append(ep_ht)
        if ep >= episodes - ANALYSIS_WINDOW:
            growth["PG_last"].append(ep_ht)
            
    # return rewards and plant growth data
    return rewards_q, rewards_pg, growth
