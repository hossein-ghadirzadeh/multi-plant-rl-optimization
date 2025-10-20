from tqdm import trange
import numpy as np


def train_agents(env_q, env_pg, q_agent, pg_agent, episodes=4500, max_steps=50):
    rewards_q, rewards_pg = [], []# track total rewards per episode
    growth = {"Q_first": [], "Q_last": [], "PG_first": [], "PG_last": []}

    for ep in trange(episodes, desc="Training"):
        # Q‑learning Agent
        s = env_q.reset()
        total_r, ep_ht = 0, [] # track episode reward and heights
        for _ in range(max_steps):
            a = q_agent.choose_action(s)
            s2, r, d = env_q.step(a)
            q_agent.learn(s, a, r, s2) # update q-table
            s, total_r = s2, total_r + r
            ep_ht.append(env_q.heights.copy())
            if d:
                break
        rewards_q.append(total_r)
        if ep < 25:
            growth["Q_first"].append(ep_ht) # save early episodes
        if ep >= episodes - 25:
            growth["Q_last"].append(ep_ht) # save late episodes
        q_agent.decay_epsilon()

        
        # Policy Gradient agent
       
        s = env_pg.reset()
        total_r, ep_ht = 0, []
        for _ in range(max_steps):
            a = pg_agent.choose_action(s)
            s2, r, d = env_pg.step(a)
            pg_agent.store_reward(r)
            s, total_r = s2, total_r + r
            ep_ht.append(env_pg.heights.copy())
            if d:
                break
        pg_agent.finish_episode()
        rewards_pg.append(total_r)
        if ep < 25:
            growth["PG_first"].append(ep_ht)
        if ep >= episodes - 25:
            growth["PG_last"].append(ep_ht)
    # return rewards and plant growth data
    return rewards_q, rewards_pg, growth
