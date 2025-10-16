import sys
from pathlib import Path
import numpy as np
import time
import random

random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from agents.q_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from evaluation.eval_arena import evaluate
from evaluation.watch import watch_agent_play
from config import MAX_POINTS, RENDER_DURING_TRAIN, Q_ALPHA, DISCOUNT_FACTOR, EPSILON_START_Q, EPSILON_START_DQN, MAX_EPISODES, EPSILON_END_DQN, EPSILON_DECAY_DQN, EPSILON_END_Q, EPSILON_DECAY_Q, EVAL_EVERY_EPISODES, MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train():
    env = PongEnv(render_mode=RENDER_DURING_TRAIN, max_points=MAX_POINTS)
    state_size = 5
    action_size = 3

    # Agents: left = Q-learning, right = DQN
    q_agent = QLearningAgent(action_size=action_size, alpha=Q_ALPHA, gamma=DISCOUNT_FACTOR, epsilon_start=EPSILON_START_Q)
    dqn_agent = DQNAgent(state_size, action_size)

    eps_dqn = EPSILON_START_DQN
    eps_q = EPSILON_START_Q

    episode_rewards_left = []
    episode_rewards_right = []
    best_avg = -9999
    start_time = time.time()
    best_model_path = MODELS_DIR / "pong_dqn_right_best.weights.h5"

    for ep in range(1, MAX_EPISODES + 1):
        state_left, state_right = env.reset()
        total_reward_left = 0.0
        total_reward_right = 0.0

        # play a match until someone reaches MAX_POINTS
        done = False
        while not done:
            if RENDER_DURING_TRAIN:
                env.render()

            # Agents choose actions
            action_left, s_disc_left = q_agent.act(state_left)
            action_right = dqn_agent.act(state_right, eps_dqn)

            next_left_state, next_right_state, reward_left, reward_right, done, info = env.step(action_left, action_right)

            # Q-learning update (left)
            q_agent.learn(s_disc_left, action_left, reward_left, next_left_state, done)

            # DQN replay buffer update (right)
            dqn_agent.step(state_right, action_right, reward_right, next_right_state, done)

            # accumulate rewards
            total_reward_left += reward_left
            total_reward_right += reward_right

            # move to next states
            state_left = next_left_state
            state_right = next_right_state

        # end of match
        episode_rewards_left.append(total_reward_left)
        episode_rewards_right.append(total_reward_right)

        # decay epsilons
        eps_dqn = max(EPSILON_END_DQN, eps_dqn * EPSILON_DECAY_DQN)
        q_agent.epsilon = max(EPSILON_END_Q, q_agent.epsilon * EPSILON_DECAY_Q)

        # Logging
        if ep % 5 == 0:
            avg_left = np.mean(episode_rewards_left[-50:]) if len(episode_rewards_left) > 0 else 0.0
            avg_right = np.mean(episode_rewards_right[-50:]) if len(episode_rewards_right) > 0 else 0.0
            elapsed = time.time() - start_time
            print(f"Episode {ep:4d} | Left(Q) LastR {total_reward_left:.3f} Avg50 {avg_left:.3f} | Right(DQN) LastR {total_reward_right:.3f} Avg50 {avg_right:.3f} | EpsDQN {eps_dqn:.3f} EpsQ {q_agent.epsilon:.3f} | Time {elapsed:.1f}s")

        # Periodic evaluation + save DQN
        if ep % EVAL_EVERY_EPISODES == 0:
            win_rate_right = evaluate(dqn_agent, q_agent, render=True, episodes=10)
            print(f"--- Eval after ep {ep}: right(DQN) win rate {win_rate_right:.3f} ---")
            if win_rate_right > best_avg:
                best_avg = win_rate_right
                dqn_agent.main_Qnetwork.save_weights(best_model_path)
                print(f"✅ Saved best DQN model to {best_model_path}")

    env.close()
    print(f"\n🏁 Best win rate achieved by DQN: {best_avg:.3f}")
    print("Training finished.")

    # Load and watch best model if exists
    if best_model_path.exists():
        print("\n🎮 Loading best trained DQN model (Right) and watching it play against Q-learning (Left)...")
        dqn_agent.main_Qnetwork.load_weights(best_model_path)
        watch_agent_play(dqn_agent, q_agent)
    else:
        print("⚠️ No trained DQN weights found. Skipping demo.")
