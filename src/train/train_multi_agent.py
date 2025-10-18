import sys
from pathlib import Path
import numpy as np, time, os
import time
import random
from itertools import product

random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from agents.q_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from evaluation.eval_arena import evaluate
from evaluation.watch import watch_agent_play

from config import (
    MAX_POINTS, Q_ALPHA, UPDATE_EVERY, TARGET_UPDATE_EVERY, REPLAY_BUFFER_SIZE,
    EPSILON_START_Q, RENDER_DURING_TRAIN, EPSILON_START_DQN, MAX_EPISODES,
    EPSILON_END_DQN, EPSILON_DECAY_DQN, EPSILON_END_Q, EVAL_EVERY_EPISODES,
    MODELS_DIR
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train(state_size,
            action_size,
            q_agent,
            dqn_agent,
            render_during_train=RENDER_DURING_TRAIN,
            eval_every=EVAL_EVERY_EPISODES,
            max_episodes=MAX_EPISODES,
            model_path=None):
    env = PongEnv(render_mode=render_during_train, max_points=MAX_POINTS)

    eps_dqn = EPSILON_START_DQN
    eps_q = q_agent.epsilon  # initial epsilon from q_agent

    episode_rewards_left = []
    episode_rewards_right = []

    rally_lengths_left = []
    rally_lengths_right = []
    reaction_times_left = []
    reaction_times_right = []

    best_avg = -9999
    start_time = time.time()
    if model_path is None:
        model_path = MODELS_DIR / "pong_dqn_right_best.weights.h5"

    for ep in range(1, max_episodes + 1):
        state_left, state_right = env.reset()
        total_reward_left = 0.0
        total_reward_right = 0.0
        done = False

        ep_rally_len_left = 0
        ep_rally_len_right = 0
        ep_react_left = 0
        ep_react_right = 0

        while not done:
            if render_during_train:
                env.render()

            action_left, s_disc_left = q_agent.act(state_left)
            action_right = dqn_agent.act(state_right, eps_dqn)

            next_left_state, next_right_state, reward_left, reward_right, done, info = env.step(action_left, action_right)

            # Q-learning update (left)
            q_agent.learn(s_disc_left, action_left, reward_left, next_left_state, done)

            # DQN replay buffer update (right)
            dqn_agent.step(state_right, action_right, reward_right, next_right_state, done)

            total_reward_left += reward_left
            total_reward_right += reward_right

            if 'rally_length_left' in info:
                ep_rally_len_left = info['rally_length_left']
            if 'rally_length_right' in info:
                ep_rally_len_right = info['rally_length_right']
            if 'reaction_time_left' in info:
                ep_react_left = info['reaction_time_left']
            if 'reaction_time_right' in info:
                ep_react_right = info['reaction_time_right']

            state_left = next_left_state
            state_right = next_right_state

        episode_rewards_left.append(total_reward_left)
        episode_rewards_right.append(total_reward_right)
        rally_lengths_left.append(ep_rally_len_left)
        rally_lengths_right.append(ep_rally_len_right)
        reaction_times_left.append(ep_react_left)
        reaction_times_right.append(ep_react_right)

        # Decay epsilons
        eps_dqn = max(EPSILON_END_DQN, eps_dqn * EPSILON_DECAY_DQN)
        q_agent.epsilon = max(EPSILON_END_Q, q_agent.epsilon * q_agent.epsilon_decay)

        elapsed = time.time() - start_time

        score_tuple = (env.left_score, env.right_score)
        print(f"Ep {ep:4d} | Left(Q) LastR {total_reward_left:.3f} | "
              f"Right(DQN) LastR {total_reward_right:.3f} | "
              f"Score {score_tuple} | Time {elapsed:.1f}s")

        # Every eval_every episodes print averages and evaluate
        if ep % eval_every == 0:
            avg_left_reward = np.mean(episode_rewards_left[-eval_every:])
            avg_right_reward = np.mean(episode_rewards_right[-eval_every:])
            avg_rally_left = np.mean(rally_lengths_left[-eval_every:]) if rally_lengths_left else 0
            avg_rally_right = np.mean(rally_lengths_right[-eval_every:]) if rally_lengths_right else 0
            avg_react_left = np.mean(reaction_times_left[-eval_every:]) if reaction_times_left else 0
            avg_react_right = np.mean(reaction_times_right[-eval_every:]) if reaction_times_right else 0

            print(f"--- Episode {ep} Averages (last {eval_every}) ---")
            print(f"Left(Q): AvgR {avg_left_reward:.3f} | AvgRallyLen {avg_rally_left:.2f} | AvgReact {avg_react_left:.2f}")
            print(f"Right(DQN): AvgR {avg_right_reward:.3f} | AvgRallyLen {avg_rally_right:.2f} | AvgReact {avg_react_right:.2f}")

            # Evaluate on 50 episodes (no render)
            win_rate_right, metrics = evaluate(dqn_agent, q_agent, render=False, episodes=50)
            print(f"--- Eval after ep {ep}: Right(DQN) win rate {win_rate_right:.3f} | "
                  f"Avg RallyLen Left {metrics['avg_rally_left']:.2f} | Avg RallyLen Right {metrics['avg_rally_right']:.2f} | "
                  f"Avg Reaction Left {metrics['avg_reaction_left']:.2f} | Avg Reaction Right {metrics['avg_reaction_right']:.2f}")

            # Save best model if improved
            if win_rate_right > best_avg:
                best_avg = win_rate_right
                dqn_agent.main_Qnetwork.save_weights(model_path)
                print(f"Saved best DQN model to {model_path}")

    env.close()
    print("Training finished.")

    # Watch best model if saved
    if os.path.exists(model_path):
        print("\n Loading best trained DQN model (Right) and watching it play against Q-learning (Left)...")
        dqn_agent.main_Qnetwork.load_weights(model_path)
        watch_agent_play(dqn_agent, q_agent, games=10)
    else:
        print(" No trained DQN weights found. Skipping demo.")

    # Return final best win rate found during this run (best_avg) and model path
    return best_avg if best_avg > -9999 else 0.0, model_path


search_space = {
    "LEARNING_RATE": [5e-4, 6e-4],
    "MINIBATCH_SIZE": [64, 128],
    "DISCOUNT_FACTOR": [0.95, 0.99],
    "EPSILON_DECAY_DQN": [0.99, 0.995],
    "EPSILON_DECAY_Q": [0.99, 0.995],
}

def run_training_and_evaluation(current_params, combo_idx, total_combos):
    """
    Builds fresh agents using current_params, runs train_run() which trains, evaluates and watches,
    and returns the final win rate for this combo.
    """
    # Map params to local hyperparameters
    lr = current_params["LEARNING_RATE"]
    minibatch = current_params["MINIBATCH_SIZE"]
    discount = current_params["DISCOUNT_FACTOR"]
    eps_decay_dqn = current_params["EPSILON_DECAY_DQN"]
    eps_decay_q = current_params["EPSILON_DECAY_Q"]

    state_size, action_size = 5, 3

    q_agent = QLearningAgent(
        action_size=action_size,
        alpha=Q_ALPHA,
        gamma=discount,
        epsilon_start=EPSILON_START_Q,
        epsilon_decay=eps_decay_q
    )

    dqn_agent = DQNAgent(
        state_size, action_size,
        learning_rate=lr,
        minibatch_size=minibatch,
        discount_factor=discount,
        update_every=UPDATE_EVERY,
        target_update_every=TARGET_UPDATE_EVERY,
        replay_buffer_size=REPLAY_BUFFER_SIZE
    )

    model_path = MODELS_DIR / f"pong_dqn_best_combo_{combo_idx}_of_{total_combos}.weights.h5"
    print(f"\nStarting training for combination {combo_idx}/{total_combos} (model -> {model_path})\n")

    best_win_rate, _ = train(
        state_size=state_size,
        action_size=action_size,
        q_agent=q_agent,
        dqn_agent=dqn_agent,
        render_during_train=RENDER_DURING_TRAIN,
        eval_every=EVAL_EVERY_EPISODES,
        max_episodes=MAX_EPISODES,
        model_path=model_path
    )

    final_win_rate, metrics = evaluate(dqn_agent, q_agent, render=False, episodes=50)
    print(f"Final eval (50 eps) Right(DQN) win rate: {final_win_rate:.3f} | metrics: {metrics}")
    return final_win_rate

def grid_search():
    keys = list(search_space.keys())
    combinations = list(product(*[search_space[k] for k in keys]))
    best_results = {"win_rate": -1.0, "params": None}

    print(f"\n=== Starting Grid Search: {len(combinations)} combinations ===\n")

    total_combos = len(combinations)
    for idx, values in enumerate(combinations, 1):
        params = dict(zip(keys, values))
        print(f"\n=== Combination {idx}/{len(combinations)} ===")
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        print("=" * 50)

        # ---- Run training with these params ----
        win_rate = run_training_and_evaluation(params, idx, total_combos)

        # ---- Save best ----
        if win_rate > best_results["win_rate"]:
            best_results["win_rate"] = win_rate
            best_results["params"] = params

    print("\n=== Grid Search Complete ===")
    print(f"Best Win Rate: {best_results['win_rate']:.3f}")
    print("Best Parameters:")
    for k, v in best_results["params"].items():
        print(f"  {k} = {v}")

    return best_results

