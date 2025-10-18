import sys
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from config import MAX_POINTS


def watch_agent_play(dqn_agent, q_agent, games=10):
    env = PongEnv(render_mode=True, max_points=MAX_POINTS)
    right_wins = 0

    for g in range(games):
        state_left, state_right = env.reset()
        done = False
        while not done:
            env.render()
            # Fully greedy actions for evaluation
            action_left, _ = q_agent.act(state_left, epsilon=0.0)
            action_right = dqn_agent.act(state_right, epsilon=0.0)
            state_left, state_right, _, _, done, info = env.step(action_left, action_right)

        winner = info.get('winner')
        print(f" Match {g+1}: Winner = {winner}")
        if winner == 'right':
            right_wins += 1
        time.sleep(1.0)

    env.close()
    win_rate = right_wins / games
    print(f"\nRight(DQN) win rate over {games} matches: {win_rate:.3f}")
    return win_rate
