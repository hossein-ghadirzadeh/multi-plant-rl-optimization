import sys
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from config import MAX_POINTS


def watch_agent_play(dqn_agent, q_agent, games=5):
    env = PongEnv(render_mode=True, max_points=MAX_POINTS)
    for g in range(games):
        left_state, right_state = env.reset()
        done = False
        while not done:
            env.render()
            action_left, _ = q_agent.act(left_state)
            action_right = dqn_agent.act(right_state, epsilon=0.0)
            left_state, right_state, rleft, rright, done, info = env.step(action_left, action_right)

        print(f"🏁 Match {g+1}: Winner = {info.get('winner')} | Final score (Left Q, Right DQN) = {info.get('final_score')}")
        time.sleep(1.0)
    env.close()