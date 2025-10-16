import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.pong_env import PongEnv
from config import MAX_POINTS

def evaluate(dqn_agent, q_agent, render=False, episodes=20):
    env = PongEnv(render_mode=render, max_points=MAX_POINTS)
    right_wins = 0
    for ep in range(episodes):
        left_state, right_state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            # Q agent uses epsilon=0 during eval to be deterministic (greedy)
            action_left, _ = q_agent.act(left_state)
            action_right = dqn_agent.act(right_state, epsilon=0.0)
            left_state, right_state, rleft, rright, done, info = env.step(action_left, action_right)

        winner = info.get('winner')
        if winner == 'right':
            right_wins += 1
        print(f"[Eval] Episode {ep+1}: Winner = {winner}, Score {info.get('final_score')}")
    env.close()
    return right_wins / episodes