# pong_multiagent.py
import pygame
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import time
import os
from collections import deque

# -------------------- Hyperparameters --------------------
LEARNING_RATE = 5e-4
MINIBATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 4                 # learn every n steps
TARGET_UPDATE_EVERY = 1000       # hard copy frequency (steps)
MAX_EPISODES = 2000
MAX_TIMESTEPS = 10000            # allow long matches because match now contains multiple rallies
EVAL_EVERY_EPISODES = 50
RENDER_DURING_TRAIN = True       # NOTE: slows training a lot
MAX_POINTS = 3                   # first to MAX_POINTS wins the match

# Epsilon-greedy parameters for DQN (right) and Q (left)
EPSILON_START_DQN = 1.0
EPSILON_END_DQN = 0.01
EPSILON_DECAY_DQN = 0.995

EPSILON_START_Q = 1.0
EPSILON_END_Q = 0.01
EPSILON_DECAY_Q = 0.995

# Q-learning specific
Q_ALPHA = 0.1                    # learning rate for tabular Q
Q_DISCRETE_BINS = {
    'bx': 6, 'by': 6, 'vx': 3, 'vy': 3, 'py': 6
}

# Game geometry
WIDTH = 400
HEIGHT = 300
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 50
PADDLE_SPEED = 4.0
BALL_RADIUS = 5
FPS = 60

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Q-Network (for DQN) --------------------------
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.hidden_layer1 = layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.hidden_layer2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

# -------------------- Replay Buffer (for DQN) ----------------------
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0

    def add_events(self, events):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(events)
        else:
            self.buffer[self.position] = events
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------- DQN Agent (Right Paddle) --------------------------
class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=LEARNING_RATE,
                 minibatch_size=MINIBATCH_SIZE,
                 discount_factor=DISCOUNT_FACTOR,
                 update_every=UPDATE_EVERY,
                 target_update_every=TARGET_UPDATE_EVERY,
                 replay_buffer_size=REPLAY_BUFFER_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.update_every = update_every
        self.target_update_every = target_update_every

        self.main_Qnetwork = QNetwork(state_size, action_size)
        self.target_Qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.buffer = ReplayBuffer(replay_buffer_size)
        self.t_step = 0
        self.global_steps = 0

        # build networks
        dummy = np.zeros((1, state_size), dtype=np.float32)
        _ = self.main_Qnetwork(dummy)
        _ = self.target_Qnetwork(dummy)
        self.update_target_network()

    def update_target_network(self):
        self.target_Qnetwork.set_weights(self.main_Qnetwork.get_weights())

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_size))
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        action_values = self.main_Qnetwork(state_tf)
        return int(np.argmax(action_values.numpy()))

    def step(self, state, action, reward, next_state, done):
        self.buffer.add_events((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        self.global_steps += 1

        if self.t_step == 0 and len(self.buffer) >= self.minibatch_size:
            experiences = self.buffer.sample(self.minibatch_size)
            self.learn(experiences)

        if self.global_steps % self.target_update_every == 0:
            self.update_target_network()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_q = self.target_Qnetwork(next_states_tf)
        max_next_q = tf.reduce_max(next_q, axis=1)
        q_targets = rewards_tf + self.discount_factor * max_next_q * (1.0 - dones_tf)

        with tf.GradientTape() as tape:
            q_values = self.main_Qnetwork(states_tf)
            indices = tf.stack([tf.range(tf.shape(actions_tf)[0]), actions_tf], axis=1)
            predicted_q = tf.gather_nd(q_values, indices)
            loss = tf.keras.losses.MSE(q_targets, predicted_q)

        gradients = tape.gradient(loss, self.main_Qnetwork.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_Qnetwork.trainable_variables))

# -------------------- Q-learning Agent (Left Paddle) --------------------------
class QLearningAgent:
    def __init__(self, action_size=3, alpha=Q_ALPHA, gamma=DISCOUNT_FACTOR, epsilon_start=EPSILON_START_Q):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.q_table = {}  # dict: state_tuple -> np.array(action_size)

        # discretization bins
        self.bins = {
            'bx': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['bx'] - 1),
            'by': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['by'] - 1),
            'vx': np.array([-1.0, 0.0, 1.0]),  # treat as sign (-,0,+) thresholds
            'vy': np.array([-1.0, 0.0, 1.0]),
            'py': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['py'] - 1)
        }

    def discretize(self, state):
        # state: [bx, by, vx, vy, py] normalized as in env
        bx, by, vx, vy, py = state
        bx_idx = int(np.digitize(bx, self.bins['bx']))
        by_idx = int(np.digitize(by, self.bins['by']))
        # vx, vy are in [-1,1]; we convert to sign buckets: -1,0,1
        vx_sign = int(np.sign(vx)) + 1   # map -1->0, 0->1, 1->2
        vy_sign = int(np.sign(vy)) + 1
        py_idx = int(np.digitize(py, self.bins['py']))
        return (bx_idx, by_idx, vx_sign, vy_sign, py_idx)

    def get_qs(self, s_disc):
        if s_disc not in self.q_table:
            self.q_table[s_disc] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[s_disc]

    def act(self, state):
        s_disc = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size), s_disc
        qs = self.get_qs(s_disc)
        return int(np.argmax(qs)), s_disc

    def learn(self, s_disc, action, reward, next_state, done):
        next_disc = self.discretize(next_state)
        q_vals = self.get_qs(s_disc)
        q_next = self.get_qs(next_disc)
        target = reward
        if not done:
            target = reward + self.gamma * np.max(q_next)
        q_vals[action] += self.alpha * (target - q_vals[action])
        # store back
        self.q_table[s_disc] = q_vals

# -------------------- Pong Environment (multi-agent, score-limit) ------------------
class PongEnv:
    def __init__(self, render_mode=False, max_points=MAX_POINTS):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.paddle_w = PADDLE_WIDTH
        self.paddle_h = PADDLE_HEIGHT
        self.ball_radius = BALL_RADIUS
        self.paddle_speed = PADDLE_SPEED
        self.fps = FPS
        self.render_mode = render_mode
        self.max_points = max_points

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong Multi-Agent (Left: Q-learning, Right: DQN)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    def reset(self):
        # left paddle (Q-learning) and right paddle (DQN)
        self.left_y = (self.height - self.paddle_h) / 2.0
        self.right_y = (self.height - self.paddle_h) / 2.0

        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0

        angle = random.uniform(-0.25 * np.pi, 0.25 * np.pi)
        direction = random.choice([-1, 1])
        speed = random.uniform(3.0, 4.5)
        self.ball_vel_x = direction * speed * np.cos(angle)
        self.ball_vel_y = speed * np.sin(angle)

        # match scoring
        self.left_score = 0
        self.right_score = 0

        self.steps = 0
        self.done = False
        # return observations for both agents
        return self._get_state_left(), self._get_state_right()

    def _get_state_left(self):
        bx = self.ball_x / self.width
        by = self.ball_y / self.height
        vx = np.clip(self.ball_vel_x / 6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y / 6.0, -1.0, 1.0)
        py = self.left_y / (self.height - self.paddle_h)
        return np.array([bx, by, vx, vy, py], dtype=np.float32)

    def _get_state_right(self):
        # mirror frame so right agent sees normalized coordinates from its perspective
        bx = self.ball_x / self.width
        by = self.ball_y / self.height
        vx = np.clip(self.ball_vel_x / 6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y / 6.0, -1.0, 1.0)
        py = self.right_y / (self.height - self.paddle_h)
        # same features; agents can have symmetric state spaces
        return np.array([bx, by, vx, vy, py], dtype=np.float32)

    def step(self, left_action, right_action):
        # Actions: 0 up, 1 stay, 2 down
        if left_action == 0:
            self.left_y -= self.paddle_speed
        elif left_action == 2:
            self.left_y += self.paddle_speed
        if right_action == 0:
            self.right_y -= self.paddle_speed
        elif right_action == 2:
            self.right_y += self.paddle_speed

        # clamp paddles
        self.left_y = float(np.clip(self.left_y, 0, self.height - self.paddle_h))
        self.right_y = float(np.clip(self.right_y, 0, self.height - self.paddle_h))

        # Update ball
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Collide top/bottom
        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vel_y *= -1
        elif self.ball_y >= self.height - self.ball_radius:
            self.ball_y = self.height - self.ball_radius
            self.ball_vel_y *= -1

        # Rewards for each agent this timestep (rally-level)
        reward_left = 0.0
        reward_right = 0.0
        info = {}
        rally_ended = False

        # Check left paddle collision / left miss (right scores)
        if self.ball_x - self.ball_radius <= self.paddle_w:
            if (self.left_y <= self.ball_y <= self.left_y + self.paddle_h):
                # bounce
                self.ball_x = self.paddle_w + self.ball_radius
                self.ball_vel_x = abs(self.ball_vel_x) + 0.1
                offset = (self.ball_y - (self.left_y + self.paddle_h / 2.0)) / (self.paddle_h / 2.0)
                self.ball_vel_y += offset * 2.0
                reward_left += 0.05  # small reward for hitting
            else:
                # right scores a point
                self.right_score += 1
                reward_right += 1.0
                rally_ended = True
                info['rally_winner'] = 'right'

        # Check right paddle collision / right miss (left scores)
        if self.ball_x + self.ball_radius >= self.width - self.paddle_w:
            if (self.right_y <= self.ball_y <= self.right_y + self.paddle_h):
                # bounce
                self.ball_x = self.width - self.paddle_w - self.ball_radius
                self.ball_vel_x = -abs(self.ball_vel_x) - 0.1
                offset = (self.ball_y - (self.right_y + self.paddle_h / 2.0)) / (self.paddle_h / 2.0)
                self.ball_vel_y += offset * 2.0
                reward_right += 0.05
            else:
                # left scores a point
                self.left_score += 1
                reward_left += 1.0
                rally_ended = True
                info['rally_winner'] = 'left'

        # small time penalty to encourage scoring
        reward_left -= 0.001
        reward_right -= 0.001

        # Cap speeds
        speed = np.sqrt(self.ball_vel_x ** 2 + self.ball_vel_y ** 2)
        max_speed = 8.0
        if speed > max_speed:
            scale = max_speed / speed
            self.ball_vel_x *= scale
            self.ball_vel_y *= scale

        # If a rally ended but nobody reached match max points -> reset rally (not done)
        done = False
        winner = None
        if rally_ended:
            # check match end
            if self.left_score >= self.max_points:
                done = True
                winner = 'left'
            elif self.right_score >= self.max_points:
                done = True
                winner = 'right'
            else:
                # reset ball for next rally in center with random direction
                self.ball_x = self.width / 2.0
                self.ball_y = self.height / 2.0
                angle = random.uniform(-0.25 * np.pi, 0.25 * np.pi)
                direction = random.choice([-1, 1])
                speed = random.uniform(3.0, 4.5)
                self.ball_vel_x = direction * speed * np.cos(angle)
                self.ball_vel_y = speed * np.sin(angle)
                # keep paddles where they are (or optionally reset) and continue
        self.steps += 1
        if self.steps >= MAX_TIMESTEPS:
            done = True
            winner = 'timeout'

        if done:
            self.done = True
            info['winner'] = winner
            info['final_score'] = (self.left_score, self.right_score)

        # return states for both agents
        return (self._get_state_left(), self._get_state_right(),
                float(reward_left), float(reward_right),
                done, info)

    def render(self):
        if not self.render_mode or not pygame.get_init():
            return
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
            self.screen.fill((0, 0, 0))
            # middle line
            pygame.draw.line(self.screen, (255, 255, 255), (self.width//2, 0), (self.width//2, self.height), 1)
            # paddles
            pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(0, int(self.left_y), self.paddle_w, self.paddle_h))
            pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.width - self.paddle_w, int(self.right_y), self.paddle_w, self.paddle_h))
            # ball
            pygame.draw.circle(self.screen, (255,255,255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)

            # Draw scores and labels
            left_label = self.font.render(f"Left: Q-learning ({self.left_score})", True, (200,200,255))
            right_label = self.font.render(f"Right: DQN ({self.right_score})", True, (200,200,255))
            self.screen.blit(left_label, (10, 10))
            self.screen.blit(right_label, (self.width - 180, 10))

            pygame.display.flip()
            self.clock.tick(self.fps)
        except pygame.error:
            # Handle case when video system was closed
            print("[Warning] Render skipped — pygame window not active.")
            return


    def close(self):
        if self.render_mode:
            pygame.quit()

# -------------------- Training Loop (multi-agent) --------------------
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
    best_model_path = os.path.join(MODEL_DIR, "pong_dqn_right_best.weights.h5")

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
    print("Training finished.")

    # Load and watch best model if exists
    if os.path.exists(best_model_path):
        print("\n🎮 Loading best trained DQN model (Right) and watching it play against Q-learning (Left)...")
        dqn_agent.main_Qnetwork.load_weights(best_model_path)
        watch_agent_play(dqn_agent, q_agent)
    else:
        print("⚠️ No trained DQN weights found. Skipping demo.")

# -------------------- Evaluation --------------------
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

# -------------------- Watch Trained Agents (rendered match) --------------------
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

# -------------------- Entry point --------------------
if __name__ == "__main__":
    print("Multi-agent Pong: LEFT = Q-learning  |  RIGHT = DQN")
    train()
