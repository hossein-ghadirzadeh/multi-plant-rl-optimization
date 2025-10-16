import pygame
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import time
import os
from itertools import product

# -------------------- Hyperparameters (defaults; overwritten by grid_search) --------------------
LEARNING_RATE = 5e-4
MINIBATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 1000
MAX_EPISODES = 50
MAX_TIMESTEPS = 10000
EVAL_EVERY_EPISODES = 50
RENDER_DURING_TRAIN = True
MAX_POINTS = 3

# Epsilon-greedy parameters (defaults; overwritten by grid_search)
EPSILON_START_DQN = 1.0
EPSILON_END_DQN = 0.01
EPSILON_DECAY_DQN = 0.995

EPSILON_START_Q = 1.0
EPSILON_END_Q = 0.01
EPSILON_DECAY_Q = 0.995

# Q-learning specific
Q_ALPHA = 0.1
Q_DISCRETE_BINS = {'bx': 6, 'by': 6, 'vx': 3, 'vy': 3, 'py': 6}

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

# -------------------- Q-Network (DQN) --------------------
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Do not set input_shape here; let the first call build weights.
        self.hidden_layer1 = layers.Dense(64, activation='relu')
        self.hidden_layer2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

# -------------------- Replay Buffer --------------------
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
        # If buffer still filling, we don't move position beyond appended items.

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

# -------------------- DQN Agent --------------------
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

        # Build networks by calling with a dummy state
        dummy = np.zeros((1, state_size), dtype=np.float32)
        self.main_Qnetwork(dummy)
        self.target_Qnetwork(dummy)
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

        # Standard DQN target (you can substitute Double-DQN here if desired)
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

# -------------------- Q-learning Agent --------------------
class QLearningAgent:
    def __init__(self, action_size=3, alpha=Q_ALPHA, gamma=DISCOUNT_FACTOR, epsilon_start=EPSILON_START_Q, epsilon_decay=EPSILON_DECAY_Q):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.bins = {
            'bx': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['bx'] - 1),
            'by': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['by'] - 1),
            'vx': np.array([-1.0, 0.0, 1.0]),
            'vy': np.array([-1.0, 0.0, 1.0]),
            'py': np.linspace(0.0, 1.0, Q_DISCRETE_BINS['py'] - 1)
        }

    def discretize(self, state):
        bx, by, vx, vy, py = state
        bx_idx = int(np.digitize(bx, self.bins['bx']))
        by_idx = int(np.digitize(by, self.bins['by']))
        vx_sign = int(np.sign(vx)) + 1
        vy_sign = int(np.sign(vy)) + 1
        py_idx = int(np.digitize(py, self.bins['py']))
        return (bx_idx, by_idx, vx_sign, vy_sign, py_idx)

    def get_qs(self, s_disc):
        if s_disc not in self.q_table:
            self.q_table[s_disc] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[s_disc]

    # <-- Updated act() method -->
    def act(self, state, epsilon=None):
        s_disc = self.discretize(state)
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
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
        self.q_table[s_disc] = q_vals
# -------------------- Pong Environment (multi-agent, score-limit, rally length + reaction time) ------------------
class PongEnv:
    def __init__(self, render_mode=False, max_points=3):
        import pygame
        pygame.init()
        self.pygame = pygame
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
            pygame.display.set_caption("Pong Multi-Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    def reset(self):
        # Paddles and ball
        self.left_y = (self.height - self.paddle_h)/2
        self.right_y = (self.height - self.paddle_h)/2
        self.ball_x = self.width/2
        self.ball_y = self.height/2
        import random, numpy as np
        angle = random.uniform(-0.25*np.pi,0.25*np.pi)
        direction = random.choice([-1,1])
        speed = random.uniform(3.0,4.5)
        self.ball_vel_x = direction*speed*np.cos(angle)
        self.ball_vel_y = speed*np.sin(angle)

        # Scores
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.done = False

        # ------------------ NEW METRICS ------------------
        self.rally_steps = 0
        self.left_rally_steps = 0
        self.right_rally_steps = 0

        # Reaction time tracking
        self.left_reaction_time = 0
        self.right_reaction_time = 0
        self.left_waiting_for_reaction = False
        self.right_waiting_for_reaction = False
        self.left_prev_y = self.left_y
        self.right_prev_y = self.right_y

        return self._get_state_left(), self._get_state_right()

    def _get_state_left(self):
        import numpy as np
        bx = self.ball_x/self.width
        by = self.ball_y/self.height
        vx = np.clip(self.ball_vel_x/6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y/6.0, -1.0, 1.0)
        py = self.left_y/(self.height-self.paddle_h)
        return np.array([bx,by,vx,vy,py], dtype=np.float32)

    def _get_state_right(self):
        import numpy as np
        bx = self.ball_x/self.width
        by = self.ball_y/self.height
        vx = np.clip(self.ball_vel_x/6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y/6.0, -1.0, 1.0)
        py = self.right_y/(self.height-self.paddle_h)
        return np.array([bx,by,vx,vy,py], dtype=np.float32)

    def step(self, left_action, right_action):
        import numpy as np
        # ------------------ Move paddles ------------------
        if left_action==0: self.left_y -= self.paddle_speed
        elif left_action==2: self.left_y += self.paddle_speed
        if right_action==0: self.right_y -= self.paddle_speed
        elif right_action==2: self.right_y += self.paddle_speed

        self.left_y = float(np.clip(self.left_y,0,self.height-self.paddle_h))
        self.right_y = float(np.clip(self.right_y,0,self.height-self.paddle_h))

        # ------------------ Move ball ------------------
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Bounce top/bottom
        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vel_y *= -1
        elif self.ball_y >= self.height - self.ball_radius:
            self.ball_y = self.height - self.ball_radius
            self.ball_vel_y *= -1

        # Rewards
        reward_left = -0.001
        reward_right = -0.001
        info = {}
        rally_ended = False
        self.rally_steps += 1

        # ------------------ Collision / Miss ------------------
        # Left paddle
        if self.ball_x - self.ball_radius <= self.paddle_w:
            if self.left_y <= self.ball_y <= self.left_y+self.paddle_h:
                self.ball_x = self.paddle_w + self.ball_radius
                self.ball_vel_x = abs(self.ball_vel_x)+0.1
                offset = (self.ball_y-(self.left_y+self.paddle_h/2.0))/(self.paddle_h/2.0)
                self.ball_vel_y += offset*2.0
                reward_left += 0.05
                self.left_rally_steps += 1
            else:
                self.right_score += 1
                reward_right += 1.0
                rally_ended = True
                info['rally_winner']='right'

        # Right paddle
        if self.ball_x + self.ball_radius >= self.width - self.paddle_w:
            if self.right_y <= self.ball_y <= self.right_y+self.paddle_h:
                self.ball_x = self.width - self.paddle_w - self.ball_radius
                self.ball_vel_x = -abs(self.ball_vel_x)-0.1
                offset = (self.ball_y-(self.right_y+self.paddle_h/2.0))/(self.paddle_h/2.0)
                self.ball_vel_y += offset*2.0
                reward_right += 0.05
                self.right_rally_steps += 1
            else:
                self.left_score += 1
                reward_left += 1.0
                rally_ended = True
                info['rally_winner']='left'

        # ------------------ Reaction Time ------------------
        ball_toward_left = self.ball_vel_x < 0
        ball_toward_right = self.ball_vel_x > 0

        # Left agent reaction
        if ball_toward_left:
            if not self.left_waiting_for_reaction:
                self.left_waiting_for_reaction = True
                self.left_reaction_time = 0
            else:
                self.left_reaction_time += 1
            if self.left_y != self.left_prev_y and self.left_waiting_for_reaction:
                info['reaction_time_left'] = self.left_reaction_time
                self.left_waiting_for_reaction = False

        # Right agent reaction
        if ball_toward_right:
            if not self.right_waiting_for_reaction:
                self.right_waiting_for_reaction = True
                self.right_reaction_time = 0
            else:
                self.right_reaction_time += 1
            if self.right_y != self.right_prev_y and self.right_waiting_for_reaction:
                info['reaction_time_right'] = self.right_reaction_time
                self.right_waiting_for_reaction = False

        self.left_prev_y = self.left_y
        self.right_prev_y = self.right_y

        # ------------------ Rally Ended Metrics ------------------
        if rally_ended:
            info['rally_length_left'] = self.left_rally_steps
            info['rally_length_right'] = self.right_rally_steps
            self.rally_steps = 0
            self.left_rally_steps = 0
            self.right_rally_steps = 0

            # Reset ball for next rally if match not finished
            import random
            if self.left_score < self.max_points and self.right_score < self.max_points:
                self.ball_x = self.width/2
                self.ball_y = self.height/2
                angle = random.uniform(-0.25*np.pi,0.25*np.pi)
                direction = random.choice([-1,1])
                speed = random.uniform(3.0,4.5)
                self.ball_vel_x = direction*speed*np.cos(angle)
                self.ball_vel_y = speed*np.sin(angle)

        # ------------------ Check Done ------------------
        self.steps += 1
        done = False
        winner = None
        if self.left_score >= self.max_points:
            done = True
            winner='left'
        elif self.right_score >= self.max_points:
            done = True
            winner='right'
        elif self.steps >= 10000:
            done = True
            winner='timeout'

        if done:
            self.done = True
            info['winner'] = winner
            info['final_score'] = (self.left_score, self.right_score)

        return self._get_state_left(), self._get_state_right(), float(reward_left), float(reward_right), done, info

    def render(self):
        if not self.render_mode: return
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()
        self.screen.fill((0,0,0))
        pygame.draw.line(self.screen,(255,255,255),(self.width//2,0),(self.width//2,self.height),1)
        pygame.draw.rect(self.screen,(0, 0, 255),pygame.Rect(0,int(self.left_y),self.paddle_w,self.paddle_h))
        pygame.draw.rect(self.screen,(255, 0, 0),pygame.Rect(self.width-self.paddle_w,int(self.right_y),self.paddle_w,self.paddle_h))
        pygame.draw.circle(self.screen,(255, 255, 0),(int(self.ball_x),int(self.ball_y)),self.ball_radius)
        if self.font:
            left_label = self.font.render(f"Q-learning Score: {self.left_score}", True, (200,200,255))
            right_label = self.font.render(f"DQN Score: {self.right_score}", True, (200,200,255))
            self.screen.blit(left_label,(10,10))
            self.screen.blit(right_label,(self.width-150,10))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.render_mode:
            self.pygame.quit()

# -------------------- Training and Evaluation (parameterized) --------------------
def train_run(state_size,
              action_size,
              q_agent,
              dqn_agent,
              render_during_train=RENDER_DURING_TRAIN,
              eval_every=EVAL_EVERY_EPISODES,
              max_episodes=MAX_EPISODES,
              model_path=None):
    import numpy as np, time, os
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
        model_path = os.path.join(MODEL_DIR, "pong_dqn_right_best.weights.h5")

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

# -------------------- Evaluation with metrics --------------------
def evaluate(dqn_agent, q_agent, render=True, episodes=50):
    env = PongEnv(render_mode=render, max_points=3)
    right_wins = 0
    left_rally_lengths = []
    right_rally_lengths = []
    left_reactions = []
    right_reactions = []

    for ep in range(episodes):
        state_left, state_right = env.reset()
        done = False
        while not done:
            action_left, _ = q_agent.act(state_left)
            action_right = dqn_agent.act(state_right, epsilon=0.0)
            state_left, state_right, _, _, done, info = env.step(action_left, action_right)
            if render:
                env.render()

            # Collect metrics if available
            if 'rally_length_left' in info:
                left_rally_lengths.append(info['rally_length_left'])
                right_rally_lengths.append(info['rally_length_right'])
            if 'reaction_time_left' in info:
                left_reactions.append(info['reaction_time_left'])
            if 'reaction_time_right' in info:
                right_reactions.append(info['reaction_time_right'])

        if info.get('winner') == 'right':
            right_wins += 1

    env.close()
    metrics = {
        'avg_rally_left': np.mean(left_rally_lengths) if left_rally_lengths else 0,
        'avg_rally_right': np.mean(right_rally_lengths) if right_rally_lengths else 0,
        'avg_reaction_left': np.mean(left_reactions) if left_reactions else 0,
        'avg_reaction_right': np.mean(right_reactions) if right_reactions else 0
    }

    return right_wins/episodes, metrics

# -------------------- Watch trained agents --------------------
# -------------------- Watch trained agents --------------------
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

# -------------------- Grid Search --------------------
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

    # Update global constants used in training loops:
    global LEARNING_RATE, MINIBATCH_SIZE, DISCOUNT_FACTOR, EPSILON_DECAY_DQN, EPSILON_DECAY_Q
    LEARNING_RATE = lr
    MINIBATCH_SIZE = minibatch
    DISCOUNT_FACTOR = discount
    EPSILON_DECAY_DQN = eps_decay_dqn
    EPSILON_DECAY_Q = eps_decay_q

    # Construct agents fresh for this combination
    state_size = 5
    action_size = 3

    q_agent = QLearningAgent(action_size=action_size, alpha=Q_ALPHA, gamma=discount, epsilon_start=EPSILON_START_Q, epsilon_decay=eps_decay_q)
    dqn_agent = DQNAgent(state_size, action_size,
                         learning_rate=lr,
                         minibatch_size=minibatch,
                         discount_factor=discount,
                         update_every=UPDATE_EVERY,
                         target_update_every=TARGET_UPDATE_EVERY,
                         replay_buffer_size=REPLAY_BUFFER_SIZE)

    # model path unique per combination
    model_filename = f"pong_dqn_best_combo_{combo_idx}_of_{total_combos}.weights.h5"
    model_path = os.path.join(MODEL_DIR, model_filename)

    print(f"\nStarting training for combination {combo_idx}/{total_combos} (model -> {model_path})\n")
    best_win_rate, saved_model_path = train_run(state_size=state_size,
                                                action_size=action_size,
                                                q_agent=q_agent,
                                                dqn_agent=dqn_agent,
                                                render_during_train=RENDER_DURING_TRAIN,
                                                eval_every=EVAL_EVERY_EPISODES,
                                                max_episodes=MAX_EPISODES,
                                                model_path=model_path)

    # Final evaluation after training (50 episodes)
    final_win_rate, metrics = evaluate(dqn_agent, q_agent, render=False, episodes=50)
    print(f"Final eval (post-train(50 episodes)) Right(DQN) win rate: {final_win_rate:.3f} | metrics: {metrics}")

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

# ----------------- Main ------------------
if __name__ == "__main__":
    print("Multi-agent Pong: LEFT = Q-learning | RIGHT = DQN")
    print("Grid-search over hyperparameters will run. This may take a long time (full training per combo).")
    grid_search()
















