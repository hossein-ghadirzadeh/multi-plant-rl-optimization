import sys
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer
from config import LEARNING_RATE, MINIBATCH_SIZE, DISCOUNT_FACTOR, UPDATE_EVERY, TARGET_UPDATE_EVERY, REPLAY_BUFFER_SIZE

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
