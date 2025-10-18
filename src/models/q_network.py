import tensorflow as tf
from tensorflow.keras import layers


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