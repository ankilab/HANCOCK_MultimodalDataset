import tensorflow as tf
import random
import numpy as np
import os


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Set the random seed
set_random_seed(42)

def create_mlp(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Output layer for binary classification
    ])
    return model

class AttentionMLPModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(AttentionMLPModel, self).__init__()
        self.attention = tf.keras.layers.Attention()
        self.mlp = create_mlp(input_dim)

    def call(self, inputs):
        # Using the same data for query, key, and value
        query = tf.expand_dims(inputs, axis=1)  # Adding a dummy dimension
        value = tf.expand_dims(inputs, axis=1)  # Adding a dummy dimension
        key = tf.expand_dims(inputs, axis=1)    # Adding a dummy dimension

        context = self.attention([query, value, key])
        context = tf.squeeze(context, axis=1)  # Removing the dummy dimension

        output = self.mlp(context)
        return output
