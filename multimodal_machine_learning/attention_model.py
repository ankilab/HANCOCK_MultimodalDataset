import tensorflow as tf
import random
import numpy as np


class AttentionMLPModel(tf.keras.Model):
    def __init__(self, input_dim, random_seed: int = 42):
        self.set_random_seed(random_seed)
        super(AttentionMLPModel, self).__init__()
        self.attention_layer = AttentionLayer()
        self.mlp = self.create_mlp()

    def call(self, inputs):
        context = self.attention_layer(inputs)
        output = self.mlp(context)
        return output

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=(input_shape,))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def create_mlp():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # Output layer for binary classification
        ])
        return model

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.attention = tf.keras.layers.Attention()

    def call(self, inputs):
        query = tf.expand_dims(inputs, axis=1)
        value = tf.expand_dims(inputs, axis=1)
        key = tf.expand_dims(inputs, axis=1)
        attention_output = self.attention([query, value, key])
        context = tf.squeeze(attention_output, axis=1)
        return context
