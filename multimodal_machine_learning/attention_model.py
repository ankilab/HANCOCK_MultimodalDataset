import tensorflow as tf
import random
import numpy as np


class AttentionMLPModel(tf.keras.Model):
    def __init__(self, input_dim, random_seed: int = 42):
        self.set_random_seed(random_seed)
        super(AttentionMLPModel, self).__init__()
        self.attention_layer = SelfAttentionWithResidual(input_dim)
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
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        return model

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


class SelfAttentionWithResidual(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, **kwargs):
        super(SelfAttentionWithResidual, self).__init__(**kwargs)
        self.input_dim = input_dim

        # Define trainable weight matrices for queries, keys, and values
        self.W_q = tf.keras.layers.Dense(1, use_bias=False)
        self.W_k = tf.keras.layers.Dense(1, use_bias=False)
        self.W_v = tf.keras.layers.Dense(1, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=True, return_attention_scores=False):
        if not training:
            self._set_layers_not_trainable()

        inputs = tf.expand_dims(inputs, axis=2)
        # Compute query, key, and value matrices
        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores /=  tf.cast(self.input_dim, tf.float32)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Apply attention weights to values
        attention_output = tf.matmul(attention_weights, v)

        output = self.layer_norm(
            tf.squeeze(inputs, axis=2) + tf.squeeze(attention_output, axis=2)
        )

        if return_attention_scores:
            return output, attention_weights
        else:
            return output

    def _set_layers_not_trainable(self):
        self.W_q.trainable = False
        self.W_k.trainable = False
        self.W_v.trainable = False
        self.layer_norm.trainable = False
