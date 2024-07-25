import tensorflow as tf
import numpy as np
    

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def positional_encoding(length: int, depth: int):
        """
            Generates a positional encoding for a given length and depth.

            Args:
                length (int): The length of the input sequence.
                depth (int): The depth that represents the dimensionality of the encoding.

            Returns:
                tf.Tensor: The positional encoding of shape (length, depth).
        """
        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
        """
        A positional embedding layer combines the input embedding with a positional encoding that helps the Transformer
        to understand the relative position of the input tokens. This layer takes the input of tokens and converts them
        into sequence of embeddings vector. Then, it adds the positional encoding to the embeddings.

        Methods:
            compute_mask: Computes the mask to be applied to the embeddings.
            call: Performs the forward pass of the layer.
        """

        def __init__(
                self, vocab_size: int, d_model: int,
                embedding: tf.keras.layers.Embedding = None, positions: int = 2048):
                """ Constructor of the PositionalEmbedding layer.

                Args:
                    vocab_size (int): The size of the vocabulary. I.e. the number of unique tokens in the input sequence.
                    d_model (int): The dimensionality of the embedding vector.
                    embedding (tf.keras.layers.Embedding): The custom embedding layer. If None, a default embedding layer will be created.
                    positions (int, optional): The length of the input sequence. Defaults to 2048.
                """
                super().__init__()
                self.d_model = d_model
                self.embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                                                           mask_zero=True) if embedding is None else embedding
                self.pos_encoding = positional_encoding(length=positions, depth=d_model)

        def compute_mask(self, *args, **kwargs):
                """ Computes the mask to be applied to the embeddings.
                """
                return self.embedding.compute_mask(*args, **kwargs)

        def call(self, x: tf.Tensor) -> tf.Tensor:
                """ Performs the forward pass of the layer.

                Args:
                    x (tf.Tensor): The input tensor of shape (batch_size, seq_length).

                Returns:
                    tf.Tensor: The output sequence of embedding vectors with added positional information. The shape is
                        (batch_size, seq_length, d_model).
                """
                x = self.embedding(x)
                length = tf.shape(x)[1]
                # This factor sets the relative scale of the embedding and positional_encoding.
                x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
                x = x + self.pos_encoding[tf.newaxis, :length, :]
                return x


class BaseAttention(tf.keras.layers.Layer):
        """
        Base class for all attention layers. It contains the common functionality of all attention layers.
        This layer contains a MultiHeadAttention layer, a LayerNormalization layer and an Add layer.
        It is used as a base class for the GlobalSelfAttention, CausalSelfAttention and CrossAttention layers.
        And it is not intended to be used directly.

        Methods:
            call: Performs the forward pass of the layer.

        Attributes:
            mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
            layer_norm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
            add (tf.keras.layers.Add): The Add layer.
        """

        def __init__(self, num_heads: int = 2, key_dim: int = 512, **kwargs: dict):
                """ Constructor of the BaseAttention layer.

                Args:
                    num_heads (int, optional): The number of heads in the MultiHeadAttention layer. Defaults to 2.
                    key_dim (int, optional): The dimensionality of the key space in the MultiHeadAttention layer. Defaults to 512.
                    **kwargs: Additional keyword arguments that are passed to the MultiHeadAttention layer
                """
                super().__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, **kwargs)
                self.layer_norm = tf.keras.layers.LayerNormalization()
                self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
        """
        A class that implements the global self-attention layer by inheriting from the BaseAttention class.
        This layer is used to process a single sequence and attends to all the tokens in the sequence.

        Methods:
            call: Performs the forward pass of the layer.

        Attributes:
            mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
            layer_norm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
            add (tf.keras.layers.Add): The Add layer.
        """

        def __init__(self, num_heads: int = 2, key_dim: int = 512, **kwargs: dict):
                """ Constructor of the BaseAttention layer.

                Args:
                    num_heads (int, optional): The number of heads in the MultiHeadAttention layer. Defaults to 2.
                    key_dim (int, optional): The dimensionality of the key space in the MultiHeadAttention layer. Defaults to 512.
                    **kwargs: Additional keyword arguments that are passed to the MultiHeadAttention layer
                """
                super().__init__(num_heads = num_heads, key_dim = key_dim, **kwargs)


        def call(self, x: tf.Tensor) -> tf.Tensor:
                """
                The call function that performs the global self-attention operation.

                Args:
                    x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

                Returns:
                    tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
                """
                attn_output = self.mha(query=x, value=x, key=x)
                x = self.add([x, attn_output])
                x = self.layer_norm(x)
                return x


if __name__ == "__main__":
        d_model = 512
        seq_len = 100
        num_classes = 2
        num_heads = 2
        ff_dim = 2048
        random_input = np.random.randint(low=0, high=3, size=(1, seq_len, d_model))
        pos_encodings = positional_encoding(seq_len, d_model)
        x = random_input + pos_encodings[tf.newaxis, :, :]

