import tensorflow as tf


class AutoEncoder(tf.keras.Model):
    def __init__(self, vocab: dict, sos_idx, pad_idx: int, device: str, args):
        super(AutoEncoder, self).__init__()
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.device = device
        self.to(device)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)
