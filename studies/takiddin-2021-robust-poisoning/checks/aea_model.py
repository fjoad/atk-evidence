"""Constructed-only AEA completion; research fitting is intentionally absent."""

from __future__ import annotations

import numpy as np


def build_aea(seed=20260920, *, encoder_units=(500, 300, 200), decoder_units=(200, 300, 500),
              timesteps=48, reconstruction_loss="mae"):
    """Build the declared I-AEA-native-table interpretation for fixtures.

    The target's AEA text omits the decoder schedule, state bridge, attention
    width, initial reconstruction and error definition. This function makes
    those choices explicit in AEA_SPECIFICATION.md; it does not claim literal
    equivalence to every printed recurrence.
    """
    import tensorflow as tf
    import keras

    if len(encoder_units) != 3 or len(decoder_units) != 3 or decoder_units != tuple(reversed(encoder_units)):
        raise ValueError("AEA requires three mirrored encoder/decoder widths")
    if timesteps < 2 or any(int(u) < 1 for u in (*encoder_units, *decoder_units)):
        raise ValueError("Invalid AEA dimensions")
    if reconstruction_loss not in ("mae", "mse"):
        raise ValueError("AEA fixture supports only declared MAE/MSE losses")
    keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

    max_norm = keras.constraints.MaxNorm(1., axis=0)

    class AEARecurrent(keras.layers.Layer):
        def __init__(self, enc, dec, steps, **kwargs):
            super().__init__(**kwargs)
            self.enc_units, self.dec_units, self.steps = tuple(enc), tuple(dec), int(steps)
            self.encoder = [keras.layers.LSTMCell(
                u, activation="sigmoid", recurrent_activation="sigmoid",
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed + i),
                recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 10 + i),
                bias_initializer="zeros", unit_forget_bias=True, dropout=0., recurrent_dropout=0.,
                kernel_constraint=max_norm, recurrent_constraint=max_norm,
                name=f"encoder_lstm_{i+1}") for i, u in enumerate(enc)]
            self.decoder = [keras.layers.LSTMCell(
                u, activation="sigmoid", recurrent_activation="sigmoid",
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 20 + i),
                recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 30 + i),
                bias_initializer="zeros", unit_forget_bias=True, dropout=0., recurrent_dropout=0.,
                kernel_constraint=max_norm, recurrent_constraint=max_norm,
                name=f"decoder_lstm_{i+1}") for i, u in enumerate(dec)]
            self.output_projection = keras.layers.Dense(
                1, activation="sigmoid", kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 40),
                bias_initializer="zeros", kernel_constraint=max_norm, name="reconstruction_projection")

        def build(self, input_shape):
            input_width = int(input_shape[-1])
            for i, cell in enumerate(self.encoder):
                cell.build((None, input_width if i == 0 else self.enc_units[i - 1]))
            for i, cell in enumerate(self.decoder):
                cell.build((None, self.enc_units[-1] + 1 if i == 0 else self.dec_units[i - 1]))
            self.output_projection.build((None, self.dec_units[-1]))
            self.attention_encoder_kernel = self.add_weight(
                name="attention_encoder_kernel", shape=(self.enc_units[-1], self.enc_units[-1]),
                initializer=keras.initializers.GlorotUniform(seed=seed + 50), constraint=max_norm)
            self.attention_decoder_kernel = self.add_weight(
                name="attention_decoder_kernel", shape=(self.dec_units[-1], self.enc_units[-1]),
                initializer=keras.initializers.GlorotUniform(seed=seed + 51), constraint=max_norm)
            self.attention_bias = self.add_weight(
                name="attention_bias", shape=(self.enc_units[-1],), initializer="zeros")
            self.attention_vector = self.add_weight(
                name="attention_vector", shape=(self.enc_units[-1],),
                initializer=keras.initializers.GlorotUniform(seed=seed + 52), constraint=max_norm)
            super().build(input_shape)

        def call(self, inputs, training=False):
            batch = tf.shape(inputs)[0]
            encoder_states = [[tf.zeros((batch, cell.units), inputs.dtype),
                               tf.zeros((batch, cell.units), inputs.dtype)] for cell in self.encoder]
            top_outputs = []
            for t in range(self.steps):
                value = inputs[:, t, :]
                for i, cell in enumerate(self.encoder):
                    value, state = cell(value, encoder_states[i], training=training)
                    encoder_states[i] = state
                top_outputs.append(value)
            memory = tf.stack(top_outputs, axis=1)
            memory_projection = tf.einsum("btd,dk->btk", memory, self.attention_encoder_kernel)
            decoder_states = [
                [encoder_states[2][0], encoder_states[2][1]],
                [encoder_states[1][0], encoder_states[1][1]],
                [encoder_states[0][0], encoder_states[0][1]],
            ]
            previous = tf.zeros((batch, 1), inputs.dtype)
            outputs, attention = [], []
            for _ in range(self.steps):
                query = tf.einsum("bd,dk->bk", decoder_states[-1][0], self.attention_decoder_kernel)
                energy = tf.einsum(
                    "btk,k->bt", tf.tanh(memory_projection + query[:, None, :] + self.attention_bias),
                    self.attention_vector)
                weights = tf.nn.softmax(energy, axis=1)
                context = tf.einsum("bt,btd->bd", weights, memory)
                value = tf.concat((context, previous), axis=-1)
                for i, cell in enumerate(self.decoder):
                    value, state = cell(value, decoder_states[i], training=training)
                    decoder_states[i] = state
                previous = self.output_projection(value)
                outputs.append(previous)
                attention.append(weights)
            self.last_attention = tf.stack(attention, axis=1)
            return tf.stack(outputs, axis=1)

    inputs = keras.Input(shape=(timesteps, 1), dtype="float32", name="daily_profile")
    recurrent = AEARecurrent(encoder_units, decoder_units, timesteps, name="aea_recurrent")
    outputs = recurrent(inputs)
    model = keras.Model(inputs, outputs, name="aea_native_table")
    loss = keras.losses.MeanAbsoluteError() if reconstruction_loss == "mae" else keras.losses.MeanSquaredError()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=.01, momentum=0., nesterov=False),
                  loss=loss, jit_compile=False)
    return model


def attention_weights(model, x, training=False):
    """Return the latest attention tensor after a forward pass."""
    output = model(x, training=training)
    layer = model.get_layer("aea_recurrent")
    return output, layer.last_attention


def parameter_formula(encoder_units=(500, 300, 200), decoder_units=(200, 300, 500)):
    encoder = sum(4 * u * (d + u + 1) for d, u in zip((1, *encoder_units[:-1]), encoder_units))
    decoder = sum(4 * u * (d + u + 1) for d, u in zip((encoder_units[-1] + 1, *decoder_units[:-1]), decoder_units))
    attention = encoder_units[-1] ** 2 + decoder_units[-1] * encoder_units[-1] + 2 * encoder_units[-1]
    projection = decoder_units[-1] + 1
    return {"encoder": encoder, "decoder": decoder, "attention": attention, "projection": projection,
            "total": encoder + decoder + attention + projection}
