"""Direct Keras implementations of the architectures printed in Paper 1."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder: tuple[int, ...]
    decoder: tuple[int, ...]
    optimizer: str
    dropout: float
    hidden_activation: str
    output_activation: str
    threshold: float
    anomaly_score: str
    anomaly_direction: str


SPECS = {
    "fc_sae": ModelSpec(
        "FC-SAE", (400, 300, 200, 100), (100, 200, 300, 400),
        "Adam", 0.4, "sigmoid", "softmax", 0.58, "mse", "higher",
    ),
    "lstm_sae": ModelSpec(
        "LSTM-SAE", (500, 300), (300, 500),
        "Adam", 0.2, "sigmoid", "sigmoid", 0.61, "mse", "higher",
    ),
    "fc_vae": ModelSpec(
        "FC-VAE", (500, 400, 300, 100), (100, 300, 400, 500),
        "Adam", 0.4, "relu", "softmax", 0.43,
        "reconstruction_probability", "lower",
    ),
    "lstm_vae": ModelSpec(
        "LSTM-VAE", (400, 300), (300, 400),
        "SGD", 0.0, "tanh", "sigmoid", 0.47,
        "reconstruction_probability", "lower",
    ),
    "lstm_aea": ModelSpec(
        "LSTM-AEA", (500, 300, 200), (200, 300, 500),
        "SGD", 0.0, "sigmoid", "sigmoid", 0.51, "mse", "higher",
    ),
}


def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)


def optimizer(name: str, learning_rate: float | None = None) -> keras.optimizers.Optimizer:
    kwargs = {} if learning_rate is None else {"learning_rate": learning_rate}
    if name == "Adam":
        return keras.optimizers.Adam(**kwargs)
    if name == "SGD":
        return keras.optimizers.SGD(**kwargs)
    if name == "Adamax":
        return keras.optimizers.Adamax(**kwargs)
    raise ValueError(f"unsupported optimizer {name}")


class Sampling(layers.Layer):
    """Gaussian reparameterization plus the paper's analytic KL term."""

    def __init__(self, seed: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(
        self,
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
        training: bool | None = None,
    ) -> keras.KerasTensor:
        mean, log_variance = inputs
        kl = -0.5 * keras.ops.sum(
            1 + log_variance - keras.ops.square(mean) - keras.ops.exp(log_variance),
            axis=-1,
        )
        self.add_loss(keras.ops.mean(kl))
        if training is False:
            return mean
        epsilon = keras.random.normal(
            keras.ops.shape(mean), seed=self.seed_generator
        )
        return mean + keras.ops.exp(0.5 * log_variance) * epsilon


def _lstm_encoder(
    inputs: keras.KerasTensor,
    widths: tuple[int, ...],
    *,
    activation: str,
    dropout: float,
    prefix: str = "encoder",
) -> tuple[keras.KerasTensor, keras.KerasTensor, list[tuple[keras.KerasTensor, keras.KerasTensor]]]:
    value = layers.Reshape((48, 1), name=f"{prefix}_time_steps")(inputs)
    states: list[tuple[keras.KerasTensor, keras.KerasTensor]] = []
    for index, width in enumerate(widths, start=1):
        value, hidden, cell = layers.LSTM(
            width,
            activation=activation,
            dropout=dropout,
            return_sequences=True,
            return_state=True,
            name=f"{prefix}_lstm_{index}_{width}",
        )(value)
        states.append((hidden, cell))
    return value, states[-1][0], states


def _lstm_decoder(
    latent: keras.KerasTensor,
    states: list[tuple[keras.KerasTensor, keras.KerasTensor]],
    widths: tuple[int, ...],
    *,
    activation: str,
    dropout: float,
    output_activation: str,
    prefix: str = "decoder",
) -> keras.KerasTensor:
    """Repeat-latent completion with Algorithm-2/4 mirrored state transfer."""

    value = layers.RepeatVector(48, name=f"{prefix}_repeat_latent")(latent)
    for index, width in enumerate(widths, start=1):
        value = layers.LSTM(
            width,
            activation=activation,
            dropout=dropout,
            return_sequences=True,
            name=f"{prefix}_lstm_{index}_{width}",
        )(value, initial_state=states[-index])
    value = layers.TimeDistributed(
        layers.Dense(1, activation=output_activation),
        name=f"{prefix}_time_distributed_output",
    )(value)
    return layers.Reshape((48,), name="reconstruction")(value)


def build_fc_sae(
    *,
    seed: int,
    learning_rate: float = 0.001,
    output_activation: str = "softmax",
) -> keras.Model:
    """Full four-layer encoder and full four-layer mirror from Table I."""

    set_seed(seed)
    spec = SPECS["fc_sae"]
    inputs = keras.Input((48,), name="daily_profile")
    x = inputs
    for side, widths in (("encoder", spec.encoder), ("decoder", spec.decoder)):
        for index, width in enumerate(widths, start=1):
            x = layers.Dense(
                width,
                activation=spec.hidden_activation,
                name=f"{side}_dense_{index}_{width}",
            )(x)
            x = layers.Dropout(
                spec.dropout, name=f"{side}_dropout_{index}"
            )(x)
    if output_activation not in {"softmax", "linear"}:
        raise ValueError(f"unsupported FC-SAE output activation {output_activation}")
    outputs = layers.Dense(48, activation=output_activation, name="reconstruction")(x)
    model = keras.Model(inputs, outputs, name="paper_fc_sae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    validate_fc_sae(model, output_activation=output_activation)
    return model


def validate_fc_sae(model: keras.Model, *, output_activation: str = "softmax") -> None:
    """Fail if the runtime model drifts beyond the named one-factor branch."""

    dense = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    dropout = [
        layer for layer in model.layers if isinstance(layer, layers.Dropout)
    ]
    units = [int(layer.units) for layer in dense]
    activations = [layer.activation.__name__ for layer in dense]
    expected_units = [400, 300, 200, 100, 100, 200, 300, 400, 48]
    expected_activations = ["sigmoid"] * 8 + [output_activation]
    if units != expected_units:
        raise AssertionError(f"FC-SAE widths drifted: {units}")
    if activations != expected_activations:
        raise AssertionError(f"FC-SAE activations drifted: {activations}")
    if len(dropout) != 8 or any(float(layer.rate) != 0.4 for layer in dropout):
        raise AssertionError("FC-SAE must apply dropout 0.4 after all 8 hidden layers")
    if int(model.count_params()) != 450_448:
        raise AssertionError(f"FC-SAE parameter count drifted: {model.count_params()}")


def build_lstm_sae(*, seed: int, learning_rate: float = 0.001) -> keras.Model:
    """Table-I LSTM-SAE plus the mirrored state transfer in Algorithm 2."""

    set_seed(seed)
    spec = SPECS["lstm_sae"]
    inputs = keras.Input((48,), name="daily_profile")
    _, latent, states = _lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    outputs = _lstm_decoder(
        latent,
        states,
        spec.decoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
        output_activation=spec.output_activation,
    )
    model = keras.Model(inputs, outputs, name="paper_lstm_sae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_fc_vae(*, seed: int, learning_rate: float = 0.001) -> keras.Model:
    """Table-I FC-VAE with the omitted latent width completed as 100."""

    set_seed(seed)
    spec = SPECS["fc_vae"]
    inputs = keras.Input((48,), name="daily_profile")
    value = inputs
    for index, width in enumerate(spec.encoder, start=1):
        value = layers.Dense(
            width,
            activation=spec.hidden_activation,
            name=f"encoder_dense_{index}_{width}",
        )(value)
        value = layers.Dropout(spec.dropout, name=f"encoder_dropout_{index}")(
            value
        )
    latent_width = spec.encoder[-1]
    mean = layers.Dense(latent_width, name="latent_mean")(value)
    log_variance = layers.Dense(latent_width, name="latent_log_variance")(value)
    value = Sampling(seed, name="latent_sample")((mean, log_variance))
    for index, width in enumerate(spec.decoder, start=1):
        value = layers.Dense(
            width,
            activation=spec.hidden_activation,
            name=f"decoder_dense_{index}_{width}",
        )(value)
        value = layers.Dropout(spec.dropout, name=f"decoder_dropout_{index}")(
            value
        )
    outputs = layers.Dense(
        48, activation=spec.output_activation, name="reconstruction"
    )(value)
    model = keras.Model(inputs, outputs, name="paper_fc_vae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_lstm_vae(*, seed: int, learning_rate: float = 0.01) -> keras.Model:
    """Table-I LSTM-VAE with a 300-wide Gaussian latent completion."""

    set_seed(seed)
    spec = SPECS["lstm_vae"]
    inputs = keras.Input((48,), name="daily_profile")
    _, encoded, states = _lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    latent_width = spec.encoder[-1]
    mean = layers.Dense(latent_width, name="latent_mean")(encoded)
    log_variance = layers.Dense(latent_width, name="latent_log_variance")(
        encoded
    )
    latent = Sampling(seed, name="latent_sample")((mean, log_variance))
    outputs = _lstm_decoder(
        latent,
        states,
        spec.decoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
        output_activation=spec.output_activation,
    )
    model = keras.Model(inputs, outputs, name="paper_lstm_vae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_lstm_aea(*, seed: int, learning_rate: float = 0.01) -> keras.Model:
    """Table-I LSTM-AEA with the smallest executable additive-attention repair."""

    set_seed(seed)
    spec = SPECS["lstm_aea"]
    inputs = keras.Input((48,), name="daily_profile")
    encoder_sequence, latent, states = _lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    value = layers.RepeatVector(48, name="decoder_repeat_latent")(latent)
    value = layers.LSTM(
        spec.decoder[0],
        activation=spec.hidden_activation,
        dropout=spec.dropout,
        return_sequences=True,
        name=f"decoder_lstm_1_{spec.decoder[0]}",
    )(value, initial_state=states[-1])
    initial_query = layers.Reshape(
        (1, spec.decoder[0]), name="initial_decoder_query"
    )(states[-1][0])
    earlier_queries = layers.Lambda(
        lambda sequence: sequence[:, :-1, :], name="earlier_decoder_queries"
    )(value)
    previous_queries = layers.Concatenate(
        axis=1, name="previous_decoder_queries"
    )([initial_query, earlier_queries])
    context = layers.AdditiveAttention(name="temporal_additive_attention")(
        [previous_queries, encoder_sequence]
    )
    value = layers.Concatenate(name="attention_context_and_decoder")(
        [context, value]
    )
    for index, width in enumerate(spec.decoder[1:], start=2):
        value = layers.LSTM(
            width,
            activation=spec.hidden_activation,
            dropout=spec.dropout,
            return_sequences=True,
            name=f"decoder_lstm_{index}_{width}",
        )(value, initial_state=states[-index])
    value = layers.TimeDistributed(
        layers.Dense(1, activation=spec.output_activation),
        name="decoder_time_distributed_output",
    )(value)
    outputs = layers.Reshape((48,), name="reconstruction")(value)
    model = keras.Model(inputs, outputs, name="paper_lstm_aea")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_supervised_feed_forward(
    *, seed: int, learning_rate: float = 0.001
) -> keras.Model:
    """Five 500-unit ReLU layers, Adamax, and the frozen two-Softmax repair."""

    set_seed(seed)
    inputs = keras.Input((48,), name="daily_profile")
    value = inputs
    for index in range(1, 6):
        value = layers.Dense(500, activation="relu", name=f"hidden_dense_{index}")(
            value
        )
    outputs = layers.Dense(2, activation="softmax", name="class_probability")(
        value
    )
    model = keras.Model(inputs, outputs, name="paper_supervised_feed_forward")
    model.compile(
        optimizer=optimizer("Adamax", learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_supervised_lstm(
    *, seed: int, learning_rate: float = 0.001
) -> keras.Model:
    """Four 300-unit ReLU LSTMs and the frozen one-Sigmoid repair."""

    set_seed(seed)
    inputs = keras.Input((48,), name="daily_profile")
    value = layers.Reshape((48, 1), name="time_steps")(inputs)
    for index in range(1, 5):
        value = layers.LSTM(
            300,
            activation="relu",
            dropout=0.2,
            return_sequences=index < 4,
            name=f"hidden_lstm_{index}",
        )(value)
    outputs = layers.Dense(1, activation="sigmoid", name="class_probability")(
        value
    )
    model = keras.Model(inputs, outputs, name="paper_supervised_lstm")
    model.compile(
        optimizer=optimizer("Adam", learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(
    name: str,
    *,
    seed: int,
    learning_rate: float | None = None,
    output_activation: str | None = None,
) -> keras.Model:
    if name == "fc_sae":
        return build_fc_sae(
            seed=seed,
            learning_rate=0.001 if learning_rate is None else learning_rate,
            output_activation=output_activation or SPECS[name].output_activation,
        )
    if output_activation is not None:
        raise ValueError("output-activation override is currently FC-SAE only")
    if name == "lstm_sae":
        return build_lstm_sae(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    if name == "fc_vae":
        return build_fc_vae(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    if name == "lstm_vae":
        return build_lstm_vae(
            seed=seed, learning_rate=0.01 if learning_rate is None else learning_rate
        )
    if name == "lstm_aea":
        return build_lstm_aea(
            seed=seed, learning_rate=0.01 if learning_rate is None else learning_rate
        )
    if name == "supervised_feed_forward":
        return build_supervised_feed_forward(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    if name == "supervised_lstm":
        return build_supervised_lstm(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    raise ValueError(f"unsupported model {name}")


def layer_inventory(model: keras.Model) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for layer in model.layers:
        config = layer.get_config()
        output = getattr(layer, "output", None)
        output_shape = (
            [str(item.shape) for item in output]
            if isinstance(output, (list, tuple))
            else str(getattr(output, "shape", None))
        )
        inventory.append(
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "units": config.get("units"),
                "rate": config.get("rate"),
                "activation": config.get("activation"),
                "output_shape": output_shape,
                "parameters": int(layer.count_params()),
            }
        )
    return inventory
