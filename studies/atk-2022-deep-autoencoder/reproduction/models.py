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
    raise ValueError(f"unsupported optimizer {name}")


def build_fc_sae(*, seed: int, learning_rate: float = 0.001) -> keras.Model:
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
    outputs = layers.Dense(
        48, activation=spec.output_activation, name="reconstruction"
    )(x)
    model = keras.Model(inputs, outputs, name="paper_fc_sae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_lstm_sae(*, seed: int, learning_rate: float = 0.001) -> keras.Model:
    """The direct 48-step×1-feature seq2seq reading of Algorithm 2/Table I."""

    set_seed(seed)
    spec = SPECS["lstm_sae"]
    inputs = keras.Input((48,), name="daily_profile")
    x = layers.Reshape((48, 1), name="time_steps")(inputs)
    x = layers.LSTM(
        500,
        activation=spec.hidden_activation,
        return_sequences=True,
        name="encoder_lstm_1_500",
    )(x)
    x = layers.Dropout(spec.dropout, name="encoder_dropout_1")(x)
    latent = layers.LSTM(
        300,
        activation=spec.hidden_activation,
        return_sequences=False,
        name="encoder_lstm_2_300",
    )(x)
    latent = layers.Dropout(spec.dropout, name="encoder_dropout_2")(latent)
    x = layers.RepeatVector(48, name="repeat_latent")(latent)
    x = layers.LSTM(
        300,
        activation=spec.hidden_activation,
        return_sequences=True,
        name="decoder_lstm_1_300",
    )(x)
    x = layers.Dropout(spec.dropout, name="decoder_dropout_1")(x)
    x = layers.LSTM(
        500,
        activation=spec.hidden_activation,
        return_sequences=True,
        name="decoder_lstm_2_500",
    )(x)
    x = layers.Dropout(spec.dropout, name="decoder_dropout_2")(x)
    x = layers.TimeDistributed(
        layers.Dense(1, activation=spec.output_activation),
        name="time_distributed_reconstruction",
    )(x)
    outputs = layers.Reshape((48,), name="reconstruction")(x)
    model = keras.Model(inputs, outputs, name="paper_lstm_sae")
    model.compile(
        optimizer=optimizer(spec.optimizer, learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_model(name: str, *, seed: int, learning_rate: float | None = None) -> keras.Model:
    if name == "fc_sae":
        return build_fc_sae(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    if name == "lstm_sae":
        return build_lstm_sae(
            seed=seed, learning_rate=0.001 if learning_rate is None else learning_rate
        )
    raise NotImplementedError(
        f"{name} follows after the FC-SAE/LSTM-SAE eligible anchors"
    )


def layer_inventory(model: keras.Model) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for layer in model.layers:
        config = layer.get_config()
        inventory.append(
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "units": config.get("units"),
                "rate": config.get("rate"),
                "activation": config.get("activation"),
                "output_shape": str(getattr(layer, "output", None).shape),
                "parameters": int(layer.count_params()),
            }
        )
    return inventory
