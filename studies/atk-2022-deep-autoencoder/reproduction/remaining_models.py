"""Approved remaining-paper model completions, isolated from historical code hashes."""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

from models import SPECS, optimizer, set_seed


REMAINING_PAPER_CONTRACT = "remaining-paper-v1"
REMAINING_MODELS = ("lstm_sae", "fc_vae", "lstm_vae", "lstm_aea")
REMAINING_PARAMETER_COUNTS = {
    "lstm_sae": 4_288_901,
    "fc_vae": 780_848,
    "lstm_vae": 3_508_201,
    "lstm_aea": 5_132_402,
}


@keras.saving.register_keras_serializable(package="atk_evidence")
class RemainingSampling(layers.Layer):
    """Gaussian reparameterization; EquationTenLoss owns the sole KL term."""

    def __init__(self, seed: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.seed_generator = keras.random.SeedGenerator(self.seed)

    def call(
        self,
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> keras.KerasTensor:
        mean, log_variance = inputs
        epsilon = keras.random.normal(
            keras.ops.shape(mean), seed=self.seed_generator
        )
        return mean + keras.ops.exp(0.5 * log_variance) * epsilon

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "seed": self.seed}


@keras.saving.register_keras_serializable(package="atk_evidence")
class EquationTenLoss(layers.Layer):
    """Attach the predeclared per-example SSE-plus-KL VAE objective."""

    def call(
        self,
        inputs: tuple[
            keras.KerasTensor,
            keras.KerasTensor,
            keras.KerasTensor,
            keras.KerasTensor,
        ],
    ) -> keras.KerasTensor:
        target, reconstruction, mean, log_variance = inputs
        reconstruction_error = keras.ops.sum(
            keras.ops.square(target - reconstruction), axis=-1
        )
        kl = -0.5 * keras.ops.sum(
            1
            + log_variance
            - keras.ops.square(mean)
            - keras.ops.exp(log_variance),
            axis=-1,
        )
        self.add_loss(keras.ops.mean(reconstruction_error + kl))
        return reconstruction


@keras.saving.register_keras_serializable(package="atk_evidence")
class FirstStepLatentSequence(layers.Layer):
    """Put the latent at decoder step one and zeros at steps 2 through 48."""

    def __init__(self, steps: int = 48, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = int(steps)

    def call(self, latent: keras.KerasTensor) -> keras.KerasTensor:
        first = keras.ops.expand_dims(latent, axis=1)
        if self.steps == 1:
            return first
        shape = keras.ops.shape(latent)
        zeros = keras.ops.zeros(
            (shape[0], self.steps - 1, shape[1]), dtype=latent.dtype
        )
        return keras.ops.concatenate((first, zeros), axis=1)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "steps": self.steps}


@keras.saving.register_keras_serializable(package="atk_evidence")
class TemporalAdditiveAttention(layers.Layer):
    """Bahdanau attention over encoder time steps for one decoder query."""

    def __init__(self, units: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.key_projection = layers.Dense(self.units, name="key_projection")
        self.query_projection = layers.Dense(
            self.units, use_bias=False, name="query_projection"
        )
        self.energy_projection = layers.Dense(
            1, use_bias=False, name="energy_projection"
        )

    def call(
        self,
        inputs: tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:
        sequence, query = inputs
        key = self.key_projection(sequence)
        projected_query = self.query_projection(query)
        energy = self.energy_projection(
            keras.ops.tanh(key + keras.ops.expand_dims(projected_query, axis=1))
        )
        weights = keras.ops.softmax(keras.ops.squeeze(energy, axis=-1), axis=1)
        context = keras.ops.sum(
            sequence * keras.ops.expand_dims(weights, axis=-1), axis=1
        )
        return context, weights

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="atk_evidence")
class AutoregressiveAttentionDecoder(layers.Layer):
    """Approved Algorithm-5 completion with previous-output feedback."""

    def __init__(
        self,
        widths: tuple[int, ...],
        *,
        steps: int = 48,
        activation: str = "sigmoid",
        output_activation: str = "sigmoid",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.widths = tuple(int(width) for width in widths)
        self.steps = int(steps)
        self.activation_name = activation
        self.output_activation_name = output_activation
        if not self.widths or self.steps < 1:
            raise ValueError("autoregressive decoder requires widths and steps")
        self.initial_query = layers.Dense(
            self.widths[-1], activation=activation, name="initial_query"
        )
        self.initial_reconstruction = layers.Dense(
            1, name="initial_reconstruction"
        )
        self.attention = TemporalAdditiveAttention(
            self.widths[0], name="temporal_attention"
        )
        self.cells = [
            layers.LSTMCell(width, activation=activation, name=f"cell_{index}")
            for index, width in enumerate(self.widths, start=1)
        ]
        self.output_projection = layers.Dense(
            1, activation=output_activation, name="output_projection"
        )

    def call(
        self,
        inputs: tuple[
            keras.KerasTensor,
            keras.KerasTensor,
            keras.KerasTensor,
        ],
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:
        encoder_sequence, encoder_hidden, encoder_cell = inputs
        batch = keras.ops.shape(encoder_sequence)[0]
        states: list[list[keras.KerasTensor]] = [
            [encoder_hidden, encoder_cell]
        ]
        for width in self.widths[1:]:
            zero = keras.ops.zeros(
                (batch, width), dtype=encoder_sequence.dtype
            )
            states.append([zero, zero])
        query = self.initial_query(encoder_hidden)
        previous = self.initial_reconstruction(encoder_hidden)
        outputs: list[keras.KerasTensor] = []
        attention_weights: list[keras.KerasTensor] = []
        for _ in range(self.steps):
            context, weights = self.attention((encoder_sequence, query))
            value = keras.ops.concatenate((context, previous), axis=-1)
            for index, cell in enumerate(self.cells):
                value, states[index] = cell(
                    value, states=states[index], training=training
                )
            query = value
            previous = self.output_projection(value)
            outputs.append(previous)
            attention_weights.append(keras.ops.expand_dims(weights, axis=1))
        return (
            keras.ops.concatenate(outputs, axis=1),
            keras.ops.concatenate(attention_weights, axis=1),
        )

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "widths": self.widths,
            "steps": self.steps,
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
        }


@dataclass(frozen=True)
class RemainingPaperBundle:
    """Training and scoring interfaces for one approved remaining model."""

    name: str
    model: keras.Model
    encoder: keras.Model | None = None
    decoder: keras.Model | None = None
    attention_model: keras.Model | None = None


def _remaining_lstm_encoder(
    inputs: keras.KerasTensor,
    widths: tuple[int, ...],
    *,
    activation: str,
    dropout: float,
) -> tuple[
    keras.KerasTensor,
    keras.KerasTensor,
    keras.KerasTensor,
]:
    value = layers.Reshape((48, 1), name="encoder_time_steps")(inputs)
    hidden: keras.KerasTensor | None = None
    cell: keras.KerasTensor | None = None
    for index, width in enumerate(widths, start=1):
        value, hidden, cell = layers.LSTM(
            width,
            activation=activation,
            dropout=dropout,
            return_sequences=True,
            return_state=True,
            name=f"encoder_lstm_{index}_{width}",
        )(value)
    assert hidden is not None and cell is not None
    return value, hidden, cell


def _remaining_lstm_decoder(
    latent_width: int,
    state_width: int,
    widths: tuple[int, ...],
    *,
    activation: str,
    dropout: float,
    output_activation: str,
    name: str,
) -> keras.Model:
    latent = keras.Input((latent_width,), name="decoder_latent")
    initial_hidden = keras.Input((state_width,), name="decoder_initial_hidden")
    initial_cell = keras.Input((state_width,), name="decoder_initial_cell")
    value = FirstStepLatentSequence(name="decoder_first_latent_then_zero")(
        latent
    )
    for index, width in enumerate(widths, start=1):
        layer = layers.LSTM(
            width,
            activation=activation,
            dropout=dropout,
            return_sequences=True,
            name=f"decoder_lstm_{index}_{width}",
        )
        value = (
            layer(value, initial_state=[initial_hidden, initial_cell])
            if index == 1
            else layer(value)
        )
    value = layers.TimeDistributed(
        layers.Dense(1, activation=output_activation),
        name="decoder_time_distributed_output",
    )(value)
    reconstruction = layers.Reshape((48,), name="reconstruction")(value)
    return keras.Model(
        (latent, initial_hidden, initial_cell), reconstruction, name=name
    )


def _build_remaining_lstm_sae(
    *, seed: int, latent_seed: int, learning_rate: float
) -> RemainingPaperBundle:
    del latent_seed
    set_seed(seed)
    spec = SPECS["lstm_sae"]
    inputs = keras.Input((48,), name="daily_profile")
    _, hidden, cell = _remaining_lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    decoder = _remaining_lstm_decoder(
        spec.encoder[-1],
        spec.encoder[-1],
        spec.decoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
        output_activation=spec.output_activation,
        name="remaining_lstm_sae_decoder",
    )
    reconstruction = decoder((hidden, hidden, cell))
    model = keras.Model(inputs, reconstruction, name="remaining_lstm_sae")
    model.compile(optimizer=optimizer(spec.optimizer, learning_rate), loss="mse")
    return RemainingPaperBundle("lstm_sae", model, decoder=decoder)


def _build_remaining_fc_vae(
    *, seed: int, latent_seed: int, learning_rate: float
) -> RemainingPaperBundle:
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
    mean = layers.Dense(spec.encoder[-1], name="latent_mean")(value)
    log_variance = layers.Dense(
        spec.encoder[-1], name="latent_log_variance"
    )(value)
    encoder = keras.Model(inputs, (mean, log_variance), name="remaining_fc_vae_encoder")
    latent = keras.Input((spec.encoder[-1],), name="decoder_latent")
    decoded = latent
    for index, width in enumerate(spec.decoder, start=1):
        decoded = layers.Dense(
            width,
            activation=spec.hidden_activation,
            name=f"decoder_dense_{index}_{width}",
        )(decoded)
        decoded = layers.Dropout(
            spec.dropout, name=f"decoder_dropout_{index}"
        )(decoded)
    decoded = layers.Dense(
        48, activation=spec.output_activation, name="reconstruction"
    )(decoded)
    decoder = keras.Model(latent, decoded, name="remaining_fc_vae_decoder")
    sampled = RemainingSampling(latent_seed, name="latent_sample")(
        (mean, log_variance)
    )
    reconstruction = decoder(sampled)
    output = EquationTenLoss(name="equation_10_loss")(
        (inputs, reconstruction, mean, log_variance)
    )
    model = keras.Model(inputs, output, name="remaining_fc_vae")
    model.compile(optimizer=optimizer(spec.optimizer, learning_rate))
    return RemainingPaperBundle("fc_vae", model, encoder=encoder, decoder=decoder)


def _build_remaining_lstm_vae(
    *, seed: int, latent_seed: int, learning_rate: float
) -> RemainingPaperBundle:
    set_seed(seed)
    spec = SPECS["lstm_vae"]
    inputs = keras.Input((48,), name="daily_profile")
    _, hidden, cell = _remaining_lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    mean = layers.Dense(spec.encoder[-1], name="latent_mean")(hidden)
    log_variance = layers.Dense(
        spec.encoder[-1], name="latent_log_variance"
    )(hidden)
    encoder = keras.Model(
        inputs,
        (mean, log_variance, hidden, cell),
        name="remaining_lstm_vae_encoder",
    )
    decoder = _remaining_lstm_decoder(
        spec.encoder[-1],
        spec.encoder[-1],
        spec.decoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
        output_activation=spec.output_activation,
        name="remaining_lstm_vae_decoder",
    )
    sampled = RemainingSampling(latent_seed, name="latent_sample")(
        (mean, log_variance)
    )
    reconstruction = decoder((sampled, hidden, cell))
    output = EquationTenLoss(name="equation_10_loss")(
        (inputs, reconstruction, mean, log_variance)
    )
    model = keras.Model(inputs, output, name="remaining_lstm_vae")
    model.compile(optimizer=optimizer(spec.optimizer, learning_rate))
    return RemainingPaperBundle(
        "lstm_vae", model, encoder=encoder, decoder=decoder
    )


def _build_remaining_lstm_aea(
    *, seed: int, latent_seed: int, learning_rate: float
) -> RemainingPaperBundle:
    del latent_seed
    set_seed(seed)
    spec = SPECS["lstm_aea"]
    inputs = keras.Input((48,), name="daily_profile")
    sequence, hidden, cell = _remaining_lstm_encoder(
        inputs,
        spec.encoder,
        activation=spec.hidden_activation,
        dropout=spec.dropout,
    )
    decoder = AutoregressiveAttentionDecoder(
        spec.decoder,
        activation=spec.hidden_activation,
        output_activation=spec.output_activation,
        name="autoregressive_attention_decoder",
    )
    reconstruction, attention = decoder((sequence, hidden, cell))
    model = keras.Model(inputs, reconstruction, name="remaining_lstm_aea")
    model.compile(optimizer=optimizer(spec.optimizer, learning_rate), loss="mse")
    attention_model = keras.Model(
        inputs, attention, name="remaining_lstm_aea_attention"
    )
    return RemainingPaperBundle(
        "lstm_aea", model, attention_model=attention_model
    )


def build_remaining_paper_model(
    name: str,
    *,
    seed: int,
    latent_seed: int | None = None,
    learning_rate: float | None = None,
) -> RemainingPaperBundle:
    """Build the approved completion without changing historical builders."""

    if name not in REMAINING_MODELS:
        raise ValueError(
            f"remaining-paper-v1 supports only {', '.join(REMAINING_MODELS)}"
        )
    resolved_rate = (
        0.01
        if learning_rate is None and SPECS[name].optimizer == "SGD"
        else (0.001 if learning_rate is None else learning_rate)
    )
    builders = {
        "lstm_sae": _build_remaining_lstm_sae,
        "fc_vae": _build_remaining_fc_vae,
        "lstm_vae": _build_remaining_lstm_vae,
        "lstm_aea": _build_remaining_lstm_aea,
    }
    bundle = builders[name](
        seed=seed,
        latent_seed=seed + 1 if latent_seed is None else latent_seed,
        learning_rate=resolved_rate,
    )
    validate_remaining_paper_bundle(bundle, learning_rate=resolved_rate)
    return bundle


def validate_remaining_paper_bundle(
    bundle: RemainingPaperBundle, *, learning_rate: float
) -> None:
    """Fail closed on the approved contract's consequential topology fields."""

    if bundle.name not in REMAINING_MODELS:
        raise AssertionError(f"unexpected remaining model {bundle.name}")
    if tuple(bundle.model.output_shape) != (None, 48):
        raise AssertionError(f"unexpected output shape {bundle.model.output_shape}")
    if int(bundle.model.count_params()) != REMAINING_PARAMETER_COUNTS[bundle.name]:
        raise AssertionError(
            f"parameter count drifted: {bundle.model.count_params()}"
        )
    observed_optimizer = bundle.model.optimizer.__class__.__name__
    if observed_optimizer != SPECS[bundle.name].optimizer:
        raise AssertionError(
            f"optimizer drifted: {observed_optimizer}"
        )
    observed_rate = float(keras.ops.convert_to_numpy(bundle.model.optimizer.learning_rate))
    if abs(observed_rate - learning_rate) > 1e-8:
        raise AssertionError(
            f"learning rate drifted: {observed_rate} versus {learning_rate}"
        )
    names = {layer.name for layer in bundle.model.layers}
    spec = SPECS[bundle.name]
    encoder_lstm = [
        layer
        for layer in bundle.model.layers
        if isinstance(layer, layers.LSTM) and layer.name.startswith("encoder_")
    ]
    if bundle.name.startswith("lstm_"):
        if [layer.units for layer in encoder_lstm] != list(spec.encoder):
            raise AssertionError("remaining encoder widths drifted")
        if any(
            layer.activation.__name__ != spec.hidden_activation
            or float(layer.dropout) != spec.dropout
            for layer in encoder_lstm
        ):
            raise AssertionError("remaining encoder activation/dropout drifted")
    if bundle.name in {"lstm_sae", "lstm_vae"}:
        if bundle.decoder is None or "decoder_first_latent_then_zero" not in {
            layer.name for layer in bundle.decoder.layers
        }:
            raise AssertionError("first-latent-then-zero decoder is missing")
        decoder_lstm = [
            layer for layer in bundle.decoder.layers if isinstance(layer, layers.LSTM)
        ]
        if [layer.units for layer in decoder_lstm] != list(spec.decoder):
            raise AssertionError("remaining decoder widths drifted")
        if any(
            layer.activation.__name__ != spec.hidden_activation
            or float(layer.dropout) != spec.dropout
            for layer in decoder_lstm
        ):
            raise AssertionError("remaining decoder activation/dropout drifted")
    if bundle.name.endswith("_vae"):
        if bundle.encoder is None or bundle.decoder is None:
            raise AssertionError("VAE encoder/decoder scoring interfaces are missing")
        if "equation_10_loss" not in names:
            raise AssertionError("Equation-(10) summed objective is missing")
    if bundle.name == "fc_vae":
        encoder_dense = [
            layer
            for layer in bundle.model.layers
            if isinstance(layer, layers.Dense)
            and layer.name.startswith("encoder_dense_")
        ]
        encoder_dropout = [
            layer
            for layer in bundle.model.layers
            if isinstance(layer, layers.Dropout)
            and layer.name.startswith("encoder_dropout_")
        ]
        decoder_dense = [
            layer for layer in bundle.decoder.layers if isinstance(layer, layers.Dense)
        ]
        decoder_dropout = [
            layer for layer in bundle.decoder.layers if isinstance(layer, layers.Dropout)
        ]
        if [layer.units for layer in encoder_dense] != list(spec.encoder):
            raise AssertionError("FC-VAE encoder widths drifted")
        if [layer.units for layer in decoder_dense] != [*spec.decoder, 48]:
            raise AssertionError("FC-VAE decoder widths drifted")
        if len(encoder_dropout) != 4 or len(decoder_dropout) != 4 or any(
            float(layer.rate) != spec.dropout
            for layer in (*encoder_dropout, *decoder_dropout)
        ):
            raise AssertionError("FC-VAE dropout placement/rate drifted")
        if any(
            layer.activation.__name__ != spec.hidden_activation
            for layer in (*encoder_dense, *decoder_dense[:-1])
        ) or decoder_dense[-1].activation.__name__ != spec.output_activation:
            raise AssertionError("FC-VAE activations drifted")
    if bundle.name == "lstm_aea":
        if bundle.attention_model is None or "autoregressive_attention_decoder" not in names:
            raise AssertionError("autoregressive attention interface is missing")
        decoder = bundle.model.get_layer("autoregressive_attention_decoder")
        if (
            tuple(decoder.widths) != spec.decoder
            or decoder.activation_name != spec.hidden_activation
            or decoder.output_activation_name != spec.output_activation
        ):
            raise AssertionError("autoregressive attention decoder drifted")
