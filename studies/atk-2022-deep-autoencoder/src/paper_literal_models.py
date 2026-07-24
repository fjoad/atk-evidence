"""Keras models for the Takiddin et al. exploratory audit.

The default builders preserve immutable implementation-v1 behavior. Passing
``architecture_contract="paper_source_v2"`` selects the source-derived
Table-I/Fig.-3--5 architecture branches without rewriting historical results.
``corrected_control_v1`` deliberately reuses those source-derived layer
topologies while the corrected track changes data isolation, selection, score,
and calibration semantics under a separate fingerprint.

All public builders accept a flat ``(batch, input_length)`` array.  Recurrent
builders reshape that array to a univariate sequence internally and return a
flat reconstruction of the same length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import layers, ops
import numpy as np


_TABLE_I_DEFAULTS: dict[str, dict[str, Any]] = {
    "fc_sae": {
        "encoder_widths": (400, 300, 200, 100),
        "optimizer": "adam",
        "dropout": 0.4,
        "hidden_activation": "sigmoid",
        "output_activation": "softmax",
    },
    "lstm_sae": {
        "encoder_widths": (500, 300),
        "optimizer": "adam",
        "dropout": 0.2,
        "hidden_activation": "sigmoid",
        "output_activation": "sigmoid",
    },
    "fc_vae": {
        "encoder_widths": (500, 400, 300, 100),
        "optimizer": "adam",
        "dropout": 0.4,
        "hidden_activation": "relu",
        "output_activation": "softmax",
    },
    "lstm_vae": {
        "encoder_widths": (400, 300),
        "optimizer": "sgd",
        "dropout": 0.0,
        "hidden_activation": "tanh",
        "output_activation": "sigmoid",
    },
    "lstm_aea": {
        "encoder_widths": (500, 300, 200),
        "optimizer": "sgd",
        "dropout": 0.0,
        "hidden_activation": "sigmoid",
        "output_activation": "sigmoid",
    },
    # Section IV-C reports one width for every hidden layer of each supervised
    # deep benchmark.  Output widths follow their printed activations.
    "supervised_feed_forward": {
        "encoder_widths": (500, 500, 500, 500, 500),
        "optimizer": "adamax",
        "dropout": 0.0,
        "hidden_activation": "relu",
        "output_activation": "softmax",
    },
    "supervised_lstm": {
        "encoder_widths": (300, 300, 300, 300),
        "optimizer": "adam",
        "dropout": 0.2,
        "hidden_activation": "relu",
        "output_activation": "sigmoid",
    },
}

_ALIASES = {
    "fc-sae": "fc_sae",
    "lstm-sae": "lstm_sae",
    "fc-vae": "fc_vae",
    "lstm-vae": "lstm_vae",
    "lstm-aea": "lstm_aea",
    "feed-forward": "supervised_feed_forward",
    "feed_forward": "supervised_feed_forward",
    "supervised-feed-forward": "supervised_feed_forward",
    "supervised-lstm": "supervised_lstm",
}
VAE_SCORE_BRANCHES = {
    "prob_learned_var_mc1",
    "prob_learned_var_mc10",
    "prob_learned_var_mc100",
    "prob_fixed_var_mc1",
    "prob_fixed_var_mc10",
    "prob_fixed_var_mc100",
    "reconstruction_mse_high",
    "mse_plus_kl_high",
}
VAE_LOSS_REDUCTIONS = {
    "sum_squared_plus_kl",
    "mean_mse_plus_kl",
}


def _canonical_name(name: str) -> str:
    candidate = name.strip().lower().replace(" ", "_")
    candidate = _ALIASES.get(candidate, candidate.replace("-", "_"))
    if candidate not in _TABLE_I_DEFAULTS:
        supported = ", ".join(sorted(_TABLE_I_DEFAULTS))
        raise ValueError(f"unknown model {name!r}; supported models: {supported}")
    return candidate


def _model_config(name: str, override: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(_TABLE_I_DEFAULTS[name])
    if override:
        config.update(override)
    widths = tuple(int(width) for width in config["encoder_widths"])
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("encoder_widths must contain positive integers")
    dropout = float(config["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    config["encoder_widths"] = widths
    config["dropout"] = dropout
    config.setdefault("architecture_contract", "implementation_v1")
    config.setdefault("latent_width", widths[-1])
    config.setdefault(
        "latent_placement", "existing_bottleneck_representation"
    )
    config.setdefault("dense_dropout_scope", "all_hidden_layers")
    config.setdefault("lstm_dropout_placement", "input_only")
    config.setdefault("vae_score", "reconstruction_mse_high")
    config.setdefault("vae_loss_reduction", "mean_mse_plus_kl")
    config.setdefault(
        "supervised_head",
        (
            "sigmoid1_binary"
            if name == "supervised_lstm"
            else "softmax2_categorical"
        ),
    )
    if config["architecture_contract"] in {
        "paper_source_v2",
        "corrected_control_v1",
    }:
        config.setdefault("lstm_input", "48_steps_1_feature")
        config.setdefault("decoder_schedule", "first_latent_then_zero")
        config.setdefault("decoder_state", "top_state_only")
        config.setdefault("attention_merge", "additive_concat")
    else:
        config.setdefault("lstm_input", "48_steps_1_feature")
        config.setdefault("decoder_schedule", "repeat_latent")
        config.setdefault("decoder_state", "mirrored_layer_states")
        config.setdefault("attention_merge", "additive_concat")
    if config["architecture_contract"] not in {
        "implementation_v1",
        "paper_source_v2",
        "corrected_control_v1",
    }:
        raise ValueError("unsupported architecture_contract")
    config["latent_width"] = int(config["latent_width"])
    if config["latent_width"] <= 0:
        raise ValueError("latent_width must be positive")
    if config["latent_placement"] not in {
        "distinct_projection",
        "existing_bottleneck_representation",
    }:
        raise ValueError("unsupported latent_placement")
    if config["dense_dropout_scope"] not in {
        "all_hidden_layers",
        "encoder_hidden_layers",
        "bottleneck_only",
    }:
        raise ValueError("unsupported dense_dropout_scope")
    if config["lstm_dropout_placement"] not in {
        "input_only",
        "recurrent_only",
        "split_input_recurrent",
    }:
        raise ValueError("unsupported lstm_dropout_placement")
    if config["lstm_input"] not in {
        "48_steps_1_feature",
        "1_step_48_features",
    }:
        raise ValueError("unsupported lstm_input")
    if config["decoder_schedule"] not in {
        "repeat_latent",
        "first_latent_then_zero",
        "autoregressive_reconstruction",
    }:
        raise ValueError("unsupported decoder_schedule")
    if config["decoder_state"] not in {
        "mirrored_layer_states",
        "top_state_only",
    }:
        raise ValueError("unsupported decoder_state")
    if config["attention_merge"] not in {
        "additive_concat",
        "literal_sum",
    }:
        raise ValueError("unsupported attention_merge")
    if config["supervised_head"] not in {
        "softmax2_categorical",
        "sigmoid1_binary",
    }:
        raise ValueError("unsupported supervised_head")
    if name.endswith("_vae"):
        parse_vae_score_branch(str(config["vae_score"]))
        if config["vae_loss_reduction"] not in VAE_LOSS_REDUCTIONS:
            raise ValueError(
                "unsupported vae_loss_reduction; expected one of "
                f"{sorted(VAE_LOSS_REDUCTIONS)}"
            )
    return config


def _optimizer(name: str) -> keras.optimizers.Optimizer:
    normalized = name.strip().lower()
    choices: dict[str, type[keras.optimizers.Optimizer]] = {
        "adam": keras.optimizers.Adam,
        "adamax": keras.optimizers.Adamax,
        "rmsprop": keras.optimizers.RMSprop,
        "sgd": keras.optimizers.SGD,
    }
    try:
        return choices[normalized]()
    except KeyError as exc:
        raise ValueError(f"unsupported optimizer: {name!r}") from exc


def _non_batch_axes(tensor: Any) -> tuple[int, ...]:
    return tuple(range(1, len(tensor.shape)))


def _per_sample_mse(target: Any, reconstruction: Any) -> Any:
    return ops.mean(
        ops.square(target - reconstruction),
        axis=_non_batch_axes(target),
    )


def _per_sample_reconstruction_loss(
    target: Any,
    reconstruction: Any,
    reduction: str,
) -> Any:
    squared_error = ops.square(target - reconstruction)
    if reduction == "sum_squared_plus_kl":
        return ops.sum(squared_error, axis=_non_batch_axes(target))
    if reduction == "mean_mse_plus_kl":
        return ops.mean(squared_error, axis=_non_batch_axes(target))
    raise ValueError(f"unsupported VAE loss reduction: {reduction!r}")


def _per_sample_kl(z_mean: Any, z_log_var: Any) -> Any:
    return -0.5 * ops.sum(
        1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1
    )


@dataclass(frozen=True)
class VaeScoreSpec:
    """Executable meaning of one frozen VAE score branch."""

    name: str
    kind: str
    positive_if: str
    variance: str | None
    monte_carlo_samples: int | None


def parse_vae_score_branch(name: str) -> VaeScoreSpec:
    """Parse one machine-lattice VAE score ID without inferring semantics."""

    if name not in VAE_SCORE_BRANCHES:
        raise ValueError(
            f"unknown VAE score branch {name!r}; "
            f"expected one of {sorted(VAE_SCORE_BRANCHES)}"
        )
    if name == "reconstruction_mse_high":
        return VaeScoreSpec(name, "reconstruction_mse", "higher", None, None)
    if name == "mse_plus_kl_high":
        return VaeScoreSpec(name, "mse_plus_kl", "higher", None, None)
    variance = "learned" if "_learned_var_" in name else "fixed"
    samples = int(name.rsplit("mc", 1)[1])
    return VaeScoreSpec(
        name,
        "reconstruction_probability",
        "lower",
        variance,
        samples,
    )


def gaussian_reconstruction_probability(
    target: Any,
    decoded_means: Any,
    *,
    decoded_log_variances: Any | None = None,
    fixed_variance: float | None = None,
) -> dict[str, np.ndarray]:
    """Average multivariate Gaussian reconstruction density over latent draws.

    ``decoded_means`` has shape ``(draws, rows, features)``. Learned-variance
    branches must supply a matching decoder log-variance tensor; fixed-variance
    branches must instead supply one positive scalar. The returned log-density
    and negative log-density remain numerically useful when the raw joint
    density underflows.

    This function implements the probability operation stated in Section
    III-B. It does not resolve how the paper's decoder obtains a variance;
    learned-head and fixed-variance model branches remain separate callers.
    """

    observed = np.asarray(target, dtype=np.float64)
    means = np.asarray(decoded_means, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("target must have shape (rows, features)")
    if means.ndim != 3 or means.shape[1:] != observed.shape:
        raise ValueError(
            "decoded_means must have shape (draws, rows, features) matching target"
        )
    if means.shape[0] < 1:
        raise ValueError("at least one Monte Carlo draw is required")
    if not np.isfinite(observed).all() or not np.isfinite(means).all():
        raise ValueError("target and decoded means must be finite")
    if (decoded_log_variances is None) == (fixed_variance is None):
        raise ValueError(
            "provide exactly one of decoded_log_variances or fixed_variance"
        )

    if decoded_log_variances is not None:
        log_variance = np.asarray(decoded_log_variances, dtype=np.float64)
        if log_variance.shape != means.shape:
            raise ValueError("decoded_log_variances must match decoded_means")
        if not np.isfinite(log_variance).all():
            raise ValueError("decoded log variances must be finite")
        variance = np.exp(log_variance)
    else:
        assert fixed_variance is not None
        if not np.isfinite(fixed_variance) or fixed_variance <= 0:
            raise ValueError("fixed_variance must be finite and positive")
        log_variance = np.full_like(means, np.log(float(fixed_variance)))
        variance = np.full_like(means, float(fixed_variance))

    residual = observed[np.newaxis, :, :] - means
    feature_log_density = -0.5 * (
        np.log(2.0 * np.pi)
        + log_variance
        + np.square(residual) / variance
    )
    draw_log_density = feature_log_density.sum(axis=-1)
    maximum = np.max(draw_log_density, axis=0)
    log_probability = maximum + np.log(
        np.mean(np.exp(draw_log_density - maximum), axis=0)
    )
    probability = np.exp(
        np.clip(
            log_probability,
            np.log(np.nextafter(0.0, 1.0)),
            np.log(np.finfo(np.float64).max),
        )
    )
    return {
        "reconstruction_probability": probability,
        "log_reconstruction_probability": log_probability,
        "negative_log_reconstruction_probability": -log_probability,
    }


@keras.saving.register_keras_serializable(package="atk_evidence")
class Sampling(layers.Layer):
    """Gaussian reparameterization with a reproducible Keras seed stream."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed or 0)

    def call(self, inputs: tuple[Any, Any]) -> Any:
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(
            shape=ops.shape(z_mean), seed=self.seed_generator
        )
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "seed": self.seed}


@keras.saving.register_keras_serializable(package="atk_evidence")
class VAELoss(layers.Layer):
    """Attach one frozen Eq. (10) reconstruction reduction plus analytic KL."""

    def __init__(self, reduction: str = "mean_mse_plus_kl", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction not in VAE_LOSS_REDUCTIONS:
            raise ValueError(
                f"unsupported VAE loss reduction {reduction!r}; "
                f"expected one of {sorted(VAE_LOSS_REDUCTIONS)}"
            )
        self.reduction = reduction

    def call(self, inputs: tuple[Any, Any, Any, Any]) -> Any:
        target, reconstruction, z_mean, z_log_var = inputs
        reconstruction_loss = ops.mean(
            _per_sample_reconstruction_loss(
                target,
                reconstruction,
                self.reduction,
            )
        )
        kl_loss = ops.mean(_per_sample_kl(z_mean, z_log_var))
        self.add_loss(reconstruction_loss + kl_loss)
        return reconstruction

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "reduction": self.reduction}


@keras.saving.register_keras_serializable(package="atk_evidence")
class GaussianVAELoss(layers.Layer):
    """Likelihood-consistent learned-variance analogue of the two reductions."""

    def __init__(self, reduction: str = "mean_mse_plus_kl", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction not in VAE_LOSS_REDUCTIONS:
            raise ValueError(
                f"unsupported VAE loss reduction {reduction!r}; "
                f"expected one of {sorted(VAE_LOSS_REDUCTIONS)}"
            )
        self.reduction = reduction

    def call(self, inputs: tuple[Any, Any, Any, Any, Any]) -> Any:
        target, mean, log_variance, z_mean, z_log_var = inputs
        feature_nll = 0.5 * (
            np.log(2.0 * np.pi)
            + log_variance
            + ops.square(target - mean) / ops.exp(log_variance)
        )
        if self.reduction == "sum_squared_plus_kl":
            gaussian_nll = ops.sum(
                feature_nll,
                axis=_non_batch_axes(target),
            )
        else:
            gaussian_nll = ops.mean(
                feature_nll,
                axis=_non_batch_axes(target),
            )
        self.add_loss(
            ops.mean(gaussian_nll) + ops.mean(_per_sample_kl(z_mean, z_log_var))
        )
        return mean

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "reduction": self.reduction}


@keras.saving.register_keras_serializable(package="atk_evidence")
class ReconstructionScore(layers.Layer):
    """Return one reconstruction-MSE anomaly score per sample."""

    def call(self, inputs: tuple[Any, Any]) -> Any:
        target, reconstruction = inputs
        return _per_sample_mse(target, reconstruction)


@keras.saving.register_keras_serializable(package="atk_evidence")
class VAEScores(layers.Layer):
    """Return the two registered high-is-anomalous VAE surrogate scores."""

    def call(self, inputs: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
        target, reconstruction, z_mean, z_log_var = inputs
        reconstruction_mse = _per_sample_mse(target, reconstruction)
        mse_plus_kl_surrogate = reconstruction_mse + _per_sample_kl(
            z_mean, z_log_var
        )
        return reconstruction_mse, mse_plus_kl_surrogate


@keras.saving.register_keras_serializable(package="atk_evidence")
class FirstStepLatentSequence(layers.Layer):
    """Place the latent vector at decoder step one and zeros thereafter."""

    def __init__(self, steps: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        self.steps = int(steps)

    def call(self, latent: Any) -> Any:
        first = ops.expand_dims(latent, axis=1)
        if self.steps == 1:
            return first
        shape = ops.shape(latent)
        zeros = ops.zeros((shape[0], self.steps - 1, shape[1]), dtype=latent.dtype)
        return ops.concatenate((first, zeros), axis=1)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "steps": self.steps}


@keras.saving.register_keras_serializable(package="atk_evidence")
class AutoregressiveLSTMDecoder(layers.Layer):
    """Unroll a stacked decoder using its previous scalar reconstruction."""

    def __init__(
        self,
        widths: tuple[int, ...],
        steps: int,
        activation: str,
        output_activation: str,
        dropout: float,
        recurrent_dropout: float,
        state_policy: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.widths = tuple(int(width) for width in widths)
        self.steps = int(steps)
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout_rate = float(dropout)
        self.recurrent_dropout_rate = float(recurrent_dropout)
        self.state_policy = state_policy
        self.start_projection = layers.Dense(1, name="start_projection")
        self.cells = [
            layers.LSTMCell(
                width,
                activation=activation,
                recurrent_activation="sigmoid",
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"cell_{index}",
            )
            for index, width in enumerate(self.widths, start=1)
        ]
        self.output_projection = layers.Dense(
            1, activation=output_activation, name="output_projection"
        )

    def call(self, inputs: tuple[Any, ...], training: bool | None = None) -> Any:
        latent, *provided_states = inputs
        batch = ops.shape(latent)[0]
        states: list[list[Any]] = []
        cursor = 0
        for index, width in enumerate(self.widths):
            receives_encoder_state = (
                self.state_policy == "mirrored_layer_states" or index == 0
            )
            if receives_encoder_state:
                states.append(
                    [provided_states[cursor], provided_states[cursor + 1]]
                )
                cursor += 2
            else:
                zero = ops.zeros((batch, width), dtype=latent.dtype)
                states.append([zero, zero])

        value = self.start_projection(latent)
        outputs: list[Any] = []
        for _ in range(self.steps):
            for index, cell in enumerate(self.cells):
                value, states[index] = cell(
                    value,
                    states=states[index],
                    training=training,
                )
            value = self.output_projection(value)
            outputs.append(value)
        return ops.concatenate(outputs, axis=1)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "widths": self.widths,
            "steps": self.steps,
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
            "dropout": self.dropout_rate,
            "recurrent_dropout": self.recurrent_dropout_rate,
            "state_policy": self.state_policy,
        }


@keras.saving.register_keras_serializable(package="atk_evidence")
class PreviousDecoderQueries(layers.Layer):
    """Shift decoder outputs so attention at ``t`` sees ``h_D,t-1``."""

    def call(self, inputs: tuple[Any, Any]) -> Any:
        decoder_sequence, initial_hidden = inputs
        initial_query = ops.expand_dims(initial_hidden, axis=1)
        return ops.concatenate(
            (initial_query, decoder_sequence[:, :-1, :]), axis=1
        )


@keras.saving.register_keras_serializable(package="atk_evidence")
class TemporalAdditiveAttention(layers.Layer):
    """Decoder-conditioned Bahdanau attention over the encoder sequence."""

    def __init__(self, attention_units: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attention_units = attention_units
        self.sequence_projection: layers.Dense | None = None
        self.query_projection: layers.Dense | None = None
        self.alignment_projection: layers.Dense | None = None

    def build(self, input_shape: tuple[Any, Any]) -> None:
        sequence_shape, _ = input_shape
        units = self.attention_units or int(sequence_shape[-1])
        self.sequence_projection = layers.Dense(units, use_bias=True)
        self.query_projection = layers.Dense(units, use_bias=False)
        self.alignment_projection = layers.Dense(1, use_bias=False)
        super().build(input_shape)

    def call(self, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
        sequence, query_sequence = inputs
        if (
            self.sequence_projection is None
            or self.query_projection is None
            or self.alignment_projection is None
        ):
            raise RuntimeError("attention layer has not been built")
        projected_sequence = self.sequence_projection(sequence)
        projected_query = self.query_projection(query_sequence)
        projected_sequence = ops.expand_dims(projected_sequence, axis=1)
        projected_query = ops.expand_dims(projected_query, axis=2)
        alignment_logits = self.alignment_projection(
            ops.tanh(projected_sequence + projected_query)
        )
        weights = ops.softmax(ops.squeeze(alignment_logits, axis=-1), axis=-1)
        context = ops.matmul(weights, sequence)
        return context, weights

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "attention_units": self.attention_units}


@keras.saving.register_keras_serializable(package="atk_evidence")
class AutoregressiveAttentionLSTMDecoder(layers.Layer):
    """Algorithm-5 decoder with attention and reconstructed-value feedback."""

    def __init__(
        self,
        widths: tuple[int, ...],
        steps: int,
        activation: str,
        output_activation: str,
        dropout: float,
        recurrent_dropout: float,
        state_policy: str,
        attention_merge: str,
        latent_width: int | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.widths = tuple(int(width) for width in widths)
        self.steps = int(steps)
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout_rate = float(dropout)
        self.recurrent_dropout_rate = float(recurrent_dropout)
        self.state_policy = state_policy
        self.attention_merge = attention_merge
        self.latent_width = (
            None if latent_width is None else int(latent_width)
        )
        if not self.widths or self.steps < 1:
            raise ValueError("autoregressive attention requires widths and steps")
        if state_policy not in {"mirrored_layer_states", "top_state_only"}:
            raise ValueError("unsupported autoregressive attention state policy")
        if attention_merge not in {"additive_concat", "literal_sum"}:
            raise ValueError("unsupported autoregressive attention merge")

        self.initial_query_projection = layers.Dense(
            self.widths[-1],
            activation=activation,
            name="initial_query_projection",
        )
        self.start_projection = layers.Dense(1, name="start_projection")
        self.attention = TemporalAdditiveAttention(
            attention_units=self.widths[0],
            name="temporal_attention",
        )
        self.latent_projection = (
            layers.Dense(
                self.latent_width,
                activation=activation,
                name="latent_projection",
            )
            if self.latent_width is not None
            else None
        )
        self.cells = [
            layers.LSTMCell(
                width,
                activation=activation,
                recurrent_activation="sigmoid",
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"cell_{index}",
            )
            for index, width in enumerate(self.widths, start=1)
        ]
        self.output_projection = layers.Dense(
            1,
            activation=output_activation,
            name="output_projection",
        )

    def call(
        self,
        inputs: tuple[Any, ...],
        training: bool | None = None,
    ) -> tuple[Any, Any]:
        encoder_sequence, initial_query, *provided_states = inputs
        batch = ops.shape(encoder_sequence)[0]
        states: list[list[Any]] = []
        cursor = 0
        for index, width in enumerate(self.widths):
            receives_encoder_state = (
                self.state_policy == "mirrored_layer_states" or index == 0
            )
            if receives_encoder_state:
                states.append(
                    [provided_states[cursor], provided_states[cursor + 1]]
                )
                cursor += 2
            else:
                zero = ops.zeros((batch, width), dtype=encoder_sequence.dtype)
                states.append([zero, zero])

        query = self.initial_query_projection(initial_query)
        previous_reconstruction = self.start_projection(query)
        outputs: list[Any] = []
        attention_weights: list[Any] = []
        for _ in range(self.steps):
            context, weights = self.attention(
                (
                    encoder_sequence,
                    ops.expand_dims(query, axis=1),
                )
            )
            context = ops.squeeze(context, axis=1)
            if self.attention_merge == "literal_sum":
                value = context + previous_reconstruction
            else:
                value = ops.concatenate(
                    (context, previous_reconstruction),
                    axis=-1,
                )
            if self.latent_projection is not None:
                value = self.latent_projection(value)
            for index, cell in enumerate(self.cells):
                value, states[index] = cell(
                    value,
                    states=states[index],
                    training=training,
                )
            query = value
            previous_reconstruction = self.output_projection(value)
            outputs.append(previous_reconstruction)
            attention_weights.append(weights)
        return (
            ops.concatenate(outputs, axis=1),
            ops.concatenate(attention_weights, axis=1),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "widths": self.widths,
            "steps": self.steps,
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
            "dropout": self.dropout_rate,
            "recurrent_dropout": self.recurrent_dropout_rate,
            "state_policy": self.state_policy,
            "attention_merge": self.attention_merge,
            "latent_width": self.latent_width,
        }


@dataclass(frozen=True)
class ModelBundle:
    """Models and stable scoring interface required by the experiment runner."""

    name: str
    model: keras.Model
    encoder: keras.Model | None = None
    decoder: keras.Model | None = None
    score_model: keras.Model | None = None
    attention_model: keras.Model | None = None
    stochastic_reconstruction_model: keras.Model | None = None
    stochastic_distribution_model: keras.Model | None = None

    def anomaly_scores(
        self, x: Any, batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        """Calculate registered anomaly scores without backend-specific code."""

        if self.score_model is None:
            raise ValueError(f"{self.name} is a supervised model without anomaly scores")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        raw = self.score_model.predict(array, batch_size=batch_size, verbose=0)
        if not isinstance(raw, dict):
            raise RuntimeError("score_model must return a dictionary")
        return {key: np.asarray(value).reshape(-1) for key, value in raw.items()}

    def vae_reconstruction_probability(
        self,
        x: Any,
        *,
        monte_carlo_samples: int,
        variance: str,
        fixed_variance: float = 1.0,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Execute the Section III-B probability operation for one VAE branch."""

        if int(monte_carlo_samples) < 1:
            raise ValueError("monte_carlo_samples must be positive")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if variance == "fixed":
            if self.stochastic_reconstruction_model is None:
                raise ValueError(
                    f"{self.name} has no stochastic VAE reconstruction"
                )
            draws = np.stack(
                [
                    np.asarray(
                        self.stochastic_reconstruction_model.predict(
                            array,
                            batch_size=batch_size,
                            verbose=0,
                        ),
                        dtype=np.float64,
                    )
                    for _ in range(int(monte_carlo_samples))
                ],
                axis=0,
            )
            return gaussian_reconstruction_probability(
                array,
                draws,
                fixed_variance=fixed_variance,
            )
        if variance != "learned":
            raise ValueError("variance must be 'fixed' or 'learned'")
        if self.stochastic_distribution_model is None:
            raise ValueError(
                "learned decoder variance requires the prose-consistent "
                "decoder-variance-head branch"
            )
        distribution_draws = [
            self.stochastic_distribution_model.predict(
                array,
                batch_size=batch_size,
                verbose=0,
            )
            for _ in range(int(monte_carlo_samples))
        ]
        means = np.stack(
            [np.asarray(draw[0], dtype=np.float64) for draw in distribution_draws],
            axis=0,
        )
        log_variances = np.stack(
            [np.asarray(draw[1], dtype=np.float64) for draw in distribution_draws],
            axis=0,
        )
        return gaussian_reconstruction_probability(
            array,
            means,
            decoded_log_variances=log_variances,
        )


def _dense_hidden(
    value: Any,
    width: int,
    activation: str,
    dropout: float,
    name: str,
) -> Any:
    value = layers.Dense(width, activation=activation, name=name)(value)
    if dropout:
        value = layers.Dropout(dropout, name=f"{name}_dropout")(value)
    return value


def _uses_source_architecture(config: Mapping[str, Any]) -> bool:
    return config["architecture_contract"] in {
        "paper_source_v2",
        "corrected_control_v1",
    }


def _dense_dropout(
    config: Mapping[str, Any],
    *,
    stage: str,
    index: int,
    total: int,
) -> float:
    rate = float(config["dropout"])
    if not _uses_source_architecture(config):
        return rate
    scope = config["dense_dropout_scope"]
    if scope == "all_hidden_layers":
        return rate
    if scope == "encoder_hidden_layers":
        return rate if stage == "encoder" else 0.0
    if config["latent_placement"] == "distinct_projection":
        return rate if stage == "latent" else 0.0
    return rate if stage == "encoder" and index == total else 0.0


def _lstm_dropouts(config: Mapping[str, Any]) -> tuple[float, float]:
    rate = float(config["dropout"])
    if not _uses_source_architecture(config):
        return rate, 0.0
    placement = config["lstm_dropout_placement"]
    if placement == "input_only":
        return rate, 0.0
    if placement == "recurrent_only":
        return 0.0, rate
    return rate / 2.0, rate / 2.0


def _compile_autoencoder(model: keras.Model, optimizer: str) -> None:
    model.compile(optimizer=_optimizer(optimizer), loss="mse")


def _build_fc_sae(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    del seed
    widths = config["encoder_widths"]
    inputs = keras.Input((input_length,), name="consumption")
    value = inputs
    for index, width in enumerate(widths, start=1):
        value = _dense_hidden(
            value,
            width,
            config["hidden_activation"],
            _dense_dropout(
                config,
                stage="encoder",
                index=index,
                total=len(widths),
            ),
            f"encoder_dense_{index}",
        )
    if (
        _uses_source_architecture(config)
        and config["latent_placement"] == "distinct_projection"
    ):
        latent = _dense_hidden(
            value,
            config["latent_width"],
            config["hidden_activation"],
            _dense_dropout(
                config,
                stage="latent",
                index=1,
                total=1,
            ),
            "latent_projection",
        )
        value = latent
    else:
        latent = value
    encoder = keras.Model(inputs, latent, name="fc_sae_encoder")
    decoder_widths = (
        tuple(reversed(widths))
        if _uses_source_architecture(config)
        else tuple(reversed(widths[:-1]))
    )
    for index, width in enumerate(decoder_widths, start=1):
        value = _dense_hidden(
            value,
            width,
            config["hidden_activation"],
            _dense_dropout(
                config,
                stage="decoder",
                index=index,
                total=len(decoder_widths),
            ),
            f"decoder_dense_{index}",
        )
    reconstruction = layers.Dense(
        input_length,
        activation=config["output_activation"],
        name="reconstruction",
    )(value)
    model = keras.Model(inputs, reconstruction, name="fc_sae")
    _compile_autoencoder(model, config["optimizer"])
    score = ReconstructionScore(name="reconstruction_mse")(
        (inputs, reconstruction)
    )
    score_model = keras.Model(
        inputs, {"reconstruction_mse": score}, name="fc_sae_scores"
    )
    return ModelBundle("fc_sae", model, encoder=encoder, score_model=score_model)


def _lstm_encoder(
    inputs: Any,
    widths: tuple[int, ...],
    activation: str,
    dropout: float,
    recurrent_dropout: float,
    *,
    input_layout: str,
    final_sequence: bool,
    prefix: str,
) -> tuple[Any, list[tuple[Any, Any]]]:
    target_shape = (
        (-1, 1)
        if input_layout == "48_steps_1_feature"
        else (1, int(inputs.shape[-1]))
    )
    value = layers.Reshape(target_shape, name=f"{prefix}_sequence")(inputs)
    states: list[tuple[Any, Any]] = []
    for index, width in enumerate(widths, start=1):
        return_sequences = final_sequence or index < len(widths)
        value, hidden, cell = layers.LSTM(
            width,
            activation=activation,
            recurrent_activation="sigmoid",
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences,
            return_state=True,
            name=f"{prefix}_lstm_{index}",
        )(value)
        states.append((hidden, cell))
    return value, states


def _lstm_decoder_layers(
    widths: tuple[int, ...],
    activation: str,
    dropout: float,
    recurrent_dropout: float,
    output_activation: str,
    prefix: str,
) -> tuple[list[layers.LSTM], layers.TimeDistributed]:
    decoder_layers = [
        layers.LSTM(
            width,
            activation=activation,
            recurrent_activation="sigmoid",
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            name=f"{prefix}_lstm_{index}",
        )
        for index, width in enumerate(reversed(widths), start=1)
    ]
    projection = layers.TimeDistributed(
        layers.Dense(1, activation=output_activation), name=f"{prefix}_output"
    )
    return decoder_layers, projection


def _decode_sequence(
    latent: Any,
    input_length: int,
    decoder_layers: list[layers.LSTM],
    projection: layers.TimeDistributed,
    encoder_states: list[tuple[Any, Any]] | None,
    schedule: str,
    state_policy: str,
    prefix: str,
) -> Any:
    if schedule == "repeat_latent":
        value = layers.RepeatVector(input_length, name=f"{prefix}_repeat")(latent)
    elif schedule == "first_latent_then_zero":
        value = FirstStepLatentSequence(
            input_length, name=f"{prefix}_first_then_zero"
        )(latent)
    else:
        raise ValueError(
            "autoregressive_reconstruction requires the dedicated "
            "autoregressive decoder implementation"
        )
    reversed_states = list(reversed(encoder_states or []))
    for index, decoder_layer in enumerate(decoder_layers):
        if state_policy == "mirrored_layer_states":
            initial_state = (
                reversed_states[index] if index < len(reversed_states) else None
            )
        else:
            initial_state = reversed_states[0] if index == 0 and reversed_states else None
        value = decoder_layer(value, initial_state=initial_state)
    value = projection(value)
    return layers.Reshape((input_length,), name=f"{prefix}_flatten")(value)


def _autoregressive_inputs(
    latent: Any,
    encoder_states: list[tuple[Any, Any]],
    state_policy: str,
) -> tuple[Any, ...]:
    return (
        latent,
        *_autoregressive_state_tensors(encoder_states, state_policy),
    )


def _autoregressive_state_tensors(
    encoder_states: list[tuple[Any, Any]],
    state_policy: str,
) -> tuple[Any, ...]:
    reversed_states = list(reversed(encoder_states))
    selected = (
        reversed_states
        if state_policy == "mirrored_layer_states"
        else reversed_states[:1]
    )
    return tuple(tensor for pair in selected for tensor in pair)


def _build_lstm_sae(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    del seed
    widths = config["encoder_widths"]
    input_dropout, recurrent_dropout = _lstm_dropouts(config)
    inputs = keras.Input((input_length,), name="consumption")
    latent, states = _lstm_encoder(
        inputs,
        widths,
        config["hidden_activation"],
        input_dropout,
        recurrent_dropout,
        input_layout=config["lstm_input"],
        final_sequence=False,
        prefix="encoder",
    )
    if (
        _uses_source_architecture(config)
        and config["latent_placement"] == "distinct_projection"
    ):
        latent = layers.Dense(
            config["latent_width"],
            activation=config["hidden_activation"],
            name="latent_projection",
        )(latent)
    encoder = keras.Model(inputs, latent, name="lstm_sae_encoder")
    if config["decoder_schedule"] == "autoregressive_reconstruction":
        reconstruction = AutoregressiveLSTMDecoder(
            tuple(reversed(widths)),
            input_length,
            config["hidden_activation"],
            config["output_activation"],
            input_dropout,
            recurrent_dropout,
            config["decoder_state"],
            name="autoregressive_decoder",
        )(_autoregressive_inputs(latent, states, config["decoder_state"]))
    else:
        decoder_layers, projection = _lstm_decoder_layers(
            widths,
            config["hidden_activation"],
            input_dropout,
            recurrent_dropout,
            config["output_activation"],
            "decoder",
        )
        reconstruction = _decode_sequence(
            latent,
            input_length,
            decoder_layers,
            projection,
            states,
            config["decoder_schedule"],
            config["decoder_state"],
            "decoder",
        )
    model = keras.Model(inputs, reconstruction, name="lstm_sae")
    _compile_autoencoder(model, config["optimizer"])
    score = ReconstructionScore(name="reconstruction_mse")(
        (inputs, reconstruction)
    )
    score_model = keras.Model(
        inputs, {"reconstruction_mse": score}, name="lstm_sae_scores"
    )
    return ModelBundle("lstm_sae", model, encoder=encoder, score_model=score_model)


def _build_fc_vae(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    widths = config["encoder_widths"]
    inputs = keras.Input((input_length,), name="consumption")
    value = inputs
    encoder_widths = widths if _uses_source_architecture(config) else widths[:-1]
    for index, width in enumerate(encoder_widths, start=1):
        value = _dense_hidden(
            value,
            width,
            config["hidden_activation"],
            _dense_dropout(
                config,
                stage="encoder",
                index=index,
                total=len(encoder_widths),
            ),
            f"encoder_dense_{index}",
        )
    latent_width = (
        config["latent_width"]
        if _uses_source_architecture(config)
        else widths[-1]
    )
    z_mean = layers.Dense(latent_width, name="z_mean")(value)
    z_log_var = layers.Dense(latent_width, name="z_log_var")(value)
    z = Sampling(seed=seed, name="z")((z_mean, z_log_var))
    encoder = keras.Model(inputs, (z_mean, z_log_var, z), name="fc_vae_encoder")

    decoder_input = keras.Input((latent_width,), name="decoder_input")
    decoded = decoder_input
    decoder_widths = (
        tuple(reversed(widths))
        if _uses_source_architecture(config)
        else tuple(reversed(widths[:-1]))
    )
    for index, width in enumerate(decoder_widths, start=1):
        decoded = _dense_hidden(
            decoded,
            width,
            config["hidden_activation"],
            _dense_dropout(
                config,
                stage="decoder",
                index=index,
                total=len(decoder_widths),
            ),
            f"decoder_dense_{index}",
        )
    decoder_mean = layers.Dense(
        input_length,
        activation=config["output_activation"],
        name="decoder_reconstruction",
    )(decoded)
    score_spec = parse_vae_score_branch(str(config["vae_score"]))
    learned_variance = score_spec.variance == "learned"
    if learned_variance:
        decoder_log_variance = layers.Dense(
            input_length,
            activation="linear",
            name="decoder_log_variance",
        )(decoded)
        decoder = keras.Model(
            decoder_input,
            (decoder_mean, decoder_log_variance),
            name="fc_vae_decoder",
        )
        sampled_reconstruction, sampled_log_variance = decoder(z)
        training_output = GaussianVAELoss(
            reduction=config["vae_loss_reduction"],
            name="gaussian_vae_loss",
        )(
            (
                inputs,
                sampled_reconstruction,
                sampled_log_variance,
                z_mean,
                z_log_var,
            )
        )
        deterministic_reconstruction, _ = decoder(z_mean)
        stochastic_distribution_model = keras.Model(
            inputs,
            (sampled_reconstruction, sampled_log_variance),
            name="fc_vae_stochastic_distribution",
        )
    else:
        decoder = keras.Model(
            decoder_input, decoder_mean, name="fc_vae_decoder"
        )
        sampled_reconstruction = decoder(z)
        training_output = VAELoss(
            reduction=config["vae_loss_reduction"],
            name="vae_loss",
        )(
            (inputs, sampled_reconstruction, z_mean, z_log_var)
        )
        deterministic_reconstruction = decoder(z_mean)
        stochastic_distribution_model = None
    model = keras.Model(inputs, training_output, name="fc_vae")
    model.compile(optimizer=_optimizer(config["optimizer"]))

    reconstruction_mse, mse_plus_kl_surrogate = VAEScores(name="vae_scores")(
        (inputs, deterministic_reconstruction, z_mean, z_log_var)
    )
    score_model = keras.Model(
        inputs,
        {
            "reconstruction_mse": reconstruction_mse,
            "mse_plus_kl_surrogate": mse_plus_kl_surrogate,
        },
        name="fc_vae_scores",
    )
    return ModelBundle(
        "fc_vae",
        model,
        encoder=encoder,
        decoder=decoder,
        score_model=score_model,
        stochastic_reconstruction_model=model,
        stochastic_distribution_model=stochastic_distribution_model,
    )


def _build_lstm_vae(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    widths = config["encoder_widths"]
    input_dropout, recurrent_dropout = _lstm_dropouts(config)
    inputs = keras.Input((input_length,), name="consumption")
    encoded, states = _lstm_encoder(
        inputs,
        widths,
        config["hidden_activation"],
        input_dropout,
        recurrent_dropout,
        input_layout=config["lstm_input"],
        final_sequence=False,
        prefix="encoder",
    )
    latent_width = (
        config["latent_width"]
        if _uses_source_architecture(config)
        else widths[-1]
    )
    z_mean = layers.Dense(latent_width, name="z_mean")(encoded)
    z_log_var = layers.Dense(latent_width, name="z_log_var")(encoded)
    z = Sampling(seed=seed, name="z")((z_mean, z_log_var))
    encoder = keras.Model(inputs, (z_mean, z_log_var, z), name="lstm_vae_encoder")
    if config["decoder_schedule"] == "autoregressive_reconstruction":
        autoregressive_decoder = AutoregressiveLSTMDecoder(
            tuple(reversed(widths)),
            input_length,
            config["hidden_activation"],
            config["output_activation"],
            input_dropout,
            recurrent_dropout,
            config["decoder_state"],
            name="autoregressive_decoder",
        )
        sampled_reconstruction = autoregressive_decoder(
            _autoregressive_inputs(z, states, config["decoder_state"])
        )
        deterministic_reconstruction = autoregressive_decoder(
            _autoregressive_inputs(z_mean, states, config["decoder_state"])
        )
    else:
        decoder_layers, projection = _lstm_decoder_layers(
            widths,
            config["hidden_activation"],
            input_dropout,
            recurrent_dropout,
            config["output_activation"],
            "decoder",
        )
        sampled_reconstruction = _decode_sequence(
            z,
            input_length,
            decoder_layers,
            projection,
            states,
            config["decoder_schedule"],
            config["decoder_state"],
            "sampled_decoder",
        )
        deterministic_reconstruction = _decode_sequence(
            z_mean,
            input_length,
            decoder_layers,
            projection,
            states,
            config["decoder_schedule"],
            config["decoder_state"],
            "deterministic_decoder",
        )
    score_spec = parse_vae_score_branch(str(config["vae_score"]))
    learned_variance = score_spec.variance == "learned"
    if learned_variance:
        variance_head = layers.Dense(
            input_length,
            activation="linear",
            name="decoder_log_variance",
        )
        sampled_log_variance = variance_head(sampled_reconstruction)
        training_output = GaussianVAELoss(
            reduction=config["vae_loss_reduction"],
            name="gaussian_vae_loss",
        )(
            (
                inputs,
                sampled_reconstruction,
                sampled_log_variance,
                z_mean,
                z_log_var,
            )
        )
        stochastic_distribution_model = keras.Model(
            inputs,
            (sampled_reconstruction, sampled_log_variance),
            name="lstm_vae_stochastic_distribution",
        )
    else:
        training_output = VAELoss(
            reduction=config["vae_loss_reduction"],
            name="vae_loss",
        )(
            (inputs, sampled_reconstruction, z_mean, z_log_var)
        )
        stochastic_distribution_model = None
    model = keras.Model(inputs, training_output, name="lstm_vae")
    model.compile(optimizer=_optimizer(config["optimizer"]))

    reconstruction_mse, mse_plus_kl_surrogate = VAEScores(name="vae_scores")(
        (inputs, deterministic_reconstruction, z_mean, z_log_var)
    )
    score_model = keras.Model(
        inputs,
        {
            "reconstruction_mse": reconstruction_mse,
            "mse_plus_kl_surrogate": mse_plus_kl_surrogate,
        },
        name="lstm_vae_scores",
    )
    return ModelBundle(
        "lstm_vae",
        model,
        encoder=encoder,
        score_model=score_model,
        stochastic_reconstruction_model=model,
        stochastic_distribution_model=stochastic_distribution_model,
    )


def _build_lstm_aea(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    del seed
    widths = config["encoder_widths"]
    input_dropout, recurrent_dropout = _lstm_dropouts(config)
    inputs = keras.Input((input_length,), name="consumption")
    encoder_sequence, states = _lstm_encoder(
        inputs,
        widths,
        config["hidden_activation"],
        input_dropout,
        recurrent_dropout,
        input_layout=config["lstm_input"],
        final_sequence=True,
        prefix="encoder",
    )
    initial_decoder_hidden = states[-1][0]
    encoder = keras.Model(inputs, initial_decoder_hidden, name="lstm_aea_encoder")
    if config["decoder_schedule"] == "autoregressive_reconstruction":
        latent_width = (
            config["latent_width"]
            if (
                _uses_source_architecture(config)
                and config["latent_placement"] == "distinct_projection"
            )
            else None
        )
        autoregressive_decoder = AutoregressiveAttentionLSTMDecoder(
            tuple(reversed(widths)),
            input_length,
            config["hidden_activation"],
            config["output_activation"],
            input_dropout,
            recurrent_dropout,
            config["decoder_state"],
            config["attention_merge"],
            latent_width,
            name="autoregressive_attention_decoder",
        )
        reconstruction, attention_weights = autoregressive_decoder(
            (
                encoder_sequence,
                initial_decoder_hidden,
                *_autoregressive_state_tensors(
                    states,
                    config["decoder_state"],
                ),
            )
        )
        attention_model = keras.Model(
            inputs,
            attention_weights,
            name="lstm_aea_attention_weights",
        )
        model = keras.Model(inputs, reconstruction, name="lstm_aea")
        _compile_autoencoder(model, config["optimizer"])
        score = ReconstructionScore(name="reconstruction_mse")(
            (inputs, reconstruction)
        )
        score_model = keras.Model(
            inputs,
            {"reconstruction_mse": score},
            name="lstm_aea_scores",
        )
        return ModelBundle(
            "lstm_aea",
            model,
            encoder=encoder,
            score_model=score_model,
            attention_model=attention_model,
        )

    decoder_layers, projection = _lstm_decoder_layers(
        widths,
        config["hidden_activation"],
        input_dropout,
        recurrent_dropout,
        config["output_activation"],
        "decoder",
    )
    if config["decoder_schedule"] == "repeat_latent":
        decoder_seed = layers.RepeatVector(
            input_length, name="decoder_repeat"
        )(initial_decoder_hidden)
    elif config["decoder_schedule"] == "first_latent_then_zero":
        decoder_seed = FirstStepLatentSequence(
            input_length, name="decoder_first_then_zero"
        )(initial_decoder_hidden)
    else:
        raise RuntimeError("unreachable decoder schedule")
    decoder_sequence = decoder_layers[0](
        decoder_seed, initial_state=states[-1]
    )
    decoder_queries = PreviousDecoderQueries(name="previous_decoder_queries")(
        (decoder_sequence, initial_decoder_hidden)
    )
    context_sequence, attention_weights = TemporalAdditiveAttention(
        attention_units=widths[-1], name="temporal_attention"
    )((encoder_sequence, decoder_queries))
    if config["attention_merge"] == "literal_sum":
        value = layers.Add(name="attention_context_plus_decoder")(
            (context_sequence, decoder_sequence)
        )
    else:
        value = layers.Concatenate(name="attention_context_and_decoder")(
            (context_sequence, decoder_sequence)
        )
    if (
        _uses_source_architecture(config)
        and config["latent_placement"] == "distinct_projection"
    ):
        value = layers.TimeDistributed(
            layers.Dense(
                config["latent_width"],
                activation=config["hidden_activation"],
            ),
            name="latent_projection",
        )(value)
    reversed_states = list(reversed(states))
    for index, decoder_layer in enumerate(decoder_layers[1:], start=1):
        initial_state = (
            reversed_states[index]
            if config["decoder_state"] == "mirrored_layer_states"
            else None
        )
        value = decoder_layer(value, initial_state=initial_state)
    value = projection(value)
    reconstruction = layers.Reshape((input_length,), name="decoder_flatten")(value)
    attention_model = keras.Model(
        inputs, attention_weights, name="lstm_aea_attention_weights"
    )
    model = keras.Model(inputs, reconstruction, name="lstm_aea")
    _compile_autoencoder(model, config["optimizer"])
    score = ReconstructionScore(name="reconstruction_mse")(
        (inputs, reconstruction)
    )
    score_model = keras.Model(
        inputs, {"reconstruction_mse": score}, name="lstm_aea_scores"
    )
    return ModelBundle(
        "lstm_aea",
        model,
        encoder=encoder,
        score_model=score_model,
        attention_model=attention_model,
    )


def _build_supervised_feed_forward(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    del seed
    inputs = keras.Input((input_length,), name="consumption")
    value = inputs
    for index, width in enumerate(config["encoder_widths"], start=1):
        value = _dense_hidden(
            value,
            width,
            config["hidden_activation"],
            config["dropout"],
            f"hidden_dense_{index}",
        )
    binary_head = config["supervised_head"] == "sigmoid1_binary"
    output = layers.Dense(
        1 if binary_head else 2,
        activation="sigmoid" if binary_head else "softmax",
        name="class_probability",
    )(value)
    model = keras.Model(inputs, output, name="supervised_feed_forward")
    model.compile(
        optimizer=_optimizer(config["optimizer"]),
        loss=(
            "binary_crossentropy"
            if binary_head
            else "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )
    return ModelBundle("supervised_feed_forward", model)


def _build_supervised_lstm(
    input_length: int, config: Mapping[str, Any], seed: int | None
) -> ModelBundle:
    del seed
    widths = config["encoder_widths"]
    input_dropout, recurrent_dropout = _lstm_dropouts(config)
    inputs = keras.Input((input_length,), name="consumption")
    sequence_shape = (
        (input_length, 1)
        if config["lstm_input"] == "48_steps_1_feature"
        else (1, input_length)
    )
    value = layers.Reshape(sequence_shape, name="sequence")(inputs)
    for index, width in enumerate(widths, start=1):
        value = layers.LSTM(
            width,
            activation=config["hidden_activation"],
            recurrent_activation="sigmoid",
            dropout=input_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=index < len(widths),
            name=f"hidden_lstm_{index}",
        )(value)
    binary_head = config["supervised_head"] == "sigmoid1_binary"
    output = layers.Dense(
        1 if binary_head else 2,
        activation="sigmoid" if binary_head else "softmax",
        name="class_probability",
    )(value)
    model = keras.Model(inputs, output, name="supervised_lstm")
    model.compile(
        optimizer=_optimizer(config["optimizer"]),
        loss=(
            "binary_crossentropy"
            if binary_head
            else "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )
    return ModelBundle("supervised_lstm", model)


_BUILDERS = {
    "fc_sae": _build_fc_sae,
    "lstm_sae": _build_lstm_sae,
    "fc_vae": _build_fc_vae,
    "lstm_vae": _build_lstm_vae,
    "lstm_aea": _build_lstm_aea,
    "supervised_feed_forward": _build_supervised_feed_forward,
    "supervised_lstm": _build_supervised_lstm,
}


def build_model(
    model_name: str,
    input_length: int,
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> ModelBundle:
    """Build and compile one paper-literal model.

    ``config`` overrides only the selected model's defaults.  Width overrides
    are intended for deterministic interface tests and resource probes; result
    runs must pass the frozen Table I widths.
    """

    if int(input_length) <= 0:
        raise ValueError("input_length must be positive")
    canonical_name = _canonical_name(model_name)
    if seed is not None:
        keras.utils.set_random_seed(seed)
    resolved = _model_config(canonical_name, config)
    return _BUILDERS[canonical_name](int(input_length), resolved, seed)


def build_fc_sae(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("fc_sae", input_length, config, seed)


def build_lstm_sae(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("lstm_sae", input_length, config, seed)


def build_fc_vae(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("fc_vae", input_length, config, seed)


def build_lstm_vae(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("lstm_vae", input_length, config, seed)


def build_lstm_aea(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("lstm_aea", input_length, config, seed)


def build_supervised_feed_forward(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("supervised_feed_forward", input_length, config, seed)


def build_supervised_lstm(
    input_length: int, config: Mapping[str, Any] | None = None, seed: int | None = None
) -> ModelBundle:
    return build_model("supervised_lstm", input_length, config, seed)
