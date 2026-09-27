"""Direct implementations of the paper's detectors, added as they are tested."""

import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC


def random_forest(seed: int = 20260920, workers: int = 4) -> RandomForestClassifier:
    """Section III-C, p.2679: 100 estimators; remaining choices are explicit."""
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=workers,
        random_state=seed,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    )


def adaboost(seed: int = 20260920) -> AdaBoostClassifier:
    """III-B.2(b)/III-C: 50 trees; historical defaults declared in ADABOOST_PILOT."""
    if sklearn.__version__ != "1.5.2":
        raise RuntimeError("SAMME.R requires the pinned AdaBoost scikit-learn 1.5.2 environment")
    return AdaBoostClassifier(
        estimator=None,  # Stock depth-one decision tree; fitted parameters saved.
        n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=seed,
    )


def svm(seed: int = 20260920) -> SVC:
    """III-C p.2679: C=1, sigmoid; omissions fixed in SVM_PILOT.md."""
    return SVC(C=1.0, kernel="sigmoid", degree=3, gamma="scale", coef0=0.0,
               shrinking=True, probability=False, tol=0.001, cache_size=200,
               class_weight=None, verbose=False, max_iter=-1,
               decision_function_shape="ovr", break_ties=False, random_state=seed)


def feed_forward(seed: int = 20260920):
    """Table II / III-B.2(d), with the explicit BCE repair in FEED_FORWARD_PILOT."""
    import tensorflow as tf
    import keras
    if tf.__version__ != "2.16.2" or keras.__version__ != "3.4.1" or keras.backend.backend() != "tensorflow":
        raise RuntimeError("Use the pinned TensorFlow feed-forward environment")
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    keras.mixed_precision.set_global_policy("float32")
    layers = [keras.layers.Input(shape=(48,), dtype="float32")]
    for index in range(6):
        layers.append(keras.layers.Dense(500, activation="relu", use_bias=True,
            kernel_initializer="glorot_uniform", bias_initializer="zeros",
            kernel_constraint=keras.constraints.MaxNorm(3., axis=0), name=f"hidden_{index + 1}"))
    layers.append(keras.layers.Dense(1, activation="sigmoid", use_bias=True,
        kernel_initializer="glorot_uniform", bias_initializer="zeros",
        kernel_constraint=keras.constraints.MaxNorm(3., axis=0), name="output"))
    model = keras.Sequential(layers, name="paper_feed_forward_bce_repair")
    model.compile(optimizer=keras.optimizers.Adamax(learning_rate=.002, beta_1=.9,
        beta_2=.999, epsilon=1e-7), loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=.5)],
        jit_compile=False)
    return model


def printed_classification_loss(y_true, probability):
    """Equation(1) as printed, for algebra/gradient fixtures only; not a fit loss."""
    import tensorflow as tf
    y_true = tf.cast(y_true, probability.dtype)
    return -tf.reduce_mean(y_true * tf.math.log(probability) + (1 - y_true) * tf.math.log(probability))


def gru(seed: int = 20260920, *, units: int = 300, hidden_layers: int = 8):
    """Table-II native-Keras interpretation; small overrides are fixture-only."""
    import tensorflow as tf
    import keras
    if tf.__version__ != "2.16.2" or keras.__version__ != "3.4.1" or keras.backend.backend() != "tensorflow":
        raise RuntimeError("Use the pinned TensorFlow neural environment")
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    keras.mixed_precision.set_global_policy("float32")
    layers = [keras.layers.Input(shape=(48, 1), dtype="float32")]
    for index in range(hidden_layers):
        layers.append(keras.layers.GRU(units, activation="relu", recurrent_activation="sigmoid",
            use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
            bias_initializer="zeros", kernel_constraint=keras.constraints.MaxNorm(5., axis=0),
            recurrent_constraint=keras.constraints.MaxNorm(5., axis=0),
            dropout=.2, recurrent_dropout=0., return_sequences=index < hidden_layers - 1,
            return_state=False, go_backwards=False, stateful=False, unroll=False,
            reset_after=False, use_cudnn=False, implementation=2, seed=seed + index,
            name=f"gru_{index + 1}"))
    layers.append(keras.layers.Dense(2, activation="softmax", use_bias=True,
        kernel_initializer="glorot_uniform", bias_initializer="zeros",
        kernel_constraint=keras.constraints.MaxNorm(5., axis=0), name="output"))
    model = keras.Sequential(layers, name="paper_gru_native_table_ce_repair")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001, beta_1=.9,
        beta_2=.999, epsilon=1e-7), loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy")], jit_compile=False)
    return model


_SEQUENTIAL_LAYER = None


def register_sequential_layer():
    """Register the serializable layer without requiring TF for classical models.

    Call this before fresh-process load_model. All scientific operations live
    here in the direct implementation; the older AEA prototype is unchanged.
    """
    global _SEQUENTIAL_LAYER
    if _SEQUENTIAL_LAYER is not None:
        return _SEQUENTIAL_LAYER
    import tensorflow as tf
    import keras

    @keras.saving.register_keras_serializable(package="atk_evidence")
    class SequentialAttentionDecoder(keras.layers.Layer):
        """I-SEQ-native-IVC: target Fig.2/IV-C and declared causal completions."""

        def __init__(self, encoder_units=(500, 300, 200), steps=48,
                     seed=20260920, **kwargs):
            super().__init__(**kwargs)
            self.encoder_units = tuple(int(u) for u in encoder_units)
            self.steps, self.seed = int(steps), int(seed)
            if len(self.encoder_units) != 3 or min(self.encoder_units) < 1 or self.steps < 2:
                raise ValueError("Need three positive encoder widths and at least two steps")
            self.decoder_units = tuple(reversed(self.encoder_units))
            constraint = keras.constraints.MaxNorm(1., axis=0)
            self.encoder = [keras.layers.LSTM(
                u, activation="relu", recurrent_activation="sigmoid",
                kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=keras.initializers.Orthogonal(seed=self.seed + 10 + i),
                bias_initializer="zeros", unit_forget_bias=True,
                kernel_constraint=constraint, recurrent_constraint=constraint,
                dropout=0., recurrent_dropout=0., return_sequences=True,
                return_state=True, use_cudnn=False, name=f"encoder_lstm_{i+1}")
                for i, u in enumerate(self.encoder_units)]
            self.decoder = [keras.layers.LSTMCell(
                u, activation="relu", recurrent_activation="sigmoid",
                kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed + 20 + i),
                recurrent_initializer=keras.initializers.Orthogonal(seed=self.seed + 30 + i),
                bias_initializer="zeros", unit_forget_bias=True,
                kernel_constraint=constraint, recurrent_constraint=constraint,
                dropout=0., recurrent_dropout=0., name=f"decoder_lstm_{i+1}")
                for i, u in enumerate(self.decoder_units)]
            self.projection = keras.layers.Dense(
                1, activation="relu", kernel_constraint=constraint,
                kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed + 40),
                bias_initializer="zeros", name="intermediate_projection")

        def build(self, input_shape):
            if tuple(input_shape[1:]) != (self.steps, 1):
                raise ValueError("Expected one scalar per declared time step")
            width = 1
            for layer in self.encoder:
                layer.build((None, self.steps, width))
                width = layer.units
            width = self.encoder_units[-1] + 1
            for cell in self.decoder:
                cell.build((None, width))
                width = cell.units
            self.projection.build((None, width))
            attention_width = self.encoder_units[-1]
            constraint = keras.constraints.MaxNorm(1., axis=0)
            for name, shape, offset in (
                ("attention_encoder_kernel", (attention_width, attention_width), 50),
                ("attention_decoder_kernel", (self.decoder_units[-1], attention_width), 51),
                ("attention_vector", (attention_width,), 52),
            ):
                setattr(self, name, self.add_weight(name=name, shape=shape,
                    initializer=keras.initializers.GlorotUniform(seed=self.seed + offset),
                    constraint=constraint))
            self.attention_bias = self.add_weight(name="attention_bias",
                shape=(attention_width,), initializer="zeros")
            super().build(input_shape)

        def get_config(self):
            return {**super().get_config(), "encoder_units": list(self.encoder_units),
                    "steps": self.steps, "seed": self.seed}

        def compute_output_shape(self, input_shape):
            return ((input_shape[0], self.steps, 1),
                    (input_shape[0], self.steps, self.steps))

        def encode(self, inputs, training=False):
            memory, states = inputs, []
            for layer in self.encoder:
                memory, hidden, cell = layer(memory, training=training)
                states.append([hidden, cell])
            projected = tf.einsum("btd,dk->btk", memory, self.attention_encoder_kernel)
            return memory, projected, list(reversed(states))

        def attend(self, memory, projected, query):
            query = tf.einsum("bd,dk->bk", query, self.attention_decoder_kernel)
            energy = tf.einsum("btk,k->bt",
                tf.tanh(projected + query[:, None, :] + self.attention_bias),
                self.attention_vector)
            weights = tf.nn.softmax(energy, axis=1)
            return tf.einsum("bt,btd->bd", weights, memory), weights

        def decode_step(self, previous, states, memory, projected, training=False):
            context, weights = self.attend(memory, projected, states[-1][0])
            value = tf.concat((context, previous), axis=-1)
            next_states = []
            for cell, state in zip(self.decoder, states):
                value, state = cell(value, state, training=training)
                next_states.append(list(state))
            return self.projection(value), next_states, weights

        def call(self, inputs, training=False):
            memory, projected, states = self.encode(inputs, training=training)
            previous = tf.zeros((tf.shape(inputs)[0], 1), dtype=inputs.dtype)
            outputs = tf.TensorArray(inputs.dtype, size=self.steps)
            attention = tf.TensorArray(inputs.dtype, size=self.steps)

            def step(t, previous, states, outputs, attention):
                value, next_states, weights = self.decode_step(
                    previous, states, memory, projected, training=training)
                return t + 1, value, next_states, outputs.write(t, value), attention.write(t, weights)

            _, _, _, outputs, attention = tf.while_loop(
                lambda t, *_: t < self.steps, step,
                (tf.constant(0), previous, states, outputs, attention), parallel_iterations=1)
            return (tf.transpose(outputs.stack(), (1, 0, 2)),
                    tf.transpose(attention.stack(), (1, 0, 2)))

    _SEQUENTIAL_LAYER = SequentialAttentionDecoder
    return _SEQUENTIAL_LAYER


def sequential_ensemble(seed: int = 20260920, *, encoder_units=(500, 300, 200),
                        gru_units=300, dense_units=500, timesteps=48):
    """Section IV-C native completion; reduced dimensions are software-only.

    The eight GRU layers and six LSTM layers are retained for every fixture.
    No standalone reconstruction loss, pretraining or 0.51 cutoff is inherited.
    """
    import tensorflow as tf
    import keras
    if tf.__version__ != "2.16.2" or keras.__version__ != "3.4.1" or keras.backend.backend() != "tensorflow":
        raise RuntimeError("Use the pinned TensorFlow neural environment")
    if gru_units < 1 or dense_units < 1:
        raise ValueError("Need positive GRU and dense widths")
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    keras.mixed_precision.set_global_policy("float32")
    inputs = keras.Input(shape=(timesteps, 1), dtype="float32", name="daily_profile")
    frontend = register_sequential_layer()(encoder_units=encoder_units, steps=timesteps,
                                          seed=seed, name="attention_decoder")
    value, _ = frontend(inputs)
    constraint = keras.constraints.MaxNorm(1., axis=0)
    for i in range(8):
        value = keras.layers.GRU(gru_units, activation="relu", recurrent_activation="sigmoid",
            use_bias=True, kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100 + i),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 200 + i),
            bias_initializer="zeros", kernel_constraint=constraint, recurrent_constraint=constraint,
            dropout=0., recurrent_dropout=0., return_sequences=i < 7,
            reset_after=False, use_cudnn=False, implementation=2, name=f"sequence_gru_{i+1}")(value)
    value = keras.layers.Dense(dense_units, activation="relu", kernel_constraint=constraint,
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 300),
        bias_initializer="zeros", name="classifier_hidden")(value)
    output = keras.layers.Dense(1, activation="sigmoid", kernel_constraint=constraint,
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 301),
        bias_initializer="zeros", name="attack_probability")(value)
    model = keras.Model(inputs, output, name="paper_sequential_native_ivc_bce_repair")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001, beta_1=.9,
        beta_2=.999, epsilon=1e-7), loss=keras.losses.BinaryCrossentropy(from_logits=False),
        jit_compile=False)
    return model
