"""The PROTGOAT architecture: one tower per input modality, then a shared trunk.

Each of the seven feature matrices gets its own two-layer tower, all towers are
concatenated, and a single sigmoid layer emits one probability per GO term::

    t5         -> Dense(600) -> BN -> LeakyReLU -> Dropout -> Dense(300) --.
    esm2_small -> ...                                                      |
    esm2_3b    -> ...                                                      |
    protbert   -> ...                                                      +--> concat
    ankh       -> ...                                                      |      |
    taxonomy   -> Dense(128) -> ... -> Dense(64) ------------------------- |      |
    abstracts  -> Dense(200) -> ... -> Dense(200) --------------------------'     |
                                                                                 v
                          BN -> Dense(800, relu) -> Dense(n_terms, sigmoid) <-----'

Keeping the towers separate before the concatenation lets each modality be
normalised on its own scale; the embeddings differ by an order of magnitude in
both dimensionality and variance.

TensorFlow is imported lazily so that ``protgoat.config``, ``protgoat.data`` and
``protgoat.submission`` stay importable (and testable) without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .config import FeatureSpec, TrainConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf


def build_model(
    feature_dims: Mapping[str, int],
    n_terms: int,
    config: TrainConfig,
    specs: Sequence[FeatureSpec],
) -> tf.keras.Model:
    """Assemble the multi-input classifier.

    Args:
        feature_dims: Embedding width of each modality, keyed by feature key.
        n_terms: Number of GO terms in this sub-ontology's label space.
        config: Hyperparameters (dropout, trunk width, ...).
        specs: Feature specs in input order. Saved checkpoints expect this exact
            order, so it must match the order used at training time.

    Returns:
        An uncompiled ``keras.Model`` taking one input per spec, in ``specs`` order.
    """
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        BatchNormalization,
        Concatenate,
        Dense,
        Dropout,
        Input,
        LeakyReLU,
    )

    missing = [spec.key for spec in specs if spec.key not in feature_dims]
    if missing:
        raise ValueError(f"no embedding width given for features: {missing}")

    inputs = []
    towers = []
    for spec in specs:
        wide, narrow = spec.hidden
        tower_input = Input(shape=(feature_dims[spec.key],), name=f"{spec.key}_input")
        tower = Dense(wide, name=f"{spec.key}_dense_1")(tower_input)
        tower = BatchNormalization(name=f"{spec.key}_batchnorm_1")(tower)
        tower = LeakyReLU(
            alpha=config.leaky_relu_alpha, name=f"{spec.key}_leakyrelu_1"
        )(tower)
        tower = Dropout(config.dropout, name=f"{spec.key}_dropout_1")(tower)
        tower = Dense(narrow, name=f"{spec.key}_dense_2")(tower)
        inputs.append(tower_input)
        towers.append(tower)

    combined = Concatenate(name="towers_concat")(towers) if len(towers) > 1 else towers[0]
    combined = BatchNormalization(name="combined_batchnorm_1")(combined)
    combined = Dense(config.trunk_dim, activation="relu", name="combined_dense_1")(
        combined
    )
    output = Dense(n_terms, activation="sigmoid", name="term_probabilities")(combined)

    return Model(inputs=inputs, outputs=output, name="protgoat")


def compile_model(model: tf.keras.Model, config: TrainConfig) -> tf.keras.Model:
    """Compile with the settings used for the submitted run."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def step_decay(config: TrainConfig):
    """Halve the learning rate every ``config.lr_decay_every`` epochs."""

    def schedule(epoch: int, lr: float) -> float:
        if epoch > 0 and epoch % config.lr_decay_every == 0:
            return lr * config.lr_decay_factor
        return lr

    return schedule
