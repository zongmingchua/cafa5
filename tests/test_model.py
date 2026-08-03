"""Architecture checks.

These pin the refactored ``build_model`` to the shape of the network that was
actually submitted: the towers were originally written out seven times by hand,
and the loop that replaced them must produce the same graph. Skipped when
TensorFlow is not installed, so the rest of the suite stays runnable without it.
"""

import numpy as np
import pytest

from protgoat.config import load_config
from protgoat.model import build_model, compile_model, step_decay

tf = pytest.importorskip("tensorflow", reason="requires the [train] extra")

# Real embedding widths of the seven modalities.
DIMS = {
    "t5": 1024,
    "esm2_small": 1280,
    "esm2_3b": 2560,
    "protbert": 1024,
    "ankh": 1536,
    "taxonomy": 200,
    "abstracts": 600,
}
N_TERMS = 1500

#: Parameter count of the submitted BPO network at the widths above, measured
#: from a verbatim transcription of the original notebook's cell.
SUBMITTED_PARAM_COUNT = 8_187_152


@pytest.fixture(scope="module")
def config():
    return load_config("BPO", data_root="/tmp/unused")


@pytest.fixture
def model(config):
    tf.keras.backend.clear_session()
    return build_model(DIMS, N_TERMS, config.training, config.feature_specs)


def test_parameter_count_matches_the_submitted_model(model):
    assert model.count_params() == SUBMITTED_PARAM_COUNT


def test_one_tower_per_feature_with_the_expected_layers(model):
    from collections import Counter

    counts = Counter(type(layer).__name__ for layer in model.layers)
    n_features = len(DIMS)
    assert counts["InputLayer"] == n_features
    assert counts["Dropout"] == n_features
    assert counts["LeakyReLU"] == n_features
    # One BatchNormalization per tower, plus one after the concatenation.
    assert counts["BatchNormalization"] == n_features + 1
    # Two Dense per tower, plus the trunk and the output layer.
    assert counts["Dense"] == 2 * n_features + 2
    assert counts["Concatenate"] == 1


def test_input_order_follows_the_config(model, config):
    # Checkpoints are positional, so this order must not drift.
    expected = [DIMS[spec.key] for spec in config.feature_specs]
    assert [int(tensor.shape[-1]) for tensor in model.inputs] == expected


def test_output_is_one_probability_per_term(model):
    assert tuple(model.outputs[0].shape) == (None, N_TERMS)
    output_layer = model.get_layer("term_probabilities")
    assert output_layer.activation is tf.keras.activations.sigmoid


def test_tower_widths_come_from_the_feature_spec(model):
    # Language models get the wide tower; taxonomy and abstracts get their own.
    assert model.get_layer("t5_dense_1").units == 600
    assert model.get_layer("t5_dense_2").units == 300
    assert model.get_layer("taxonomy_dense_1").units == 128
    assert model.get_layer("taxonomy_dense_2").units == 64
    assert model.get_layer("abstracts_dense_1").units == 200


def test_missing_feature_dimension_is_rejected(config):
    with pytest.raises(ValueError, match="no embedding width"):
        build_model({"t5": 1024}, N_TERMS, config.training, config.feature_specs)


def test_ablation_to_a_single_feature_still_builds():
    config = load_config(
        "CCO", data_root="/tmp/unused", overrides={"training": {"features": ["t5"]}}
    )
    tf.keras.backend.clear_session()
    model = build_model({"t5": 1024}, 800, config.training, config.feature_specs)
    # With one tower there is nothing to concatenate.
    assert not any(type(layer).__name__ == "Concatenate" for layer in model.layers)
    assert tuple(model.outputs[0].shape) == (None, 800)


def test_forward_pass_and_one_training_step(model, config):
    compile_model(model, config.training)
    rng = np.random.default_rng(0)
    inputs = [
        rng.random((8, DIMS[spec.key]), dtype=np.float32)
        for spec in config.feature_specs
    ]
    targets = (rng.random((8, N_TERMS)) > 0.9).astype(np.float32)

    predictions = model.predict(inputs, verbose=0)
    assert predictions.shape == (8, N_TERMS)
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0

    history = model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_step_decay_halves_every_ten_epochs(config):
    schedule = step_decay(config.training)
    assert schedule(0, 0.0003) == pytest.approx(0.0003)
    assert schedule(9, 0.0003) == pytest.approx(0.0003)
    assert schedule(10, 0.0003) == pytest.approx(0.00015)
    assert schedule(20, 0.00015) == pytest.approx(0.000075)
