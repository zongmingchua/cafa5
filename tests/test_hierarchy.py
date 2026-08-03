"""The experimental hierarchy-aware loss.

Kept under test even though the submitted model did not use it, so the recorded
negative result stays reproducible rather than bit-rotting.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="requires the [train] extra")

from protgoat.hierarchy import constrained_output, make_mcm_loss  # noqa: E402

# Three terms in a chain: 0 is the parent of 1, which is the parent of 2.
# ancestry[i, j] == 1 when j is a descendant of, or equal to, i.
CHAIN = np.array(
    [
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def test_a_parent_inherits_its_descendants_highest_score():
    scores = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)

    out = constrained_output(scores, CHAIN).numpy()

    # Every ancestor of the confident leaf is raised to the leaf's score.
    assert out[0].tolist() == pytest.approx([0.9, 0.9, 0.9])


def test_scores_are_never_lowered():
    scores = np.array([[0.8, 0.3, 0.1]], dtype=np.float32)

    out = constrained_output(scores, CHAIN).numpy()

    assert (out[0] >= scores[0]).all()
    # The leaf has no descendants, so it is unchanged.
    assert out[0][2] == pytest.approx(0.1)


def test_output_is_coherent_parents_score_at_least_their_children():
    rng = np.random.default_rng(0)
    scores = rng.random((16, 3)).astype(np.float32)

    out = constrained_output(scores, CHAIN).numpy()

    assert (out[:, 0] >= out[:, 1] - 1e-6).all()
    assert (out[:, 1] >= out[:, 2] - 1e-6).all()


def test_identity_hierarchy_is_a_no_op():
    scores = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    out = constrained_output(scores, np.eye(3, dtype=np.float32)).numpy()
    assert out[0].tolist() == pytest.approx(scores[0].tolist())


def test_batch_shape_is_preserved():
    scores = np.zeros((7, 3), dtype=np.float32)
    assert constrained_output(scores, CHAIN).shape == (7, 3)


def test_mcm_loss_is_finite_and_penalises_incoherent_predictions():
    loss = make_mcm_loss(CHAIN)
    y_true = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    coherent = float(np.mean(loss(y_true, np.array([[0.9, 0.9, 0.9]], np.float32))))
    incoherent = float(np.mean(loss(y_true, np.array([[0.1, 0.1, 0.9]], np.float32))))

    assert np.isfinite(coherent) and np.isfinite(incoherent)
    # A confident leaf under an unconfident parent is the case the constraint
    # exists to reward, so it should not score worse than the coherent one.
    assert coherent <= incoherent + 1e-6


def test_mcm_loss_returns_one_value_per_example():
    loss = make_mcm_loss(CHAIN)
    rng = np.random.default_rng(1)
    y_true = (rng.random((5, 3)) > 0.5).astype(np.float32)
    y_pred = rng.random((5, 3)).astype(np.float32)

    values = loss(y_true, y_pred).numpy()

    assert values.shape == (5,)
    assert np.isfinite(values).all()
