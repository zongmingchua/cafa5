"""Experimental hierarchy-aware loss (max constraint module).

GO is a DAG: a protein annotated with a term is implicitly annotated with all of
that term's ancestors, so a raw sigmoid output can be inconsistent (high score
for a child, low for its parent).  The max constraint module of Giunchiglia &
Lukasiewicz, *Coherent Hierarchical Multi-Label Classification Networks*
(NeurIPS 2020) enforces coherence by replacing each term's score with the max
over its subtree.

**This was not used in the submitted model** -- it did not beat plain binary
cross-entropy on the CAFA5 validation folds. It is kept because the DAG matrices
it needs are part of the published dataset and the negative result is worth
recording. See ``docs/model.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf


def constrained_output(scores: tf.Tensor, ancestry: tf.Tensor) -> tf.Tensor:
    """Raise each term's score to the max over its descendants.

    Args:
        scores: ``(batch, n_terms)`` sigmoid outputs.
        ancestry: ``(n_terms, n_terms)`` binary matrix where ``ancestry[i, j]``
            is 1 when term *j* is a descendant of (or equal to) term *i*.

    Returns:
        ``(batch, n_terms)`` scores obeying the DAG constraint.
    """
    import tensorflow as tf

    scores = tf.cast(scores, tf.float32)
    ancestry = tf.cast(ancestry, tf.float32)
    # (batch, 1, n_terms) * (1, n_terms, n_terms) broadcasts to
    # (batch, n_terms, n_terms); the max over the last axis picks, for each term
    # i, the highest score among its descendants.
    expanded = tf.expand_dims(scores, axis=1)
    masked = expanded * tf.expand_dims(ancestry, axis=0)
    return tf.reduce_max(masked, axis=2)


def make_mcm_loss(ancestry):
    """Build the max-constraint-module binary cross-entropy loss.

    For positive labels the constraint is applied to ``y_true * y_pred``, which
    keeps the gradient on the term actually annotated rather than letting a
    high-scoring descendant absorb it; for negative labels it is applied to the
    raw prediction, penalising any descendant that fires.
    """
    import tensorflow as tf

    ancestry_tensor = tf.cast(ancestry, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        negatives = constrained_output(y_pred, ancestry_tensor)
        positives = constrained_output(y_true * y_pred, ancestry_tensor)
        coherent = (1.0 - y_true) * negatives + y_true * positives
        return tf.keras.losses.binary_crossentropy(y_true, coherent)

    loss.__name__ = "mcm_binary_crossentropy"
    return loss
