"""Ensemble inference: average every cross-validation member's predictions."""

from __future__ import annotations

import gc
import logging

import numpy as np

from .config import RunConfig
from .data import load_features, load_targets, load_test_protein_ids
from .submission import write_submission

log = logging.getLogger(__name__)


def predict(config: RunConfig, *, batch_size: int = 256) -> np.ndarray:
    """Average predictions over all checkpoints for this ontology.

    Models are loaded and released one at a time -- 25 of them will not fit in
    memory together -- and the running sum is accumulated in float32.
    """
    from tensorflow.keras import backend as keras_backend
    from tensorflow.keras.models import load_model

    features = load_features(config.feature_specs, config.paths, "test", mmap=False)
    inputs = [features[spec.key] for spec in config.feature_specs]

    total: np.ndarray | None = None
    n_models = 0
    for seed in config.training.seeds:
        for fold in range(1, config.training.folds + 1):
            path = config.checkpoint(fold, seed)
            if not path.exists():
                log.warning("skipping missing checkpoint %s", path)
                continue

            model = load_model(str(path))
            predictions = model.predict(inputs, batch_size=batch_size).astype(
                np.float32
            )
            total = predictions if total is None else total + predictions
            n_models += 1
            log.info("averaged %d models so far (latest: %s)", n_models, path.name)

            keras_backend.clear_session()
            gc.collect()

    if total is None:
        raise FileNotFoundError(
            f"no checkpoints found under {config.paths.models} for "
            f"version {config.version!r} and ontology {config.ontology.name}"
        )

    return total / n_models


def predict_and_write(
    config: RunConfig, *, top_fraction: float = 0.05, batch_size: int = 256
):
    """Run ensemble inference and write this ontology's submission TSV."""
    predictions = predict(config, batch_size=batch_size)
    targets = load_targets(config.ontology, config.paths)
    protein_ids = load_test_protein_ids(config.paths)
    return write_submission(
        predictions,
        targets.term_ids,
        protein_ids,
        config.submission(),
        top_fraction=top_fraction,
    )
