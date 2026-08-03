"""Cross-validated training of the PROTGOAT ensemble.

The submitted model is an average over ``folds * len(seeds)`` networks: repeated
5-fold cross-validation with five different fold partitions.  Each member is
saved to its own checkpoint and reloaded at inference time.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RunConfig
from .data import Targets, load_training_set
from .model import build_model, compile_model, step_decay

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    seed: int
    fold: int
    checkpoint: Path
    best_val_loss: float


def _feature_dims(features: dict[str, np.ndarray]) -> dict[str, int]:
    return {key: int(array.shape[1]) for key, array in features.items()}


def _inputs_for(
    features: dict[str, np.ndarray], keys: tuple[str, ...], index: np.ndarray
) -> list[np.ndarray]:
    """Slice each feature matrix to ``index``, preserving model input order."""
    return [np.asarray(features[key])[index] for key in keys]


def train_fold(
    config: RunConfig,
    targets: Targets,
    features: dict[str, np.ndarray],
    train_index: np.ndarray,
    val_index: np.ndarray,
    *,
    seed: int,
    fold: int,
) -> FoldResult:
    """Train one cross-validation member and checkpoint its best epoch."""
    from tensorflow.keras import backend as keras_backend
    from tensorflow.keras.callbacks import (
        CSVLogger,
        LearningRateScheduler,
        ModelCheckpoint,
    )

    specs = config.feature_specs
    keys = tuple(spec.key for spec in specs)
    checkpoint_path = config.checkpoint(fold, seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = compile_model(
        build_model(_feature_dims(features), targets.n_terms, config.training, specs),
        config.training,
    )

    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(str(config.history(fold, seed))),
        LearningRateScheduler(step_decay(config.training), verbose=1),
    ]

    history = model.fit(
        _inputs_for(features, keys, train_index),
        targets.y[train_index],
        validation_data=(
            _inputs_for(features, keys, val_index),
            targets.y[val_index],
        ),
        class_weight=targets.weights if config.training.use_ia_weights else None,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        callbacks=callbacks,
    )

    best_val_loss = float(min(history.history["val_loss"]))

    # Keras leaks graph state across many sequential fits; clearing the session
    # between folds keeps memory flat over the full 25-model run.
    keras_backend.clear_session()
    gc.collect()

    return FoldResult(
        seed=seed, fold=fold, checkpoint=checkpoint_path, best_val_loss=best_val_loss
    )


def train(config: RunConfig) -> list[FoldResult]:
    """Train the full ensemble for one sub-ontology."""
    from sklearn.model_selection import KFold

    targets, features = load_training_set(config)
    n_proteins = targets.y.shape[0]
    results: list[FoldResult] = []

    for seed in config.training.seeds:
        splitter = KFold(
            n_splits=config.training.folds, shuffle=True, random_state=seed
        )
        for fold, (train_index, val_index) in enumerate(
            splitter.split(np.arange(n_proteins)), start=1
        ):
            log.info(
                "%s seed %d fold %d/%d",
                config.ontology.name,
                seed,
                fold,
                config.training.folds,
            )
            results.append(
                train_fold(
                    config,
                    targets,
                    features,
                    train_index,
                    val_index,
                    seed=seed,
                    fold=fold,
                )
            )

    log.info(
        "trained %d models; mean best val_loss %.5f",
        len(results),
        float(np.mean([r.best_val_loss for r in results])),
    )
    return results
