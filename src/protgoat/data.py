"""Loading of target matrices and per-modality embedding matrices.

Every matrix on disk is sorted by protein accession, so row *i* of every feature
file refers to the same protein as row *i* of the target matrix.  Nothing here
re-checks that alignment -- it is established once, when the embeddings are
generated and sorted against the reference ID order (see ``docs/data.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataPaths, FeatureSpec, OntologySpec, RunConfig

log = logging.getLogger(__name__)


@dataclass
class Targets:
    """Multi-hot GO term annotations for one sub-ontology."""

    #: ``(n_proteins, n_terms)`` multi-hot annotation matrix.
    y: np.ndarray
    #: GO term accessions, one per column of :attr:`y`.
    term_ids: np.ndarray
    #: Per-term information accretion weight, keyed by column index, in the form
    #: Keras' ``class_weight`` expects.
    weights: dict[int, float]

    @property
    def n_terms(self) -> int:
        return self.y.shape[1]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. See docs/data.md for how to fetch the "
            f"precomputed embeddings and target matrices."
        )
    return path


def load_targets(spec: OntologySpec, paths: DataPaths) -> Targets:
    """Load the target matrix, term accessions and IA weights for one ontology."""
    y = np.load(_require(paths.train / spec.targets), allow_pickle=True)
    term_ids = np.load(_require(paths.train / spec.term_ids), allow_pickle=True)
    weights_df = pd.read_csv(_require(paths.train / spec.ia_weights))

    if "IA_weight" not in weights_df.columns:
        raise ValueError(f"{spec.ia_weights} has no 'IA_weight' column")
    if len(weights_df) != y.shape[1] or len(term_ids) != y.shape[1]:
        raise ValueError(
            f"{spec.name}: shape mismatch between targets ({y.shape[1]} terms), "
            f"term ids ({len(term_ids)}) and weights ({len(weights_df)})"
        )

    weights = {i: float(w) for i, w in enumerate(weights_df["IA_weight"].to_numpy())}
    log.info("loaded %s targets: %d proteins x %d terms", spec.name, *y.shape)
    return Targets(y=y, term_ids=term_ids, weights=weights)


def load_features(
    specs: tuple[FeatureSpec, ...],
    paths: DataPaths,
    split: str,
    *,
    mmap: bool = True,
) -> dict[str, np.ndarray]:
    """Memory-map every feature matrix for ``split`` ('train' or 'test').

    The embeddings total tens of gigabytes, so they are mapped rather than read.
    Fancy-indexing a memory-mapped array (as :func:`drop_unannotated` does)
    materialises the selection in RAM, which is what training needs.
    """
    directory = paths.split_dir(split)
    mode = "r" if mmap else None
    features: dict[str, np.ndarray] = {}
    for spec in specs:
        filename = spec.train_file if split == "train" else spec.test_file
        array = np.load(_require(directory / filename), mmap_mode=mode)
        log.info("%-11s %-6s %s", spec.key, split, array.shape)
        features[spec.key] = array
    _check_rows_agree(features, split)
    return features


def _check_rows_agree(features: dict[str, np.ndarray], split: str) -> None:
    row_counts = {key: array.shape[0] for key, array in features.items()}
    if len(set(row_counts.values())) > 1:
        raise ValueError(
            f"{split} feature matrices disagree on protein count: {row_counts}"
        )


def drop_unannotated(
    targets: Targets, features: dict[str, np.ndarray]
) -> tuple[Targets, dict[str, np.ndarray]]:
    """Drop proteins with no annotation in this sub-ontology.

    A protein annotated only in, say, BPO carries an all-zero row in the CCO
    target matrix.  Training on those rows teaches the model to predict nothing
    for perfectly well-characterised proteins, so they are removed.
    """
    keep = ~np.all(targets.y == 0, axis=1)
    n_dropped = int((~keep).sum())
    log.info(
        "dropping %d of %d proteins with no annotation in this ontology",
        n_dropped,
        keep.size,
    )
    kept_targets = Targets(
        y=targets.y[keep], term_ids=targets.term_ids, weights=targets.weights
    )
    kept_features = {key: np.asarray(array)[keep] for key, array in features.items()}
    return kept_targets, kept_features


def load_test_protein_ids(paths: DataPaths) -> np.ndarray:
    """Accessions of the test superset, in the row order of the test features."""
    return np.load(_require(paths.test / "t5_labels_sorted.npy"), allow_pickle=True)


def load_training_set(config: RunConfig) -> tuple[Targets, dict[str, np.ndarray]]:
    """Convenience wrapper: targets + features, with unannotated rows removed."""
    targets = load_targets(config.ontology, config.paths)
    features = load_features(config.feature_specs, config.paths, "train")
    if targets.y.shape[0] != next(iter(features.values())).shape[0]:
        raise ValueError(
            "target matrix and feature matrices disagree on protein count: "
            f"{targets.y.shape[0]} vs {next(iter(features.values())).shape[0]}"
        )
    return drop_unannotated(targets, features)
