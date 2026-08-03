"""Run configuration: data locations, per-ontology settings, hyperparameters.

Everything that varied between runs in the original notebooks -- Google Drive
paths, which GO sub-ontology is being modelled, layer sizes -- is declared here
or in the bundled ``configs/*.yaml`` so the training and inference entry points
contain no hard-coded paths.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ONTOLOGIES = ("BPO", "CCO", "MFO")

#: Bundled YAML configs. Shipped as package data so that an installed (non
#: editable) copy resolves them without needing the source tree.
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: Environment variable consulted when ``--data-root`` is not given.
DATA_ROOT_ENV = "PROTGOAT_DATA_ROOT"


@dataclass(frozen=True)
class DataPaths:
    """Resolves the on-disk layout described in ``docs/data.md``."""

    root: Path

    @classmethod
    def resolve(cls, root: str | os.PathLike[str] | None = None) -> DataPaths:
        if root is None:
            root = os.environ.get(DATA_ROOT_ENV, "data")
        return cls(Path(root).expanduser())

    @property
    def train(self) -> Path:
        return self.root / "train"

    @property
    def test(self) -> Path:
        return self.root / "test"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def obo(self) -> Path:
        return self.train / "go-basic.obo"

    def split_dir(self, split: str) -> Path:
        if split == "train":
            return self.train
        if split == "test":
            return self.test
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")


@dataclass(frozen=True)
class FeatureSpec:
    """One input modality: where its embedding matrix lives and its tower size.

    ``hidden`` is the (wide, narrow) pair of ``Dense`` widths in the per-modality
    tower.  The pretrained protein language models all get the same capacity; the
    taxonomy one-hot and the sparse TF-IDF abstract vectors get their own.
    """

    key: str
    train_file: str
    test_file: str
    hidden: tuple[int, int]
    description: str


#: Input modalities in the exact order the 4th-place model consumed them.
#: The ordering is load-bearing: saved ``.hdf5`` models expect this input order.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        key="t5",
        train_file="t5_train_data_sorted_f32.npy",
        test_file="t5_test_data_sorted_f32.npy",
        hidden=(600, 300),
        description="ProtT5-XL mean-pooled per-residue embeddings",
    ),
    FeatureSpec(
        key="esm2_small",
        train_file="esm2_train_data_sorted_f32.npy",
        test_file="esm2_test_data_sorted_f32.npy",
        hidden=(600, 300),
        description="ESM-2 650M mean-pooled embeddings",
    ),
    FeatureSpec(
        key="esm2_3b",
        train_file="ESM2_3B_train_embeddings_sorted.npy",
        test_file="ESM2_3B_test_embeddings_sorted.npy",
        hidden=(600, 300),
        description="ESM-2 3B mean-pooled embeddings",
    ),
    FeatureSpec(
        key="protbert",
        train_file="pb_train_data_sorted_f32.npy",
        test_file="pb_test_data_sorted_f32.npy",
        hidden=(600, 300),
        description="ProtBERT mean-pooled embeddings",
    ),
    FeatureSpec(
        key="ankh",
        train_file="Ankh_train_embeddings_sorted.npy",
        test_file="Ankh_test_embeddings_sorted.npy",
        hidden=(600, 300),
        description="Ankh-large mean-pooled embeddings (see embeddings/plm.py)",
    ),
    FeatureSpec(
        key="taxonomy",
        train_file="protein_taxa_matrix_train.npy",
        test_file="protein_taxa_matrix_test.npy",
        hidden=(128, 64),
        description="One-hot source-organism taxonomy",
    ),
    FeatureSpec(
        key="abstracts",
        train_file="abstract_embeds_train_sorted_remade_1_3.npy",
        test_file="abstract_embeds_test_sorted_all_remade_1_3.npy",
        hidden=(200, 200),
        description="TF-IDF over PubMed abstracts, GO-name vocabulary",
    ),
)

FEATURES_BY_KEY = {spec.key: spec for spec in FEATURES}


@dataclass(frozen=True)
class OntologySpec:
    """Per-sub-ontology label set.

    Each GO domain is modelled independently: the label space is the ``n_terms``
    most frequent terms in that domain, so the target matrices and information
    accretion weights differ per domain.
    """

    name: str
    n_terms: int
    targets: str
    term_ids: str
    ia_weights: str
    dag_matrix: str
    #: Term counts per DAG depth. Only used by the experimental hierarchy-aware
    #: loss in ``hierarchy.py``; the submitted model used plain cross-entropy.
    output_split: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and architecture hyperparameters of the submitted model."""

    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    dropout: float = 0.5
    #: Width of the shared trunk after the per-modality towers are concatenated.
    trunk_dim: int = 800
    leaky_relu_alpha: float = 0.1
    #: The learning rate is halved every ``lr_decay_every`` epochs.
    lr_decay_every: int = 10
    lr_decay_factor: float = 0.5
    #: Cross-validation: ``folds`` splits repeated once per seed, so the final
    #: ensemble averages ``folds * len(seeds)`` models.
    folds: int = 5
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    features: tuple[str, ...] = tuple(spec.key for spec in FEATURES)
    #: Weight each term's loss by its information accretion.
    use_ia_weights: bool = True


@dataclass(frozen=True)
class RunConfig:
    """A fully resolved run: which ontology, which data, which hyperparameters."""

    ontology: OntologySpec
    training: TrainConfig
    paths: DataPaths
    version: str = "f02"

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(FEATURES_BY_KEY[key] for key in self.training.features)

    def checkpoint(self, fold: int, seed: int) -> Path:
        """Path of one cross-validation member's weights."""
        return self.paths.models / (
            f"{self.version}_{self.ontology.name}_fc{fold}_r{seed}.hdf5"
        )

    def history(self, fold: int, seed: int) -> Path:
        return self.checkpoint(fold, seed).with_suffix(".csv")

    def submission(self) -> Path:
        return self.paths.submissions / f"{self.version}_{self.ontology.name}.tsv"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def load_config(
    ontology: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    config_dir: str | os.PathLike[str] | None = None,
    version: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Build a :class:`RunConfig` from ``default.yaml`` plus ``<ontology>.yaml``.

    ``overrides`` is applied last and may contain ``training`` and ``ontology``
    sub-mappings, which lets callers (and the CLIs) tweak single values without
    editing the YAML files.
    """
    ontology = ontology.upper()
    if ontology not in ONTOLOGIES:
        raise ValueError(f"ontology must be one of {ONTOLOGIES}, got {ontology!r}")

    directory = Path(config_dir) if config_dir else CONFIG_DIR
    raw = _deep_merge(
        _read_yaml(directory / "default.yaml"),
        _read_yaml(directory / f"{ontology.lower()}.yaml"),
    )
    if overrides:
        raw = _deep_merge(raw, overrides)

    training_raw = dict(raw.get("training", {}))
    for key in ("seeds", "features"):
        if key in training_raw:
            training_raw[key] = tuple(training_raw[key])
    unknown = set(training_raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown training keys in config: {sorted(unknown)}")

    ontology_raw = dict(raw.get("ontology", {}))
    ontology_raw.setdefault("name", ontology)
    if "output_split" in ontology_raw:
        ontology_raw["output_split"] = tuple(ontology_raw["output_split"])

    unknown_features = set(training_raw.get("features", ())) - set(FEATURES_BY_KEY)
    if unknown_features:
        raise ValueError(f"unknown feature keys: {sorted(unknown_features)}")

    return RunConfig(
        ontology=OntologySpec(**ontology_raw),
        training=TrainConfig(**training_raw),
        paths=DataPaths.resolve(data_root),
        version=version or raw.get("version", "f02"),
    )
