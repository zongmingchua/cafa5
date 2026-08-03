"""PROTGOAT: protein function prediction, 4th place in CAFA5 (2023).

An ensemble of multi-input neural networks that fuse five protein language model
embeddings with source-organism taxonomy and TF-IDF features from the protein's
literature, trained separately for each of the three Gene Ontology domains.

Manuscript: https://www.biorxiv.org/content/10.1101/2024.04.01.587572v1

Typical use::

    from protgoat import load_config, train, predict_and_write

    config = load_config("BPO", data_root="data")
    train(config)
    predict_and_write(config)

Only :mod:`protgoat.config`, :mod:`protgoat.data`, :mod:`protgoat.ontology` and
:mod:`protgoat.submission` are imported eagerly; the TensorFlow-dependent
training and inference entry points are resolved on first attribute access so
that the light-weight modules stay usable without a deep learning stack.
"""

from __future__ import annotations

from typing import Any

from .config import ONTOLOGIES, DataPaths, RunConfig, TrainConfig, load_config
from .data import load_features, load_targets, load_training_set
from .submission import merge_submissions, write_submission

__version__ = "1.0.0"

_LAZY = {
    "train": ("protgoat.train", "train"),
    "train_fold": ("protgoat.train", "train_fold"),
    "predict": ("protgoat.predict", "predict"),
    "predict_and_write": ("protgoat.predict", "predict_and_write"),
    "build_model": ("protgoat.model", "build_model"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        module_name, attribute = _LAZY[name]
        return getattr(import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "ONTOLOGIES",
    "DataPaths",
    "RunConfig",
    "TrainConfig",
    "__version__",
    "build_model",
    "load_config",
    "load_features",
    "load_targets",
    "load_training_set",
    "merge_submissions",
    "predict",
    "predict_and_write",
    "train",
    "train_fold",
    "write_submission",
]
