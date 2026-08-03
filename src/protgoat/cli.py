"""Command line entry points.

::

    protgoat-train   --ontology BPO --data-root data
    protgoat-predict --ontology BPO --data-root data
    protgoat-merge   --data-root data
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import DATA_ROOT_ENV, ONTOLOGIES, load_config
from .submission import merge_submissions


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        help=f"Root of the data directory (default: ${DATA_ROOT_ENV}, else ./data)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Run identifier used in checkpoint and submission filenames",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory holding default.yaml and <ontology>.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def _ontology_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ontology",
        required=True,
        choices=list(ONTOLOGIES),
        help="GO domain to model (each is trained independently)",
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def train_command(argv: Sequence[str] | None = None) -> int:
    """``protgoat-train``: cross-validated training for one GO domain."""
    parser = argparse.ArgumentParser(
        prog="protgoat-train",
        description="Train the PROTGOAT ensemble for one GO domain.",
    )
    _ontology_argument(parser)
    _add_common(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Cross-validation seeds; one full fold set is trained per seed",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    overrides: dict[str, dict[str, object]] = {}
    training = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("folds", args.folds),
            ("seeds", tuple(args.seeds) if args.seeds else None),
        )
        if value is not None
    }
    if training:
        overrides["training"] = training

    config = load_config(
        args.ontology,
        data_root=args.data_root,
        config_dir=args.config_dir,
        version=args.version,
        overrides=overrides or None,
    )

    from .train import train

    results = train(config)
    print(f"trained {len(results)} models into {config.paths.models}")
    return 0


def predict_command(argv: Sequence[str] | None = None) -> int:
    """``protgoat-predict``: ensemble inference and submission TSV for one GO domain."""
    parser = argparse.ArgumentParser(
        prog="protgoat-predict",
        description="Average all checkpoints and write a per-ontology submission.",
    )
    _ontology_argument(parser)
    _add_common(parser)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.05,
        help="Fraction of the highest-scoring (protein, term) pairs to keep",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    config = load_config(
        args.ontology,
        data_root=args.data_root,
        config_dir=args.config_dir,
        version=args.version,
    )

    from .predict import predict_and_write

    frame = predict_and_write(
        config, top_fraction=args.top_fraction, batch_size=args.batch_size
    )
    print(f"wrote {len(frame)} predictions to {config.submission()}")
    return 0


def merge_command(argv: Sequence[str] | None = None) -> int:
    """``protgoat-merge``: combine the three per-ontology TSVs into one submission."""
    parser = argparse.ArgumentParser(
        prog="protgoat-merge",
        description="Merge per-ontology submissions into a single CAFA submission.",
    )
    _add_common(parser)
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV (default: <data-root>/submissions/<version>_all.tsv)",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    configs = [
        load_config(
            ontology,
            data_root=args.data_root,
            config_dir=args.config_dir,
            version=args.version,
        )
        for ontology in ONTOLOGIES
    ]
    paths = [config.submission() for config in configs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error(
            "missing per-ontology submissions: "
            + ", ".join(missing)
            + " (run protgoat-predict for each ontology first)"
        )

    output = Path(args.output) if args.output else (
        configs[0].paths.submissions / f"{configs[0].version}_all.tsv"
    )
    merged = merge_submissions(paths, output)
    print(f"wrote {len(merged)} predictions to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_command())
