"""Turning a dense probability matrix into a CAFA submission file.

CAFA expects a three-column TSV of ``protein <tab> GO term <tab> score``.  A
full prediction matrix is ``n_proteins x n_terms`` -- around 200 million rows for
BPO -- so the writer keeps only the highest-scoring cells.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ("protein_id", "go_term", "score")

#: Scores at or below this are treated as "no prediction".
MIN_SCORE = 1e-4


def to_long_frame(
    predictions: np.ndarray,
    term_ids: Sequence[str],
    protein_ids: Sequence[str],
) -> pd.DataFrame:
    """Flatten a ``(n_proteins, n_terms)`` matrix into one row per (protein, term)."""
    n_proteins, n_terms = predictions.shape
    if len(protein_ids) != n_proteins:
        raise ValueError(
            f"{len(protein_ids)} protein ids for {n_proteins} prediction rows"
        )
    if len(term_ids) != n_terms:
        raise ValueError(f"{len(term_ids)} term ids for {n_terms} prediction columns")

    return pd.DataFrame(
        {
            "protein_id": np.repeat(np.asarray(protein_ids), n_terms),
            "go_term": np.tile(np.asarray(term_ids), n_proteins),
            "score": predictions.ravel(),
        }
    )


def write_submission(
    predictions: np.ndarray,
    term_ids: Sequence[str],
    protein_ids: Sequence[str],
    path: str | Path,
    *,
    top_fraction: float = 0.05,
    min_score: float = MIN_SCORE,
    decimals: int = 3,
) -> pd.DataFrame:
    """Write the top-scoring predictions as a headerless CAFA TSV.

    Args:
        predictions: ``(n_proteins, n_terms)`` probabilities.
        term_ids: GO accessions, one per column.
        protein_ids: Protein accessions, one per row.
        path: Destination TSV.
        top_fraction: Keep this fraction of all (protein, term) cells, ranked by
            score globally rather than per protein. 0.05 was used for the
            submitted run; it comfortably covers CAFA's per-protein term cap
            while keeping the file a manageable size.
        min_score: Drop predictions at or below this score.
        decimals: Rounding applied to scores, matching CAFA's precision.

    Returns:
        The frame that was written, for inspection in a notebook.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    frame = to_long_frame(predictions, term_ids, protein_ids)
    n_keep = max(1, int(top_fraction * len(frame)))
    frame = frame.nlargest(n_keep, "score")

    n_below = int((frame["score"] <= min_score).sum())
    if n_below:
        log.info("dropping %d predictions at or below %g", n_below, min_score)
        frame = frame[frame["score"] > min_score]

    frame["score"] = frame["score"].round(decimals)
    frame = frame[frame["score"] > 0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", header=False, index=False)
    log.info("wrote %d predictions to %s", len(frame), path)
    return frame


def merge_submissions(paths: Iterable[str | Path], output: str | Path) -> pd.DataFrame:
    """Concatenate the three per-ontology TSVs into one submission, sorted by protein.

    CAFA scores a single file covering all three GO domains; each domain is
    predicted by its own model, so the per-ontology outputs are concatenated here.
    """
    frames = [
        pd.read_csv(path, sep="\t", header=None, names=COLUMNS) for path in paths
    ]
    if not frames:
        raise ValueError("no submission files given")

    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["protein_id", "go_term"], kind="stable"
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, sep="\t", header=False, index=False)
    log.info("merged %d files into %s (%d rows)", len(frames), output, len(merged))
    return merged
