"""Submission formatting: flattening, top-k filtering, and merging."""

import numpy as np
import pandas as pd
import pytest

from protgoat.submission import (
    COLUMNS,
    merge_submissions,
    to_long_frame,
    write_submission,
)

TERMS = ["GO:0000001", "GO:0000002", "GO:0000003"]
PROTEINS = ["P1", "P2"]
SCORES = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.2]])


def test_long_frame_pairs_every_protein_with_every_term():
    frame = to_long_frame(SCORES, TERMS, PROTEINS)
    assert len(frame) == len(PROTEINS) * len(TERMS)
    # Row order must be protein-major so that ravel() lines up with the labels.
    assert list(frame["protein_id"]) == ["P1"] * 3 + ["P2"] * 3
    assert list(frame["go_term"]) == TERMS * 2
    assert frame.loc[frame.index[0], "score"] == pytest.approx(0.9)


def test_long_frame_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="protein ids"):
        to_long_frame(SCORES, TERMS, ["P1"])
    with pytest.raises(ValueError, match="term ids"):
        to_long_frame(SCORES, TERMS[:2], PROTEINS)


def test_write_submission_keeps_only_the_top_fraction(tmp_path):
    path = tmp_path / "sub.tsv"
    frame = write_submission(SCORES, TERMS, PROTEINS, path, top_fraction=0.5)

    assert len(frame) == 3  # half of six cells
    assert list(frame["score"]) == sorted(frame["score"], reverse=True)
    assert set(frame["score"]) == {0.9, 0.8, 0.5}


def test_written_file_is_headerless_three_column_tsv(tmp_path):
    path = tmp_path / "sub.tsv"
    write_submission(SCORES, TERMS, PROTEINS, path, top_fraction=1.0)

    reread = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)
    assert list(reread.columns) == list(COLUMNS)
    # 0.1 survives; nothing here is below the 1e-4 floor.
    assert len(reread) == 6
    assert reread["protein_id"].iloc[0] in PROTEINS


def test_near_zero_scores_are_dropped(tmp_path):
    scores = np.array([[0.9, 1e-9, 0.0]])
    frame = write_submission(scores, TERMS, ["P1"], tmp_path / "s.tsv", top_fraction=1.0)
    assert list(frame["go_term"]) == ["GO:0000001"]


def test_rounding_removes_predictions_that_round_to_zero(tmp_path):
    scores = np.array([[0.9, 0.0004]])
    frame = write_submission(
        scores, TERMS[:2], ["P1"], tmp_path / "s.tsv", top_fraction=1.0
    )
    # 0.0004 clears the 1e-4 floor but rounds to 0.000, which CAFA would reject.
    assert list(frame["score"]) == [0.9]


def test_top_fraction_must_be_a_proper_fraction(tmp_path):
    path = tmp_path / "s.tsv"
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_fraction"):
            write_submission(SCORES, TERMS, PROTEINS, path, top_fraction=bad)


def test_output_directory_is_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "sub.tsv"
    write_submission(SCORES, TERMS, PROTEINS, path, top_fraction=1.0)
    assert path.exists()


def test_merge_concatenates_and_sorts_by_protein(tmp_path):
    first = tmp_path / "bpo.tsv"
    second = tmp_path / "cco.tsv"
    write_submission(SCORES, TERMS, ["P3", "P1"], first, top_fraction=1.0)
    write_submission(SCORES, TERMS, ["P2", "P4"], second, top_fraction=1.0)

    merged = merge_submissions([first, second], tmp_path / "all.tsv")

    assert len(merged) == 12
    assert list(merged["protein_id"]) == sorted(merged["protein_id"])
    assert (tmp_path / "all.tsv").exists()


def test_merge_requires_at_least_one_file(tmp_path):
    with pytest.raises(ValueError, match="no submission files"):
        merge_submissions([], tmp_path / "all.tsv")
