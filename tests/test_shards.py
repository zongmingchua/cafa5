"""Shard stitching and reference alignment for the embedding pipeline.

These guard the step that silently corrupted results in the original notebook:
if shards are concatenated in the wrong order, or one is missing, every
downstream feature row belongs to the wrong protein.
"""

import numpy as np
import pytest

from protgoat.embeddings.plm import concatenate_shards, sort_to_reference

PREFIX = "Ankh_test_embeddings"


def _write_shard(directory, start, end, values, ids):
    np.save(directory / f"{PREFIX}_{start}_{end}.npy", np.array(values, dtype=np.float32))
    np.save(directory / f"{PREFIX}_{start}_{end}.ids.npy", np.array(ids, dtype=object))


def test_shards_are_ordered_by_start_index_not_filename(tmp_path):
    # Lexicographic sorting would put "10_12" before "2_10"; numeric must win.
    _write_shard(tmp_path, 0, 2, [[1.0], [2.0]], ["P0", "P1"])
    _write_shard(tmp_path, 2, 10, [[3.0]] * 8, [f"P{i}" for i in range(2, 10)])
    _write_shard(tmp_path, 10, 12, [[4.0], [5.0]], ["P10", "P11"])

    embeddings, ids = concatenate_shards(tmp_path, prefix=PREFIX)

    assert embeddings.shape == (12, 1)
    assert list(ids[:2]) == ["P0", "P1"]
    assert list(ids[-2:]) == ["P10", "P11"]
    assert embeddings[0][0] == 1.0
    assert embeddings[-1][0] == 5.0


def test_a_gap_between_shards_is_an_error(tmp_path):
    _write_shard(tmp_path, 0, 2, [[1.0], [2.0]], ["P0", "P1"])
    _write_shard(tmp_path, 5, 7, [[3.0], [4.0]], ["P5", "P6"])

    with pytest.raises(ValueError, match="not contiguous"):
        concatenate_shards(tmp_path, prefix=PREFIX)


def test_overlapping_shards_are_an_error(tmp_path):
    _write_shard(tmp_path, 0, 4, [[1.0]] * 4, [f"P{i}" for i in range(4)])
    _write_shard(tmp_path, 2, 6, [[2.0]] * 4, [f"P{i}" for i in range(2, 6)])

    with pytest.raises(ValueError, match="not contiguous"):
        concatenate_shards(tmp_path, prefix=PREFIX)


def test_missing_shard_directory_content_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shards matching"):
        concatenate_shards(tmp_path, prefix=PREFIX)


def test_id_files_are_not_mistaken_for_embedding_shards(tmp_path):
    _write_shard(tmp_path, 0, 2, [[1.0], [2.0]], ["P0", "P1"])
    embeddings, ids = concatenate_shards(tmp_path, prefix=PREFIX)
    assert len(embeddings) == 2
    assert len(ids) == 2


def test_sort_to_reference_reorders_rows_to_match_reference_ids():
    embeddings = np.array([[1.0], [2.0], [3.0]])
    ids = np.array(["P2", "P0", "P1"], dtype=object)

    ordered = sort_to_reference(embeddings, ids, ["P0", "P1", "P2"])

    assert list(ordered.ravel()) == [2.0, 3.0, 1.0]


def test_sort_to_reference_zero_fills_proteins_without_embeddings():
    embeddings = np.array([[1.0], [2.0]])
    ids = np.array(["P0", "P1"], dtype=object)

    ordered = sort_to_reference(embeddings, ids, ["P0", "P_absent", "P1"])

    assert list(ordered.ravel()) == [1.0, 0.0, 2.0]
