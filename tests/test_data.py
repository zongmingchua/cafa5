"""Target/feature loading, shape validation, and unannotated-row filtering."""

import numpy as np
import pandas as pd
import pytest

from protgoat.config import DataPaths, FeatureSpec, OntologySpec
from protgoat.data import (
    Targets,
    drop_unannotated,
    load_features,
    load_targets,
)

SPEC = OntologySpec(
    name="CCO",
    n_terms=3,
    targets="Y.npy",
    term_ids="Y_labels.npy",
    ia_weights="weights.csv",
    dag_matrix="dag.npy",
)


@pytest.fixture
def paths(tmp_path):
    """A data root holding a tiny but internally consistent CCO dataset."""
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    np.save(tmp_path / "train" / "Y.npy", np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1]]))
    np.save(
        tmp_path / "train" / "Y_labels.npy",
        np.array(["GO:1", "GO:2", "GO:3"], dtype=object),
    )
    pd.DataFrame({"IA_weight": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / "train" / "weights.csv", index=False
    )
    return DataPaths.resolve(tmp_path)


def _feature(key, filename, rows, cols):
    return FeatureSpec(
        key=key, train_file=filename, test_file=filename, hidden=(4, 2), description=key
    ), np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


def test_load_targets_reads_matrix_labels_and_weights(paths):
    targets = load_targets(SPEC, paths)
    assert targets.y.shape == (3, 3)
    assert targets.n_terms == 3
    assert list(targets.term_ids) == ["GO:1", "GO:2", "GO:3"]
    # Keras' class_weight wants {column_index: weight}.
    assert targets.weights == {0: 1.0, 1: 2.0, 2: 3.0}


def test_missing_file_points_at_the_data_docs(paths):
    spec = OntologySpec(
        name="CCO",
        n_terms=3,
        targets="absent.npy",
        term_ids="Y_labels.npy",
        ia_weights="weights.csv",
        dag_matrix="dag.npy",
    )
    with pytest.raises(FileNotFoundError, match="docs/data.md"):
        load_targets(spec, paths)


def test_weights_of_the_wrong_length_are_rejected(paths):
    pd.DataFrame({"IA_weight": [1.0, 2.0]}).to_csv(
        paths.train / "weights.csv", index=False
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_targets(SPEC, paths)


def test_weights_column_must_be_present(paths):
    pd.DataFrame({"wrong_name": [1.0, 2.0, 3.0]}).to_csv(
        paths.train / "weights.csv", index=False
    )
    with pytest.raises(ValueError, match="IA_weight"):
        load_targets(SPEC, paths)


def test_load_features_returns_one_array_per_spec(paths):
    spec_a, array_a = _feature("t5", "a.npy", 3, 4)
    spec_b, array_b = _feature("ankh", "b.npy", 3, 2)
    np.save(paths.train / "a.npy", array_a)
    np.save(paths.train / "b.npy", array_b)

    features = load_features((spec_a, spec_b), paths, "train")

    assert set(features) == {"t5", "ankh"}
    assert features["t5"].shape == (3, 4)
    assert features["ankh"].shape == (3, 2)


def test_features_disagreeing_on_protein_count_are_rejected(paths):
    spec_a, array_a = _feature("t5", "a.npy", 3, 4)
    spec_b, array_b = _feature("ankh", "b.npy", 2, 2)
    np.save(paths.train / "a.npy", array_a)
    np.save(paths.train / "b.npy", array_b)

    with pytest.raises(ValueError, match="disagree on protein count"):
        load_features((spec_a, spec_b), paths, "train")


def test_drop_unannotated_removes_all_zero_target_rows():
    targets = Targets(
        y=np.array([[1, 0], [0, 0], [0, 1]]),
        term_ids=np.array(["GO:1", "GO:2"]),
        weights={0: 1.0, 1: 1.0},
    )
    features = {"t5": np.array([[10.0], [20.0], [30.0]])}

    kept_targets, kept_features = drop_unannotated(targets, features)

    assert kept_targets.y.shape == (2, 2)
    # The surviving feature rows must be the ones whose targets survived.
    assert list(kept_features["t5"].ravel()) == [10.0, 30.0]
    assert list(kept_targets.term_ids) == ["GO:1", "GO:2"]


def test_drop_unannotated_is_a_no_op_when_everything_is_annotated():
    targets = Targets(
        y=np.array([[1, 0], [0, 1]]),
        term_ids=np.array(["GO:1", "GO:2"]),
        weights={0: 1.0, 1: 1.0},
    )
    kept, features = drop_unannotated(targets, {"t5": np.zeros((2, 3))})
    assert kept.y.shape == (2, 2)
    assert features["t5"].shape == (2, 3)
