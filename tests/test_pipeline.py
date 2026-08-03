"""End-to-end pipeline test on a tiny synthetic dataset.

Exercises the real ``train`` -> ``predict`` -> ``merge`` path, so that the
plumbing between config, data loading, checkpoint naming and submission writing
is checked rather than each part in isolation. Uses two small features, a handful
of proteins, and one epoch -- this asserts nothing about accuracy, only that the
pipeline runs and produces well-formed output.
"""

import numpy as np
import pandas as pd
import pytest

from protgoat.config import load_config
from protgoat.submission import merge_submissions

pytest.importorskip("tensorflow", reason="requires the [train] extra")

N_TRAIN, N_TEST, N_TERMS = 40, 12, 6
DIMS = {"t5": 16, "taxonomy": 5}

OVERRIDES = {
    "training": {
        "epochs": 1,
        "batch_size": 8,
        "folds": 2,
        "seeds": [1],
        "features": ["t5", "taxonomy"],
        "trunk_dim": 8,
    }
}


@pytest.fixture
def dataset(tmp_path):
    """A miniature but structurally faithful CAFA5 data root."""
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        (tmp_path / split).mkdir()

    # Every protein has at least one annotation except the last two, which
    # exercise the drop-unannotated path.
    y = (rng.random((N_TRAIN, N_TERMS)) > 0.6).astype(np.float32)
    y[0] = 0
    y[1] = 0
    y[2:, 0] = 1
    np.save(tmp_path / "train" / "Y.npy", y)
    np.save(
        tmp_path / "train" / "Y_labels.npy",
        np.array([f"GO:{i:07d}" for i in range(N_TERMS)], dtype=object),
    )
    pd.DataFrame({"IA_weight": rng.random(N_TERMS) + 0.5}).to_csv(
        tmp_path / "train" / "weights.csv", index=False
    )

    for key, dim in DIMS.items():
        np.save(
            tmp_path / "train" / f"{key}_train.npy",
            rng.random((N_TRAIN, dim)).astype(np.float32),
        )
        np.save(
            tmp_path / "test" / f"{key}_test.npy",
            rng.random((N_TEST, dim)).astype(np.float32),
        )

    np.save(
        tmp_path / "test" / "t5_labels_sorted.npy",
        np.array([f"P{i:05d}" for i in range(N_TEST)], dtype=object),
    )
    return tmp_path


@pytest.fixture
def configs(dataset, tmp_path, monkeypatch):
    """Point the feature specs at the synthetic files and build per-ontology configs."""
    from protgoat import config as config_module

    specs = tuple(
        config_module.FeatureSpec(
            key=key,
            train_file=f"{key}_train.npy",
            test_file=f"{key}_test.npy",
            hidden=(8, 4),
            description=key,
        )
        for key in DIMS
    )
    monkeypatch.setattr(config_module, "FEATURES", specs)
    monkeypatch.setattr(
        config_module, "FEATURES_BY_KEY", {spec.key: spec for spec in specs}
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("version: test\n")
    built = {}
    for ontology in ("BPO", "CCO", "MFO"):
        (config_dir / f"{ontology.lower()}.yaml").write_text(
            f"ontology:\n"
            f"  name: {ontology}\n"
            f"  n_terms: {N_TERMS}\n"
            f"  targets: Y.npy\n"
            f"  term_ids: Y_labels.npy\n"
            f"  ia_weights: weights.csv\n"
            f"  dag_matrix: dag.npy\n"
        )
        built[ontology] = load_config(
            ontology,
            data_root=dataset,
            config_dir=config_dir,
            overrides=OVERRIDES,
        )
    return built


def test_train_writes_one_checkpoint_per_fold_and_seed(configs):
    from protgoat.train import train

    config = configs["BPO"]
    results = train(config)

    assert len(results) == config.training.folds * len(config.training.seeds)
    for result in results:
        assert result.checkpoint.exists(), f"missing {result.checkpoint}"
        assert config.history(result.fold, result.seed).exists()
        assert np.isfinite(result.best_val_loss)
    # Checkpoint names must be unique per ensemble member.
    assert len({r.checkpoint for r in results}) == len(results)


def test_predict_averages_the_ensemble_into_test_shaped_output(configs):
    from protgoat.predict import predict
    from protgoat.train import train

    config = configs["CCO"]
    train(config)
    predictions = predict(config)

    assert predictions.shape == (N_TEST, N_TERMS)
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0
    assert np.isfinite(predictions).all()


def test_predict_without_checkpoints_reports_where_it_looked(configs):
    from protgoat.predict import predict

    with pytest.raises(FileNotFoundError, match="no checkpoints found"):
        predict(configs["MFO"])


def test_full_run_produces_a_merged_three_column_submission(configs):
    from protgoat.predict import predict_and_write
    from protgoat.train import train

    for ontology in ("BPO", "CCO", "MFO"):
        config = configs[ontology]
        train(config)
        frame = predict_and_write(config, top_fraction=0.5)
        assert len(frame) > 0
        assert config.submission().exists()

    output = configs["BPO"].paths.submissions / "test_all.tsv"
    merged = merge_submissions(
        [configs[o].submission() for o in ("BPO", "CCO", "MFO")], output
    )

    assert output.exists()
    reread = pd.read_csv(output, sep="\t", header=None)
    assert reread.shape[1] == 3
    assert len(reread) == len(merged)
    # Every predicted protein must be one of the test proteins, and every score
    # a valid CAFA confidence.
    assert set(reread[0]) <= {f"P{i:05d}" for i in range(N_TEST)}
    assert reread[2].between(0, 1).all()
