"""Config loading: YAML merge, validation, and derived paths."""

from pathlib import Path

import pytest

from protgoat.config import CONFIG_DIR, FEATURES, ONTOLOGIES, DataPaths, load_config


def test_bundled_configs_ship_inside_the_package():
    # Regression guard: CONFIG_DIR must resolve within the installed package, or
    # a non-editable install cannot find its YAML files at all.
    assert CONFIG_DIR.is_dir(), f"{CONFIG_DIR} is missing"
    assert CONFIG_DIR.parent.name == "protgoat"
    expected = {"default.yaml", "bpo.yaml", "cco.yaml", "mfo.yaml"}
    assert expected <= {path.name for path in CONFIG_DIR.iterdir()}


@pytest.mark.parametrize("ontology", ONTOLOGIES)
def test_every_ontology_config_loads(ontology):
    config = load_config(ontology, data_root="/tmp/data")
    assert config.ontology.name == ontology
    assert config.ontology.n_terms > 0
    assert config.training.folds == 5
    assert len(config.training.seeds) == 5


def test_bpo_has_the_widest_label_space():
    n_terms = {o: load_config(o, data_root="/tmp/d").ontology.n_terms for o in ONTOLOGIES}
    assert n_terms["BPO"] > n_terms["CCO"]
    assert n_terms["BPO"] > n_terms["MFO"]


def test_output_split_sums_below_label_count():
    for ontology in ONTOLOGIES:
        spec = load_config(ontology, data_root="/tmp/d").ontology
        assert sum(spec.output_split) < spec.n_terms


def test_default_features_are_all_known_and_ordered():
    config = load_config("BPO", data_root="/tmp/d")
    assert config.training.features == tuple(spec.key for spec in FEATURES)
    assert [spec.key for spec in config.feature_specs] == list(config.training.features)


def test_unknown_ontology_is_rejected():
    with pytest.raises(ValueError, match="ontology must be one of"):
        load_config("XYZ", data_root="/tmp/d")


def test_unknown_training_key_is_rejected(tmp_path):
    (tmp_path / "default.yaml").write_text("training:\n  learning_rate_typo: 1\n")
    (tmp_path / "bpo.yaml").write_text("ontology:\n  name: BPO\n  n_terms: 1\n")
    with pytest.raises(ValueError, match="unknown training keys"):
        load_config("BPO", data_root="/tmp/d", config_dir=tmp_path)


def test_unknown_feature_key_is_rejected(tmp_path):
    (tmp_path / "default.yaml").write_text("training:\n  features: [t5, nonesuch]\n")
    (tmp_path / "bpo.yaml").write_text("ontology:\n  name: BPO\n  n_terms: 1\n")
    with pytest.raises(ValueError, match="unknown feature keys"):
        load_config("BPO", data_root="/tmp/d", config_dir=tmp_path)


def test_overrides_win_over_yaml():
    config = load_config(
        "MFO", data_root="/tmp/d", overrides={"training": {"epochs": 3, "folds": 2}}
    )
    assert config.training.epochs == 3
    assert config.training.folds == 2
    # Untouched values still come from the YAML.
    assert config.training.batch_size == 256


def test_feature_ablation_via_overrides():
    config = load_config(
        "CCO", data_root="/tmp/d", overrides={"training": {"features": ["t5", "ankh"]}}
    )
    assert [spec.key for spec in config.feature_specs] == ["t5", "ankh"]


def test_data_root_from_environment(monkeypatch):
    monkeypatch.setenv("PROTGOAT_DATA_ROOT", "/mnt/cafa")
    assert DataPaths.resolve().root == Path("/mnt/cafa")
    # An explicit argument still takes precedence.
    assert DataPaths.resolve("/other").root == Path("/other")


def test_checkpoint_and_submission_paths_are_distinct_per_member():
    config = load_config("BPO", data_root="/tmp/d", version="v9")
    first = config.checkpoint(fold=1, seed=1)
    second = config.checkpoint(fold=1, seed=2)
    assert first != second
    assert first.name == "v9_BPO_fc1_r1.hdf5"
    assert config.history(1, 1).suffix == ".csv"
    assert config.submission().name == "v9_BPO.tsv"


def test_split_dir_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        DataPaths.resolve("/tmp/d").split_dir("validation")
