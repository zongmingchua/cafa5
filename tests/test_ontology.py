"""OBO parsing: domain mapping, term names, obsolete-term handling."""

import pytest

from protgoat.ontology import term_names, term_ontologies

OBO = """format-version: 1.2
data-version: releases/2023-01-01

[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process

[Term]
id: GO:0000002
name: mitochondrial genome maintenance
namespace: cellular_component

[Term]
id: GO:0000003
name: reproduction
namespace: molecular_function

[Term]
id: GO:0000004
name: ancient thing
namespace: biological_process
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture
def obo_file(tmp_path):
    path = tmp_path / "go-basic.obo"
    path.write_text(OBO)
    return path


def test_namespaces_map_to_cafa_abbreviations(obo_file):
    assert term_ontologies(obo_file) == {
        "GO:0000001": "BPO",
        "GO:0000002": "CCO",
        "GO:0000003": "MFO",
    }


def test_obsolete_terms_are_excluded(obo_file):
    assert "GO:0000004" not in term_ontologies(obo_file)
    assert "ancient thing" not in term_names(obo_file)


def test_typedef_stanzas_are_not_treated_as_terms(obo_file):
    assert "part of" not in term_names(obo_file)


def test_term_names_are_returned_in_file_order(obo_file):
    assert term_names(obo_file) == [
        "mitochondrion inheritance",
        "mitochondrial genome maintenance",
        "reproduction",
    ]


def test_file_without_terms_yields_empty_results(tmp_path):
    path = tmp_path / "empty.obo"
    path.write_text("format-version: 1.2\n")
    assert term_ontologies(path) == {}
    assert term_names(path) == []
