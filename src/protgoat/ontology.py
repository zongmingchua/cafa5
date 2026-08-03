"""Reading the Gene Ontology OBO file.

Only the two facts the pipeline needs are extracted: which GO domain each term
belongs to, and the term names used as the TF-IDF vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

#: OBO namespace -> CAFA sub-ontology abbreviation.
NAMESPACE_TO_ONTOLOGY = {
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "molecular_function": "MFO",
}

_TERM_STANZA = re.compile(r"^\[Term\]$(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
_ID = re.compile(r"^id: (GO:\d+)", re.MULTILINE)
_NAMESPACE = re.compile(r"^namespace: (\w+)", re.MULTILINE)
_NAME = re.compile(r"^name: (.+)$", re.MULTILINE)
_OBSOLETE = re.compile(r"^is_obsolete: true", re.MULTILINE)


def term_ontologies(obo_path: str | Path) -> dict[str, str]:
    """Map each non-obsolete GO term id to 'BPO', 'CCO' or 'MFO'."""
    content = Path(obo_path).read_text()
    mapping: dict[str, str] = {}
    for stanza in _TERM_STANZA.findall(content):
        if _OBSOLETE.search(stanza):
            continue
        term_id = _ID.search(stanza)
        namespace = _NAMESPACE.search(stanza)
        if term_id and namespace:
            ontology = NAMESPACE_TO_ONTOLOGY.get(namespace.group(1))
            if ontology:
                mapping[term_id.group(1)] = ontology
    return mapping


def term_names(obo_path: str | Path) -> list[str]:
    """Human-readable names of all non-obsolete terms, in file order."""
    content = Path(obo_path).read_text()
    names = []
    for stanza in _TERM_STANZA.findall(content):
        if _OBSOLETE.search(stanza):
            continue
        name = _NAME.search(stanza)
        if name:
            names.append(name.group(1).strip())
    return names
