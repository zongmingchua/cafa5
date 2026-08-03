"""Literature features: TF-IDF over each protein's PubMed abstracts.

The intuition is that a protein's literature already names its functions, so a
bag-of-words restricted to the vocabulary of GO term names is a cheap, strong
signal that is independent of sequence.  Restricting the vocabulary this way
keeps the matrix narrow and drops words that could not correspond to any GO term.

Not every protein has literature; those rows end up all-zero, which the model
handles as "no evidence from this modality".
"""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..ontology import term_names

log = logging.getLogger(__name__)

PROTEIN_ID_COLUMN = "Protein ID"
ABSTRACT_COLUMN = "Preprocessed Abstract"

#: Words shorter than this are dropped from the vocabulary as uninformative.
MIN_WORD_LENGTH = 3


def go_vocabulary(obo_path: str | Path) -> list[str]:
    """Build the TF-IDF vocabulary from GO term names.

    Term names are lower-cased, split on non-word characters, and filtered of
    stopwords, punctuation, digits and very short tokens. The result is sorted so
    that the column order of the TF-IDF matrix is reproducible across runs --
    the original notebook used ``set()`` iteration order, which is not.
    """
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    text = " ".join(term_names(obo_path)).lower()
    text = re.sub(r"\W", " ", text)
    tokens = re.split(r"\s+", text.strip())

    stop = set(stopwords.words("english"))
    vocabulary = {
        token
        for token in tokens
        if token not in stop
        and token not in string.punctuation
        and not token.isdigit()
        and len(token) >= MIN_WORD_LENGTH
    }
    log.info("GO name vocabulary: %d terms", len(vocabulary))
    return sorted(vocabulary)


def fit_vectorizer(abstracts: pd.DataFrame, vocabulary: Sequence[str]):
    """Fit a TF-IDF vectorizer on the union of train and test abstracts.

    Fitting on both splits is deliberate: the vocabulary is fixed in advance from
    GO, so only the IDF statistics are learned, and computing those over all
    available abstracts gives a better estimate. No label information is involved.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(vocabulary=list(vocabulary))
    vectorizer.fit(abstracts[ABSTRACT_COLUMN].fillna(""))
    return vectorizer


def combine_abstracts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Stack the train and test abstract tables, keeping one row per protein."""
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=[PROTEIN_ID_COLUMN])
    log.info(
        "combined abstracts: %d rows -> %d unique proteins", n_before, len(combined)
    )
    return combined


def align_to_reference(
    matrix, protein_ids: Sequence[str], reference_ids: Sequence[str]
) -> np.ndarray:
    """Densify a TF-IDF matrix onto the reference protein order, zero-filling gaps.

    Proteins with no abstract get an all-zero row, which is the honest encoding
    of "no literature evidence".
    """
    frame = pd.DataFrame.sparse.from_spmatrix(matrix)
    frame[PROTEIN_ID_COLUMN] = np.asarray(protein_ids)
    frame = frame.drop_duplicates(subset=[PROTEIN_ID_COLUMN]).set_index(
        PROTEIN_ID_COLUMN
    )

    aligned = frame.reindex(pd.Index(reference_ids)).fillna(0.0)
    n_missing = int(frame.index.isin(reference_ids).sum())
    log.info(
        "aligned abstracts to %d reference proteins (%d had literature)",
        len(reference_ids),
        n_missing,
    )
    return aligned.to_numpy(dtype=np.float32)


def build_abstract_features(
    train_csv: str | Path,
    test_csv: str | Path,
    obo_path: str | Path,
    train_reference_ids: Sequence[str],
    test_reference_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end abstract features for both splits, in reference row order."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    for name, frame in (("train", train), ("test", test)):
        missing = {PROTEIN_ID_COLUMN, ABSTRACT_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} abstracts CSV is missing columns: {sorted(missing)}"
            )

    vectorizer = fit_vectorizer(combine_abstracts(train, test), go_vocabulary(obo_path))

    train_matrix = vectorizer.transform(train[ABSTRACT_COLUMN].fillna(""))
    test_matrix = vectorizer.transform(test[ABSTRACT_COLUMN].fillna(""))

    return (
        align_to_reference(train_matrix, train[PROTEIN_ID_COLUMN], train_reference_ids),
        align_to_reference(test_matrix, test[PROTEIN_ID_COLUMN], test_reference_ids),
    )


def top_words(
    features: np.ndarray, row: int, vocabulary: Sequence[str], limit: int = 20
) -> list[tuple[str, float]]:
    """Highest-weighted vocabulary terms for one protein: a feature sanity check."""
    scores = features[row]
    nonzero = np.flatnonzero(scores)
    ranked = sorted(nonzero, key=lambda i: scores[i], reverse=True)[:limit]
    return [(vocabulary[i], float(scores[i])) for i in ranked]
