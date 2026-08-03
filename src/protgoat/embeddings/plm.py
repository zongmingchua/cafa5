"""Protein language model embeddings.

Each protein is encoded to a single vector by mean-pooling the model's final
hidden states over residues.  The reference implementation is Ankh-large; the
other four modalities (ProtT5, ESM-2 650M, ESM-2 3B, ProtBERT) were produced the
same way and are published in the Kaggle dataset -- see ``docs/data.md``.

Embedding the full 142k-protein test superset takes far longer than a single
Kaggle session allows, so this runs over a ``[start, end)`` slice and appends to
its shard file every ``flush_every`` proteins.  Interrupted shards can be
resumed from the checkpoint file, and the shards are stitched together by
:func:`concatenate_shards`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Residues beyond this are truncated. Long proteins (e.g. A2ASS6, ~35k
#: residues) otherwise exhaust GPU memory on a single forward pass.
MAX_RESIDUES = 3000

SHARD_PATTERN = re.compile(r"_(\d+)[_-](\d+)\.npy$")


@dataclass
class Shard:
    """One contiguous ``[start, end)`` slice of the protein list."""

    start: int
    end: int

    @property
    def suffix(self) -> str:
        return f"{self.start}_{self.end}"


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    """Return ``(accessions, sequences)`` from a FASTA file, in file order."""
    from Bio import SeqIO

    accessions: list[str] = []
    sequences: list[str] = []
    for record in SeqIO.parse(str(path), "fasta"):
        accessions.append(record.id)
        sequences.append(str(record.seq))
    log.info("read %d sequences from %s", len(sequences), path)
    return accessions, sequences


def embed_sequences(
    sequences: Sequence[str],
    *,
    device: str = "cuda",
    max_residues: int = MAX_RESIDUES,
) -> Iterator[np.ndarray]:
    """Yield one mean-pooled Ankh-large embedding per input sequence."""
    import ankh
    import torch

    model, tokenizer = ankh.load_large_model()
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        device = "cpu"
    model = model.to(device)

    for sequence in sequences:
        residues = [list(sequence)[:max_residues]]
        encoded = tokenizer.batch_encode_plus(
            residues,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        hidden = outputs.last_hidden_state[0].cpu().numpy()
        yield hidden.mean(axis=0)


def embed_shard(
    fasta: str | Path,
    output_dir: str | Path,
    shard: Shard,
    *,
    prefix: str = "Ankh_test_embeddings",
    flush_every: int = 50,
    device: str = "cuda",
) -> Path:
    """Embed ``shard`` of ``fasta``, flushing partial results as it goes.

    Returns the path of the shard's ``.npy`` file. A sibling ``.ids.npy`` holds
    the matching accessions and a ``.checkpoint`` file records how many proteins
    of the shard are done, so a killed session can be restarted from there.
    """
    accessions, sequences = read_fasta(fasta)
    end = min(shard.end, len(sequences))
    if not 0 <= shard.start < end:
        raise ValueError(
            f"shard [{shard.start}, {shard.end}) is empty or out of range for "
            f"{len(sequences)} sequences"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_path = output_dir / f"{prefix}_{shard.suffix}.npy"
    ids_path = _ids_path(embed_path)
    checkpoint_path = output_dir / f"{prefix}_{shard.suffix}.checkpoint"

    done = 0
    if checkpoint_path.exists():
        done = int(checkpoint_path.read_text().strip())
        log.info("resuming shard %s after %d proteins", shard.suffix, done)

    pending_embeddings: list[np.ndarray] = []
    pending_ids: list[str] = []

    def flush() -> None:
        if not pending_embeddings:
            return
        _append(embed_path, np.asarray(pending_embeddings))
        _append(ids_path, np.asarray(pending_ids))
        checkpoint_path.write_text(str(done))
        pending_embeddings.clear()
        pending_ids.clear()

    start = shard.start + done
    for offset, embedding in enumerate(
        embed_sequences(sequences[start:end], device=device)
    ):
        pending_embeddings.append(embedding)
        pending_ids.append(accessions[start + offset])
        done += 1
        if done % flush_every == 0:
            flush()
    flush()

    log.info("shard %s complete: %d proteins -> %s", shard.suffix, done, embed_path)
    return embed_path


def _ids_path(embed_path: Path) -> Path:
    """Sibling accession file for a shard's embedding file."""
    return embed_path.with_name(embed_path.name[: -len(".npy")] + ".ids.npy")


def _append(path: Path, block: np.ndarray) -> None:
    """Append ``block`` to the array stored at ``path``, creating it if needed."""
    if path.exists():
        block = np.concatenate([np.load(path, allow_pickle=True), block])
    np.save(path, block)


def concatenate_shards(
    shard_dir: str | Path, *, prefix: str = "Ankh_test_embeddings"
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch every shard in ``shard_dir`` back into one array, ordered by start index.

    Replaces the original notebook's ``a1 .. a59`` variable-per-shard listing:
    shards are discovered on disk and sorted by their start offset, and a gap or
    overlap between consecutive shards is an error rather than a silent
    misalignment of every downstream row.
    """
    shard_dir = Path(shard_dir)
    found: list[tuple[int, int, Path]] = []
    for path in shard_dir.glob(f"{prefix}_*.npy"):
        if path.name.endswith(".ids.npy"):
            continue
        match = SHARD_PATTERN.search(path.name)
        if not match:
            log.warning("ignoring unparseable shard name %s", path.name)
            continue
        found.append((int(match.group(1)), int(match.group(2)), path))

    if not found:
        raise FileNotFoundError(f"no shards matching {prefix}_*.npy in {shard_dir}")
    found.sort()

    pairs = zip(found, found[1:], strict=False)
    for (_, prev_end, prev_path), (start, _, path) in pairs:
        if start != prev_end:
            raise ValueError(
                f"shards are not contiguous: {prev_path.name} ends at {prev_end} "
                f"but {path.name} starts at {start}"
            )

    embeddings = np.concatenate([np.load(p, allow_pickle=True) for *_, p in found])
    ids = np.concatenate(
        [np.load(_ids_path(p), allow_pickle=True) for *_, p in found]
    )
    if len(embeddings) != len(ids):
        raise ValueError(
            f"{len(embeddings)} embeddings but {len(ids)} ids across shards"
        )
    log.info("concatenated %d shards into %d embeddings", len(found), len(embeddings))
    return embeddings, ids


def sort_to_reference(
    embeddings: np.ndarray, ids: np.ndarray, reference_ids: Sequence[str]
) -> np.ndarray:
    """Reorder ``embeddings`` to match ``reference_ids``, zero-filling the missing.

    All feature matrices must share one row order for the multi-input model to
    line up, so every modality is projected onto the same reference accession list.
    """
    position = {accession: i for i, accession in enumerate(ids)}
    output = np.zeros((len(reference_ids), embeddings.shape[1]), dtype=embeddings.dtype)
    missing = 0
    for row, accession in enumerate(reference_ids):
        index = position.get(accession)
        if index is None:
            missing += 1
            continue
        output[row] = embeddings[index]
    if missing:
        log.warning(
            "%d of %d reference proteins had no embedding",
            missing,
            len(reference_ids),
        )
    return output
