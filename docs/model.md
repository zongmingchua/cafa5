# Model

## Why fuse modalities

Protein function is predictable from sequence, but different pretrained language
models make different mistakes, and some function is not recoverable from
sequence at all. PROTGOAT combines three kinds of evidence:

- **Sequence** — five protein language models (ProtT5-XL, ESM-2 650M, ESM-2 3B,
  ProtBERT, Ankh-large), each mean-pooled over residues into one vector per
  protein. Their errors are correlated but not identical, so the ensemble of
  representations beats any single one.
- **Taxonomy** — a one-hot encoding of the source organism. A strong prior:
  which functions are plausible at all depends on the organism.
- **Literature** — TF-IDF over the protein's PubMed abstracts, restricted to the
  vocabulary of GO term names. Where a protein has been studied, its papers tend
  to name its functions outright; this signal is independent of sequence.

## Architecture

Each modality gets its own two-layer tower before anything is combined, because
the embeddings differ by an order of magnitude in both width and variance —
concatenating them raw would let the widest one dominate. Each tower normalises
its own input scale, and only then are they fused:

```
t5         (1024) -> Dense(600) -> BN -> LeakyReLU(0.1) -> Dropout(0.5) -> Dense(300) --.
esm2_small (1280) -> Dense(600) -> ...                                 -> Dense(300) --|
esm2_3b    (2560) -> Dense(600) -> ...                                 -> Dense(300) --|
protbert   (1024) -> Dense(600) -> ...                                 -> Dense(300) --+--> concat
ankh       (1536) -> Dense(600) -> ...                                 -> Dense(300) --|      |
taxonomy    (200) -> Dense(128) -> ...                                 -> Dense(64)  --|      |
abstracts   (600) -> Dense(200) -> ...                                 -> Dense(200) --'      |
                                                                                              v
                                              BN -> Dense(800, relu) -> Dense(n_terms, sigmoid)
```

Around 8.2M parameters for BPO. One sigmoid per GO term: this is multi-label, not
multi-class, since a protein has many functions at once.

**Input order is load-bearing.** Checkpoints are `.hdf5` files that take inputs
positionally, so reordering `features` in `src/protgoat/configs/default.yaml` invalidates
every saved model. `tests/test_model.py` pins the order, the parameter count and
the layer composition against the original notebook's hand-written version.

## Training

Each GO domain is a separate model with its own label space — the target
matrices, term counts and IA weights all differ — so `--ontology` selects one and
the three are trained independently.

- **Label space**: the most frequent terms per domain (1500 BPO, 800 CCO, 800
  MFO). The tail is too sparse to learn and contributes little to the CAFA score.
- **IA weighting**: each term's loss is weighted by its information accretion, so
  that specific, rare terms are not drowned out by the near-universal ones close
  to the DAG root. CAFA's own metric is information-weighted, so this aligns the
  objective with the evaluation.
- **Unannotated rows are dropped**: a protein annotated only in BPO has an
  all-zero CCO target row. Training on those teaches the CCO model to predict
  nothing for well-characterised proteins, so `drop_unannotated` removes them.
- **Ensemble**: 5-fold cross-validation repeated over 5 seeds gives 25 models per
  domain, averaged at inference. Different fold partitions decorrelate the
  members more than different initialisations alone.
- **Schedule**: Adam at 3e-4, halved every 10 epochs, 20 epochs, batch 256. The
  best epoch by validation loss is checkpointed rather than the last.

Memory is the practical constraint: 25 sequential `fit` calls leak Keras graph
state, so the session is cleared between folds, and the 25 checkpoints are loaded
one at a time at inference.

## Inference and submission

Predictions are averaged over all 25 checkpoints, then the top 5% of
(protein, term) cells by score are written as a headerless three-column TSV. The
full matrix would be ~200M rows for BPO; 5% comfortably covers CAFA's per-protein
term cap. Scores are rounded to three decimals, and anything rounding to zero is
dropped. `protgoat-merge` concatenates the three domains into the single file
CAFA scores.

## What did not work

`protgoat/hierarchy.py` implements the max constraint module of Giunchiglia &
Lukasiewicz, [*Coherent Hierarchical Multi-Label Classification
Networks*](https://arxiv.org/abs/2010.10151) (NeurIPS 2020), which enforces GO's
DAG structure by raising each term's score to the maximum over its descendants —
a protein annotated with a term is implicitly annotated with all its ancestors,
so raw sigmoid outputs can be incoherent.

**It did not beat plain binary cross-entropy on the CAFA5 validation folds.** It
is kept, tested, and unused: the DAG matrices it needs are part of the published
dataset, and the negative result is worth recording. `output_split` in the
ontology configs (term counts per DAG depth) exists only for this path.
