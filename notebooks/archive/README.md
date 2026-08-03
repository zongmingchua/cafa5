# Archived notebooks

These are the original Colab and Kaggle notebooks used for the CAFA5 submission,
kept unmodified as the provenance record of the 4th-place result. They contain
absolute `/content/drive/MyDrive/...` and `/kaggle/input/...` paths and will not
run outside those environments.

The logic has been extracted into the `protgoat` package; use that instead.

| Notebook | Superseded by |
| --- | --- |
| `CAFA5_Train_and_infer_f02.ipynb` | `protgoat.train`, `protgoat.predict`, `protgoat.model`, `protgoat.submission` |
| `ankh-embeddings-make.ipynb` | `protgoat.embeddings.plm` |
| `train-test-generate-abstract-embeddings.ipynb` | `protgoat.embeddings.abstracts` |

Differences worth knowing about, if you are comparing the two:

- **Shard concatenation.** The Ankh notebook listed 59 shard files into variables
  `a1 .. a59` and concatenated them by name. Two entries loaded `''` (a crash),
  one was skipped, and the ordering depended on that hand-written list.
  `concatenate_shards` discovers shards on disk, orders them numerically, and
  raises on a gap or overlap instead of silently misaligning every row.
- **TF-IDF vocabulary order.** The abstracts notebook built the vocabulary with
  `list(set(...))`, whose order varies between interpreter runs, so the feature
  columns were not reproducible. `go_vocabulary` sorts it.
- **Duplicate code.** `PredictionPreparation` was defined twice, identically, in
  the same cell; the seven per-modality towers were written out by hand.
- **Dead cells.** The last cells of the training notebook reference an undefined
  `prot_P62259` and would raise `NameError` on a clean run.

`tests/test_model.py` pins the refactored architecture to the same parameter
count and layer composition as the notebook version.
