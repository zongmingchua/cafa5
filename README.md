# PROTGOAT

Protein function prediction from fused protein language model, taxonomy and
literature features. **4th place** in the [CAFA5 protein function prediction
competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
(2023), out of roughly 1,600 submitted methods.

[Manuscript](https://www.biorxiv.org/content/10.1101/2024.04.01.587572v1) ·
[Data](https://www.kaggle.com/datasets/zmcxjt/cafa5-train-test-data)

## Approach

Five protein language models (ProtT5-XL, ESM-2 650M, ESM-2 3B, ProtBERT,
Ankh-large) are each mean-pooled into one embedding per protein. Those are fused
with the source organism's taxonomy and with TF-IDF features over the protein's
PubMed abstracts, restricted to the vocabulary of GO term names — where a protein
has been studied, its literature names its functions directly, which is evidence
sequence alone cannot provide.

Each modality passes through its own tower before the towers are concatenated
and mapped to one sigmoid per GO term. Each of the three GO domains gets its own
model, and each is an average of 25 networks (5-fold cross-validation × 5 seeds).

See [docs/model.md](docs/model.md) for the architecture, the training recipe, and
what was tried and rejected.

## Install

```bash
conda create -n protgoat python=3.11 && conda activate protgoat
pip install -e ".[train]"          # add ",dev" for the test suite
```

`train` pins TensorFlow `<2.16`: the checkpoints are `.hdf5`, which Keras 3 no
longer loads by default. The `embeddings` extra (torch, ankh, biopython, nltk) is
only needed to regenerate features from raw sequences — the published embeddings
mean you can skip it.

## Use

Download the embeddings and target matrices into the layout described in
[docs/data.md](docs/data.md), then:

```bash
export PROTGOAT_DATA_ROOT=/path/to/data

for ONTOLOGY in BPO CCO MFO; do
    protgoat-train   --ontology $ONTOLOGY     # 25 models -> $PROTGOAT_DATA_ROOT/models/
    protgoat-predict --ontology $ONTOLOGY     # ensemble  -> .../submissions/f02_$ONTOLOGY.tsv
done

protgoat-merge                                # one CAFA submission across all three domains
```

Training all three domains is a multi-GPU-hour job. To check the plumbing first,
run a single fold for one epoch:

```bash
protgoat-train --ontology CCO --epochs 1 --folds 2 --seeds 1
```

Or from Python:

```python
from protgoat import load_config, train, predict_and_write

config = load_config("BPO", data_root="/path/to/data")
train(config)
predict_and_write(config)
```

Hyperparameters live in `src/protgoat/configs/`: `default.yaml` is shared, `bpo.yaml` /
`cco.yaml` / `mfo.yaml` hold each domain's label space. Ablate a modality by
dropping it from `features`:

```python
config = load_config("BPO", overrides={"training": {"features": ["t5", "ankh"]}})
```

## Layout

```
src/protgoat/
  config.py        run configuration: data paths, per-ontology specs, hyperparameters
  data.py          loading targets and the seven feature matrices
  model.py         the multi-input architecture
  train.py         cross-validated training loop
  predict.py       ensemble inference
  submission.py    prediction matrix -> CAFA TSV
  ontology.py      GO OBO parsing
  hierarchy.py     experimental DAG-aware loss (not used in the submission)
  cli.py           protgoat-train / -predict / -merge
  configs/         default.yaml + one file per GO domain (shipped as package data)
  embeddings/
    plm.py         protein language model embeddings, sharded and resumable
    abstracts.py   TF-IDF literature features
docs/              data layout and model documentation
tests/             pytest suite
notebooks/archive/ the original competition notebooks, unmodified
```

The [archived notebooks](notebooks/archive/README.md) are the provenance record
of the submitted result. They hard-code Colab and Kaggle paths and will not run
elsewhere; the package supersedes them.

## Tests

```bash
pip install -e ".[train,dev]"
pytest
```

The suite covers config loading, data validation, submission formatting, OBO
parsing, shard stitching, and an end-to-end train → predict → merge run on
synthetic data. `tests/test_model.py` pins the refactored architecture to the
same parameter count and layer composition as the original notebook. Tests
needing TensorFlow skip cleanly without it.

## Citation

```bibtex
@article{protgoat2024,
  title   = {PROTGOAT: Improved automated protein function predictions using
             Protein Language Models},
  author  = {Chua, Z. M. and Rajesh, A. and Sinha, S. and Adams, P. D.},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.04.01.587572}
}
```

MIT licensed.
