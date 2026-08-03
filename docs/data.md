# Data layout

Every command takes `--data-root` (or reads `$PROTGOAT_DATA_ROOT`, defaulting to
`./data`) and expects this layout:

```
<data-root>/
  train/
    go-basic.obo                              # Gene Ontology, from the CAFA5 competition data
    Y_BP_1500.npy   Y_BP_labels_1500.npy      # BPO targets + GO term ids
    Y_CC_800.npy    Y_CC_labels_800.npy       # CCO
    Y_MF_800.npy    Y_MF_labels_800.npy       # MFO
    BPO_1500_freq_weights.csv                 # information accretion weights ('IA_weight' column)
    CCO_800_freq_weights.csv
    MFO_800_freq_weights.csv
    BPO_1500_DAG_matrix.npy                   # only needed for the experimental hierarchy loss
    CCO_800_DAG_matrix.npy
    MFO_800_DAG_matrix.npy
    t5_train_data_sorted_f32.npy              # the seven feature matrices
    esm2_train_data_sorted_f32.npy
    ESM2_3B_train_embeddings_sorted.npy
    pb_train_data_sorted_f32.npy
    Ankh_train_embeddings_sorted.npy
    protein_taxa_matrix_train.npy
    abstract_embeds_train_sorted_remade_1_3.npy
  test/
    t5_labels_sorted.npy                      # test protein accessions, in feature row order
    t5_test_data_sorted_f32.npy               # the same seven matrices for the test superset
    esm2_test_data_sorted_f32.npy
    ESM2_3B_test_embeddings_sorted.npy
    pb_test_data_sorted_f32.npy
    Ankh_test_embeddings_sorted.npy
    protein_taxa_matrix_test.npy
    abstract_embeds_test_sorted_all_remade_1_3.npy
  models/                                     # written by protgoat-train
  submissions/                                # written by protgoat-predict and protgoat-merge
```

Filenames are declared in `src/protgoat/configs/*.yaml` and `protgoat/config.py`; nothing is
hard-coded in the pipeline code, so renaming a file means editing the config
rather than hunting through modules.

## Getting the data

The embeddings for all 142,246 training and 141,864 test-superset proteins are
published, so a training run does not require re-embedding anything:

<https://www.kaggle.com/datasets/zmcxjt/cafa5-train-test-data>

The target matrices, IA weights, DAG matrices and `go-basic.obo` come from the
same dataset and the original competition data:

<https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data>

## The row-order invariant

**Row *i* of every matrix — targets and all seven features, within a split —
refers to the same protein.** The model concatenates features positionally, so a
misalignment does not raise; it silently trains on mismatched pairs.

The invariant is established once, when features are generated: each modality is
projected onto a reference accession list (`sort_to_reference` for the language
models, `align_to_reference` for the abstracts), with a zero row for any protein
that modality lacks. `t5_labels_sorted.npy` is that reference order for the test
split, and is what labels the rows of the submission file.

`load_features` checks that all matrices agree on row *count*, and
`load_training_set` checks them against the target matrix. Neither can detect a
permutation, so if you regenerate any feature matrix, re-derive its order from
the reference list rather than assuming it.

## Sizes

The five language model matrices dominate: at 142k proteins and 1024–2560
dimensions in float32 they are roughly 0.6–1.5 GB each per split. They are
memory-mapped during training (`mmap_mode='r'`) and only the rows selected by the
current fold are materialised.

## Regenerating features

Only needed if you are changing the feature set. Requires the `embeddings` extra
(`pip install -e ".[embeddings]"`), a GPU, and considerable patience.

```python
from protgoat.embeddings.plm import Shard, embed_shard, concatenate_shards, sort_to_reference

# Embedding 142k proteins exceeds any single Kaggle session, so work in shards.
embed_shard("testsuperset.fasta", "shards/", Shard(0, 3000))
embed_shard("testsuperset.fasta", "shards/", Shard(3000, 6000))
# ... a killed shard resumes from its .checkpoint file on re-run

embeddings, ids = concatenate_shards("shards/")   # errors on any gap or overlap
ordered = sort_to_reference(embeddings, ids, reference_ids)
```

```python
from protgoat.embeddings.abstracts import build_abstract_features

train_features, test_features = build_abstract_features(
    "preprocessed_data.csv", "preprocessed_data_test.csv", "go-basic.obo",
    train_reference_ids, test_reference_ids,
)
```

The abstract CSVs need a `Protein ID` and a `Preprocessed Abstract` column.
