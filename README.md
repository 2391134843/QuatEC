# QuatEC

**Parameter-Efficient Quaternion Hyperbolic Networks for Hierarchical Enzyme Function Prediction**

QuatEC combines protein sequence embeddings with structure and active-site
graphs using quaternion hyperbolic networks for hierarchical EC classification.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

We use the graph version of the [CARE benchmark](https://github.com/jsunn-y/CARE)
prepared by [PoinnCARE](https://github.com/kkkkk001/PoinnCARE), with 29,531
proteins and 4,963 EC labels.

### Download and prepare

1. Download the graph release from [Zenodo](https://zenodo.org/records/18813316).
   If access is restricted, follow the upstream authors' access instructions.
2. Keep the graphs, labels, mappings, and splits already included in `data/`.
   The only missing input is `feats.npy`: a `(29531, 1280)` float32 matrix of
   ESM-2 embeddings, approximately 144 MiB.
3. Save that matrix as `data/features.npy`. Preserve its original node order:
   row `i` must match the protein assigned index `i` in `entry2index.json`.

```text
data/
├── features.npy           # download separately
├── labels.npz             # multi-hot EC labels
├── structure_graph.npz    # structure-similarity graph
├── active_site_graph.npz  # active-site-similarity graph
├── entry2index.json       # protein ID to node row
├── ec2idx.json            # EC number to label column
├── train.tsv
└── test_*.tsv             # four evaluation splits
```

To keep the upstream feature filename instead, place `feats.npy` in `data/`
and run:

```bash
bash run.sh S --features-file feats.npy
```

### Data preprocessing

The loader in [`quatec/data.py`](quatec/data.py) performs these steps automatically:

1. **Align samples.** Resolve the `Entry` column of each TSV through
   `entry2index.json`. Features, label rows, and both graph axes use this same
   node order; label columns follow `ec2idx.json`.
2. **Prepare features and labels.** Convert both to float32, divide each feature
   row by its sum, and retain binary multi-hot labels. Sparse labels are
   converted to dense tensors for training. Supply the original embeddings;
   no manual normalization is needed.
3. **Prepare graphs.** Apply the selected variant's PPR diffusion where enabled,
   add self-loops, row-normalize the adjacency matrices, and convert them to
   sparse PyTorch tensors. The supplied graphs are already constructed; this
   loader does not run structure searches or extract ESM-2 embeddings.
4. **Create validation data.** Hold out 5% of `train.tsv` using `--seed`, keeping
   the four test splits for final evaluation. All repeats share that holdout.
   The full graph includes test nodes, while supervised loss uses training
   labels only: this is a transductive setting.

See [data/README.md](data/README.md) for file dimensions, diffusion settings,
and instructions for using a different data directory.

## Training

After activating the environment and preparing the data, use [run.sh](run.sh):

```bash
# Run all three variants sequentially
bash run.sh

# QuatEC-S: <30% and 30–50% sequence identity
bash run.sh S

# QuatEC-D: difficult / Price
bash run.sh D

# QuatEC-M: multifunctional / Promiscuous
bash run.sh M

# Pass additional training options, for example a GPU and seed count
bash run.sh all --device cuda:0 --seed 1234 --repeats 3
```

The script calls `train.py` and runs three seeds per variant by default.
Checkpoints and metrics are saved under `outputs/QuatEC-S/`, `outputs/QuatEC-D/`,
and `outputs/QuatEC-M/`. Relative paths are resolved from the repository root.
Use `bash run.sh --help` for script usage or `python train.py --help` for training
options. You can also run `python train.py --variant S` directly.

## Acknowledgements

We thank the authors of [PoinnCARE](https://github.com/kkkkk001/PoinnCARE)
for releasing the baseline implementation and benchmark resources on which
this work builds.
