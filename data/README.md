# Benchmark data

These files package the graph benchmark used by
[PoinnCARE](https://github.com/kkkkk001/PoinnCARE), built on the
[CARE enzyme-classification benchmark](https://github.com/jsunn-y/CARE).
The node identifiers, EC vocabulary, and split indexing are shared by the
graphs, labels, and feature matrix.

## Included files

| File | Contents | Shape or count |
| --- | --- | --- |
| `labels.npz` | Level-4 multi-hot EC labels; float32 sparse CSR | `(29531, 4963)`, 29,933 nonzeros |
| `structure_graph.npz` | Structure-similarity graph | `(29531, 29531)`, 517,544 nonzeros |
| `active_site_graph.npz` | Active-site-similarity graph | `(29531, 29531)`, 14,923 nonzeros |
| `entry2index.json` | Protein identifier → node row | 29,531 entries |
| `ec2idx.json` | EC number → label column | 4,963 entries |
| `train.tsv` | Training pool, including the validation holdout | 28,182 proteins |
| `test_lt30.tsv` | <30% sequence identity | 432 proteins |
| `test_30_50.tsv` | 30–50% sequence identity | 560 proteins |
| `test_difficult.tsv` | Difficult / Price | 148 proteins |
| `test_multifunction.tsv` | Multifunctional / Promiscuous | 209 proteins |

The five split files are disjoint and cover all 29,531 nodes. Each TSV has
three columns: `Entry`, `EC number`, and `Sequence`. The loader resolves
identifiers from the first column through `entry2index.json`.

`labels.npz` stores binary multi-hot labels as a float32 sparse matrix.
The training loader converts it to a dense float32 tensor in memory.

## Required external file: node features

`features.npy` is **not included**. The expected input is a `(29531, 1280)`
float32 matrix of ESM-2 sequence embeddings, approximately 144 MiB.

1. Download the PoinnCARE graph release from
   [Zenodo](https://zenodo.org/records/18813316). It contains the original
   `feats.npy` matrix.
2. Place that matrix in this directory as `features.npy`, or retain its name
   and pass `--features-file feats.npy` to `train.py`.
3. Preserve node order: row `i` must describe the protein whose
   `entry2index.json` value is `i`. Use the original unnormalized features;
   the loader performs its own row-sum normalization.

Example using the upstream filename:

```bash
# Run from the repository root after placing feats.npy in data/.
bash run.sh S --features-file feats.npy --repeats 3
```

## Data processing

[`quatec/data.py`](../quatec/data.py) prepares the supplied matrices at load time.
Graph construction and embedding extraction have already been performed
upstream; they are not run by `train.py`.

### 1. Align identifiers and matrices

`entry2index.json` maps protein identifiers to integer rows `0` through `29530`.
Row `i` of the features and labels, and row/column `i` of both graphs, must all
refer to the same protein. `ec2idx.json` maps EC labels to columns `0` through
`4962`. Do not independently sort a matrix or either mapping.

For each TSV, the loader uses only its first column (`Entry`) to look up node
indices. It reads targets from `labels.npz` and embeddings from `features.npy`;
it does not regenerate either from the TSV's `EC number` or `Sequence` column.

### 2. Prepare features and labels

Features and labels are converted to float32. Each feature vector `x` is
divided by `sum(x)`; when that sum is zero, the denominator is set to one.
This is row-sum normalization, not L2 normalization or standardization.
Use the original feature matrix and let the loader apply this step once.

Labels remain binary multi-hot vectors. Their sparse on-disk representation
is converted to a dense tensor during loading, so training needs more memory
than the compressed file sizes suggest.

### 3. Diffuse and normalize graphs

The two sparse adjacency matrices are processed separately. Where enabled,
the finite PPR diffusion starts at `D = alpha * I` and performs the configured
number of updates `D = (1 - alpha) * D @ A + alpha * I`, using the input graph
`A`. The current presets in [`quatec/config.py`](../quatec/config.py) are:

| Variant | Structure graph: alpha / steps | Active-site graph: alpha / steps |
| --- | --- | --- |
| QuatEC-S | No diffusion | 0.95 / 4 |
| QuatEC-D | No diffusion | 0.80 / 4 |
| QuatEC-M | 0.80 / 1 | 0.80 / 2 |

After diffusion, or directly on the input graph when diffusion is disabled,
the loader adds the identity matrix for self-loops and divides each adjacency
row by its sum. The results become coalesced sparse PyTorch tensors with
float32 edge weights. No additional manual diffusion or normalization is needed.

### 4. Form training, validation, and test splits

The loader shuffles the 28,182 training identifiers with `--seed` and reserves
`--validation-fraction` of them for validation (default: 0.05). The same
holdout is used for every repeat in that invocation. The four provided test
splits remain separate and are used for final evaluation.

Training uses the complete graph, including validation and test nodes, while
supervised loss uses only training labels. This is a transductive benchmark.

## Using another data directory

Put the complete set of aligned files in your chosen directory and pass it
through `--data-dir`, for example:

```bash
# Replace /path/to/graph-data with the directory containing all dataset files.
bash run.sh S --data-dir /path/to/graph-data --features-file feats.npy
```

The default names for labels, graphs, mappings, and splits are listed above.
Use the corresponding options from `python train.py --help` if they differ.
Paths set by these options are resolved relative to `--data-dir`. Use the
same node indexing and EC vocabulary for all input files.

## Supported formats

Features and labels accept dense numeric `.npy` files or SciPy sparse `.npz`
files. Graphs use SciPy sparse `.npz`; mappings use JSON objects. File names
can be changed through the command-line options in `python train.py --help`.
