# Neutrino Direction Reconstruction — KM3NeT Open Data

A small, modular research framework for reconstructing neutrino arrival
directions from [KM3NeT](https://www.km3net.org/) CERN Open Data, comparing
an MLP baseline against three graph neural network architectures (GCN,
GraphSAGE, GAT) under an identical, fully config-driven training procedure.

## Contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Data format](#data-format)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The k-NN population graph](#the-k-nn-population-graph)
- [Configuration system](#configuration-system)
- [Hyperparameter sweeps](#hyperparameter-sweeps)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Extending the framework](#extending-the-framework)
- [Module reference](#module-reference)

## Overview

Each KM3NeT event has 18 reconstructed input features (JMuon/JGandalf/
JShowerfit fit quantities) and a 3D unit target direction vector
(`dir_x`, `dir_y`, `dir_z`). This project trains and compares four
architectures, all consuming the exact same features/targets, splits, loss
function, and training procedure:

| Model       | Config                    | Structure used                          |
|-------------|----------------------------|-------------------------------------------|
| MLP         | `config/mlp.yaml`           | None — every event treated independently     |
| GCN         | `config/gcn.yaml`             | k-NN population graph over events               |
| GraphSAGE   | `config/graphsage.yaml`         | k-NN population graph over events                 |
| GAT         | `config/gat.yaml`                 | k-NN population graph over events (with attention)   |

## Project structure

```
project/
├── config/
│   ├── base.yaml         # Shared data / training procedure / loss / logging
│   ├── mlp.yaml           # MLP architecture only
│   ├── gcn.yaml             # GCN architecture only
│   ├── graphsage.yaml         # GraphSAGE architecture only
│   ├── gat.yaml                 # GAT architecture only
│   └── sweep.yaml                 # Example hyperparameter grid for sweep.py
│
├── data/                # Place KM3NeT .h5 files here
├── checkpoints/          # Created automatically: weights, config, history, metrics per run
├── figures/                # Created automatically: PNG figures per run + comparisons
├── runs/                     # Created automatically: TensorBoard logs per run
│
├── dataset.py       # HDF5 loading, splitting, scaling, k-NN graph construction
├── models.py           # MLP / GCN / GraphSAGE / GAT + factory
├── losses.py              # Cosine loss, Cosine + lambda*MSE + factory
├── trainer.py                # Training loop, TensorBoard logging, checkpointing, early stopping
├── train.py                     # CLI: train a single model from YAML config(s)
├── sweep.py                        # CLI: run a hyperparameter grid over train.py
├── evaluate.py                        # CLI: generate figures + metrics for a trained checkpoint
├── compare.py                            # CLI: aggregate metrics across runs into a comparison
├── utils.py           # Seeding, config I/O, checkpointing, metrics, logging
│
├── requirements.txt
└── README.md
```

## Data format

Place one or more `.h5` files under `data/`. Each file is expected to
contain a single pandas DataFrame (as written by `DataFrame.to_hdf`), one
row per event, with (at least) these columns:

**Input features** (18):
`jmuon_E`, `jmuon_t`, `jmuon_likelihood`, `jmuon_JENERGY_ENERGY`,
`jmuon_JENERGY_CHI2`, `jmuon_JENERGY_NDF`, `jmuon_pos_x`, `jmuon_pos_y`,
`jmuon_pos_z`, `jmuon_dir_x`, `jmuon_dir_y`, `jmuon_dir_z`,
`jmuon_JGANDALF_BETA0_RAD`, `jmuon_JGANDALF_BETA1_RAD`,
`jmuon_JGANDALF_CHI2`, `jmuon_JGANDALF_NUMBER_OF_HITS`,
`jmuon_JSHOWERFIT_ENERGY`, `jmuon_AASHOWERFIT_ENERGY`

**Targets** (3): `dir_x`, `dir_y`, `dir_z`

## Installation

```bash
# 1) Install PyTorch and PyTorch Geometric for your platform/CUDA version --
#    see requirements.txt header for exact commands.
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA index
pip install torch_geometric

# 2) Install the rest of the dependencies
pip install -r requirements.txt
```

## Quick start

```bash
# Train each model (identical data/training procedure, different architecture)
python train.py config/mlp.yaml
python train.py config/gcn.yaml
python train.py config/graphsage.yaml
python train.py config/gat.yaml

# Common flags: --epochs N (shortcut for train.epochs), --no-scheduler
# (shortcut for train.scheduler.type=none), --seed N, --device cpu|cuda,
# --resume <checkpoint.pt>. Anything else: --override dotted.key=value.
python train.py config/mlp.yaml --epochs 30 --no-scheduler

# Evaluate a trained model: generates PNG figures + metrics.json
python evaluate.py --checkpoint checkpoints/gat_20260701-113045/best.pt

# Compare all evaluated runs: table + comparison plots
python compare.py

# Monitor training live -- train.py prints the exact command to use at the
# end of every run (see "Viewing a grid in TensorBoard" below)
tensorboard --logdir runs/
```

`--base-config` defaults to `config/baseline.yaml` (this repo's actual base
config), so it normally doesn't need to be passed explicitly.

Every `train.py` run prints its exact `checkpoints/<run_name>/` path at the
end, along with the ready-to-copy `evaluate.py` command for it.

## The k-NN population graph

Each KM3NeT event is a single flat feature vector — there's no natural
intra-event graph structure the way there is for, say, a point cloud of PMT
hits. To give GCN/GraphSAGE/GAT something structural to exploit (and make
the comparison against the MLP baseline meaningful rather than an arbitrary
reshaping of tabular data), `dataset.py` builds a **population graph**:
nodes are events, edges connect each event to its `k` nearest neighbors in
standardized feature space (`graph.k` in `base.yaml`). GCN/GraphSAGE/GAT
then perform node-level regression on this graph, aggregating information
from similar reconstructed events via message passing. The MLP baseline
sees identical features and targets but no graph structure at all — the
*only* variable being tested is whether neighbor-aggregation over similar
events helps direction reconstruction.

To prevent any leakage between splits, this graph is built **separately**
for train/val/test — neighbors are only ever drawn from within the same
split. Mini-batches over each split's graph are produced via PyTorch
Geometric's `NeighborLoader`, giving GNNs the same `batch_size` semantics
as the MLP's plain `DataLoader`.

## Configuration system

`config/baseline.yaml` holds everything that must be identical across models
for a fair comparison: data location/splitting, graph construction, batch
size, optimizer, scheduler, epochs, early stopping, and loss function.
Each `config/{mlp,gcn,graphsage,gat}.yaml` overrides *only* the `model:`
section (architecture name + hyperparameters). They're deep-merged at
runtime:

```python
from utils import load_experiment_config
config = load_experiment_config("config/baseline.yaml", "config/gat.yaml")
```

**Nothing is hardcoded in Python** — optimizer, learning rate, scheduler,
batch size, hidden dims, dropout, number of layers, activation function,
weight decay, and early-stopping patience all come from these YAML files.

### One-off overrides without editing YAML

```bash
python train.py config/gat.yaml \
    --override train.lr=0.0005 \
    --override model.activation=relu \
    --override train.optimizer=sgd
```

Any dotted config path can be overridden this way — this is also the
mechanism `sweep.py` uses internally.

## Hyperparameter sweeps

To compare multiple optimizers, learning rates, activations (or any other
config field) for a given architecture, define a grid in a sweep YAML
file (see `config/sweep.yaml`):

```yaml
sweep:
  train.optimizer: [adam, adamw, sgd]
  train.lr: [0.01, 0.001, 0.0001]
  model.activation: [relu, elu, gelu, tanh]
```

Then run:

```bash
# Preview the planned runs first (no training executed)
python sweep.py --model-config config/gat.yaml --sweep-config config/sweep.yaml --dry-run

# Run the full grid, evaluating each combination automatically
python sweep.py --model-config config/gat.yaml --sweep-config config/sweep.yaml --evaluate

# Quick smoke test: only the first 3 combinations, with a short epoch budget
python sweep.py --model-config config/mlp.yaml --sweep-config config/sweep.yaml \
    --max-runs 3 --override train.epochs=5 --evaluate
```

Each combination trains in its own isolated subprocess and gets a
self-describing run name (e.g. `gat_optimizeradamw_lr0.001_activationelu_...`).
Once the sweep finishes:

> **Note:** `sweep.py` is documented here as the intended way to run grids
> spanning *multiple* architectures via isolated subprocesses, but isn't
> yet present in this repo. For grids within a single architecture (same
> `model.name`), use the inline list syntax in a model config file (see
> above) with `train.py` directly — no separate script needed.

### Viewing a grid in TensorBoard

Whether the grid comes from an inline list in a model config file (see
above) or from `sweep.py`, every combination trained by **one invocation**
logs to TensorBoard under **one shared group directory**,
`runs/<model>_<invocation_timestamp>/<combo_run_name>/`, instead of a
separate top-level `runs/` entry per combination. At the end of training,
the console prints the exact command, e.g.:

```bash
tensorboard --logdir runs/mlp_20260703-193726
```

That single TensorBoard session gives two complementary views of the same
grid:

- **Scalars tab** — every combination's train/val loss, cosine similarity,
  and learning-rate curves overlaid on the same charts, with a sidebar
  checklist to toggle individual combinations on/off.
- **HParams tab** — one row per combination, with its hyperparameters
  (learning rate, optimizer, `num_layers`, `activation`, ...) and final
  metrics (`hparam/best_val_loss`, `hparam/final_val_cosine_similarity`,
  ...) side by side in a sortable table, plus a parallel-coordinates plot
  — the fastest way to spot which configuration actually won.

For a single (non-grid) run, logs go directly to `runs/<run_name>/` as
before, and the same `tensorboard --logdir ...` command is printed.
Checkpoints are **not** grouped this way — they stay flat under
`checkpoints/<run_name>/` so `evaluate.py`/`compare.py` are unaffected.

```bash
python compare.py                    # every combination
python compare.py --best-per-model   # collapse to the best run per architecture
```

## Outputs

For a run named `<run_name>` (e.g. `gat_20260701-113045`):

```
checkpoints/<run_name>/
├── config.yaml            # Fully resolved config that produced this run
├── last.pt                  # Most recent checkpoint (weights, optimizer, scheduler, epoch)
├── best.pt                    # Best checkpoint by validation loss
├── history.json                 # Per-epoch metrics (updated every epoch)
├── train.log                       # Training console log
├── metrics.json                       # Test-set summary metrics (after evaluate.py)
└── angular_errors.npy                    # Raw per-event test angular errors (after evaluate.py)

runs/<run_name>/            # TensorBoard event files

figures/<run_name>/
├── loss_curves.png
├── cosine_similarity_curves.png
├── prediction_vs_target.png
└── angular_error_histogram.png

figures/comparison/          # After compare.py
├── comparison_table.csv
├── comparison_table.md
├── comparison_angular_error_bar.png
├── comparison_cosine_similarity_bar.png
└── comparison_angular_error_boxplot.png
```

TensorBoard automatically receives, per epoch: training loss, validation
loss, training/validation cosine similarity, learning rate, and epoch time
(under `Loss/`, `CosineSimilarity/`, `LearningRate`, `Time/`).

## Reproducibility

- `seed` (top-level in `base.yaml`) seeds Python, NumPy, and PyTorch
  (CPU + CUDA); `deterministic: true` additionally forces deterministic
  cuDNN algorithms.
- The train/val/test split and feature scaler are both derived
  deterministically from `(data_dir, ratios, seed)` — re-running with the
  same config always reproduces the same split and preprocessing, which is
  how `evaluate.py` reconstructs the exact test set without any extra
  saved state.
- Every checkpoint stores the fully-resolved config that produced it
  (`checkpoints/<run_name>/config.yaml`), so any run can be inspected or
  reproduced later without needing to remember which flags were used.

## Extending the framework

**Add a new model architecture:**

```python
# in models.py
class GIN(DirectionRegressor):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3,
                 dropout=0.1, activation="relu"):
        ...

    def forward(self, x, edge_index=None):
        ...

    @property
    def requires_graph(self) -> bool:
        return True

MODEL_REGISTRY["gin"] = GIN
```

Then add `config/gin.yaml` with a `model:` section, and
`python train.py config/gin.yaml` works immediately —
`dataset.py`, `trainer.py`, `evaluate.py`, and `compare.py` require no changes.

**Add a new loss function:** subclass `losses.DirectionLoss`, implement
`compute_components`, and register it in `losses.LOSS_REGISTRY`.

## Module reference

| Module | Responsibility |
|---|---|
| `utils.py` | Seeding, config load/merge/override, device selection, checkpointing, early stopping, angular error, logging |
| `dataset.py` | HDF5 loading, splitting, feature scaling, k-NN population graph, tabular/graph DataLoaders |
| `models.py` | `DirectionRegressor` base class, MLP/GCN/GraphSAGE/GAT, `build_model` factory |
| `loss.py` | `DirectionLoss` base class, Cosine / Cosine+λMSE losses, `build_loss` factory |
| `trainer.py` | `Trainer`: optimizer/scheduler construction, train/val loop, TensorBoard logging, checkpointing, early stopping |
| `train.py` | CLI: single training run from YAML config(s) + optional overrides |
| `sweep.py` | CLI: hyperparameter grid search over `train.py` |
| `evaluate.py` | CLI: test-set inference, PNG figures, `metrics.json` |
| `compare.py` | CLI: aggregate `metrics.json` across runs into a comparison table + plots |

## Acknowledgments

Data: [KM3NeT Collaboration](https://www.km3net.org/) CERN Open Data.