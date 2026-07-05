"""
dataset.py
==========

Data loading and preprocessing for KM3NeT neutrino direction reconstruction.

Each KM3NeT HDF5 file contains one pandas DataFrame where each row is a
single reconstructed neutrino event. This module:

1. Loads and concatenates all ``.h5`` files in a data directory.
2. Extracts the 18 input features and 3 target direction components.
3. Produces a reproducible, seed-controlled train/val/test split.
4. Standardizes features (zero mean, unit variance), fit on the training
   split only, and applied identically to val/test.
5. Exposes the data in two forms:

   - **Tabular** (:meth:`NeutrinoDataModule.tabular_dataloaders`): plain
     ``(X, y)`` DataLoaders, used by the MLP baseline.
   - **Graph** (:meth:`NeutrinoDataModule.graph_dataloaders`): a k-NN
     "population graph" per split, used by GCN / GraphSAGE / GAT.

Why a k-NN population graph
----------------------------
Each event here is a single feature vector, not an inherently structured
object (unlike, e.g., a point cloud of PMT hits). To meaningfully compare
graph neural networks against an MLP baseline on this data, we construct a
graph **over events**: nodes are events, and edges connect each event to its
``k`` nearest neighbors in standardized feature space. GCN / GraphSAGE / GAT
then perform *node regression* on this graph -- predicting each node's
direction vector using both its own features and those of similar
("nearby") events, aggregated via message passing.

This is the standard "population graph" construction used when applying
GNNs to tabular data (common in medical imaging / tabular-GNN literature).
The MLP baseline uses the identical features and targets but ignores this
graph structure entirely, which isolates *exactly one variable* in the
comparison: does neighbor-aggregation over similar events help direction
reconstruction, or not?

To avoid any information leakage between splits, the k-NN graph is built
**separately per split** (train, val, test each get their own self-contained
graph) -- neighbors are only ever drawn from within the same split.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from utils import PathLike, seed_worker

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS: List[str] = [
    "jmuon_E",
    "jmuon_t",
    "jmuon_likelihood",
    "jmuon_JENERGY_ENERGY",
    "jmuon_JENERGY_CHI2",
    "jmuon_JENERGY_NDF",
    "jmuon_pos_x",
    "jmuon_pos_y",
    "jmuon_pos_z",
    "jmuon_dir_x",
    "jmuon_dir_y",
    "jmuon_dir_z",
    "jmuon_JGANDALF_BETA0_RAD",
    "jmuon_JGANDALF_BETA1_RAD",
    "jmuon_JGANDALF_CHI2",
    "jmuon_JGANDALF_NUMBER_OF_HITS",
    "jmuon_JSHOWERFIT_ENERGY",
    "jmuon_AASHOWERFIT_ENERGY",
]

TARGET_COLUMNS: List[str] = ["dir_x", "dir_y", "dir_z"]

# Added to every loaded DataFrame (see `infer_particle_type`) to record which
# source file each event came from. Not a model input -- deliberately kept
# out of FEATURE_COLUMNS -- but carried through concatenation, splitting, and
# (optionally) used to stratify the train/val/test split.
PARTICLE_TYPE_COLUMN: str = "particle_type"

logger = logging.getLogger("neutrino_reco")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def infer_particle_type(path: PathLike, regex: Optional[str] = None) -> str:
    """Infer a short sample/particle-type label from a source filename.

    This is what lets every event still be traced back to its source file
    after multiple files are concatenated into one DataFrame (see
    :func:`load_h5_files`), and is what :func:`train_val_test_split` can
    optionally stratify on.

    Args:
        path: Path to the source ``.h5`` file.
        regex: Optional regex applied to the file's stem (filename without
            extension). If the match has a named group ``type``, that
            group's text is used as the label; otherwise, if the match has
            any capturing group, the first one is used; if it has none,
            the whole match is used. If ``regex`` is ``None`` or doesn't
            match, the full file stem is used as-is. Example: with files
            named like ``KM3NeT_00000133_numuCC_23.h5``, the regex
            ``r"_(numuCC|nueCC|anue|atm_muon)_"`` extracts just
            ``"numuCC"`` instead of the whole filename.

    Returns:
        A label string identifying which file/sample this event came from.
    """
    stem = Path(path).stem
    if regex:
        match = re.search(regex, stem)
        if match:
            if "type" in match.groupdict():
                return match.group("type")
            if match.groups():
                return match.group(1)
            return match.group(0)
    return stem
def discover_h5_files(data_dir: PathLike, pattern: str = "*.h5") -> List[Path]:
    """List HDF5 files in a directory matching a glob pattern, in sorted order.

    Sorted order gives reproducible, deterministic file ordering across runs
    and platforms (directory iteration order is not guaranteed otherwise).

    Args:
        data_dir: Directory to search.
        pattern: Glob pattern used to select files within ``data_dir``.

    Returns:
        A sorted list of matching file paths.

    Raises:
        FileNotFoundError: If no files match ``pattern`` in ``data_dir``.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    return files


def load_single_h5_file(
    path: PathLike, key: Optional[str] = None, particle_type_regex: Optional[str] = None
) -> pd.DataFrame:
    """Load exactly one HDF5 file into a DataFrame, tagged with its particle type.

    Args:
        path: Path to a single ``.h5`` file.
        key: Optional HDF5 key. If ``None``, ``pandas.read_hdf`` auto-detects
            the key, which requires the file to contain exactly one dataset
            (as stated in the project data format).
        particle_type_regex: Passed to :func:`infer_particle_type` to derive
            the ``particle_type`` column value for every row of this file.

    Returns:
        The file's contents as a DataFrame, with an added
        ``particle_type`` column (see :data:`PARTICLE_TYPE_COLUMN`) so
        provenance survives later concatenation with other files.
    """
    path = Path(path)
    df = pd.read_hdf(path, key=key) if key is not None else pd.read_hdf(path)
    label = infer_particle_type(path, particle_type_regex)
    df[PARTICLE_TYPE_COLUMN] = label
    logger.info(f"Loaded {len(df)} events from {path.name} (particle_type={label!r})")
    return df


def load_h5_files(
    data_dir: PathLike,
    pattern: str = "*.h5",
    key: Optional[str] = None,
    particle_type_regex: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate ALL HDF5 files in a directory into one DataFrame.

    .. warning::
        Only use this when pooling events across files is physically
        justified (e.g. multiple files that are genuinely fragments of the
        same homogeneous run/sample). If your files represent different
        detector runs, calibration periods, or samples that should **not**
        be statistically mixed, use per-file mode instead: pass a specific
        file directly to :class:`NeutrinoDataModule` via its ``data_file``
        argument (this is what ``train.py`` does by default -- see its
        ``data.combine_files`` config flag). Combining incompatible files
        into one train/val/test split can introduce subtle systematic
        biases that per-file analysis avoids entirely.

    Each file is expected to contain a single pandas DataFrame (as written by
    ``DataFrame.to_hdf``). Files are read in sorted filename order for
    reproducibility, and concatenated along rows. Every file is tagged with
    a ``particle_type`` column (see :func:`infer_particle_type`) **before**
    concatenation, so even after pooling, every event can still be traced
    back to its source file -- and, if desired, the train/val/test split
    can be stratified by it (see ``NeutrinoDataModule``'s
    ``data.stratify_by_particle_type`` config flag).

    Args:
        data_dir: Directory containing ``.h5`` files.
        pattern: Glob pattern used to select files within ``data_dir``.
        key: Optional HDF5 key to read from each file. If ``None``,
            ``pandas.read_hdf`` will auto-detect the key, which requires each
            file to contain exactly one dataset (as stated in the project
            data format).
        particle_type_regex: Passed to :func:`infer_particle_type` for every
            file; see that function for how it's applied.

    Returns:
        A single concatenated DataFrame with a reset integer index and a
        ``particle_type`` column.

    Raises:
        FileNotFoundError: If no files match ``pattern`` in ``data_dir``.
    """
    files = discover_h5_files(data_dir, pattern)
    frames = [load_single_h5_file(f, key=key, particle_type_regex=particle_type_regex) for f in files]
    full_df = pd.concat(frames, axis=0, ignore_index=True)
    counts = full_df[PARTICLE_TYPE_COLUMN].value_counts().to_dict()
    logger.info(f"Combined {len(files)} files into {len(full_df)} events. particle_type counts: {counts}")
    return full_df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def train_val_test_split(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce reproducible, non-overlapping train/val/test index arrays.

    Args:
        n: Total number of samples.
        train_ratio: Fraction of samples assigned to training.
        val_ratio: Fraction of samples assigned to validation.
        test_ratio: Fraction of samples assigned to testing.
        seed: Random seed controlling the shuffle (a dedicated
            ``numpy.random.Generator`` is used, independent of global RNG
            state, so calling this repeatedly with the same seed always
            yields the same split regardless of other random calls made
            elsewhere). Also used as sklearn's ``random_state`` in the
            stratified path, for the same reproducibility guarantee.
        stratify_labels: Optional per-sample group labels of length ``n``
            (e.g. ``particle_type``). When given, splitting is done in two
            stratified stages -- train vs. (val+test), then val vs. test --
            each preserving every group's proportion, so e.g. a rare
            sample type can't be accidentally under- or over-represented
            in validation or test. Falls back to a plain (non-stratified)
            random split, with a warning, if the rarest group has too few
            members to be split reliably (fewer than 3 -- sklearn requires
            at least 2 members per class per split, and this is split
            twice).

    Returns:
        A tuple ``(train_idx, val_idx, test_idx)`` of integer index arrays.

    Raises:
        ValueError: If the ratios do not sum to ~1.0, or if
            ``stratify_labels`` is given with a length other than ``n``.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    if stratify_labels is not None:
        labels = np.asarray(stratify_labels)
        if len(labels) != n:
            raise ValueError(f"stratify_labels length ({len(labels)}) must match n ({n}).")

        _, group_counts = np.unique(labels, return_counts=True)
        if group_counts.min() < 3:
            logger.warning(
                f"Rarest particle_type group has only {group_counts.min()} event(s) -- too few to "
                "stratify a train/val/test split reliably (need >= 3). Falling back to a plain "
                "(non-stratified) random split."
            )
        else:
            all_idx = np.arange(n)
            train_idx, rest_idx = sk_train_test_split(
                all_idx, train_size=train_ratio, stratify=labels, random_state=seed
            )
            rest_labels = labels[rest_idx]
            relative_val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = sk_train_test_split(
                rest_idx, train_size=relative_val_size, stratify=rest_labels, random_state=seed
            )
            return train_idx, val_idx, test_idx

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Feature scaling
# --------------------------------------------------------------------------- #
@dataclass
class FeatureScaler:
    """Standardization (zero mean, unit variance) fit on training features.

    Attributes:
        mean: Per-feature mean, shape ``(num_features,)``.
        std: Per-feature standard deviation, shape ``(num_features,)``
            (zero/near-zero std is clamped to 1.0 to avoid division by zero
            for constant features).
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "FeatureScaler":
        """Fit mean/std statistics from a feature matrix.

        Args:
            X: Feature matrix, shape ``(num_samples, num_features)``. Should
                be the **training split only** to avoid leaking val/test
                statistics into preprocessing.
            eps: Threshold below which std is treated as zero (constant
                feature) and clamped to 1.0.

        Returns:
            A fitted :class:`FeatureScaler`.
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization: ``(X - mean) / std``."""
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert standardization: ``X * std + mean``."""
        return X * self.std + self.mean

    def save(self, path: PathLike) -> None:
        """Persist scaler statistics to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: PathLike) -> "FeatureScaler":
        """Load scaler statistics previously saved with :meth:`save`."""
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])


# --------------------------------------------------------------------------- #
# k-NN population graph construction
# --------------------------------------------------------------------------- #
def build_knn_edge_index(features: np.ndarray, k: int, symmetric: bool = True) -> torch.Tensor:
    """Build a k-NN graph edge index over a set of feature vectors.

    Each node (row of ``features``) is connected to its ``k`` nearest
    neighbors in Euclidean space. The self-edge (a point being its own
    nearest neighbor) is filtered out.

    Args:
        features: Standardized feature matrix, shape ``(num_nodes, num_features)``.
        k: Number of nearest neighbors per node. Automatically clamped to
            ``num_nodes - 1`` if larger (e.g. for very small splits).
        symmetric: If ``True`` (recommended for GCN/GraphSAGE/GAT, which
            assume undirected graphs), also add the reverse of every edge so
            the resulting graph is undirected, then deduplicate.

    Returns:
        A ``LongTensor`` of shape ``(2, num_edges)`` in PyTorch Geometric's
        ``edge_index`` format.

    Raises:
        ValueError: If ``num_nodes`` is smaller than 2 (cannot build a graph).
    """
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to build a graph, got {n_samples}")

    k_eff = min(k, n_samples - 1)
    nn_model = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(features)
    _, neighbor_idx = nn_model.kneighbors(features)

    rows = np.repeat(np.arange(n_samples), k_eff + 1)
    cols = neighbor_idx.reshape(-1)

    # Remove self-loops (a node appearing as its own neighbor).
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]

    edge_index = np.stack([rows, cols], axis=0)
    if symmetric:
        edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)

    edge_index_t = torch.from_numpy(edge_index).long()
    edge_index_t = torch.unique(edge_index_t, dim=1)
    # torch.unique(..., dim=1) sorts/deduplicates by internally transposing
    # and transposing back, which returns a non-contiguous view (correct
    # values, but non-standard strides). NeighborLoader's pure-Python
    # fallback sampler tolerates this, but pyg_lib's compiled C++ sampler
    # does a strict contiguity check and raises "Input should be
    # contiguous" otherwise -- so this is required once pyg_lib is
    # installed, and harmless (a cheap no-op if already contiguous)
    # whether or not it is.
    edge_index_t = edge_index_t.contiguous()
    return edge_index_t


# --------------------------------------------------------------------------- #
# Data module
# --------------------------------------------------------------------------- #
class NeutrinoDataModule:
    """Loads, splits, scales, and serves KM3NeT event data for both MLP and GNN models.

    Expected configuration structure (see ``configs/base.yaml``)::

        seed: 42
        data:
          data_dir: "data/"
          file_pattern: "*.h5"
          hdf_key: null
          dropna: true
          particle_type_regex: null   # e.g. "_(numuCC|nueCC|anue|atm_muon)_"
          stratify_by_particle_type: true
          train_ratio: 0.7
          val_ratio: 0.15
          test_ratio: 0.15
          batch_size: 256
          num_workers: 0
        graph:
          k: 8
          symmetric: true
          num_neighbors: [10, 10]

    Usage::

        dm = NeutrinoDataModule(config)
        dm.setup()
        train_loader, val_loader, test_loader = dm.tabular_dataloaders()   # MLP
        train_loader, val_loader, test_loader = dm.graph_dataloaders()     # GNNs

    Note on reproducibility:
        Rather than persisting the fitted :class:`FeatureScaler` and split
        indices to disk, this module regenerates them deterministically from
        ``(data_dir, ratios, seed)`` every time ``setup()`` is called. As long
        as the config and the contents of ``data_dir`` are unchanged, this
        yields bit-identical splits and scaling across ``train.py`` and
        ``evaluate.py`` runs, with zero extra state to manage. See "possible
        improvements" for when you would instead want to persist the scaler.
    Note on per-file vs. combined data:
        By default (``data_file=None``), this data module combines *every*
        file matching ``data.file_pattern`` in ``data.data_dir`` into one
        pooled train/val/test split -- appropriate only when combining
        those files is physically justified. When it isn't (e.g. the files
        are different detector runs or samples that shouldn't be
        statistically mixed), pass a specific file via ``data_file`` to
        scope this data module to exactly that one file; ``train.py``'s
        default (``data.combine_files: false`` in ``base.yaml``) does this
        automatically, training one independent model per file.
    Note on particle_type provenance:
        Every loaded event is tagged with a ``particle_type`` column
        (see :func:`infer_particle_type`) derived from its source
        filename, *before* any concatenation across files -- so even a
        pooled dataset never loses track of which file each event came
        from. This column is metadata, never a model input (it's kept out
        of :data:`FEATURE_COLUMNS`), but is used, when
        ``data.stratify_by_particle_type`` is enabled (default), to
        stratify the train/val/test split so each sample type keeps its
        overall proportion in every split -- see
        :func:`train_val_test_split`. It's exposed on the instance via
        :attr:`particle_type` after :meth:`setup`.
    """

    def __init__(self, config: Dict[str, Any], data_file: Optional[PathLike] = None) -> None:
        self.config = config
        self.data_cfg: Dict[str, Any] = config.get("data", {})
        self.graph_cfg: Dict[str, Any] = config.get("graph", {})
        self.seed: int = config.get("seed", 42)
        self.data_file: Optional[Path] = Path(data_file) if data_file is not None else None

        self.scaler: Optional[FeatureScaler] = None
        self._splits: Dict[str, np.ndarray] = {}
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._particle_type: Optional[np.ndarray] = None
        self._is_setup: bool = False

    @property
    def particle_type(self) -> np.ndarray:
        """Per-event ``particle_type`` labels, aligned with rows of ``X``/``y``.

        Populated by :meth:`setup`. Useful for tracing predictions/errors
        back to their source file, e.g. breaking down angular error by
        sample type in a custom analysis script.
        """
        self._check_setup()
        return self._particle_type

    @property
    def input_dim(self) -> int:
        """Number of input features (18)."""
        return len(FEATURE_COLUMNS)

    @property
    def output_dim(self) -> int:
        """Number of target dimensions (3: dir_x, dir_y, dir_z)."""
        return len(TARGET_COLUMNS)

    def setup(self) -> None:
        """Load data from disk, split, and fit the feature scaler.

        Idempotent: calling this multiple times on the same instance is a
        no-op after the first call.

        Raises:
            KeyError: If any expected feature/target column is missing from
                the loaded data.
        """
        if self._is_setup:
            return

        hdf_key = self.data_cfg.get("hdf_key", None)
        particle_type_regex = self.data_cfg.get("particle_type_regex", None)

        if self.data_file is not None:
            df = load_single_h5_file(self.data_file, key=hdf_key, particle_type_regex=particle_type_regex)
            logger.info(f"NeutrinoDataModule scoped to a single file: {self.data_file.name}")
        else:
            data_dir = self.data_cfg.get("data_dir", "data/")
            pattern = self.data_cfg.get("file_pattern", "*.h5")
            df = load_h5_files(data_dir, pattern=pattern, key=hdf_key, particle_type_regex=particle_type_regex)

        required_cols = FEATURE_COLUMNS + TARGET_COLUMNS
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in dataset: {missing}")

        if self.data_cfg.get("dropna", True):
            before = len(df)
            df = df.dropna(subset=required_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped}/{before} rows containing NaNs in required columns.")

        X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        particle_type = df[PARTICLE_TYPE_COLUMN].to_numpy()
        n = len(df)

        train_ratio = self.data_cfg.get("train_ratio", 0.7)
        val_ratio = self.data_cfg.get("val_ratio", 0.15)
        test_ratio = self.data_cfg.get("test_ratio", 0.15)
        stratify = self.data_cfg.get("stratify_by_particle_type", True)

        train_idx, val_idx, test_idx = train_val_test_split(
            n,
            train_ratio,
            val_ratio,
            test_ratio,
            seed=self.seed,
            stratify_labels=particle_type if stratify else None,
        )
        self._splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        self.scaler = FeatureScaler.fit(X[train_idx])
        self._X = self.scaler.transform(X)
        self._y = y
        self._particle_type = particle_type
        self._is_setup = True

        for split_name, idx in self._splits.items():
            split_counts = pd.Series(particle_type[idx]).value_counts().to_dict()
            logger.info(f"particle_type distribution [{split_name}]: {split_counts}")

        logger.info(
            f"NeutrinoDataModule ready: {n} events total "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}), "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )

    def _check_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError("NeutrinoDataModule.setup() must be called before accessing data.")

    def tabular_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build plain ``(X, y)`` DataLoaders for the MLP baseline.

        Returns:
            ``(train_loader, val_loader, test_loader)``. Only the training
            loader shuffles; a seeded ``torch.Generator`` makes the shuffle
            order reproducible.
        """
        self._check_setup()
        batch_size = self.data_cfg.get("batch_size", 256)
        num_workers = self.data_cfg.get("num_workers", 0)

        loaders = []
        for split_name, shuffle in (("train", True), ("val", False), ("test", False)):
            idx = self._splits[split_name]
            X_t = torch.from_numpy(self._X[idx])
            y_t = torch.from_numpy(self._y[idx])
            dataset = TensorDataset(X_t, y_t)

            generator = torch.Generator().manual_seed(self.seed) if shuffle else None
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=seed_worker if num_workers > 0 else None,
                generator=generator,
                drop_last=False,
            )
            loaders.append(loader)

        return tuple(loaders)  # type: ignore[return-value]

    def graph_datasets(self) -> Dict[str, Data]:
        """Build one self-contained k-NN population graph per split.

        Returns:
            A dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping
            to a ``torch_geometric.data.Data`` object with ``x`` (node
            features), ``y`` (node targets), and ``edge_index`` (k-NN edges
            computed independently within that split -- no cross-split edges).
        """
        self._check_setup()
        k = self.graph_cfg.get("k", 8)
        symmetric = self.graph_cfg.get("symmetric", True)

        datasets: Dict[str, Data] = {}
        for split_name in ("train", "val", "test"):
            idx = self._splits[split_name]
            X_split = self._X[idx]
            y_split = self._y[idx]

            edge_index = build_knn_edge_index(X_split, k=k, symmetric=symmetric)
            data = Data(
                x=torch.from_numpy(X_split),
                y=torch.from_numpy(y_split),
                edge_index=edge_index,
            )
            datasets[split_name] = data
            logger.info(
                f"k-NN graph [{split_name}]: {data.num_nodes} nodes, "
                f"{data.num_edges} edges (k={k}, symmetric={symmetric})"
            )
        return datasets

    def graph_dataloaders(self) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
        """Build mini-batch ``NeighborLoader`` loaders for GCN/GraphSAGE/GAT.

        Each batch is a subgraph sampled around a set of "seed" nodes: the
        first ``batch.batch_size`` nodes in the returned ``Data`` object are
        the seed (target) nodes for which loss/metrics should be computed;
        the remaining nodes are sampled neighbors included only to provide
        message-passing context. See ``trainer.py`` for how this convention
        is used (``out[:batch.batch_size]`` vs. ``batch.y[:batch.batch_size]``).

        Returns:
            ``(train_loader, val_loader, test_loader)``.
        """
        self._check_setup()
        datasets = self.graph_datasets()
        batch_size = self.data_cfg.get("batch_size", 256)
        num_neighbors = self.graph_cfg.get("num_neighbors", [10, 10])

        loaders = []
        for split_name, shuffle in (("train", True), ("val", False), ("test", False)):
            data = datasets[split_name]
            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            loaders.append(loader)

        return tuple(loaders)  # type: ignore[return-value]

    def __repr__(self) -> str:
        status = "setup" if self._is_setup else "not setup"
        return f"NeutrinoDataModule(status={status}, data_dir={self.data_cfg.get('data_dir')!r})"