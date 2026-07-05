"""
utils.py
========

General-purpose, domain-agnostic utilities shared across the neutrino
direction-reconstruction framework.

This module intentionally has **no dependency on any other project module**
(``dataset.py``, ``models.py``, ``losses.py``, ``trainer.py``, ...). It only
depends on the standard library, NumPy, PyTorch and PyYAML. Keeping it a leaf
dependency avoids circular imports and makes it trivially reusable in other
projects.

Contents
--------
- Reproducibility: :func:`set_seed`
- Configuration:   :func:`load_config`, :func:`merge_configs`, :func:`build_run_name`
- Device handling: :func:`get_device`
- Bookkeeping:     :class:`AverageMeter`, :class:`EpochTimer`
- Checkpointing:   :func:`save_checkpoint`, :func:`load_checkpoint`
- Early stopping:  :class:`EarlyStopping`
- Geometry:        :func:`normalize_vectors`, :func:`angular_error_deg`
- Logging:         :func:`setup_logger`
- Misc:            :func:`count_parameters`
"""

from __future__ import annotations

import copy
import itertools
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducible experiments.

    This seeds Python's ``random``, NumPy, and PyTorch (CPU and all CUDA
    devices), and optionally forces deterministic cuDNN algorithms.

    Args:
        seed: The seed value to use everywhere.
        deterministic: If ``True``, disables cuDNN autotuning/non-determinism
            (``torch.backends.cudnn.deterministic = True``,
            ``torch.backends.cudnn.benchmark = False``) and sets
            ``PYTHONHASHSEED``. This can slow down training slightly but is
            required for bit-for-bit reproducible runs, which matters when
            comparing MLP / GCN / GraphSAGE / GAT under identical conditions.

    Note:
        Full determinism also depends on downstream code (e.g. DataLoader
        ``worker_init_fn`` and ``generator``) being seeded consistently.
        ``dataset.py`` and ``trainer.py`` are responsible for wiring the
        DataLoader-level seeding.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Available on newer PyTorch versions; fail silently otherwise.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch signature without `warn_only`.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` for reproducible multi-process loading.

    PyTorch's docs recommend deriving each worker's seed from
    ``torch.initial_seed()`` so that every worker gets a distinct but
    deterministic seed. Pass this to ``DataLoader(..., worker_init_fn=seed_worker)``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a single YAML file into a dictionary.

    Args:
        path: Path to a ``.yaml``/``.yml`` file.

    Returns:
        Parsed YAML content as a dictionary. An empty file yields ``{}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively deep-merge two config dictionaries.

    Keys in ``override`` take precedence over ``base``. Nested dictionaries
    are merged key-by-key rather than replaced wholesale, so a model config
    only needs to specify the keys it changes relative to ``base.yaml``.

    Args:
        base: The base configuration (e.g. loaded from ``configs/base.yaml``).
        override: The overriding configuration (e.g. ``configs/gat.yaml``).

    Returns:
        A new merged dictionary. Neither input is mutated.

    Example:
        >>> base = {"train": {"lr": 0.001, "epochs": 100}, "model": {"name": "mlp"}}
        >>> override = {"model": {"name": "gat", "heads": 4}}
        >>> merge_configs(base, override)
        {'train': {'lr': 0.001, 'epochs': 100}, 'model': {'name': 'gat', 'heads': 4}}
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment_config(base_path: PathLike, model_path: PathLike) -> Dict[str, Any]:
    """Load and merge a base config with a model-specific config.

    This is the single entry point ``train.py`` and ``evaluate.py`` should
    use to obtain the final, effective configuration for a run.

    Args:
        base_path: Path to ``configs/base.yaml`` (shared defaults: data,
            training loop, logging, seed, etc.).
        model_path: Path to a model-specific config (e.g. ``configs/gat.yaml``)
            containing at least a ``model.name`` key and any hyperparameters
            that differ from the base.

    Returns:
        The merged configuration dictionary.
    """
    base_cfg = load_config(base_path)
    model_cfg = load_config(model_path)
    return merge_configs(base_cfg, model_cfg)


def save_config(config: Dict[str, Any], path: PathLike) -> None:
    """Persist a (typically fully-resolved/merged) config dictionary to YAML.

    Used by ``train.py`` to save the effective configuration next to each
    checkpoint, so a given run's exact hyperparameters are always
    recoverable later without needing to remember which base/model YAML
    files (and any CLI overrides) produced it.

    Args:
        config: The configuration dictionary to save.
        path: Destination ``.yaml`` path; parent directories are created
            automatically if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def build_run_name(cfg: Dict[str, Any]) -> str:
    """Build a unique, human-readable run identifier from a config.

    The name encodes the model type, an optional hyperparameter tag, and a
    timestamp, e.g. ``gat_20260701-113045`` or, when ``cfg["run_tag"]`` is
    set (as done by ``sweep.py`` for each grid combination),
    ``gat_lr0.001_optimizeradamw_20260701-113045``. This name is used both
    as the TensorBoard run subdirectory (under ``runs/``) and the
    checkpoint subdirectory (under ``checkpoints/``), so a given
    experiment's logs and weights are always easy to correlate -- and, for
    sweeps, easy to tell apart at a glance.

    Args:
        cfg: The effective (merged) experiment configuration. Expects a
            ``model.name`` key (falls back to ``"model"`` if absent) and an
            optional top-level ``run_tag`` string.

    Returns:
        A filesystem-safe run name string.
    """
    model_name = cfg.get("model", {}).get("name", "model")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = cfg.get("run_tag")
    if tag:
        safe_tag = str(tag).replace(" ", "").replace("/", "-").replace(":", "-")
        return f"{model_name}_{safe_tag}_{timestamp}"
    return f"{model_name}_{timestamp}"


def set_by_dotted_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested config dict using a dotted key path, in place.

    Intermediate dictionaries are created automatically if they don't yet
    exist. Used by :func:`apply_overrides` to implement ``--override
    train.lr=0.0005``-style CLI arguments.

    Args:
        config: The config dictionary to mutate in place.
        dotted_key: Dotted path, e.g. ``"train.lr"`` or ``"model.activation"``.
        value: The value to set at that path.
    """
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply a list of ``"dotted.key=value"`` CLI-style overrides to a config.

    Values are parsed with YAML's own scalar rules (via ``yaml.safe_load``),
    so ``"0.001"`` becomes a float, ``"true"`` becomes a bool, ``"adamw"``
    stays a string, and ``"[1,2,3]"`` becomes a list -- matching how the
    same value would be written directly in a YAML file. This is what lets
    ``train.py --override train.optimizer=adamw --override train.lr=0.0005``
    behave identically to hand-editing those two keys in the YAML.

    Args:
        config: The base configuration dictionary (not mutated; a deep copy
            is modified and returned).
        overrides: A list of strings like ``"train.lr=0.0005"``.

    Returns:
        A new configuration dictionary with all overrides applied.

    Raises:
        ValueError: If an override string doesn't contain ``'='``.
    """
    cfg = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected format 'dotted.key=value'")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_path(cfg, dotted_key.strip(), value)
    return cfg


def expand_config_grid(
    config: Dict[str, Any],
    sections: Optional[List[str]] = None,
    exclude_keys: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Expand list-valued keys across one or more config sections into a full grid.

    Lets a single config file double as an inline hyperparameter grid:
    instead of a single value, any top-level key within any section in
    ``sections`` (e.g. ``model.num_layers``, ``model.activation``,
    ``train.lr``, ``train.optimizer``) may be given as a YAML list of
    candidate values, and this function expands that into one concrete
    config per combination -- the Cartesian product across *every*
    list-valued key found, across *all* scanned sections at once::

        model:
          name: mlp
          hidden_dim: 256
          num_layers: [4, 8, 16]
          activation: [relu, gelu]
        train:
          lr: [0.01, 0.001, 0.0001]

    expands to 3 x 2 x 3 = 18 concrete configs, each with one fixed
    ``num_layers``, ``activation``, and ``lr``. Nested dicts (e.g.
    ``train.scheduler``) are never treated as sweep candidates -- only keys
    whose *value* is itself a plain list are expanded.

    Each returned config also gets a ``run_tag`` describing which
    combination it is (e.g. ``"num_layers4_activationrelu_lr0.001"``),
    appended to any existing ``run_tag``, so ``utils.build_run_name`` gives
    every combination a distinct, self-describing run directory automatically.

    Args:
        config: The effective (merged) experiment configuration.
        sections: Which top-level config sections to scan for list-valued
            keys. Defaults to ``["model", "train"]``.
        exclude_keys: Per-section keys that must always be treated as a
            single scalar value even if given as a list, e.g.
            ``{"model": ["name"]}`` (the architecture name -- different
            architectures have different hyperparameter sets and can't be
            swept together). Defaults to ``{"model": ["name"]}``.

    Returns:
        A list of full config dicts. If no list-valued keys are found in
        any scanned section (the common case), returns a single-element
        list containing an unmodified deep copy of ``config`` -- so this
        function is always safe to call, even on a config with no sweep
        intent at all.
    """
    sections = sections if sections is not None else ["model", "train"]
    exclude_keys = exclude_keys if exclude_keys is not None else {"model": ["name"]}

    grid_locations: List[Tuple[str, str]] = []  # (section, key) pairs
    grid_values: List[List[Any]] = []

    for section in sections:
        section_cfg = config.get(section, {}) or {}
        section_exclude = exclude_keys.get(section, [])
        for key, value in section_cfg.items():
            if key in section_exclude:
                continue
            if isinstance(value, list):
                grid_locations.append((section, key))
                grid_values.append(value)

    if not grid_locations:
        return [copy.deepcopy(config)]

    combinations: List[Dict[str, Any]] = []
    for values in itertools.product(*grid_values):
        cfg = copy.deepcopy(config)
        tag_parts = []
        for (section, key), value in zip(grid_locations, values):
            cfg[section][key] = value
            tag_parts.append(f"{key}{value}")

        tag = "_".join(tag_parts).replace(" ", "")
        existing_tag = cfg.get("run_tag")
        cfg["run_tag"] = f"{existing_tag}_{tag}" if existing_tag else tag

        combinations.append(cfg)
    return combinations


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select the compute device.

    Args:
        prefer_cuda: If ``True`` and a CUDA device is available, use it.
            Otherwise fall back to CPU.

    Returns:
        A ``torch.device`` instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Tracks and computes the running average of a scalar metric.

    Typical usage inside a training loop::

        loss_meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(...)
            loss_meter.update(loss.item(), n=batch_size)
        print(loss_meter.avg)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all running statistics to zero."""
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update the running average with a new value.

        Args:
            val: The new (batch-level) metric value, typically already
                averaged over the batch (e.g. ``loss.item()``).
            n: The number of samples this value represents (e.g. batch size),
                used to weight the running average correctly for
                variable-sized batches (e.g. the last, smaller batch).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class EpochTimer:
    """Simple context manager / stopwatch for timing epochs.

    Example:
        >>> timer = EpochTimer()
        >>> with timer:
        ...     train_one_epoch(...)
        >>> print(timer.elapsed)  # seconds
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "EpochTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: The PyTorch module to inspect.
        trainable_only: If ``True``, only count parameters with
            ``requires_grad=True``.

    Returns:
        The total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: PathLike,
    filename: str = "last.pt",
    is_best: bool = False,
    best_filename: str = "best.pt",
) -> Path:
    """Save a training checkpoint to disk.

    Args:
        state: Arbitrary state dictionary to save, typically containing
            ``epoch``, ``model_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict``, ``best_val_loss`` and ``config``.
        checkpoint_dir: Directory to save into (created if missing).
        filename: Filename for the "latest" checkpoint, saved every epoch.
        is_best: If ``True``, additionally save (copy) this state as the
            best checkpoint under ``best_filename``.
        best_filename: Filename for the best checkpoint.

    Returns:
        The path to the "latest" checkpoint that was written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_path = checkpoint_dir / filename
    torch.save(state, last_path)

    if is_best:
        best_path = checkpoint_dir / best_filename
        torch.save(state, best_path)

    return last_path


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into a model (and optionally optimizer/scheduler).

    Args:
        path: Path to a ``.pt`` checkpoint file created by
            :func:`save_checkpoint`.
        model: Model to load ``model_state_dict`` into (in-place).
        optimizer: Optional optimizer to load ``optimizer_state_dict`` into.
        scheduler: Optional LR scheduler to load ``scheduler_state_dict`` into.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        The full checkpoint dictionary (useful for reading back ``epoch``,
        ``best_val_loss``, ``config``, etc.).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement after which training
            is stopped.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement.
        mode: ``"min"`` if lower is better (e.g. validation loss), ``"max"``
            if higher is better (e.g. cosine similarity).

    Example:
        >>> stopper = EarlyStopping(patience=10, mode="min")
        >>> for epoch in range(num_epochs):
        ...     val_loss = validate(...)
        ...     if stopper.step(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        return current > (self.best_score + self.min_delta)

    def step(self, current: float) -> bool:
        """Update state with the latest metric value.

        Args:
            current: The latest value of the monitored metric.

        Returns:
            ``True`` if training should stop now, ``False`` otherwise.
        """
        if self._is_improvement(current):
            self.best_score = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop


# --------------------------------------------------------------------------- #
# Geometry (shared by losses.py, trainer.py, evaluate.py)
# --------------------------------------------------------------------------- #
def normalize_vectors(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a batch of direction vectors to unit length.

    Args:
        vectors: Tensor of shape ``(N, 3)`` (or ``(N, D)`` in general).
        eps: Small constant added to the norm for numerical stability
            against division by (near) zero.

    Returns:
        Tensor of the same shape as ``vectors``, with each row having unit
        L2 norm.
    """
    norm = vectors.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the angular error (in degrees) between predicted and true directions.

    Both inputs are normalized to unit vectors before computing the angle,
    so callers do not need to pre-normalize raw model outputs.

    Args:
        pred: Predicted direction vectors, shape ``(N, 3)``.
        target: Ground-truth direction vectors, shape ``(N, 3)``.
        eps: Numerical stability constant, forwarded to normalization and
            used to clamp the cosine similarity into a valid ``arccos`` domain.

    Returns:
        Tensor of shape ``(N,)`` with the per-event angular error in degrees.
    """
    pred_n = normalize_vectors(pred, eps=eps)
    target_n = normalize_vectors(target, eps=eps)
    cos_sim = (pred_n * target_n).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    angle_rad = torch.acos(cos_sim)
    return torch.rad2deg(angle_rad)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(
    name: str = "neutrino_reco",
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a logger with a console handler and optional file handler.

    Safe to call multiple times with the same ``name``; handlers are not
    duplicated on repeated calls.

    Args:
        name: Logger name. Use the same name across a script to retrieve the
            same configured logger.
        log_file: If provided, also log to this file (in addition to stdout).
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured; avoid duplicate handlers on repeated calls.
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger