"""
train.py
========

CLI entry point for training neutrino direction reconstruction model(s).

Usage
-----
    python train.py config/gat.yaml
    python train.py config/mlp.yaml --seed 123
    python train.py config/gcn.yaml --resume checkpoints/gcn_20260701-101500/last.pt
    python train.py config/graphsage.yaml --device cpu
    python train.py config/mlp.yaml --epochs 30 --no-scheduler

Every hyperparameter (optimizer, lr, scheduler, batch size, hidden dims,
dropout, num layers, activation, weight decay, early stopping, loss type
and lambda, ...) comes from YAML (``config/baseline.yaml`` merged with the
model config you pass positionally). CLI flags here only control
operational concerns: which config to use, whether to resume, and
device/seed/epochs/scheduler overrides for quick experimentation --
``--epochs``/``--no-scheduler`` are shortcuts for the two overrides used
most often; anything else goes through ``--override dotted.key=value``.

Inline hyperparameter grids
----------------------------
Any top-level field within ``model:`` (other than ``name``) or ``train:``
may be given as a YAML list instead of a single value, e.g.::

    model:
      name: mlp
      hidden_dim: 256
      num_layers: [4, 8, 16]
      dropout: 0.2
      activation: [relu, leaky_relu, gelu, elu, tanh]
    train:
      lr: [0.01, 0.001, 0.0001]

Running ``python train.py config/mlp.yaml`` against a
config like this automatically trains every combination (here,
3 x 5 x 3 = 45 runs) in sequence -- each gets its own self-describing
checkpoint directory under ``checkpoints/``, and each is a complete,
independent training of ``train.epochs`` epochs at one fixed learning rate
(plus whatever ``train.scheduler`` does *within* that run -- see the note
below). The dataset (and, for GNNs, the k-NN graph) is built once and
reused across all combinations, since only architecture/training
hyperparameters differ between them.

One TensorBoard per config file
--------------------------------
Every combination trained from a single ``train.py`` invocation logs to
TensorBoard under one shared *group* directory,
``runs/<model>_<invocation_timestamp>/<combo_run_name>/`` -- one group per
model-config file you run, not one directory per combination. At the
end of the run, the log prints the exact command to view all of them
together, e.g.::

    tensorboard --logdir runs/mlp_20260703-193726

Opening that single TensorBoard session shows every combination's
train/val loss, cosine similarity, and learning-rate curves overlaid in
the Scalars tab (toggle individual combinations on/off in the sidebar),
plus a sortable table and parallel-coordinates view of every
combination's hyperparameters vs. final metrics in the HParams tab -- the
fastest way to see which configuration in the grid actually performed
best. For a single (non-grid) config, the run logs directly to
``runs/<run_name>/`` as before, and the same ``tensorboard --logdir``
command is printed for it.

Checkpoints are deliberately *not* grouped this way -- they stay flat
under ``checkpoints/<combo_run_name>/`` so ``evaluate.py`` and
``compare.py`` work unchanged.

Comparing learning rates vs. scheduling within a run -- don't confuse them
----------------------------------------------------------------------------
There are two *independent* mechanisms that both affect the learning rate,
and they answer different questions:

- ``train.lr`` (optionally a list, as above) is the **starting** learning
  rate of one full, independent training run of ``train.epochs`` epochs.
  Giving it as a list is how you compare different learning rates against
  each other -- e.g. does 0.01 converge better than 0.0001 -- each as its
  own complete, directly comparable training.
- ``train.scheduler`` changes the learning rate *within* a single one of
  those runs, epoch by epoch (e.g. ``type: step`` with ``step_size: 50``
  cuts the LR every 50 epochs; ``type: plateau`` cuts it when validation
  loss stops improving). This is a within-run optimization detail, not a
  way to "try several learning rates" -- with ``epochs: 150`` and
  ``step_size: 50`` you get one run that passes through 3 different LR
  values, not 3 comparable runs.

If you're sweeping ``train.lr`` to compare learning rates, and you want
each run's LR curve to be a clean, uninterrupted read on that starting LR
(no scheduler intervening), disable the scheduler for the sweep:

    python train.py config/mlp.yaml --no-scheduler

Otherwise, every swept LR value is still subject to the *same* schedule
policy from ``base.yaml`` (e.g. still gets cut by the same plateau rule) --
which is usually what you want, since it keeps "same training procedure"
true across the comparison, but it does mean two runs' LR curves won't be
flat lines at different heights.

"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dataset import NeutrinoDataModule
from models import build_model
from trainer import Trainer
from utils import (
    apply_overrides,
    build_run_name,
    count_parameters,
    expand_config_grid,
    get_device,
    load_experiment_config,
    save_config,
    set_seed,
    setup_logger,
)

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a neutrino direction reconstruction model (MLP / GCN / GraphSAGE / GAT)."
    )
    parser.add_argument(
        "model_config",
        type=str,
        help="Path to the model-specific YAML config (e.g. config/mlp.yaml). May contain inline "
        "hyperparameter lists (see module docstring) to train several combinations in one run.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="config/baseline.yaml",
        help="Path to the shared base YAML config. Defaults to this project's actual base config; "
        "override only if you keep multiple base configs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from. Only valid for a single-combination run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the 'seed' value from config, for quick reproducibility experiments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override automatic device selection.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Shortcut for --override train.epochs=N.",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Shortcut for --override train.scheduler.type=none (disable the LR scheduler for this run).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config value using a dotted path, e.g. "
            "--override train.lr=0.0005 --override model.activation=elu. "
            "Repeatable. Applied before inline grid expansion. --epochs and "
            "--no-scheduler are shortcuts for the two most common overrides; "
            "use --override for anything else."
        ),
    )
    return parser.parse_args()


def resolve_device(config: dict, cli_override: Optional[str]) -> torch.device:
    """Resolve the compute device from config and optional CLI override.

    Args:
        config: The effective (merged) experiment configuration. Reads the
            top-level ``device`` key (``"auto"`` | ``"cuda"`` | ``"cpu"``,
            defaulting to ``"auto"``).
        cli_override: If provided, takes precedence over the config value.

    Returns:
        The resolved ``torch.device``.
    """
    device_str = cli_override or str(config.get("device", "auto")).lower()
    if device_str == "cpu":
        return torch.device("cpu")
    return get_device(prefer_cuda=True)  # "auto" or "cuda": use CUDA if available


def get_or_build_loaders(
    model_requires_graph: bool,
    dm: NeutrinoDataModule,
    cache: Dict[str, Optional[Tuple[Any, Any, Any]]],
) -> Tuple[Any, Any]:
    """Fetch (building + caching on first use) the DataLoaders matching a model's needs.

    Building the k-NN graph / DataLoaders is identical across every
    combination in an inline hyperparameter grid (only ``model:``
    hyperparameters differ), so this avoids redundant work by caching each
    loader type the first time it's needed.

    Args:
        model_requires_graph: ``model.requires_graph`` for the model about
            to be trained.
        dm: The (already set up) data module to pull loaders from.
        cache: A dict with keys ``"tabular"`` and ``"graph"``, mutated in
            place to memoize loader tuples across calls.

    Returns:
        ``(train_loader, val_loader)`` of the appropriate type.
    """
    key = "graph" if model_requires_graph else "tabular"
    if cache.get(key) is None:
        loaders = dm.graph_dataloaders() if model_requires_graph else dm.tabular_dataloaders()
        cache[key] = loaders
    train_loader, val_loader, _test_loader = cache[key]
    return train_loader, val_loader


def run_single_config(
    run_config: Dict[str, Any],
    dm: NeutrinoDataModule,
    loader_cache: Dict[str, Optional[Tuple[Any, Any, Any]]],
    device: torch.device,
    resume_from: Optional[str],
    combo_index: int,
    combo_total: int,
    group_name: str,
) -> Dict[str, Any]:
    """Train one concrete configuration (one point in the grid) end to end.

    Args:
        run_config: A single, fully concrete (no list-valued model fields)
            effective configuration.
        dm: The (already set up) shared data module.
        loader_cache: Shared loader cache, see :func:`get_or_build_loaders`.
        device: Compute device.
        resume_from: Optional checkpoint path to resume from (only
            meaningful when ``combo_total == 1``).
        combo_index: 1-based index of this combination, for logging.
        combo_total: Total number of combinations in this grid, for logging.
        group_name: Shared identifier for this whole train.py invocation
            (one per model-config file run, computed once in
            :func:`main`). When ``combo_total > 1``, every combination's
            TensorBoard logs are nested under
            ``<runs_dir>/<group_name>/<run_name>`` instead of directly
            under ``<runs_dir>/<run_name>``, so a single
            ``tensorboard --logdir <runs_dir>/<group_name>`` shows every
            configuration from this config file's grid together (overlaid
            scalar curves + one row per configuration in the HParams tab).
            Checkpoints are intentionally not nested this way -- they stay
            flat under ``<checkpoints_dir>/<run_name>`` so
            ``evaluate.py``/``compare.py`` need no changes.

    Returns:
        A summary dict with ``run_name``, ``checkpoint_dir``, and the
        ``Trainer.fit()`` result (``best_val_loss``, ``epochs_trained``).
    """
    seed = run_config.get("seed", 42)
    deterministic = run_config.get("deterministic", True)
    set_seed(seed, deterministic=deterministic)  # reset per combination for independent, reproducible init

    run_name = build_run_name(run_config)
    logging_cfg = run_config.get("logging", {})
    runs_root = Path(logging_cfg.get("runs_dir", "runs"))
    run_dir = (runs_root / group_name / run_name) if combo_total > 1 else (runs_root / run_name)
    checkpoint_dir = Path(logging_cfg.get("checkpoints_dir", "checkpoints")) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_logger = setup_logger(name=run_name, log_file=checkpoint_dir / "train.log")
    run_logger.info(f"[{combo_index}/{combo_total}] Run name: {run_name}")

    save_config(run_config, checkpoint_dir / "config.yaml")

    model = build_model(run_config, dm.input_dim, dm.output_dim)
    run_logger.info(
        f"[{combo_index}/{combo_total}] Model: {model.__class__.__name__} | "
        f"requires_graph={model.requires_graph} | trainable_params={count_parameters(model):,} | "
        f"model_config={ {k: v for k, v in run_config.get('model', {}).items()} }"
    )

    train_loader, val_loader = get_or_build_loaders(model.requires_graph, dm, loader_cache)

    trainer = Trainer(
        model=model,
        config=run_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        logger=run_logger,
    )
    fit_summary = trainer.fit()

    run_logger.info(
        f"[{combo_index}/{combo_total}] Training complete. Best val loss: "
        f"{fit_summary['best_val_loss']:.5f} after {fit_summary['epochs_trained']} epochs."
    )
    run_logger.info(f"[{combo_index}/{combo_total}] Checkpoints: {checkpoint_dir}")
    run_logger.info(f"[{combo_index}/{combo_total}] Next: python evaluate.py --checkpoint {checkpoint_dir / 'best.pt'}")

    # Release the model/trainer and any cached CUDA memory before the next
    # combination -- important when sweeping several architecture variants
    # (e.g. num_layers) in one process.
    del trainer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "checkpoint_dir": str(checkpoint_dir),
        "status": "ok",
        **fit_summary,
    }


def main() -> None:
    """Load config, expand any inline hyperparameter grid, and train each combination."""
    args = parse_args()

    overrides = list(args.override)
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.no_scheduler:
        overrides.append("train.scheduler.type=none")

    config = load_experiment_config(args.base_config, args.model_config)
    config = apply_overrides(config, overrides)

    if args.seed is not None:
        config["seed"] = args.seed

    configs_to_run = expand_config_grid(config, sections=["model", "train"], exclude_keys={"model": ["name"]})
    n_combos = len(configs_to_run)

    # One group per train.py invocation (i.e. per model-config file run),
    # computed once here rather than per combination, so every combination
    # in an inline grid shares the same TensorBoard group directory. See
    # `run_single_config`'s `group_name` docstring for how this is used.
    model_name = config.get("model", {}).get("name", "model")
    group_name = build_run_name({"model": {"name": model_name}})
    runs_root = Path(config.get("logging", {}).get("runs_dir", "runs"))

    logger.info(f"Base config: {args.base_config} | Model config: {args.model_config}")
    if overrides:
        logger.info(f"CLI overrides applied: {overrides}")
    if n_combos > 1:
        logger.info(f"Detected an inline hyperparameter grid: {n_combos} combinations will be trained in sequence.")
        logger.info(f"All combinations will log to TensorBoard under: {runs_root / group_name}")
        if args.resume:
            logger.warning("--resume is ignored when training a hyperparameter grid (ambiguous target); disabling it.")

    device = resolve_device(config, args.device)
    logger.info(f"Using device: {device}")

    # --- Data (built once; identical across every combination) --------- #
    dm = NeutrinoDataModule(config)
    dm.setup()
    loader_cache: Dict[str, Optional[Tuple[Any, Any, Any]]] = {"tabular": None, "graph": None}

    resume_from = args.resume if n_combos == 1 else None

    results: List[Dict[str, Any]] = []
    for i, run_config in enumerate(configs_to_run, start=1):
        try:
            result = run_single_config(run_config, dm, loader_cache, device, resume_from, i, n_combos, group_name)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one bad combo shouldn't kill the grid
            logger.error(f"[{i}/{n_combos}] Combination failed: {exc}")
            results.append({"run_name": None, "status": "failed", "error": str(exc)})
            continue
        results.append(result)

    if n_combos > 1:
        logger.info("Grid summary:")
        for r in results:
            if r["status"] == "ok":
                logger.info(f"  OK     {r['run_name']:60s} best_val_loss={r['best_val_loss']:.5f}")
            else:
                logger.info(f"  FAILED {r.get('error', 'unknown error')}")
        logger.info("Aggregate results (after running evaluate.py on each) with: python compare.py")
        logger.info(
            f"View every configuration from this run in one place (overlaid Scalars + a sortable "
            f"HParams table) with: tensorboard --logdir {runs_root / group_name}"
        )
    else:
        only_result = results[0] if results else None
        if only_result and only_result["status"] == "ok":
            single_run_dir = runs_root / only_result["run_name"]
            logger.info(f"View this run's TensorBoard logs with: tensorboard --logdir {single_run_dir}")


if __name__ == "__main__":
    main()