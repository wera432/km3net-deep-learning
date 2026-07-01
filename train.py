"""
train.py
========

CLI entry point for training a single neutrino direction reconstruction
model.

Usage
-----
    python train.py --model-config configs/gat.yaml
    python train.py --model-config configs/mlp.yaml --seed 123
    python train.py --model-config configs/gcn.yaml --resume checkpoints/gcn_20260701-101500/last.pt
    python train.py --model-config configs/graphsage.yaml --device cpu

Every hyperparameter (optimizer, lr, scheduler, batch size, hidden dims,
dropout, num layers, activation, weight decay, early stopping, loss type
and lambda, ...) comes from YAML (``configs/base.yaml`` merged with the
chosen ``--model-config``). CLI flags here only control operational
concerns: which config to use, whether to resume, and device/seed
overrides for quick experimentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import NeutrinoDataModule
from models import build_model
from trainer import Trainer
from utils import (
    build_run_name,
    count_parameters,
    get_device,
    load_experiment_config,
    save_config,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a neutrino direction reconstruction model (MLP / GCN / GraphSAGE / GAT)."
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to the shared base YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model-specific YAML config (e.g. configs/gat.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
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
    return parser.parse_args()


def resolve_device(config: dict, cli_override: str | None) -> torch.device:
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


def main() -> None:
    """Load config, build the data module / model / trainer, and run training."""
    args = parse_args()
    config = load_experiment_config(args.base_config, args.model_config)

    if args.seed is not None:
        config["seed"] = args.seed
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", True)
    set_seed(seed, deterministic=deterministic)

    run_name = build_run_name(config)
    logging_cfg = config.get("logging", {})
    run_dir = Path(logging_cfg.get("runs_dir", "runs")) / run_name
    checkpoint_dir = Path(logging_cfg.get("checkpoints_dir", "checkpoints")) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=checkpoint_dir / "train.log")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Base config: {args.base_config} | Model config: {args.model_config}")

    save_config(config, checkpoint_dir / "config.yaml")
    logger.info(f"Resolved config saved to {checkpoint_dir / 'config.yaml'}")

    device = resolve_device(config, args.device)
    logger.info(f"Using device: {device}")

    # --- Data --------------------------------------------------------- #
    dm = NeutrinoDataModule(config)
    dm.setup()

    # --- Model ---------------------------------------------------------- #
    model = build_model(config, dm.input_dim, dm.output_dim)
    logger.info(
        f"Model: {model.__class__.__name__} | requires_graph={model.requires_graph} | "
        f"trainable_params={count_parameters(model):,}"
    )

    # --- Loaders (tabular for MLP, graph mini-batches for GNNs) -------- #
    if model.requires_graph:
        train_loader, val_loader, _test_loader = dm.graph_dataloaders()
    else:
        train_loader, val_loader, _test_loader = dm.tabular_dataloaders()

    # --- Train ----------------------------------------------------------- #
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
        logger=logger,
    )
    summary = trainer.fit()

    logger.info(
        f"Training complete. Best val loss: {summary['best_val_loss']:.5f} "
        f"after {summary['epochs_trained']} epochs."
    )
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"TensorBoard logs: {run_dir}")
    logger.info(
        f"Next: python evaluate.py --checkpoint {checkpoint_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()