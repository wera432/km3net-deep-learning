"""
evaluate.py
===========

Evaluation script: loads a trained checkpoint, reconstructs the exact test
split it was trained against, runs inference, and produces the required
diagnostic figures:

1. Loss curves (train vs. val, over epochs) -- from ``history.json``.
2. Cosine similarity curves (train vs. val, over epochs) -- from ``history.json``.
3. Prediction vs. target scatter, per direction component (dir_x/dir_y/dir_z).
4. Angular error histogram (degrees), on the held-out test set.

It also writes a small ``metrics.json`` summary (mean/median/std angular
error, mean cosine similarity, test loss) that ``compare.py`` aggregates
across multiple runs.

Usage
-----
    python evaluate.py --checkpoint checkpoints/gat_20260701-113045/best.pt

The checkpoint's saved ``config.yaml`` (written by ``train.py`` next to the
checkpoint) is used to exactly reproduce the data split and preprocessing,
so evaluation never depends on remembering which config/flags were used
originally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive backend, safe for headless/CI runs

from dataset import NeutrinoDataModule
from models import build_model
from utils import (
    PathLike,
    angular_error_deg,
    get_device,
    load_checkpoint,
    load_config,
    normalize_vectors,
    setup_logger,
)

logger = setup_logger()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: Any, device: torch.device, is_graph_model: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over an entire loader and collect predictions/targets.

    Uses the same seed-node slicing convention as ``trainer.Trainer`` for
    graph models (see ``trainer._forward_batch``), so evaluation metrics are
    computed identically to how training/validation metrics were computed.

    Args:
        model: A trained :class:`models.DirectionRegressor` in eval mode.
        loader: A tabular ``DataLoader`` or graph ``NeighborLoader``.
        device: Device to run inference on.
        is_graph_model: Whether ``model`` requires graph structure (i.e.
            ``model.requires_graph``).

    Returns:
        A tuple ``(predictions, targets)`` of NumPy arrays, each shape
        ``(num_samples, 3)``, concatenated over all batches.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if is_graph_model:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            seed_size = batch.batch_size
            pred = out[:seed_size]
            target = batch.y[:seed_size]
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            target = y

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute summary metrics on the test set.

    Args:
        predictions: Raw (unnormalized) model outputs, shape ``(N, 3)``.
        targets: Ground-truth unit direction vectors, shape ``(N, 3)``.

    Returns:
        A dict with ``mean_angular_error_deg``, ``median_angular_error_deg``,
        ``std_angular_error_deg``, and ``mean_cosine_similarity``.
    """
    pred_t = torch.from_numpy(predictions)
    target_t = torch.from_numpy(targets)

    ang_err = angular_error_deg(pred_t, target_t).numpy()
    pred_n = normalize_vectors(pred_t).numpy()
    target_n = normalize_vectors(target_t).numpy()
    cos_sim = (pred_n * target_n).sum(axis=-1)

    return {
        "mean_angular_error_deg": float(np.mean(ang_err)),
        "median_angular_error_deg": float(np.median(ang_err)),
        "std_angular_error_deg": float(np.std(ang_err)),
        "mean_cosine_similarity": float(np.mean(cos_sim)),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_loss_curves(history: list, output_path: PathLike, model_name: str) -> None:
    """Plot train vs. validation loss over epochs and save as PNG.

    Args:
        history: List of per-epoch metric dicts, as written by
            ``trainer.Trainer`` to ``history.json``.
        output_path: Destination PNG path.
        model_name: Model name, used in the plot title.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="Train loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val loss", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name}: Training / Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cosine_similarity_curves(history: list, output_path: PathLike, model_name: str) -> None:
    """Plot train vs. validation cosine similarity over epochs and save as PNG.

    Args:
        history: List of per-epoch metric dicts, as written by
            ``trainer.Trainer`` to ``history.json``.
        output_path: Destination PNG path.
        model_name: Model name, used in the plot title.
    """
    epochs = [h["epoch"] for h in history]
    train_cos = [h["train_cosine_similarity"] for h in history]
    val_cos = [h["val_cosine_similarity"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_cos, label="Train cosine similarity", color="#2ca02c")
    ax.plot(epochs, val_cos, label="Val cosine similarity", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"{model_name}: Training / Validation Cosine Similarity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction_vs_target(
    predictions: np.ndarray, targets: np.ndarray, output_path: PathLike, model_name: str
) -> None:
    """Plot predicted vs. true value for each direction component (dir_x, dir_y, dir_z).

    Predictions are normalized to unit vectors before plotting, so the
    comparison is on the same physical scale as the targets.

    Args:
        predictions: Raw (unnormalized) model outputs, shape ``(N, 3)``.
        targets: Ground-truth unit direction vectors, shape ``(N, 3)``.
        output_path: Destination PNG path.
        model_name: Model name, used in the plot title.
    """
    pred_n = normalize_vectors(torch.from_numpy(predictions)).numpy()
    component_names = ["dir_x", "dir_y", "dir_z"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        ax.scatter(targets[:, i], pred_n[:, i], s=2, alpha=0.15, color="#1f77b4")
        ax.plot([-1, 1], [-1, 1], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{model_name}: Prediction vs. Target (Test Set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_angular_error_histogram(
    predictions: np.ndarray, targets: np.ndarray, output_path: PathLike, model_name: str
) -> None:
    """Plot a histogram of per-event angular error (degrees) on the test set.

    Args:
        predictions: Raw (unnormalized) model outputs, shape ``(N, 3)``.
        targets: Ground-truth unit direction vectors, shape ``(N, 3)``.
        output_path: Destination PNG path.
        model_name: Model name, used in the plot title.
    """
    ang_err = angular_error_deg(torch.from_numpy(predictions), torch.from_numpy(targets)).numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ang_err, bins=60, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.axvline(
        float(np.median(ang_err)),
        color="red",
        linestyle="--",
        label=f"Median = {np.median(ang_err):.2f} deg",
    )
    ax.set_xlabel("Angular error (degrees)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"{model_name}: Angular Error Distribution (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained neutrino direction reconstruction model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint .pt file (typically checkpoints/<run_name>/best.pt).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save figures into. Defaults to figures/<run_name>/.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint, run test-set evaluation, and save figures + metrics."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    run_name = run_dir.name

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected a resolved config at {config_path} (written by train.py). "
            f"Cannot evaluate without it."
        )
    config = load_config(config_path)

    figures_dir = Path(args.figures_dir) if args.figures_dir else Path("figures") / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=True)
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Using device: {device}")

    # --- Rebuild data (same seed/config -> identical split + scaler) --- #
    dm = NeutrinoDataModule(config)
    dm.setup()

    # --- Rebuild model architecture, then load trained weights ------------- #
    model = build_model(config, dm.input_dim, dm.output_dim).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model_name = config.get("model", {}).get("name", "model")
    logger.info(f"Loaded weights from epoch {checkpoint.get('epoch', '?')}")

    # --- Test-set inference --------------------------------------------- #
    if model.requires_graph:
        _train_loader, _val_loader, test_loader = dm.graph_dataloaders()
    else:
        _train_loader, _val_loader, test_loader = dm.tabular_dataloaders()

    predictions, targets = run_inference(model, test_loader, device, model.requires_graph)
    logger.info(f"Ran inference on {len(predictions)} test events.")

    metrics = compute_metrics(predictions, targets)
    logger.info(f"Test metrics: {metrics}")

    # Persist raw per-event angular errors (not just summary stats) so
    # compare.py can build distribution-level comparisons (e.g. box plots)
    # across models, not only bar charts of means.
    ang_err_array = angular_error_deg(torch.from_numpy(predictions), torch.from_numpy(targets)).numpy()
    np.save(run_dir / "angular_errors.npy", ang_err_array)

    # --- Figures ---------------------------------------------------------- #
    history_path = run_dir / "history.json"
    if history_path.is_file():
        with history_path.open("r") as f:
            history = json.load(f)
        plot_loss_curves(history, figures_dir / "loss_curves.png", model_name)
        plot_cosine_similarity_curves(history, figures_dir / "cosine_similarity_curves.png", model_name)
        logger.info("Saved loss_curves.png and cosine_similarity_curves.png")
    else:
        logger.warning(f"No history.json found at {history_path}; skipping training curve plots.")

    plot_prediction_vs_target(predictions, targets, figures_dir / "prediction_vs_target.png", model_name)
    plot_angular_error_histogram(predictions, targets, figures_dir / "angular_error_histogram.png", model_name)
    logger.info("Saved prediction_vs_target.png and angular_error_histogram.png")

    # --- Metrics summary (consumed by compare.py) ---------------------- #
    metrics_out = {
        "run_name": run_name,
        "model": model_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        **metrics,
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Saved metrics summary to {metrics_path}")
    logger.info(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()