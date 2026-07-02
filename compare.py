"""
compare.py
==========

Aggregates evaluation results across multiple trained runs (MLP, GCN,
GraphSAGE, GAT, ...) into a single comparison: a summary table (CSV +
Markdown) and comparison plots (mean angular error, mean cosine similarity,
and a box plot of full angular-error distributions).

This script only reads artifacts already written by ``evaluate.py``
(``metrics.json`` and ``angular_errors.npy`` inside each
``checkpoints/<run_name>/`` directory) -- it does not load models or run
inference itself, so it's fast and has no GPU dependency.

Usage
-----
    # Compare every evaluated run found under checkpoints/
    python compare.py

    # Compare specific runs by name
    python compare.py --runs mlp_20260701-090000 gat_20260701-113045

    # If multiple runs exist for the same model type, keep only the best
    python compare.py --best-per-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import PathLike, setup_logger

logger = setup_logger()


# --------------------------------------------------------------------------- #
# Discovery / loading
# --------------------------------------------------------------------------- #
def discover_runs(checkpoints_dir: PathLike, run_names: Optional[List[str]] = None) -> List[Path]:
    """Find run directories that have been evaluated (i.e. contain ``metrics.json``).

    Args:
        checkpoints_dir: Root checkpoints directory (e.g. ``checkpoints/``).
        run_names: If provided, restrict to these specific run directory
            names instead of auto-discovering every evaluated run.

    Returns:
        A sorted list of run directory paths.

    Raises:
        FileNotFoundError: If a requested run name doesn't exist or has no
            ``metrics.json`` (i.e. ``evaluate.py`` hasn't been run on it yet).
    """
    checkpoints_dir = Path(checkpoints_dir)

    if run_names:
        run_dirs = []
        for name in run_names:
            run_dir = checkpoints_dir / name
            if not (run_dir / "metrics.json").is_file():
                raise FileNotFoundError(
                    f"No metrics.json found for run '{name}' at {run_dir}. "
                    f"Run evaluate.py on it first."
                )
            run_dirs.append(run_dir)
        return run_dirs

    run_dirs = sorted(
        d for d in checkpoints_dir.iterdir() if d.is_dir() and (d / "metrics.json").is_file()
    )
    return run_dirs


def load_metrics_table(run_dirs: List[Path]) -> pd.DataFrame:
    """Load and concatenate ``metrics.json`` from each run directory into a DataFrame.

    Args:
        run_dirs: Run directories, each expected to contain ``metrics.json``
            as written by ``evaluate.py``.

    Returns:
        A DataFrame with one row per run, including an internal
        ``_run_dir`` column (not for display) pointing back to the source
        directory, used later to locate ``angular_errors.npy``.
    """
    rows = []
    for run_dir in run_dirs:
        with (run_dir / "metrics.json").open("r") as f:
            row = json.load(f)
        row["_run_dir"] = str(run_dir)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best (lowest mean angular error) run per model type.

    Useful when multiple training runs exist for the same architecture
    (e.g. from hyperparameter tuning) and only the best should be shown in
    the final comparison.

    Args:
        df: The full metrics table, as returned by :func:`load_metrics_table`.

    Returns:
        A DataFrame with exactly one row per unique ``model`` value.
    """
    return (
        df.sort_values("mean_angular_error_deg")
        .groupby("model", as_index=False)
        .first()
    )


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #
_DISPLAY_COLUMNS = [
    "model",
    "run_name",
    "mean_angular_error_deg",
    "median_angular_error_deg",
    "std_angular_error_deg",
    "mean_cosine_similarity",
]


def dataframe_to_markdown(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (no external deps).

    Args:
        df: The DataFrame to render.
        columns: Which columns to include, in order. Defaults to
            :data:`_DISPLAY_COLUMNS`, restricted to columns present in ``df``.

    Returns:
        A Markdown table as a single string.
    """
    cols = columns or [c for c in _DISPLAY_COLUMNS if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        formatted = []
        for c in cols:
            v = row[c]
            formatted.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: PathLike,
    std_metric: Optional[str] = None,
) -> None:
    """Plot a bar chart comparing one metric across models.

    Args:
        df: Comparison table with a ``model`` column and the metric column.
        metric: Column name to plot as bar height.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: Destination PNG path.
        std_metric: Optional column name to use as symmetric error bars.
    """
    models = df["model"].tolist()
    values = df[metric].tolist()
    errors = df[std_metric].tolist() if std_metric and std_metric in df.columns else None

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap("tab10").colors
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]
    bars = ax.bar(models, values, yerr=errors, capsize=5, color=bar_colors)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_angular_error_boxplot(run_dir_by_model: Dict[str, Path], output_path: PathLike) -> None:
    """Plot a box plot comparing full angular-error distributions across models.

    Reads ``angular_errors.npy`` (per-event angular errors, in degrees) from
    each run directory, as saved by ``evaluate.py``.

    Args:
        run_dir_by_model: Mapping from model name to its run directory.
        output_path: Destination PNG path.
    """
    data, labels = [], []
    for model_name, run_dir in run_dir_by_model.items():
        arr_path = Path(run_dir) / "angular_errors.npy"
        if arr_path.is_file():
            data.append(np.load(arr_path))
            labels.append(model_name)
        else:
            logger.warning(f"No angular_errors.npy for '{model_name}' at {arr_path}; excluding from box plot.")

    if not data:
        logger.warning("No angular_errors.npy files found for any model; skipping box plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Angular error (degrees)")
    ax.set_title("Model Comparison: Angular Error Distribution")
    ax.grid(axis="y", alpha=0.3)
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
    parser = argparse.ArgumentParser(
        description="Compare evaluated neutrino direction reconstruction models."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Root directory containing per-run checkpoint subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/comparison",
        help="Directory to save the comparison table and plots into.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="*",
        default=None,
        help="Specific run directory names to compare (default: auto-discover all evaluated runs).",
    )
    parser.add_argument(
        "--best-per-model",
        action="store_true",
        help="If multiple runs exist for the same model type, keep only the best (lowest mean angular error).",
    )
    return parser.parse_args()


def main() -> None:
    """Load metrics from all evaluated runs, build the comparison table and plots."""
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(checkpoints_dir, args.runs)
    if not run_dirs:
        raise FileNotFoundError(
            f"No evaluated runs (with metrics.json) found under {checkpoints_dir}. "
            f"Run train.py then evaluate.py first."
        )

    df = load_metrics_table(run_dirs)
    if args.best_per_model:
        df = select_best_per_model(df)
    df = df.sort_values("mean_angular_error_deg").reset_index(drop=True)

    # --- Table -------------------------------------------------------- #
    csv_path = output_dir / "comparison_table.csv"
    df.drop(columns=["_run_dir"]).to_csv(csv_path, index=False)

    md_table = dataframe_to_markdown(df)
    md_path = output_dir / "comparison_table.md"
    md_path.write_text(md_table)

    logger.info(f"Comparison table ({len(df)} runs):\n{md_table}")

    # --- Plots ---------------------------------------------------------- #
    plot_metric_bar(
        df,
        metric="mean_angular_error_deg",
        ylabel="Mean angular error (degrees)",
        title="Model Comparison: Mean Angular Error",
        output_path=output_dir / "comparison_angular_error_bar.png",
        std_metric="std_angular_error_deg",
    )
    plot_metric_bar(
        df,
        metric="mean_cosine_similarity",
        ylabel="Mean cosine similarity",
        title="Model Comparison: Mean Cosine Similarity",
        output_path=output_dir / "comparison_cosine_similarity_bar.png",
    )

    run_dir_by_model = {row["model"]: Path(row["_run_dir"]) for _, row in df.iterrows()}
    plot_angular_error_boxplot(run_dir_by_model, output_dir / "comparison_angular_error_boxplot.png")

    logger.info(f"Comparison table: {csv_path}")
    logger.info(f"Comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    main()