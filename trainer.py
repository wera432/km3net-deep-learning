"""
trainer.py
==========

The training engine: optimizer/scheduler construction from config, the
train/validation loop, TensorBoard logging, checkpoint saving, early
stopping, and a plain-JSON training history for downstream plotting.

``Trainer`` is deliberately model-agnostic: it does not know whether it is
training an MLP or a GAT. It asks ``model.requires_graph`` (see
``models.py``) to decide how to unpack a batch, and treats every batch --
tabular ``(x, y)`` tuple or PyTorch Geometric mini-batch from
``NeighborLoader`` -- through the same uniform ``_forward_batch`` path. This
means the exact same ``Trainer`` code trains all four architectures under
identical conditions, which is the whole point of the comparison this
project is built around.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import NeighborLoader

from loss import DirectionLoss, build_loss
from utils import (
    AverageMeter,
    EarlyStopping,
    EpochTimer,
    PathLike,
    count_parameters,
    load_checkpoint,
    normalize_vectors,
    save_checkpoint,
    setup_logger,
)

BatchLoader = Union[DataLoader, NeighborLoader]


# --------------------------------------------------------------------------- #
# Optimizer / scheduler factories
# --------------------------------------------------------------------------- #
_OPTIMIZERS: Dict[str, Any] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def build_optimizer(parameters: Any, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer purely from configuration.

    Reads the ``train`` section::

        train:
          optimizer: adamw     # adam | adamw | sgd | rmsprop
          lr: 0.001
          weight_decay: 0.0001
          momentum: 0.9         # only used by sgd

    Args:
        parameters: Model parameters (e.g. ``model.parameters()``).
        config: The effective (merged) experiment configuration.

    Returns:
        An instantiated ``torch.optim.Optimizer``.

    Raises:
        ValueError: If ``train.optimizer`` is unrecognized.
    """
    train_cfg = config.get("train", {})
    name = str(train_cfg.get("optimizer", "adam")).lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_OPTIMIZERS)}")

    kwargs: Dict[str, Any] = {
        "lr": train_cfg.get("lr", 1e-3),
        "weight_decay": train_cfg.get("weight_decay", 0.0),
    }
    if name == "sgd":
        kwargs["momentum"] = train_cfg.get("momentum", 0.9)

    return _OPTIMIZERS[name](parameters, **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], str]:
    """Build a learning-rate scheduler purely from configuration.

    Reads the ``train.scheduler`` section::

        train:
          epochs: 100
          scheduler:
            type: cosine        # none | step | cosine | plateau | exponential
            step_size: 30        # step
            gamma: 0.1            # step / exponential
            t_max: 100             # cosine (defaults to train.epochs)
            factor: 0.5              # plateau
            patience: 5                # plateau

    Args:
        optimizer: The optimizer to attach the scheduler to.
        config: The effective (merged) experiment configuration.

    Returns:
        A tuple ``(scheduler, scheduler_type)``. ``scheduler`` is ``None``
        when ``type: none``. ``scheduler_type`` is returned separately
        because ``ReduceLROnPlateau`` must be stepped with a metric
        (``scheduler.step(val_loss)``) while all others are stepped with no
        arguments (``scheduler.step()``) -- the caller needs to know which.

    Raises:
        ValueError: If ``train.scheduler.type`` is unrecognized.
    """
    train_cfg = config.get("train", {})
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    name = str(sched_cfg.get("type", "none")).lower()

    if name in ("none", ""):
        return None, "none"

    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", train_cfg.get("epochs", 100)),
        )
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
        )
    elif name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=sched_cfg.get("gamma", 0.95)
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Available: none, step, cosine, plateau, exponential"
        )

    return scheduler, name


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Trains a :class:`models.DirectionRegressor` and manages the full training loop.

    Args:
        model: A model instance (e.g. from ``models.build_model``). Its
            ``requires_graph`` property determines how batches are unpacked.
        config: The effective (merged) experiment configuration.
        train_loader: Training DataLoader -- either a plain ``DataLoader``
            (tabular models) or a ``NeighborLoader`` (graph models), matching
            ``model.requires_graph``.
        val_loader: Validation loader, same type as ``train_loader``.
        device: Compute device to train on.
        run_dir: Directory for TensorBoard logs (e.g. ``runs/<run_name>``).
        checkpoint_dir: Directory for checkpoints (e.g. ``checkpoints/<run_name>``).
        resume_from: Optional path to a checkpoint to resume training from.
        logger: Optional pre-configured logger; a default one is created if omitted.

    Attributes:
        history: List of per-epoch metric dicts, appended to during
            :meth:`fit` and persisted as ``<checkpoint_dir>/history.json``
            after every epoch (so partial history survives an interrupted run).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        device: torch.device,
        run_dir: PathLike,
        checkpoint_dir: PathLike,
        resume_from: Optional[PathLike] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.train_cfg = config.get("train", {})
        self.device = device
        self.logger = logger or setup_logger()

        self.model = model.to(device)
        self.is_graph_model: bool = getattr(model, "requires_graph", False)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn: DirectionLoss = build_loss(config).to(device)
        self.optimizer = build_optimizer(self.model.parameters(), config)
        self.scheduler, self.scheduler_type = build_scheduler(self.optimizer, config)

        self.epochs: int = self.train_cfg.get("epochs", 100)
        self.grad_clip: Optional[float] = self.train_cfg.get("grad_clip", None)

        early_stop_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.early_stopping: Optional[EarlyStopping] = None
        if early_stop_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=early_stop_cfg.get("patience", 15),
                min_delta=early_stop_cfg.get("min_delta", 0.0),
                mode="min",
            )

        self.run_dir = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        self.start_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.history: List[Dict[str, float]] = []

        if resume_from is not None:
            self._resume(resume_from)

        self.logger.info(
            f"Trainer ready: model={self.model.__class__.__name__} "
            f"({count_parameters(self.model):,} trainable params), "
            f"device={self.device}, graph_model={self.is_graph_model}"
        )

    # ----------------------------------------------------------------- #
    # Resume
    # ----------------------------------------------------------------- #
    def _resume(self, path: PathLike) -> None:
        """Load a checkpoint and restore optimizer/scheduler/epoch state."""
        checkpoint = load_checkpoint(
            path, self.model, optimizer=self.optimizer, scheduler=self.scheduler, map_location=self.device
        )
        self.start_epoch = checkpoint.get("epoch", -1) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.logger.info(f"Resumed from {path} at epoch {self.start_epoch} (best_val_loss={self.best_val_loss:.5f})")

    # ----------------------------------------------------------------- #
    # Batch handling (unifies tabular and graph models)
    # ----------------------------------------------------------------- #
    def _forward_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Run the model forward on one batch, returning (pred, target, n).

        For graph models, ``batch`` is a ``NeighborLoader``-produced
        ``Data`` object where the first ``batch.batch_size`` nodes are the
        "seed" nodes the batch was sampled around; the remainder are
        neighbors included only for message-passing context. Loss and
        metrics must only be computed on the seed nodes, so predictions and
        targets are sliced to ``[:batch.batch_size]`` before being returned.

        For tabular models, ``batch`` is a plain ``(x, y)`` tuple from a
        standard ``DataLoader`` and every row is used directly.

        Args:
            batch: A batch yielded by ``self.train_loader``/``self.val_loader``.

        Returns:
            ``(pred, target, n)`` where ``pred``/``target`` have shape
            ``(n, output_dim)`` and ``n`` is the number of samples actually
            contributing to the loss for this batch.
        """
        if self.is_graph_model:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            seed_size = batch.batch_size
            pred = out[:seed_size]
            target = batch.y[:seed_size]
            return pred, target, seed_size

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.model(x)
        return pred, y, x.size(0)

    # ----------------------------------------------------------------- #
    # Epoch loops
    # ----------------------------------------------------------------- #
    def _run_epoch(self, loader: BatchLoader, train: bool) -> Dict[str, float]:
        """Run one full pass over ``loader``, either training or evaluating.

        Args:
            loader: The DataLoader/NeighborLoader to iterate.
            train: If ``True``, put the model in train mode, compute
                gradients, and step the optimizer. If ``False``, run in
                ``eval()`` mode under ``torch.no_grad()``.

        Returns:
            A dict of epoch-averaged metrics: at least ``"loss"`` and
            ``"cosine_similarity"``, plus every component returned by
            ``self.loss_fn.compute_components`` (e.g. ``"mse_loss"`` for
            :class:`losses.CosineMSELoss`).
        """
        self.model.train(mode=train)

        loss_meter = AverageMeter()
        cos_meter = AverageMeter()
        component_meters: Dict[str, AverageMeter] = {}

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()

                pred, target, n = self._forward_batch(batch)
                components = self.loss_fn.compute_components(pred, target)
                loss = components["total_loss"]

                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                loss_meter.update(loss.item(), n)

                pred_n = normalize_vectors(pred.detach())
                target_n = normalize_vectors(target.detach())
                cos_sim = (pred_n * target_n).sum(dim=-1).mean().item()
                cos_meter.update(cos_sim, n)

                for name, value in components.items():
                    if name not in component_meters:
                        component_meters[name] = AverageMeter()
                    component_meters[name].update(value.item(), n)

        metrics: Dict[str, float] = {"loss": loss_meter.avg, "cosine_similarity": cos_meter.avg}
        for name, meter in component_meters.items():
            metrics[name] = meter.avg
        return metrics

    # ----------------------------------------------------------------- #
    # Full training loop
    # ----------------------------------------------------------------- #
    def fit(self) -> Dict[str, Any]:
        """Run the full training loop: epochs, logging, checkpointing, early stopping.

        Returns:
            A summary dict with ``"best_val_loss"`` and the number of epochs
            actually completed (``"epochs_trained"``).
        """
        self.logger.info(
            f"Starting training: epochs={self.epochs}, start_epoch={self.start_epoch}, "
            f"optimizer={self.optimizer.__class__.__name__}, scheduler={self.scheduler_type}"
        )

        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            timer = EpochTimer()
            with timer:
                train_metrics = self._run_epoch(self.train_loader, train=True)
                val_metrics = self._run_epoch(self.val_loader, train=False)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            self._log_tensorboard(epoch, train_metrics, val_metrics, current_lr, timer.elapsed)
            self._log_console(epoch, train_metrics, val_metrics, current_lr, timer.elapsed)

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)
            self._append_history(epoch, train_metrics, val_metrics, current_lr, timer.elapsed)

            if self.early_stopping is not None and self.early_stopping.step(val_metrics["loss"]):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(no improvement for {self.early_stopping.patience} epochs)"
                )
                break

        self.writer.close()
        return {"best_val_loss": self.best_val_loss, "epochs_trained": epoch + 1}

    # ----------------------------------------------------------------- #
    # Logging / checkpointing helpers
    # ----------------------------------------------------------------- #
    def _log_tensorboard(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float,
    ) -> None:
        """Write one epoch's metrics to TensorBoard."""
        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        self.writer.add_scalar("CosineSimilarity/train", train_metrics["cosine_similarity"], epoch)
        self.writer.add_scalar("CosineSimilarity/val", val_metrics["cosine_similarity"], epoch)
        self.writer.add_scalar("LearningRate", lr, epoch)
        self.writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        for name, value in train_metrics.items():
            if name not in ("loss", "cosine_similarity"):
                self.writer.add_scalar(f"LossComponents/train_{name}", value, epoch)
        for name, value in val_metrics.items():
            if name not in ("loss", "cosine_similarity"):
                self.writer.add_scalar(f"LossComponents/val_{name}", value, epoch)

    def _log_console(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float,
    ) -> None:
        """Log a concise one-line epoch summary to the console/log file."""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.epochs} | "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} | "
            f"train_cos={train_metrics['cosine_similarity']:.4f} val_cos={val_metrics['cosine_similarity']:.4f} | "
            f"lr={lr:.2e} | time={epoch_time:.1f}s"
        )

    def _save_checkpoint(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], is_best: bool
    ) -> None:
        """Save the 'last' checkpoint every epoch, and 'best' when validation loss improves."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(state, self.checkpoint_dir, filename="last.pt", is_best=is_best, best_filename="best.pt")

    def _append_history(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float,
    ) -> None:
        """Append one epoch's metrics to ``self.history`` and persist to ``history.json``.

        Persisting every epoch (rather than only at the end of training)
        means an interrupted run still leaves a usable, plottable history
        file behind.
        """
        record = {"epoch": epoch + 1, "lr": lr, "epoch_time_sec": epoch_time}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        self.history.append(record)

        history_path = self.checkpoint_dir / "history.json"
        with history_path.open("w") as f:
            json.dump(self.history, f, indent=2)