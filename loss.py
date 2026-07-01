"""
losses.py
=========

Loss functions for neutrino direction reconstruction.

Two losses are supported, both operating on raw (unnormalized) model output
directions against unit-vector targets:

- :class:`CosineLoss` -- ``1 - cos(pred, target)``, averaged over the batch.
  Purely direction-based: invariant to the predicted vector's magnitude.
- :class:`CosineMSELoss` -- ``CosineLoss + lambda * MSE(pred, target)``.
  Adds a magnitude-sensitive term: since targets are unit vectors, penalizing
  raw (non-normalized) MSE against them also pushes the model's *output
  magnitude* toward 1, which is a useful auxiliary signal on top of pure
  angular alignment. ``lambda`` is fully configurable via YAML.

Both share a common interface via :class:`DirectionLoss`, and are looked up
through :func:`build_loss` purely from configuration -- consistent with the
model registry pattern in ``models.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import normalize_vectors


# --------------------------------------------------------------------------- #
# Common interface
# --------------------------------------------------------------------------- #
class DirectionLoss(nn.Module, ABC):
    """Abstract base class for direction-regression losses.

    Subclasses implement :meth:`compute_components`, which returns a dict of
    named loss terms including a mandatory ``"total_loss"`` key (the value
    actually backpropagated). :meth:`forward` is implemented once here in
    terms of :meth:`compute_components`, so subclasses never need to
    duplicate the "return total_loss" boilerplate -- and callers (like
    ``trainer.py``) can always get either just the scalar (``loss_fn(...)``)
    or the full breakdown for logging (``loss_fn.compute_components(...)``).

    Args:
        eps: Numerical stability constant used when normalizing vectors.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    @abstractmethod
    def compute_components(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        """Compute all named loss terms for this loss function.

        Args:
            pred: Raw (unnormalized) predicted direction vectors, shape ``(N, 3)``.
            target: Ground-truth unit direction vectors, shape ``(N, 3)``.

        Returns:
            A dict of scalar tensors. Must include a ``"total_loss"`` key.
        """
        raise NotImplementedError

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the scalar total loss to backpropagate.

        Args:
            pred: Raw (unnormalized) predicted direction vectors, shape ``(N, 3)``.
            target: Ground-truth unit direction vectors, shape ``(N, 3)``.

        Returns:
            A 0-dim tensor: ``compute_components(pred, target)["total_loss"]``.
        """
        return self.compute_components(pred, target)["total_loss"]


# --------------------------------------------------------------------------- #
# Cosine Loss
# --------------------------------------------------------------------------- #
class CosineLoss(DirectionLoss):
    """Pure cosine-similarity loss: ``1 - mean(cos(pred, target))``.

    Both ``pred`` and ``target`` are L2-normalized before computing cosine
    similarity, so this loss is invariant to the predicted vector's
    magnitude and depends only on its direction -- the physically relevant
    quantity for direction reconstruction.

    Args:
        eps: Numerical stability constant used when normalizing vectors.
    """

    def compute_components(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        pred_n = normalize_vectors(pred, eps=self.eps)
        target_n = normalize_vectors(target, eps=self.eps)
        cos_sim = (pred_n * target_n).sum(dim=-1)
        cosine_loss = (1.0 - cos_sim).mean()
        return {"cosine_loss": cosine_loss, "total_loss": cosine_loss}


# --------------------------------------------------------------------------- #
# Cosine + lambda * MSE
# --------------------------------------------------------------------------- #
class CosineMSELoss(DirectionLoss):
    """Combined loss: ``cosine_loss + lambda * mse_loss``.

    Args:
        lam: Weight applied to the MSE term (``lambda`` in the project spec;
            named ``lam`` in code since ``lambda`` is a Python keyword).
        eps: Numerical stability constant used when normalizing vectors.
        normalize_mse: If ``False`` (default), the MSE term is computed
            between the *raw* model output and the (unit-norm) target, which
            additionally penalizes incorrect output magnitude -- a useful
            auxiliary signal since well-calibrated confidence/magnitude can
            be informative even though only direction is ultimately scored.
            If ``True``, the MSE term uses the normalized prediction instead,
            making it a purely direction-based (magnitude-invariant)
            complement to the cosine term -- closer to penalizing squared
            chord distance between two points on the unit sphere.
    """

    def __init__(self, lam: float = 1.0, eps: float = 1e-8, normalize_mse: bool = False) -> None:
        super().__init__(eps=eps)
        self.lam = lam
        self.normalize_mse = normalize_mse

    def compute_components(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        pred_n = normalize_vectors(pred, eps=self.eps)
        target_n = normalize_vectors(target, eps=self.eps)

        cos_sim = (pred_n * target_n).sum(dim=-1)
        cosine_loss = (1.0 - cos_sim).mean()

        mse_input = pred_n if self.normalize_mse else pred
        mse_loss = F.mse_loss(mse_input, target)

        total_loss = cosine_loss + self.lam * mse_loss
        return {
            "cosine_loss": cosine_loss,
            "mse_loss": mse_loss,
            "total_loss": total_loss,
        }


# --------------------------------------------------------------------------- #
# Registry + factory
# --------------------------------------------------------------------------- #
LOSS_REGISTRY: Dict[str, Type[DirectionLoss]] = {
    "cosine": CosineLoss,
    "cosine_mse": CosineMSELoss,
}


def build_loss(config: Dict[str, Any]) -> DirectionLoss:
    """Instantiate a loss function purely from configuration.

    Reads the ``loss`` section of the config::

        loss:
          type: cosine_mse   # or "cosine"
          lambda: 0.5         # only used by cosine_mse
          eps: 1.0e-8
          normalize_mse: false  # only used by cosine_mse

    Args:
        config: The effective (merged) experiment configuration.

    Returns:
        An instantiated :class:`DirectionLoss` subclass.

    Raises:
        ValueError: If ``loss.type`` is unrecognized.
    """
    loss_cfg = config.get("loss", {})
    name = str(loss_cfg.get("type", "cosine")).lower()

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type '{name}'. Available: {list(LOSS_REGISTRY)}")

    eps = loss_cfg.get("eps", 1e-8)

    if name == "cosine":
        return CosineLoss(eps=eps)

    # cosine_mse
    lam = loss_cfg.get("lambda", 1.0)
    normalize_mse = loss_cfg.get("normalize_mse", False)
    return CosineMSELoss(lam=lam, eps=eps, normalize_mse=normalize_mse)