"""
models.py
=========

Model architectures for neutrino direction reconstruction.

Every model predicts a raw (unnormalized) 3D direction vector ``(N, 3)`` from
per-event input features. Normalization to a unit vector is deliberately
*not* done inside the model -- it happens once, consistently, in
``losses.py`` / ``utils.normalize_vectors`` / evaluation code. This keeps the
model's job purely "predict a direction" and avoids subtly different
normalization behaviour across architectures.

Four architectures are provided, all built from the same config keys
(``hidden_dim``, ``num_layers``, ``dropout``, ``activation``), so they are
directly comparable under identical training procedures:

- :class:`MLP` -- baseline, operates on each event independently.
- :class:`GCN` -- Graph Convolutional Network (Kipf & Welling) over the k-NN
  population graph from ``dataset.py``.
- :class:`GraphSAGE` -- inductive neighbor-sampling GNN (Hamilton et al.).
- :class:`GAT` -- Graph Attention Network (Velickovic et al.), with an
  additional ``heads`` hyperparameter.

New models can be added by subclassing :class:`DirectionRegressor` and
registering the class in :data:`MODEL_REGISTRY` -- no other file needs to
change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# --------------------------------------------------------------------------- #
# Activation registry
# --------------------------------------------------------------------------- #
ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
}


def get_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name.

    Args:
        name: Activation name, case-insensitive. One of the keys of
            :data:`ACTIVATIONS` (``"relu"``, ``"leaky_relu"``, ``"elu"``,
            ``"gelu"``, ``"tanh"``, ``"selu"``).

    Returns:
        A new instance of the requested activation module.

    Raises:
        ValueError: If ``name`` is not a recognized activation.
    """
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


# --------------------------------------------------------------------------- #
# Common interface
# --------------------------------------------------------------------------- #
class DirectionRegressor(nn.Module, ABC):
    """Abstract base class for all direction-reconstruction models.

    Subclasses must implement :meth:`forward` and declare whether they need
    graph structure (``edge_index``) via the :attr:`requires_graph` property.
    This property is what ``trainer.py`` uses to decide which kind of
    DataLoader (tabular vs. graph) to pull from ``NeutrinoDataModule``.

    Args:
        input_dim: Number of input features per event/node.
        output_dim: Number of output (target) dimensions (3 for a direction
            vector).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Predict raw (unnormalized) direction vectors.

        Args:
            x: Node/event feature matrix, shape ``(N, input_dim)``.
            edge_index: Graph connectivity in PyTorch Geometric format,
                shape ``(2, num_edges)``. Ignored by non-graph models.

        Returns:
            Predicted direction vectors, shape ``(N, output_dim)``.
        """
        raise NotImplementedError

    @property
    def requires_graph(self) -> bool:
        """Whether this model needs ``edge_index`` to run (i.e. is a GNN)."""
        return False


# --------------------------------------------------------------------------- #
# MLP baseline
# --------------------------------------------------------------------------- #
class MLP(DirectionRegressor):
    """Multi-layer perceptron baseline: treats every event independently.

    Architecture: ``num_layers`` blocks of ``[Linear -> Activation -> Dropout]``
    with width ``hidden_dim``, followed by a final ``Linear`` projection to
    ``output_dim``.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output dimensions.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden ``[Linear -> Activation -> Dropout]`` blocks.
        dropout: Dropout probability applied after each hidden activation.
        activation: Name of the activation function (see :data:`ACTIVATIONS`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__(input_dim, output_dim)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        blocks: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            blocks.append(nn.Linear(in_dim, hidden_dim))
            blocks.append(get_activation(activation))
            blocks.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        blocks.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward pass. ``edge_index`` is accepted for interface compatibility but ignored."""
        return self.net(x)

    @property
    def requires_graph(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Graph Convolutional Network
# --------------------------------------------------------------------------- #
class GCN(DirectionRegressor):
    """Graph Convolutional Network (Kipf & Welling, 2017) for node-level direction regression.

    Architecture: ``num_layers`` ``GCNConv`` layers (each followed by
    activation + dropout), then a final ``Linear`` regression head from
    ``hidden_dim`` to ``output_dim``. A separate linear head (rather than a
    final ``GCNConv``) is used so the regression output isn't constrained by
    graph-convolution semantics, and so the "backbone + head" pattern stays
    identical across GCN/GraphSAGE/GAT for a fair comparison.

    Args:
        input_dim: Number of input features per node.
        output_dim: Number of output dimensions.
        hidden_dim: Width of each hidden ``GCNConv`` layer.
        num_layers: Number of ``GCNConv`` layers.
        dropout: Dropout probability applied after each hidden activation.
        activation: Name of the activation function (see :data:`ACTIVATIONS`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__(input_dim, output_dim)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.activations.append(get_activation(activation))
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward pass through stacked GCN layers followed by a linear head.

        Raises:
            ValueError: If ``edge_index`` is ``None`` (GCN requires graph structure).
        """
        if edge_index is None:
            raise ValueError("GCN.forward requires a non-None edge_index.")

        h = x
        for conv, act in zip(self.convs, self.activations):
            h = conv(h, edge_index)
            h = act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    @property
    def requires_graph(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# GraphSAGE
# --------------------------------------------------------------------------- #
class GraphSAGE(DirectionRegressor):
    """GraphSAGE (Hamilton et al., 2017) for node-level direction regression.

    Same backbone+head pattern as :class:`GCN`, but using ``SAGEConv`` layers,
    which aggregate sampled-neighbor features via a (mean, by default)
    aggregator rather than GCN's symmetrically-normalized adjacency. This is
    the architecture ``NeighborLoader``-style mini-batch training was
    originally designed around, making it a natural fit for the mini-batch
    graph loaders built in ``dataset.py``.

    Args:
        input_dim: Number of input features per node.
        output_dim: Number of output dimensions.
        hidden_dim: Width of each hidden ``SAGEConv`` layer.
        num_layers: Number of ``SAGEConv`` layers.
        dropout: Dropout probability applied after each hidden activation.
        activation: Name of the activation function (see :data:`ACTIVATIONS`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__(input_dim, output_dim)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.activations.append(get_activation(activation))
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward pass through stacked SAGEConv layers followed by a linear head.

        Raises:
            ValueError: If ``edge_index`` is ``None`` (GraphSAGE requires graph structure).
        """
        if edge_index is None:
            raise ValueError("GraphSAGE.forward requires a non-None edge_index.")

        h = x
        for conv, act in zip(self.convs, self.activations):
            h = conv(h, edge_index)
            h = act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    @property
    def requires_graph(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Graph Attention Network
# --------------------------------------------------------------------------- #
class GAT(DirectionRegressor):
    """Graph Attention Network (Velickovic et al., 2018) for node-level direction regression.

    Same backbone+head pattern as :class:`GCN`/:class:`GraphSAGE`, using
    multi-head ``GATConv`` layers. Each hidden ``GATConv`` uses ``heads``
    attention heads with ``concat=True``; the per-head output width is set
    to ``hidden_dim // heads`` so the concatenated output of each layer is
    exactly ``hidden_dim``, keeping layer widths comparable to the other
    architectures.

    Args:
        input_dim: Number of input features per node.
        output_dim: Number of output dimensions.
        hidden_dim: Total width of each hidden layer's (concatenated) output.
            Must be divisible by ``heads``.
        num_layers: Number of ``GATConv`` layers.
        dropout: Dropout probability applied after each hidden activation,
            and also passed to ``GATConv`` as attention-coefficient dropout.
        activation: Name of the activation function (see :data:`ACTIVATIONS`).
        heads: Number of attention heads per ``GATConv`` layer.

    Raises:
        ValueError: If ``hidden_dim`` is not divisible by ``heads``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        heads: int = 4,
    ) -> None:
        super().__init__(input_dim, output_dim)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}) "
                f"so that concatenated multi-head output matches hidden_dim."
            )

        self.dropout = dropout
        out_per_head = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(
                GATConv(in_dim, out_per_head, heads=heads, concat=True, dropout=dropout)
            )
            self.activations.append(get_activation(activation))
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward pass through stacked GATConv layers followed by a linear head.

        Raises:
            ValueError: If ``edge_index`` is ``None`` (GAT requires graph structure).
        """
        if edge_index is None:
            raise ValueError("GAT.forward requires a non-None edge_index.")

        h = x
        for conv, act in zip(self.convs, self.activations):
            h = conv(h, edge_index)
            h = act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    @property
    def requires_graph(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Registry + factory
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, Type[DirectionRegressor]] = {
    "mlp": MLP,
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
}


def build_model(config: Dict[str, Any], input_dim: int, output_dim: int) -> DirectionRegressor:
    """Instantiate a model purely from configuration.

    Reads the ``model`` section of the config, expecting at least a
    ``model.name`` key matching one of :data:`MODEL_REGISTRY`'s keys
    (``"mlp"``, ``"gcn"``, ``"graphsage"``, ``"gat"``). All architectural
    hyperparameters (``hidden_dim``, ``num_layers``, ``dropout``,
    ``activation``, and ``heads`` for GAT) are read from the same section,
    with sane defaults if omitted -- never hardcoded in Python.

    Args:
        config: The effective (merged) experiment configuration.
        input_dim: Number of input features, typically ``dm.input_dim`` from
            a :class:`dataset.NeutrinoDataModule`.
        output_dim: Number of output dimensions, typically ``dm.output_dim``.

    Returns:
        An instantiated :class:`DirectionRegressor` subclass, ready to move
        to a device and train.

    Raises:
        ValueError: If ``model.name`` is missing or unrecognized.

    Example (``configs/gat.yaml`` excerpt)::

        model:
          name: gat
          hidden_dim: 128
          num_layers: 3
          dropout: 0.2
          activation: elu
          heads: 4
    """
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "")).lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[name]
    kwargs: Dict[str, Any] = dict(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    )
    if name == "gat":
        kwargs["heads"] = model_cfg.get("heads", 4)

    return model_cls(**kwargs)