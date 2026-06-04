"""PyTorch detector backend -- runs the original DeepFly2D checkpoint directly.

Presents the backend interface (``HourglassNet``, :func:`load_model`,
:func:`predict_heatmaps`) plus :func:`state_dict_from_torch_checkpoint`. See
:mod:`deeperfly.pose2d.backends` for the contract. torch is a core dependency and
uses CUDA / Metal (MPS) automatically when a GPU is present.
"""

from __future__ import annotations

from .model import (
    Bottleneck,
    HourglassNet,
    device,
    predict_heatmaps,
    set_precision,
)
from .weights import load_model, state_dict_from_torch_checkpoint

__all__ = [
    "HourglassNet",
    "Bottleneck",
    "device",
    "predict_heatmaps",
    "set_precision",
    "load_model",
    "state_dict_from_torch_checkpoint",
]
