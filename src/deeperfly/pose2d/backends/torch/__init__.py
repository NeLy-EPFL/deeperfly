"""PyTorch detector backend -- the reference that runs the original checkpoint.

Presents the uniform backend interface (``HourglassNet``, :func:`load_model`,
:func:`predict_heatmaps`) plus :func:`state_dict_from_torch_checkpoint`, which
the JAX backend's converter consumes. See :mod:`deeperfly.pose2d.backends` for
the backend contract. JAX is the default (faster on GPU); this backend needs the
``torch`` dependency (installed by default).
"""

from __future__ import annotations

from .model import Bottleneck, HourglassNet, device, predict_heatmaps
from .weights import load_model, state_dict_from_torch_checkpoint

__all__ = [
    "HourglassNet",
    "Bottleneck",
    "device",
    "predict_heatmaps",
    "load_model",
    "state_dict_from_torch_checkpoint",
]
