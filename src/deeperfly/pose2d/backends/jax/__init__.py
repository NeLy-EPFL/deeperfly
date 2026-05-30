"""JAX (Equinox) detector backend -- the default, and the faster one on GPU.

Presents the uniform backend interface (``HourglassNet``, :func:`load_model`,
:func:`predict_heatmaps`) plus the PyTorch-weight bridge
(:func:`convert_state_dict` / :func:`save_checkpoint` / :func:`export_state_dict`).
See :mod:`deeperfly.pose2d.backends` for the backend contract.
"""

from __future__ import annotations

from .model import (
    Bottleneck,
    FrozenBatchNorm,
    HourglassNet,
    predict_heatmaps,
    to_dtype,
)
from .weights import (
    convert_state_dict,
    export_state_dict,
    load_model,
    save_checkpoint,
)

__all__ = [
    "HourglassNet",
    "FrozenBatchNorm",
    "Bottleneck",
    "to_dtype",
    "predict_heatmaps",
    "load_model",
    "save_checkpoint",
    "convert_state_dict",
    "export_state_dict",
]
