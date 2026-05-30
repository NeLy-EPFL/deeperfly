"""JAX (Equinox) stacked-hourglass 2D pose detector for DeepFly2D.

- :class:`~deeperfly.pose2d.model.HourglassNet` -- the network (port of
  DeepFly2D's stacked hourglass); build the canonical fly model with
  ``HourglassNet.deepfly2d(key=...)``.
- :mod:`~deeperfly.pose2d.weights` -- convert/serialise the original PyTorch
  weights (``convert_state_dict``, ``save_checkpoint`` / ``load_checkpoint``).
- :mod:`~deeperfly.pose2d.inference` -- ``preprocess``, ``predict_heatmaps``,
  ``heatmap_to_points``, ``assemble_skeleton``.
- :mod:`~deeperfly.pose2d.download` -- fetch/cache pretrained weights.
- :mod:`~deeperfly.pose2d.torch_backend` -- the co-equal PyTorch detector
  (runs the original ``.tar`` weights directly).

The detector ships two co-equal backends behind one interface: the JAX/Equinox
:class:`HourglassNet` and the PyTorch :mod:`torch_backend`. Either runs the
published ``sh8`` weights; :func:`~deeperfly.pose2d.inference.detect` dispatches
on the model type, so calibration/triangulation downstream are identical.
"""

from __future__ import annotations

from . import inference, weights
from .inference import (
    assemble_skeleton,
    heatmap_to_points,
    predict_heatmaps,
    preprocess,
)
from .model import HourglassNet

__all__ = [
    "HourglassNet",
    "inference",
    "weights",
    "preprocess",
    "predict_heatmaps",
    "heatmap_to_points",
    "assemble_skeleton",
]
