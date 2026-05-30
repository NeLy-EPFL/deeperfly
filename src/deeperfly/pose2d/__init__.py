"""Stacked-hourglass 2D pose detector for DeepFly2D, with two backends.

The detector ships two interchangeable backends behind one interface, in
:mod:`~deeperfly.pose2d.backends`:

* :mod:`~deeperfly.pose2d.backends.jax` -- the **default** Equinox port; runs the
  published ``sh8`` weights as a pure JAX PyTree and is the faster backend on GPU.
* :mod:`~deeperfly.pose2d.backends.torch` -- a faithful copy of the original
  DeepFly2D network that loads the released ``.tar`` weights directly.

Both expose ``HourglassNet`` / ``load_model`` / ``predict_heatmaps`` and the same
heatmap contract, so the shared orchestration in
:mod:`~deeperfly.pose2d.inference` (``preprocess`` -> ``predict_heatmaps`` ->
``heatmap_to_points`` -> ``assemble_skeleton`` -> ``detect``) and everything
downstream (calibration, triangulation) are identical whichever you pick. Choose
one with :func:`~deeperfly.pose2d.backends.load_detector` (or ``--backend``).
:mod:`~deeperfly.pose2d.download` fetches/caches the pretrained weights.

``HourglassNet`` re-exported here is the JAX (default) network; reach the PyTorch
one as ``pose2d.backends.torch.HourglassNet``.
"""

from __future__ import annotations

from . import backends, download, inference
from .backends import infer_num_stacks, load_detector, predict_heatmaps
from .backends.jax import HourglassNet
from .inference import (
    assemble_skeleton,
    detect,
    detect_sequence,
    fly_camera_layout,
    heatmap_to_points,
    preprocess,
)

__all__ = [
    "HourglassNet",
    "backends",
    "download",
    "inference",
    "load_detector",
    "predict_heatmaps",
    "infer_num_stacks",
    "preprocess",
    "heatmap_to_points",
    "assemble_skeleton",
    "detect",
    "detect_sequence",
    "fly_camera_layout",
]
