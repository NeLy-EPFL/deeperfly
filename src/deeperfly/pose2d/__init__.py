"""Stacked-hourglass 2D pose detector for DeepFly2D (PyTorch).

The detector is a faithful copy of the original DeepFly2D network in PyTorch
(:mod:`~deeperfly.pose2d.model`); it loads the released DeepFly2D weights
directly (:mod:`~deeperfly.pose2d.weights`) and uses CUDA / Metal (MPS)
automatically. The shared orchestration in :mod:`~deeperfly.pose2d.inference`
(``preprocess`` -> ``predict_heatmaps`` -> ``heatmap_to_points`` ->
``assemble_skeleton`` -> ``detect``) drives it through the torch-free seam
(:mod:`~deeperfly.pose2d.detector`), and everything downstream (calibration,
triangulation) consumes its output. :mod:`~deeperfly.pose2d.download`
fetches/caches the pretrained weights.

The torch modules (:mod:`~deeperfly.pose2d.model`, :mod:`~deeperfly.pose2d.weights`)
are not imported here so that ``import deeperfly.pose2d`` never imports torch.
"""

from __future__ import annotations

from . import detector, download, inference
from .inference import (
    assemble_skeleton,
    detect,
    detect_sequence,
    expand_passes,
    fly_camera_layout,
    heatmap_to_points,
    preprocess,
)

__all__ = [
    "detector",
    "download",
    "inference",
    "preprocess",
    "heatmap_to_points",
    "assemble_skeleton",
    "detect",
    "detect_sequence",
    "expand_passes",
    "fly_camera_layout",
]
