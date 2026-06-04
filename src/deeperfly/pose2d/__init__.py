"""Stacked-hourglass 2D pose detector for DeepFly2D (PyTorch).

The detector is a faithful copy of the original DeepFly2D network in PyTorch
(:mod:`~deeperfly.pose2d.backends`); it loads the released DeepFly2D weights
directly and uses CUDA / Metal (MPS) automatically. The shared orchestration in
:mod:`~deeperfly.pose2d.inference` (``preprocess`` -> ``predict_heatmaps`` ->
``heatmap_to_points`` -> ``assemble_skeleton`` -> ``detect``) drives it, and
everything downstream (calibration, triangulation) consumes its output.
:mod:`~deeperfly.pose2d.download` fetches/caches the pretrained weights.

The detector network is :class:`pose2d.backends.torch.HourglassNet`; it is not
re-exported here so that ``import deeperfly.pose2d`` never imports torch.
"""

from __future__ import annotations

from . import backends, download, inference
from .backends import (
    auto_batch_size,
    gpu_memory_bytes,
    infer_num_stacks,
    load_detector,
    predict_heatmaps,
)
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
    "backends",
    "download",
    "inference",
    "load_detector",
    "predict_heatmaps",
    "infer_num_stacks",
    "auto_batch_size",
    "gpu_memory_bytes",
    "preprocess",
    "heatmap_to_points",
    "assemble_skeleton",
    "detect",
    "detect_sequence",
    "expand_passes",
    "fly_camera_layout",
]
