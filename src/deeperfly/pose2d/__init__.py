"""Stacked-hourglass 2D pose detector for DeepFly2D (PyTorch).

The detector is a faithful copy of the original DeepFly2D network in PyTorch
(:mod:`~deeperfly.pose2d.model`); it loads the released DeepFly2D weights
directly (:mod:`~deeperfly.pose2d.weights`) and uses CUDA / Metal (MPS)
automatically. A :class:`~deeperfly.pose2d.pathways.DetectionPlan` drives the
shared orchestration in :mod:`~deeperfly.pose2d.inference` (``detect`` /
``detect_sequence``) through the torch-free seam
(:mod:`~deeperfly.pose2d.detector`) and the model registry
(:mod:`~deeperfly.pose2d.models`); everything downstream (bundle adjustment,
triangulation) consumes its ``(V, T, P, 2)`` output.
:mod:`~deeperfly.pose2d.download` fetches/caches the pretrained weights.

The torch modules (:mod:`~deeperfly.pose2d.model`, :mod:`~deeperfly.pose2d.weights`)
are not imported here so that ``import deeperfly.pose2d`` never imports torch.
"""

from __future__ import annotations

from . import detector, download, inference, models, pathways
from .inference import detect, detect_sequence, heatmap_to_points
from .models import LoadedModel, ModelSpec, load_model
from .pathways import DetectionPlan, Pathway, Source

__all__ = [
    "detector",
    "download",
    "inference",
    "models",
    "pathways",
    "heatmap_to_points",
    "detect",
    "detect_sequence",
    "DetectionPlan",
    "Pathway",
    "Source",
    "ModelSpec",
    "LoadedModel",
    "load_model",
]
