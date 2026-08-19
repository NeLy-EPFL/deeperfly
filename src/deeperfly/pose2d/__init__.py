"""The 2D pose detectors: dense per-view and multiview (PyTorch).

Two classes, both DENSE -- one channel per tracked point, in every view.
:mod:`~deeperfly.pose2d.hrnet` predicts each view alone (and runs both single-view
checkpoints, selecting its feature maps by stride rather than by index);
:mod:`~deeperfly.pose2d.mvt` encodes a frame's views TOGETHER, so a joint only one camera
can see informs the cameras that cannot. Both take ONE grayscale plane and use CUDA /
Metal (MPS) automatically.

A :class:`~deeperfly.pose2d.pathways.DetectionPlan` drives the shared orchestration in
:mod:`~deeperfly.pose2d.inference` (``detect`` / ``detect_sequence``) through the
torch-free seam (:mod:`~deeperfly.pose2d.detector`) and the model registry
(:mod:`~deeperfly.pose2d.models`); everything downstream (bundle adjustment,
triangulation) consumes its ``(V, T, P, 2)`` output.
:mod:`~deeperfly.pose2d.download` finds a checkpoint on ``$DEEPERFLY_MODELS`` -- every
detector is trained per project, so nothing is downloaded.

The torch modules are imported lazily, so ``import deeperfly.pose2d`` never imports torch.
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
