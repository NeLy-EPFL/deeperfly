"""deeperfly: JAX multi-view geometry and bundle adjustment for camera rigs.

Public surface:

- :mod:`deeperfly.geometry` -- low-level projection / triangulation / Rodrigues
  primitives (JAX, JIT- and grad-friendly).
- :class:`deeperfly.cameras.Camera` / :class:`deeperfly.cameras.CameraGroup` --
  camera models and config-driven rigs.
- :func:`deeperfly.bundle_adjustment.bundle_adjust` and
  :func:`deeperfly.bundle_adjustment.bundle_adjust_from_config` -- bundle
  adjustment over a ``CameraGroup``.
"""

from __future__ import annotations

from . import correction, geometry, pipeline, triangulate
from .bundle_adjustment import bundle_adjust, bundle_adjust_from_config
from .cameras import Camera, CameraGroup
from .io import PoseResult
from .pipeline import run_from_points2d
from .skeleton import Skeleton

__all__ = [
    "geometry",
    "triangulate",
    "correction",
    "pipeline",
    "Camera",
    "CameraGroup",
    "Skeleton",
    "PoseResult",
    "bundle_adjust",
    "bundle_adjust_from_config",
    "run_from_points2d",
]
