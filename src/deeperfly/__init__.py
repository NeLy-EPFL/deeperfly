"""deeperfly: JAX multi-view geometry and bundle adjustment for camera rigs.

Public surface:

- :mod:`deeperfly.geometry` -- low-level projection / triangulation / Rodrigues
  primitives (JIT- and grad-friendly JAX on the CPU; see :mod:`deeperfly._jax_cpu`).
  Also used by bundle adjustment, which needs autodiff for the Jacobian.
- :class:`deeperfly.cameras.Camera` / :class:`deeperfly.cameras.CameraGroup` --
  camera models and config-driven rigs (built on :mod:`deeperfly.geometry`).
- :func:`deeperfly.bundle_adjustment.bundle_adjust` and
  :func:`deeperfly.bundle_adjustment.bundle_adjust_from_config` -- bundle
  adjustment over a ``CameraGroup`` (also on the CPU).

JAX (CPU) powers that geometry; the 2D detector (:mod:`deeperfly.pose2d`) is
PyTorch and uses the GPU when one is available.
"""

from __future__ import annotations

from . import correction, geometry, pictorial, pipeline, triangulate
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
    "pictorial",
    "Camera",
    "CameraGroup",
    "Skeleton",
    "PoseResult",
    "bundle_adjust",
    "bundle_adjust_from_config",
    "run_from_points2d",
]
