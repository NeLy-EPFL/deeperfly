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

from . import geometry
from .bundle_adjustment import bundle_adjust, bundle_adjust_from_config
from .cameras import Camera, CameraGroup

__all__ = [
    "geometry",
    "Camera",
    "CameraGroup",
    "bundle_adjust",
    "bundle_adjust_from_config",
    "main",
]


def main() -> None:
    print("Hello from deeperfly!")
