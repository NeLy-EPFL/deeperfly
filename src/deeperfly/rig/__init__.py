"""The camera rig: the cameras, how they are solved, and what they reconstruct.

Everything here is about *where the cameras are* and what follows from knowing that.
:mod:`~deeperfly.rig.cameras` is the model -- a :class:`~deeperfly.rig.cameras.Camera`
and the ordered :class:`~deeperfly.rig.cameras.CameraGroup` whose order defines the ``V``
axis of every points array in the project. The other modules are the three things one
does with it:

- :mod:`~deeperfly.rig.solve` solves a rig from hand labels alone (the from-scratch
  calibration), and :mod:`~deeperfly.rig.bundle_adjustment` refines one that already
  roughly stands;
- :mod:`~deeperfly.rig.calibration` is the artifact a solved rig is saved as -- a
  standalone, shareable ``calibration.toml`` that records the footage frame its
  intrinsics describe, so it can refuse footage it does not fit;
- :mod:`~deeperfly.rig.triangulation` turns per-view 2D into 3D through the rig,
  plainly or by consensus (RANSAC).

The primitives underneath -- projection, Rodrigues, the DLT -- are JAX and live in
:mod:`deeperfly.geometry`, which knows nothing about a rig.
"""

from __future__ import annotations

from . import bundle_adjustment, calibration, cameras, solve, triangulation
from .bundle_adjustment import bundle_adjust, bundle_adjust_from_config
from .calibration import Calibration
from .cameras import Camera, CameraGroup, resolve_extrinsics
from .triangulation import reprojection_error, triangulate, triangulate_ransac

__all__ = [
    "bundle_adjustment",
    "calibration",
    "cameras",
    "solve",
    "triangulation",
    "Camera",
    "CameraGroup",
    "Calibration",
    "resolve_extrinsics",
    "triangulate",
    "triangulate_ransac",
    "reprojection_error",
    "bundle_adjust",
    "bundle_adjust_from_config",
]
