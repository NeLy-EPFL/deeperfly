"""deeperfly: JAX multi-view geometry and bundle adjustment for camera rigs.

Public surface:

- :mod:`deeperfly.geometry` -- low-level projection / triangulation / Rodrigues
  primitives (JIT- and grad-friendly JAX on the CPU). Also used by bundle
  adjustment, which needs autodiff for the Jacobian.
- :class:`deeperfly.cameras.Camera` / :class:`deeperfly.cameras.CameraGroup` --
  camera models and config-driven rigs (built on :mod:`deeperfly.geometry`).
- :func:`deeperfly.bundle_adjustment.bundle_adjust` and
  :func:`deeperfly.bundle_adjustment.bundle_adjust_from_config` -- bundle
  adjustment over a ``CameraGroup`` (also on the CPU).

JAX (CPU) powers that geometry; the 2D detector (:mod:`deeperfly.pose2d`) is
PyTorch and uses the GPU when one is available.

The end-to-end pipeline is reusable without the CLI:

- :func:`deeperfly.resolve_recordings` -- expand recording dirs / wildcards into the
  per-camera footage to process (:mod:`deeperfly.recordings`).
- :func:`deeperfly.load_detector` -- load a PyTorch detector model
  (:mod:`deeperfly.pose2d.detector`); :func:`deeperfly.detect_2d` streams 2D
  detection over a recording given a detection plan + loaded models
  (:mod:`deeperfly.pose2d.stream`).
- :func:`deeperfly.run_recording` -- run a recording's enabled stages against an
  output directory, reusing cached results (the staged run behind ``deeperfly run``);
  :func:`deeperfly.run_from_points2d` is the lower-level 2D-to-3D pass over arrays.
"""

from __future__ import annotations

from . import geometry, pictorial, pipeline, recordings, triangulation
from .bundle_adjustment import bundle_adjust, bundle_adjust_from_config
from .cameras import Camera, CameraGroup
from .config import Config
from .pipeline import run_from_points2d, run_recording
from .pose2d.detector import load_detector
from .pose2d.stream import detect_2d
from .recordings import Recording, resolve_recordings
from .results import PoseResult
from .skeleton import Skeleton

__all__ = [
    "geometry",
    "triangulation",
    "pipeline",
    "pictorial",
    "recordings",
    "Camera",
    "CameraGroup",
    "Config",
    "Skeleton",
    "PoseResult",
    "Recording",
    "bundle_adjust",
    "bundle_adjust_from_config",
    "run_from_points2d",
    "run_recording",
    "resolve_recordings",
    "detect_2d",
    "load_detector",
]
