"""End-to-end orchestration: 2D points -> calibration -> 3D.

The pure array functions live in :mod:`deeperfly.pipeline.core` (re-exported here):

- :func:`calibrate` -- treat the animal as the calibration target and refine the
  cameras with bundle adjustment (confidence weights, Huber loss, bone-length prior).
- :func:`reconstruct` -- triangulate a 2D sequence and *greedily* reject
  high-reprojection-error observations, re-triangulating from the survivors.
- :func:`reconstruct_ransac` -- the default: triangulate each point from its largest
  multi-view consensus set (RANSAC) instead of a contaminated fit.
- :func:`run_from_points2d` -- the whole pipeline from a 2D sequence to a saved
  :class:`~deeperfly.io.PoseResult`.

On top of those, the *staged* run (shared by the CLI and a library caller):

- :mod:`deeperfly.pipeline.stages` -- the per-stage wrappers (``stage_pose2d``,
  ``stage_bundle_adjustment``, ...) and the cache/overwrite bookkeeping.
- :func:`run_recording` -- run a single recording's enabled stages against an output
  directory, reusing cached results and recomputing only what changed.
"""

from __future__ import annotations

from .core import (  # noqa: F401  (re-exported)
    _bone_prior,
    _resolve_triangulation,
    _subsample,
    calibrate,
    reconstruct,
    reconstruct_ransac,
    run_from_points2d,
)
from .run import run_recording
from .stages import (
    _OVERWRITE_ALL,  # noqa: F401  (re-exported)
    config_camera_rig,
    overwrite_stages,
    render_videos,
    source_view_frames,
    stage_bundle_adjustment,
    stage_cached,
    stage_pictorial_structures,
    stage_pose2d,
    stage_triangulation,
    visualization_cached,
)

__all__ = [
    "calibrate",
    "reconstruct",
    "reconstruct_ransac",
    "run_from_points2d",
    "run_recording",
    "overwrite_stages",
    "stage_pose2d",
    "config_camera_rig",
    "stage_bundle_adjustment",
    "stage_pictorial_structures",
    "stage_triangulation",
    "source_view_frames",
    "render_videos",
    "stage_cached",
    "visualization_cached",
]
