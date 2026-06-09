"""Bundle adjustment over a :class:`~deeperfly.cameras.CameraGroup`.

:func:`bundle_adjust` takes a (bundle-adjustment-unaware) ``CameraGroup`` plus
the observed 2D points and ``fixed`` / ``shared`` specifications, builds the
packed state (:func:`deeperfly.bundle_adjustment.state.build_state`), runs the
core solver (:mod:`deeperfly.bundle_adjustment.core`), and returns an optimized
``CameraGroup`` alongside the refined 3D points.

:func:`bundle_adjust_from_config` is the config-driven entry point: it reads the
``[pipeline.bundle_adjustment]`` section of a TOML config (``fixed``, ``shared``
and the flat scipy ``least_squares`` kwargs such as ``max_nfev`` / ``loss``) and
dispatches to :func:`bundle_adjust`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float
from numpy import ndarray
from scipy.optimize import OptimizeResult

from ..cameras import CameraGroup
from . import core
from .state import BAState, build_state, initialize_pts3d

if TYPE_CHECKING:
    from ..config import Config

__all__ = [
    "bundle_adjust",
    "bundle_adjust_from_config",
    "build_state",
    "initialize_pts3d",
    "BAState",
    "CameraGroup",
    "core",
]


def bundle_adjust(
    cameras: CameraGroup,
    pts2d: Float[ndarray, "V N 2"],
    *,
    fixed: list[str] = (),
    shared: list[list[str]] = (),
    pts3d: Float[ndarray, "N 3"] | None = None,
    **solver_kwargs,
) -> tuple[OptimizeResult, CameraGroup, Float[ndarray, "N 3"]]:
    """Bundle-adjust a camera group against observed 2D points.

    Parameters
    ----------
    cameras
        Initial cameras. Their stacked parameters seed the optimization.
    pts2d
        Observed 2D points of shape ``(V, N, 2)`` with NaNs for missing.
    fixed, shared
        Parameter references to hold constant / tie together; see
        :func:`deeperfly.bundle_adjustment.state.build_state`.
    pts3d
        Initial 3D points; triangulated from ``cameras`` if omitted.
    **solver_kwargs
        Forwarded to the core solver (e.g. ``max_nfev``, ``loss``, ``f_scale``).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The raw scipy least-squares result.
    optimized_cameras : CameraGroup
        A camera group carrying the refined parameters.
    pts3d : np.ndarray
        The refined 3D points of shape ``(N, 3)``.
    """
    state = build_state(
        cameras.rvecs,
        cameras.tvecs,
        cameras.intrs,
        cameras.dists,
        pts2d,
        cameras.names,
        fixed=fixed,
        shared=shared,
        pts3d=pts3d,
    )
    result, solution = core.bundle_adjust(*state, **solver_kwargs)
    optimized = CameraGroup.from_arrays(
        cameras.names,
        solution.rvecs,
        solution.tvecs,
        solution.intrs,
        solution.dists,
    )
    return result, optimized, solution.pts3d


def bundle_adjust_from_config(
    config: "Config | dict | str | Path",
    pts2d: Float[ndarray, "V N 2"],
) -> tuple[OptimizeResult, CameraGroup, Float[ndarray, "N 3"]]:
    """Run :func:`bundle_adjust` driven by a TOML config.

    The ``[pipeline.bundle_adjustment]`` section supplies ``fixed`` / ``shared`` and
    the flat scipy ``least_squares`` kwargs (e.g. ``max_nfev`` / ``loss``). The
    ``keypoints`` key (which restricts the calibration keypoints) is a pipeline-level
    concern handled by :func:`deeperfly.pipeline.calibrate`, not here.

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`, parsed ``dict`` or path to a config
        TOML file.
    pts2d
        Observed 2D points of shape ``(V, N, 2)`` with NaNs for missing.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The raw scipy least-squares result.
    optimized_cameras : CameraGroup
        A camera group carrying the refined parameters.
    pts3d : np.ndarray
        The refined 3D points of shape ``(N, 3)``.
    """
    from ..config import Config

    cfg = Config.coerce(config)
    cameras = CameraGroup.from_config(cfg)
    ba = cfg.bundle_adjustment
    return bundle_adjust(
        cameras,
        pts2d,
        fixed=ba.fixed,
        shared=ba.shared,
        **ba.least_squares,
    )
