"""Bundle adjustment over a :class:`~deeperfly.cameras.CameraGroup`.

:func:`bundle_adjust` takes a (bundle-adjustment-unaware) ``CameraGroup`` plus
the observed 2D points and ``fixed`` / ``shared`` specifications, builds the
packed state (:func:`deeperfly.bundle_adjustment.state.build_state`), runs the
core solver (:mod:`deeperfly.bundle_adjustment.core`), and returns an optimized
``CameraGroup`` alongside the refined 3D points.

:func:`bundle_adjust_from_config` is the config-driven entry point: it reads the
``[bundle_adjustment]`` section of a TOML config (``fixed``, ``shared``,
``solver`` and a solver-named kwargs sub-table such as
``[bundle_adjustment.scipy.least_squares]``) and dispatches to
:func:`bundle_adjust`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from jaxtyping import Float
from numpy import ndarray
from scipy.optimize import OptimizeResult

from ..cameras import CameraGroup
from . import core
from .state import BAState, build_state, initialize_pts3d

__all__ = [
    "bundle_adjust",
    "bundle_adjust_from_config",
    "build_state",
    "initialize_pts3d",
    "BAState",
    "CameraGroup",
    "core",
]

_SUPPORTED_SOLVERS = ("scipy.least_squares",)


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
    ``(result, optimized_cameras, pts3d)`` -- the raw scipy ``OptimizeResult``,
    a ``CameraGroup`` carrying the refined parameters, and the refined points.
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
    config: dict | str | Path,
    pts2d: Float[ndarray, "V N 2"],
) -> tuple[OptimizeResult, CameraGroup, Float[ndarray, "N 3"]]:
    """Run :func:`bundle_adjust` driven by a TOML config (dict or file path).

    The ``[bundle_adjustment]`` section supplies ``fixed``, ``shared`` and
    ``solver``; solver kwargs live in a sub-table named after the solver, e.g.
    ``[bundle_adjustment.scipy.least_squares]``.
    """
    if not isinstance(config, dict):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    cameras = CameraGroup.from_config(config)
    ba = config.get("bundle_adjustment", {})

    solver = ba.get("solver", "scipy.least_squares")
    if solver not in _SUPPORTED_SOLVERS:
        raise ValueError(
            f"unsupported solver {solver!r}; expected one of {_SUPPORTED_SOLVERS}"
        )
    solver_kwargs = ba
    for part in solver.split("."):  # e.g. ba["scipy"]["least_squares"]
        solver_kwargs = solver_kwargs.get(part, {})

    return bundle_adjust(
        cameras,
        pts2d,
        fixed=ba.get("fixed", []),
        shared=ba.get("shared", []),
        **solver_kwargs,
    )
