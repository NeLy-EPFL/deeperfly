"""Shared constants and helpers for the test suite.

Importable as a top-level module thanks to ``pythonpath = ["tests"]`` in
``pyproject.toml``. Pure helpers live here; pytest fixtures live in
``conftest.py``.
"""

from __future__ import annotations

import numpy as np

# Reference rig parameters (shared with examples/cameras.toml).
FOCAL_PX = 22388.125
DISTANCE_MM = 107.463
WIDTH, HEIGHT = 1024, 512
AZIMUTHS_DEG = [-120, -90, -45, 0, 45, 90, 120]
CAMERA_NAMES = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]


def leg_indices(skeleton, side: str) -> np.ndarray:
    """Point indices of one body side's leg points (``side`` is ``"r"`` or ``"l"``).

    Leg points are named ``"{side}{f|m|h}_..."`` (front / mid / hind leg); the
    antennae and abdominal markers (``"{side}_..."``) are excluded.
    """
    return np.array(
        [
            i
            for i, name in enumerate(skeleton.point_names)
            if name[:1] == side and name[1:2] in "fmh"
        ],
        dtype=np.int64,
    )


def point_sources_table(point_names, specs):
    """Build a ``[point_sources.<view>]`` mapping from per-pathway channel lists.

    Parameters
    ----------
    point_names
        The skeleton's ordered point names.
    specs
        Iterable of ``(view, pathway, points)`` where ``points[i]`` is the point
        index output channel ``i`` of ``pathway`` fills in ``view`` (``-1`` drops
        the channel) -- the old per-pathway ``points`` list.

    Returns
    -------
    dict
        ``{view: {point_name: {"pathway": ..., "out_channel": i}}}``.
    """
    table: dict[str, dict] = {}
    for view, pathway, points in specs:
        entries = table.setdefault(view, {})
        for ch, p in enumerate(points):
            if p >= 0:
                entries[point_names[p]] = {"pathway": pathway, "out_channel": ch}
    return table


def reference_rmat(yaw_rad: float) -> np.ndarray:
    """Reference world->camera rotation for a camera at azimuth ``yaw_rad``.

    This is the project's ground-truth convention that :mod:`deeperfly.cameras`
    must reproduce from an orbit (``look_at`` / ``azimuth`` / ``distance``) spec.
    The camera looks toward the origin with image-down along world ``-z``.
    """
    y = np.array([0.0, 0.0, -1.0])
    z = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
    return np.array([np.cross(y, z), y, z])


def small_rotation(sigma: float, seed: int) -> np.ndarray:
    """A random rotation matrix close to identity (axis-angle std ``sigma``)."""
    from scipy.linalg import expm

    o = np.random.default_rng(seed).normal(scale=sigma, size=3)
    skew = np.array([[0, -o[2], o[1]], [o[2], 0, -o[0]], [-o[1], o[0], 0]])
    return expm(skew)
