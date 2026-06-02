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
