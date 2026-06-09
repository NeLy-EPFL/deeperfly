"""Pytest fixtures for the deeperfly test suite.

The canonical test rig mirrors the example in ``examples/`` and the project's
``get_rmat`` reference convention: seven cameras orbiting the world origin,
looking inward, with a long focal length (a microscope-like setup). Pure
constants and helpers live in ``helpers.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import geometry as geom
from deeperfly.cameras import CameraGroup
from deeperfly.results import PoseResult
from deeperfly.skeleton import Skeleton
from helpers import (
    AZIMUTHS_DEG,
    CAMERA_NAMES,
    DISTANCE_MM,
    FOCAL_PX,
    HEIGHT,
    WIDTH,
    reference_rmat,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def rig():
    """A 7-camera orbit rig as plain arrays.

    Returns a dict with ``names``, ``rvecs``, ``tvecs``, ``intrs`` (4-vector
    ``[fx, fy, cx, cy]``) and ``dists`` (empty, i.e. no distortion).
    """
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    rmats = np.array([reference_rmat(t) for t in np.deg2rad(AZIMUTHS_DEG)])
    rvecs = np.asarray(geom.rmat_to_rvec(rmats))
    tvecs = np.array([[0.0, 0.0, DISTANCE_MM]] * len(rmats))
    intrs = np.tile([FOCAL_PX, FOCAL_PX, cx, cy], (len(rmats), 1))
    dists = np.zeros((len(rmats), 0))
    return {
        "names": CAMERA_NAMES,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "intrs": intrs,
        "dists": dists,
    }


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


@pytest.fixture
def result(cameras, fly, rng) -> PoseResult:
    """A small synthetic 7-camera fly result with 2D + 3D points."""
    pts3d = rng.uniform(-1.5, 1.5, size=(6, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    return PoseResult(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=rng.uniform(0, 1, size=pts2d.shape[:3]),
        pts3d=pts3d,
        reproj_error=np.zeros(pts2d.shape[:3]),
    )
