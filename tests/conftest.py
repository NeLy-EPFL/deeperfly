"""Pytest fixtures for the deeperfly test suite.

The canonical test rig mirrors the example in ``examples/`` and the project's
``get_rmat`` reference convention: seven cameras orbiting the world origin,
looking inward, with a long focal length (a microscope-like setup). Pure
constants and helpers live in ``helpers.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_cameras, rig_arrays

from deeperfly.cameras import CameraGroup
from deeperfly.results import PoseResult
from deeperfly.skeleton import Skeleton


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def rig():
    """A 7-camera orbit rig as plain arrays (see :func:`helpers.rig_arrays`)."""
    return rig_arrays()


@pytest.fixture
def cameras() -> CameraGroup:
    return make_cameras()


@pytest.fixture
def fly() -> Skeleton:
    """Whatever skeleton the package currently ships (``fly38b``)."""
    return Skeleton.fly()


@pytest.fixture
def fly38() -> Skeleton:
    """The ``fly38`` preset specifically, for the tests that are about ITS layout.

    ``fly38`` is the only skeleton whose 38 points are two mirrored 19-point halves, so
    the block-layout properties (partner of ``i`` is ``i + 19``, the flip permutation is a
    roll by 19) are facts about this skeleton and not about "the default". Pinning them to
    the preset keeps them meaningful when the shipped default changes again.
    """
    from deeperfly.config import Config

    return Config.from_dict({"skeleton": {"name": "fly38"}}).skeleton()


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
