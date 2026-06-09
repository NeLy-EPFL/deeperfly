"""Round-trip tests for the PoseResult HDF5 container."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.results import PoseResult
from deeperfly.skeleton import Skeleton


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


def _result(cameras, rng):
    v, t, n = len(cameras), 4, 38
    pts2d = rng.normal(size=(v, t, n, 2))
    pts2d[0, 0, 0] = np.nan  # missing observation
    conf = rng.uniform(size=(v, t, n))
    pts3d = rng.normal(size=(t, n, 3))
    pts3d[1, 5] = np.nan  # un-triangulated point
    return PoseResult(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        reproj_error=rng.uniform(size=(v, t, n)),
        meta={"fps": 100.0, "source": "synthetic"},
    )


def test_roundtrip_preserves_arrays(cameras, rng, tmp_path):
    res = _result(cameras, rng)
    path = tmp_path / "result.h5"
    res.save(path)
    loaded = PoseResult.load(path)

    np.testing.assert_array_equal(loaded.pts2d, res.pts2d)  # NaNs preserved
    np.testing.assert_array_equal(loaded.conf, res.conf)
    np.testing.assert_array_equal(loaded.pts3d, res.pts3d)
    np.testing.assert_array_equal(loaded.reproj_error, res.reproj_error)


def test_roundtrip_preserves_meta(cameras, rng, tmp_path):
    res = _result(cameras, rng)
    path = tmp_path / "result.h5"
    res.save(path)
    loaded = PoseResult.load(path)
    assert loaded.meta["fps"] == 100.0
    assert loaded.meta["source"] == "synthetic"
    assert "created_utc" in loaded.meta
    assert "deeperfly_format_version" not in loaded.meta  # stripped on load


def test_roundtrip_reconstructs_cameras(cameras, rng, tmp_path):
    res = _result(cameras, rng)
    path = tmp_path / "result.h5"
    res.save(path)
    loaded = PoseResult.load(path)
    assert loaded.cameras.names == cameras.names
    np.testing.assert_allclose(loaded.cameras.rvecs, cameras.rvecs)
    np.testing.assert_allclose(loaded.cameras.tvecs, cameras.tvecs)
    np.testing.assert_allclose(loaded.cameras.intrs, cameras.intrs)


def test_roundtrip_reconstructs_skeleton(cameras, rng, tmp_path):
    res = _result(cameras, rng)
    path = tmp_path / "result.h5"
    res.save(path)
    sk = PoseResult.load(path).skeleton
    assert sk.name == "fly38"
    assert sk.joint_names == Skeleton.fly().joint_names
    assert sk.palette == Skeleton.fly().palette
    np.testing.assert_array_equal(sk.bones, Skeleton.fly().bones)
    np.testing.assert_array_equal(
        sk.visibility_mask(cameras.names),
        Skeleton.fly().visibility_mask(cameras.names),
    )


def test_optional_fields_absent(cameras, rng, tmp_path):
    res = PoseResult(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=rng.normal(size=(len(cameras), 2, 38, 2)),
    )
    path = tmp_path / "minimal.h5"
    res.save(path)
    loaded = PoseResult.load(path)
    assert loaded.conf is None
    assert loaded.pts3d is None
    assert loaded.reproj_error is None
