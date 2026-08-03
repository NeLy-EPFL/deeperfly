"""Round-trip tests for the PoseResult HDF5 container and the StageStore."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.pictorial import Candidates
from deeperfly.results import PoseResult, StageStore
from deeperfly.skeleton import Skeleton


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


# -- PoseResult.save / load round-trips ----------------------------------------


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
    assert sk.point_names == Skeleton.fly().point_names
    assert sk.palette == Skeleton.fly().palette
    np.testing.assert_array_equal(sk.bones, Skeleton.fly().bones)
    np.testing.assert_array_equal(sk.limb_id, Skeleton.fly().limb_id)
    # The editor reads its skeleton from here, so losing the pairs on the way would leave
    # the chirality check falling back to name inference for every run.
    np.testing.assert_array_equal(sk.symmetries, Skeleton.fly().symmetries)


def test_a_results_file_written_before_symmetry_existed_still_loads(
    cameras, rng, tmp_path
):
    """The ``symmetries`` dataset is additive, so its absence must not be an error.

    Such a file loads as a skeleton with no declared pairs, which disables the pair-driven
    features for it rather than breaking it -- and the editor's chirality check then falls
    back to inferring pairs by name (``Skeleton.symmetries_or_inferred``).
    """
    import h5py

    path = tmp_path / "old.h5"
    _result(cameras, rng).save(path)
    with h5py.File(path, "r+") as f:
        del f["skeleton/symmetries"]
    sk = PoseResult.load(path).skeleton
    assert sk.n_symmetries == 0
    assert sk.point_names == Skeleton.fly().point_names
    np.testing.assert_array_equal(
        sk.symmetries_or_inferred(), Skeleton.fly().symmetries
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


def test_load_rejects_old_format(cameras, rng, tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps({"deeperfly_format_version": 1})
    with pytest.raises(ValueError, match="format version"):
        PoseResult.load(path)


# -- StageStore ----------------------------------------------------------------


def _image_sizes(cameras):
    return {name: (512, 1024) for name in cameras.names}


def _write_base(store, cameras, rng, *, candidates=None):
    v, t, n = len(cameras), 4, 38
    pts2d = rng.normal(size=(v, t, n, 2))
    conf = rng.uniform(size=(v, t, n))
    store.write_pose2d(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=pts2d,
        conf=conf,
        image_sizes=_image_sizes(cameras),
        candidates=candidates,
    )
    return pts2d, conf


def test_store_pose2d_roundtrip(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    assert not store.has("pose2d")
    pts2d, conf = _write_base(store, cameras, rng)

    assert store.has("pose2d")
    assert not store.has("triangulation")
    assert not store.has("visualization")  # no h5 group for visualization
    got2d, gotconf = store.read_pose2d()
    np.testing.assert_array_equal(got2d, pts2d)
    np.testing.assert_array_equal(gotconf, conf)
    assert store.read_cameras("pose2d").names == cameras.names
    assert store.read_image_sizes() == _image_sizes(cameras)
    assert store.read_skeleton().point_names == Skeleton.fly().point_names


def test_store_candidates_roundtrip(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    v, t, n, k = len(cameras), 4, 38, 3
    cand = Candidates(
        xy=rng.normal(size=(v, t, n, k, 2)), score=rng.uniform(size=(v, t, n, k))
    )
    _write_base(store, cameras, rng, candidates=cand)
    assert store.has_candidates()
    got = store.read_candidates()
    np.testing.assert_array_equal(got.xy, cand.xy)
    np.testing.assert_array_equal(got.score, cand.score)

    _write_base(store, cameras, rng)  # rewrite without candidates -> gone
    assert not store.has_candidates()
    assert store.read_candidates() is None


def test_store_stage_groups_do_not_touch_pose2d(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    pts2d, _ = _write_base(store, cameras, rng)

    refined = CameraGroup.from_arrays(
        cameras.names,
        cameras.rvecs + 0.01,
        cameras.tvecs,
        cameras.intrs,
        cameras.dists,
    )
    store.write_cameras("bundle_adjustment", refined)
    v, t, n = pts2d.shape[:3]
    tri2d = rng.normal(size=(v, t, n, 2))
    tri3d = rng.normal(size=(t, n, 3))
    reproj = rng.uniform(size=(v, t, n))
    store.write_points("triangulation", pts2d=tri2d, pts3d=tri3d, reproj_error=reproj)

    # pose2d stays pristine
    np.testing.assert_array_equal(store.read_pose2d()[0], pts2d)
    np.testing.assert_allclose(store.read_cameras("pose2d").rvecs, cameras.rvecs)
    # the stage groups round-trip
    np.testing.assert_allclose(
        store.read_cameras("bundle_adjustment").rvecs, refined.rvecs
    )
    got2d, got3d, gotrep = store.read_points("triangulation")
    np.testing.assert_array_equal(got2d, tri2d)
    np.testing.assert_array_equal(got3d, tri3d)
    np.testing.assert_array_equal(gotrep, reproj)


def test_load_prefers_most_derived(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    pts2d, conf = _write_base(store, cameras, rng)
    v, t, n = pts2d.shape[:3]

    ps2d = rng.normal(size=(v, t, n, 2))
    ps3d = rng.normal(size=(t, n, 3))
    store.write_points(
        "pictorial_structures", pts2d=ps2d, pts3d=ps3d, reproj_error=None
    )
    loaded = PoseResult.load(store.path)
    np.testing.assert_array_equal(loaded.pts2d, ps2d)  # pictorial over pose2d
    np.testing.assert_array_equal(loaded.pts3d, ps3d)
    np.testing.assert_array_equal(loaded.conf, conf)  # conf always from pose2d

    tri2d = rng.normal(size=(v, t, n, 2))
    tri3d = rng.normal(size=(t, n, 3))
    store.write_points("triangulation", pts2d=tri2d, pts3d=tri3d, reproj_error=None)
    loaded = PoseResult.load(store.path)
    np.testing.assert_array_equal(loaded.pts2d, tri2d)  # triangulation over pictorial
    np.testing.assert_array_equal(loaded.pts3d, tri3d)

    refined = CameraGroup.from_arrays(
        cameras.names,
        cameras.rvecs + 0.01,
        cameras.tvecs,
        cameras.intrs,
        cameras.dists,
    )
    store.write_cameras("bundle_adjustment", refined)
    loaded = PoseResult.load(store.path)
    np.testing.assert_allclose(loaded.cameras.rvecs, refined.rvecs)  # BA over config


def test_truncate_from_drops_stage_and_downstream(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    pts2d, _ = _write_base(store, cameras, rng)
    v, t, n = pts2d.shape[:3]
    store.write_cameras("bundle_adjustment", cameras)
    args = dict(
        pts2d=rng.normal(size=(v, t, n, 2)),
        pts3d=rng.normal(size=(t, n, 3)),
        reproj_error=None,
    )
    store.write_points("pictorial_structures", **args)
    store.write_points("triangulation", **args)

    store.truncate_from("pictorial_structures")
    assert store.has("pose2d")
    assert store.has("bundle_adjustment")
    assert not store.has("pictorial_structures")
    assert not store.has("triangulation")


def test_store_treats_old_format_as_empty(tmp_path):
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps({"deeperfly_format_version": 1})
        f.create_group("pose2d").create_dataset("points", data=np.zeros((1, 1, 1, 2)))
    store = StageStore(path)
    assert not store.has("pose2d")
    assert store.read_pose2d() is None
    assert store.read_image_sizes() is None


def test_store_missing_file_reads_empty(tmp_path):
    store = StageStore(tmp_path / "nope.h5")
    assert not store.has("pose2d")
    assert store.read_pose2d() is None
    assert store.read_candidates() is None
    store.truncate_from("triangulation")  # no-op, no error


def test_store_footage_paths_roundtrip(cameras, rng, tmp_path):
    import os

    outdir = tmp_path / "deeperfly_outputs"
    outdir.mkdir()
    store = StageStore(outdir / "results.h5")
    vids = {name: tmp_path / f"{name}.mp4" for name in cameras.names}
    for path in vids.values():
        path.write_bytes(b"x")
    v, t, n = len(cameras), 2, 38
    store.write_pose2d(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=rng.normal(size=(v, t, n, 2)),
        conf=rng.uniform(size=(v, t, n)),
        image_sizes=_image_sizes(cameras),
        footage={name: [path] for name, path in vids.items()},
    )
    got = store.read_footage()
    assert set(got) == set(cameras.names)
    name0 = cameras.names[0]
    assert got[name0]["abs"] == [str(vids[name0].resolve())]
    # relative paths are anchored at the results.h5 directory
    assert got[name0]["rel"] == [os.path.relpath(vids[name0].resolve(), outdir)]


def test_store_footage_absent_when_not_written(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)  # no footage argument
    assert store.read_footage() is None


# -- inverse_kinematics group --------------------------------------------------


def test_store_ik_roundtrip(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    assert not store.has("inverse_kinematics")
    t, n, d = 4, 38, 5
    angles = rng.normal(size=(t, d))
    names = [f"Angle_RF_J{i}" for i in range(d)]
    model = rng.normal(size=(t, n, 3))
    model[1, 7] = np.nan  # an unobserved model joint
    store.write_ik(
        angles=angles,
        angle_names=names,
        model_pts3d=model,
        meta={"template": "neuromechfly"},
    )

    assert store.has("inverse_kinematics")
    got_angles, got_names, got_model = store.read_ik()
    np.testing.assert_array_equal(got_angles, angles)
    assert got_names == names
    np.testing.assert_array_equal(got_model, model)  # NaN preserved


def test_poseresult_load_picks_up_nmf(cameras, rng, tmp_path):
    """PoseResult.load surfaces the fitted model joints as nmf_pts3d."""
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    store.write_points(
        "triangulation",
        pts2d=rng.normal(size=(len(cameras), 4, 38, 2)),
        pts3d=rng.normal(size=(4, 38, 3)),
        reproj_error=rng.uniform(size=(len(cameras), 4, 38)),
    )
    model = rng.normal(size=(4, 38, 3))
    store.write_ik(
        angles=rng.normal(size=(4, 5)),
        angle_names=["a"] * 5,
        model_pts3d=model,
        meta={"chain_scales": {"head": 1.25, "abdomen": 2.1}, "body_scale": 0.87},
    )
    res = PoseResult.load(store.path)
    assert res.nmf_pts3d is not None
    np.testing.assert_array_equal(res.nmf_pts3d, model)
    # the data-estimated head/abdomen size rides along on the IK group meta
    assert res.nmf_chain_scales == {"head": 1.25, "abdomen": 2.1}
    assert res.nmf_head_scale == 1.25 and res.nmf_abdomen_scale == 2.1
    # the per-recording constant body scale rides along too (defaults to 1.0)
    assert res.nmf_body_scale == 0.87


def test_store_truncate_from_drops_ik(cameras, rng, tmp_path):
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    store.write_ik(
        angles=rng.normal(size=(4, 5)),
        angle_names=["a"] * 5,
        model_pts3d=rng.normal(size=(4, 38, 3)),
    )
    assert store.has("inverse_kinematics")
    store.truncate_from("inverse_kinematics")
    assert not store.has("inverse_kinematics")
    assert store.read_ik() is None


# -- the animal/ group: which keypoints are not on this specimen --------------


def test_absent_roundtrips_and_is_omitted_when_empty(tmp_path, result):
    import h5py

    path = tmp_path / "r.h5"
    result.save(path)
    with h5py.File(path, "r") as f:
        assert "animal" not in f  # an ordinary run's file is unchanged

    result.absent = np.zeros(result.pts2d.shape[2], dtype=bool)
    result.absent[4] = True
    result.subject_id = "Fly2"
    result.save(path)
    back = PoseResult.load(path)
    assert back.absent is not None and back.absent[4] and not back.absent[5]
    assert back.subject_id == "Fly2"


def test_absent_survives_a_pose2d_rewrite_and_a_stage_truncation(tmp_path, result):
    # `write_pose2d` truncates the whole file and `truncate_from` drops stage groups, so
    # both are places an operator-authored fact could silently vanish. `animal/` is not a
    # stage, and write_pose2d carries it across its own truncation.
    from deeperfly.results import StageStore

    path = tmp_path / "r.h5"
    result.save(path)
    store = StageStore(path)
    absent = np.zeros(result.pts2d.shape[2], dtype=bool)
    absent[4] = True
    store.write_animal(absent=absent, subject_id="Fly2")

    store.truncate_from("triangulation")
    assert store.read_animal()[0][4]

    store.write_pose2d(
        cameras=result.cameras,
        skeleton=result.skeleton,
        pts2d=result.pts2d,
        conf=result.conf,
        image_sizes={n: (256, 256) for n in result.cameras.names},
    )
    carried, subject = store.read_animal()
    assert carried is not None and carried[4]
    assert subject == "Fly2"
