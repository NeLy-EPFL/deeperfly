"""Round-trip tests for the PoseResult HDF5 container and the StageStore."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.pictorial import Candidates
from deeperfly.results import FORMAT_VERSION, PoseResult, StageStore
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
    # To float32 resolution, not bit-exactly: big point arrays are narrowed on the way in.
    # Not asserted against a fixed dtype either, because whether a given array is narrowed
    # depends on its *size* -- these fixtures straddle the threshold, ``xy`` above it and
    # ``score`` below -- and this test is about the round trip being faithful. What the
    # policy does by size is ``test_storage_policy_narrows_and_deflates_only_big_arrays``.
    np.testing.assert_allclose(got.xy, cand.xy, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got.score, cand.score, rtol=1e-6, atol=1e-7)

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


def test_store_refuses_a_newer_format_instead_of_reading_it_as_empty(tmp_path):
    """A newer file must RAISE, not read as absent.

    Treating it as absent is what made this destructive: every ``has(stage)`` reports
    false, so a run recomputes ``pose2d``, and ``write_pose2d`` truncates the whole file --
    destroying a newer result and reporting success. Older files stay regenerable; only
    this direction is refused.
    """
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps({"deeperfly_format_version": FORMAT_VERSION + 1})
        f.create_group("pose2d").create_dataset("points", data=np.zeros((1, 1, 1, 2)))
    store = StageStore(path)
    for call in (
        store.has,
        lambda _: store.read_pose2d(),
        lambda _: store.read_image_sizes(),
        lambda _: store.read_footage(),
        lambda _: store.read_animal(),
    ):
        with pytest.raises(ValueError, match="newer deeperfly"):
            call("pose2d")


def test_load_refuses_a_newer_format_with_the_shared_message(tmp_path):
    """And it must not advise re-running the pipeline, which would destroy the file."""
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps({"deeperfly_format_version": FORMAT_VERSION + 1})
    with pytest.raises(ValueError, match="newer deeperfly") as exc:
        PoseResult.load(path)
    assert "re-run the pipeline" not in str(exc.value)


def test_load_of_a_foreign_hdf5_diagnoses_itself(tmp_path):
    """A file that is not a deeperfly result at all gets this loader's message, not KeyError."""
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("something", data=[1, 2, 3])
    with pytest.raises(ValueError, match="format version"):
        PoseResult.load(path)


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
        meta={
            "chain_scales": {"head": 1.25, "abdomen": 2.1},
            "chain_offsets": {"head": [0.03, -0.02, -0.13]},
            "body_scale": 0.87,
        },
    )
    res = PoseResult.load(store.path)
    assert res.nmf_pts3d is not None
    np.testing.assert_array_equal(res.nmf_pts3d, model)
    # the data-estimated head/abdomen size rides along on the IK group meta
    assert res.nmf_chain_scales == {"head": 1.25, "abdomen": 2.1}
    assert res.nmf_head_scale == 1.25 and res.nmf_abdomen_scale == 2.1
    # ...and so does where each chain's base was measured, which the overlay needs for
    # the same reason: the angles were fitted about the shifted pivot, so a head drawn
    # about the model's anchor is off by exactly this vector.
    np.testing.assert_allclose(res.nmf_chain_offsets["head"], [0.03, -0.02, -0.13])
    # the per-recording constant body scale rides along too (defaults to 1.0)
    assert res.nmf_body_scale == 0.87


def test_ik_meta_without_chain_offsets_reads_as_no_shift(cameras, rng, tmp_path):
    """A results.h5 written before chains had base landmarks must mean *no* shift.

    The overlay adds this vector to its node transforms, so a missing key has to arrive
    as "nothing to add" rather than as anything the renderer has to guess at.
    """
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    store.write_ik(
        angles=rng.normal(size=(4, 5)),
        angle_names=["a"] * 5,
        model_pts3d=rng.normal(size=(4, 38, 3)),
        meta={"chain_scales": {"head": 1.25}},
    )
    assert PoseResult.load(store.path).nmf_chain_offsets == {}


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


# -- the rig as the same record calibration.toml carries -----------------------


def test_the_camera_group_carries_the_rigs_units_and_provenance(tmp_path, cameras, fly):
    """The HDF5 group had no slot for units/scale/provenance/quality, so both exporters
    INVENTED them -- hardcoding units="config", scale_source="orbit_prior". That re-labels a
    millimeter board calibration as an arbitrary-scale orbit guess.
    """
    store = StageStore(tmp_path / "results.h5")
    v, t, n = len(cameras.names), 2, len(fly.point_names)
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=np.zeros((v, t, n, 2)),
        conf=np.ones((v, t, n)),
        image_sizes={name: (48, 64) for name in cameras.names},
    )
    store.write_cameras(
        "bundle_adjustment",
        cameras,
        image_sizes={name: (48, 64) for name in cameras.names},
        meta={
            "units": "mm",
            "scale_source": "board",
            "provenance": {"method": "labels_ba", "intrinsics": "board"},
            "quality": {"rms_px": 1.25},
        },
    )
    meta = store.read_camera_meta("bundle_adjustment")
    assert meta["units"] == "mm"
    assert meta["scale_source"] == "board"
    assert meta["provenance"]["intrinsics"] == "board"
    assert meta["quality"]["rms_px"] == 1.25
    assert meta["image_sizes"][cameras.names[0]] == (48, 64)


def test_a_camera_group_written_without_metadata_reads_as_empty(tmp_path, cameras, fly):
    """Additive: an older file has none of it, and {} is what makes a caller pass through."""
    store = StageStore(tmp_path / "results.h5")
    v, t, n = len(cameras.names), 2, len(fly.point_names)
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=np.zeros((v, t, n, 2)),
        conf=np.ones((v, t, n)),
        image_sizes={name: (48, 64) for name in cameras.names},
    )
    store.write_cameras("bundle_adjustment", cameras)
    meta = store.read_camera_meta("bundle_adjustment")
    assert "units" not in meta and "scale_source" not in meta
    assert store.read_camera_meta("triangulation") == {}


def test_the_true_distortion_lengths_survive_the_round_trip(tmp_path, fly):
    """`CameraGroup.dists` pads to the group max by contract (for the JAX call sites), so a
    camera authored `dist = []` read back as five zeros -- a textual round-trip failure.
    Numerically harmless, but it made the HDF5 group a lossy copy of a calibration.toml.
    """
    from deeperfly.cameras import Camera

    # Built from Cameras directly: `from_arrays` cannot express a ragged dist, which is
    # itself the reason the padded array is the only thing that ever reached disk.
    group = CameraGroup(
        {
            name: Camera(
                rvec=np.zeros(3),
                tvec=np.array([0.0, 0.0, 10.0]),
                intr=np.array([100.0, 100.0, 32.0, 24.0]),
                dist=np.asarray(dist, dtype=float),
                name=name,
            )
            for name, dist in (("a", []), ("b", [0.1, 0.2, 0.0, 0.0, 0.3]))
        }
    )
    assert group.dists.shape == (2, 5)  # padded, by contract, for the JAX call sites
    store = StageStore(tmp_path / "results.h5")
    v, t, n = 2, 2, len(fly.point_names)
    store.write_pose2d(
        cameras=group,
        skeleton=fly,
        pts2d=np.zeros((v, t, n, 2)),
        conf=np.ones((v, t, n)),
        image_sizes={"a": (48, 64), "b": (48, 64)},
    )
    assert store.read_camera_meta("pose2d")["dist_lengths"] == [0, 5]


# -- v3: what a stage stores, and what a reader rebuilds ------------------------


def _stage_arrays(cameras, rng, *, t=6):
    """``(pts2d, pts3d, proj)`` for a stage write: an independent 2D and a projected one.

    ``t`` is deliberately small; the arrays that need to clear the compression threshold
    say so themselves.
    """
    n = 38
    pts3d = rng.uniform(-1.5, 1.5, size=(t, n, 3))
    proj = np.array(cameras.project(pts3d), dtype=float)
    assert np.isfinite(proj).all(), "fixture rig projects the fixture points off-camera"
    pts2d = proj + rng.normal(scale=3.0, size=proj.shape)  # an independent measurement
    return pts2d, pts3d, proj


def test_a_2d_that_is_its_3d_reprojected_is_not_stored(cameras, rng, tmp_path):
    """The smoother's 2D is exactly ``project(points3d)``, so v3 keeps only the 3D."""
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    _, pts3d, proj = _stage_arrays(cameras, rng)
    store.write_points("eks", pts2d=proj, pts3d=pts3d, reproj_error=None)

    with h5py.File(store.path, "r") as f:
        assert "points" not in f["eks"]
        assert f["eks"].attrs["points2d_storage"] == "derived"
    got2d, got3d, _ = store.read_points("eks")
    np.testing.assert_allclose(got2d, proj, rtol=0, atol=1e-4)
    np.testing.assert_allclose(got3d, pts3d, rtol=0, atol=1e-6)


def test_a_2d_frozen_over_time_is_stored_as_an_override(cameras, rng, tmp_path):
    """The correction chain freezes a few columns in pixel space; only those are stored.

    They are not recoverable from the 3D -- that is the whole point of ``freeze_2d``,
    which takes each view's own temporal center rather than reprojecting -- but they are
    constant over time, so the payload is per-view constants and not an array per frame.
    """
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    _, pts3d, proj = _stage_arrays(cameras, rng)
    frozen = [3, 11]
    pts2d = proj.copy()
    pts2d[:, :, frozen, :] = proj[:, :1, frozen, :] + 5.0  # per view, constant in time
    store.write_points("postprocess", pts2d=pts2d, pts3d=pts3d, reproj_error=None)

    with h5py.File(store.path, "r") as f:
        g = f["postprocess"]
        assert "points" not in g
        assert g.attrs["points2d_storage"] == "override"
        assert list(g["points2d_override_cols"][()]) == frozen
        # (V, len(frozen), 2) and nothing per frame -- the saving is the whole reason.
        assert g["points2d_override"].shape == (len(cameras), len(frozen), 2)
    got2d, _, _ = store.read_points("postprocess")
    np.testing.assert_allclose(got2d, pts2d, rtol=0, atol=1e-4)


def test_a_2d_that_differs_per_frame_is_stored_whole(cameras, rng, tmp_path):
    """An override is only for columns held *constant*; per-frame information is not one."""
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    _, pts3d, proj = _stage_arrays(cameras, rng)
    pts2d = proj.copy()
    pts2d[:, :, 4, :] += rng.normal(scale=2.0, size=(len(cameras), proj.shape[1], 2))
    store.write_points("postprocess", pts2d=pts2d, pts3d=pts3d, reproj_error=None)

    with h5py.File(store.path, "r") as f:
        assert f["postprocess"].attrs["points2d_storage"] == "full"
        assert "points" in f["postprocess"]


def test_an_independent_2d_is_stored_whole(cameras, rng, tmp_path):
    """Triangulation's cleaned observations are a measurement, not a reprojection."""
    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    pts2d, pts3d, _ = _stage_arrays(cameras, rng)
    pts2d[0, 0, 0] = np.nan  # a rejected observation
    store.write_points("triangulation", pts2d=pts2d, pts3d=pts3d, reproj_error=None)

    with h5py.File(store.path, "r") as f:
        assert f["triangulation"].attrs["points2d_storage"] == "full"
    got2d, _, _ = store.read_points("triangulation")
    assert np.isnan(got2d[0, 0, 0]).all()  # the rejection survives
    np.testing.assert_allclose(got2d[1], pts2d[1], rtol=0, atol=1e-4)


def test_a_stored_2d_keeps_its_reproj_error_as_an_audit_record(cameras, rng, tmp_path):
    """A stage whose 2D is stored whole keeps its error even when it is recomputable.

    Not an oversight: that 2D is the one an outside tool can overwrite, and a recomputed
    error agrees with the stored 3D by construction, so it could never reveal the
    substitution. ``acquisition.stored_vs_pose2d`` reads the stored value for exactly that.
    """
    store = StageStore(tmp_path / "results.h5")
    obs2d, _ = _write_base(store, cameras, rng)
    pts2d, pts3d, proj = _stage_arrays(cameras, rng, t=obs2d.shape[1])
    recomputable = np.linalg.norm(proj - obs2d, axis=-1)
    store.write_points(
        "triangulation", pts2d=pts2d, pts3d=pts3d, reproj_error=recomputable
    )

    with h5py.File(store.path, "r") as f:
        assert f["triangulation"].attrs["points2d_storage"] == "full"
        assert "reproj_error" in f["triangulation"]


def test_a_derived_2d_drops_a_recomputable_reproj_error(cameras, rng, tmp_path):
    """The smoother's error is measured against the detections, so a reader recomputes it."""
    store = StageStore(tmp_path / "results.h5")
    obs2d, _ = _write_base(store, cameras, rng)
    _, pts3d, proj = _stage_arrays(cameras, rng, t=obs2d.shape[1])
    reproj = np.linalg.norm(proj - obs2d, axis=-1)
    store.write_points("eks", pts2d=proj, pts3d=pts3d, reproj_error=reproj)

    with h5py.File(store.path, "r") as f:
        assert "reproj_error" not in f["eks"]
    _, _, got = store.read_points("eks")
    np.testing.assert_allclose(got, reproj, rtol=0, atol=1e-4)


def test_a_reproj_error_measured_against_something_else_is_stored(
    cameras, rng, tmp_path
):
    """Dropping it is conditional on a recomputation *reproducing* it, never assumed."""
    store = StageStore(tmp_path / "results.h5")
    obs2d, _ = _write_base(store, cameras, rng)
    _, pts3d, proj = _stage_arrays(cameras, rng, t=obs2d.shape[1])
    store.write_points(
        "eks", pts2d=proj, pts3d=pts3d, reproj_error=np.full(obs2d.shape[:3], 7.0)
    )

    with h5py.File(store.path, "r") as f:
        assert f["eks"].attrs["points2d_storage"] == "derived"  # the 2D still went
        assert "reproj_error" in f["eks"]  # the error did not
    _, _, got = store.read_points("eks")
    np.testing.assert_allclose(got, 7.0)


def test_storage_policy_narrows_and_deflates_only_big_arrays(cameras, rng, tmp_path):
    """float32 + deflate for point arrays; small arrays, and so every rig, stay float64.

    The size threshold is what keeps the policy away from the arrays whose precision is
    load-bearing without this having to know which those are.
    """
    store = StageStore(tmp_path / "results.h5")
    v, t, n = len(cameras), 40, 38
    store.write_pose2d(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=rng.normal(size=(v, t, n, 2)),
        conf=rng.uniform(size=(v, t, n)),
        image_sizes=_image_sizes(cameras),
    )
    with h5py.File(store.path, "r") as f:
        big = f["pose2d/points"]
        assert big.size >= 4096 and big.dtype == np.float32
        assert big.compression == "gzip"
        for name in ("rvecs", "tvecs", "intrs"):
            rig = f[f"pose2d/cameras/{name}"]
            assert rig.dtype == np.float64, f"{name} must keep full precision"
            assert rig.compression is None


# -- repack --------------------------------------------------------------------


def _write_v2(path, cameras, rng, *, t=6):
    """A schema-v2 file: every stage storing every array, the way older builds wrote them.

    Built by hand rather than by the store, because the store only writes the current
    schema -- which is the thing :func:`repack` has to be fed an older file to test.
    """
    from deeperfly.results import _write_cameras, _write_skeleton

    v, n = len(cameras), 38
    pts3d = rng.uniform(-1.5, 1.5, size=(t, n, 3))
    proj = np.array(cameras.project(pts3d), dtype=float)
    pts2d = proj + rng.normal(scale=3.0, size=proj.shape)
    pts2d[0, 0, 0] = np.nan
    err = np.linalg.norm(proj - pts2d, axis=-1)
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(
            {"deeperfly_format_version": 2, "created_utc": "2026-01-01T00:00:00+00:00"}
        )
        _write_skeleton(f.create_group("skeleton"), Skeleton.fly())
        g = f.create_group("pose2d")
        g.create_dataset("points", data=pts2d)
        g.create_dataset("conf", data=rng.uniform(size=(v, t, n)))
        _write_cameras(g.create_group("cameras"), cameras)
        _write_cameras(
            f.create_group("bundle_adjustment").create_group("cameras"), cameras
        )
        for stage, s2 in (
            ("triangulation", pts2d),
            ("eks", proj),
            ("postprocess", proj),
        ):
            gs = f.create_group(stage)
            gs.create_dataset("points", data=s2)
            gs.create_dataset("points3d", data=pts3d)
            gs.create_dataset("reproj_error", data=err)
    return pts2d, pts3d, proj


def test_repack_preserves_the_pose_and_shrinks_the_file(cameras, rng, tmp_path):
    from deeperfly.results import repack

    path = tmp_path / "results.h5"
    _write_v2(path, cameras, rng, t=40)
    before_result = PoseResult.load(path)
    before, after = repack(path)

    assert after < before
    got = PoseResult.load(path)
    # Reconstructed, not reread -- and still the same pose to the float32 storage step.
    np.testing.assert_allclose(got.pts2d, before_result.pts2d, rtol=0, atol=1e-3)
    np.testing.assert_allclose(got.pts3d, before_result.pts3d, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        got.reproj_error, before_result.reproj_error, rtol=0, atol=1e-3
    )
    assert np.array_equal(np.isnan(got.pts2d), np.isnan(before_result.pts2d))
    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["meta"])["deeperfly_format_version"] == FORMAT_VERSION
        assert f["triangulation"].attrs["points2d_storage"] == "full"
        assert f["eks"].attrs["points2d_storage"] == "derived"
        assert "points" not in f["eks"]


def test_repack_keeps_groups_it_does_not_know_about(cameras, rng, tmp_path):
    """A foreign tool's record of what it did to this file is not a space saving.

    ``dfpose_predict/`` is written by the labeling pipeline and read back by
    ``acquisition`` to detect reseeded cells; a repack that dropped it would destroy the
    provenance the reprojection-error rule exists to protect.
    """
    from deeperfly.results import repack

    path = tmp_path / "results.h5"
    _write_v2(path, cameras, rng)
    with h5py.File(path, "a") as f:
        g = f.create_group("dfpose_predict")
        g.create_dataset("contra_seed_source", data=np.arange(12, dtype=np.int32))
        g["contra_seed_source"].attrs["legend"] = json.dumps({"reprojection": 3})
        g.attrs["note"] = "written elsewhere"
    repack(path)

    with h5py.File(path, "r") as f:
        assert np.array_equal(
            f["dfpose_predict/contra_seed_source"][()], np.arange(12, dtype=np.int32)
        )
        assert json.loads(f["dfpose_predict/contra_seed_source"].attrs["legend"]) == {
            "reprojection": 3
        }
        assert f["dfpose_predict"].attrs["note"] == "written elsewhere"


def test_repack_refuses_a_newer_file_and_leaves_it_alone(cameras, rng, tmp_path):
    from deeperfly.results import repack

    path = tmp_path / "results.h5"
    _write_v2(path, cameras, rng)
    with h5py.File(path, "a") as f:
        f.attrs["meta"] = json.dumps({"deeperfly_format_version": FORMAT_VERSION + 1})
    size = path.stat().st_size

    with pytest.raises(ValueError, match="newer deeperfly"):
        repack(path)
    assert path.stat().st_size == size  # untouched
    assert not list(path.parent.glob(".*.repack"))  # and no debris left behind


def test_an_older_file_reads_but_does_not_let_the_run_skip_a_stage(
    cameras, rng, tmp_path
):
    """The two halves of the version split, which is what makes repack worth having.

    Reads work, so a viewer or an annotation session can open the corpus as it stands.
    :meth:`StageStore.has` still reports incomplete, so a *run* recomputes rather than
    appending current-schema groups into an older-schema file.
    """
    path = tmp_path / "results.h5"
    pts2d, _, _ = _write_v2(path, cameras, rng)
    store = StageStore(path)

    assert not store.has("pose2d")  # a run would recompute from the root
    assert not store.has("triangulation")
    got = store.read_pose2d()  # but the detections are still readable
    assert got is not None
    np.testing.assert_array_equal(got[0], pts2d)
    assert store.read_points("eks") is not None
    assert PoseResult.load(path).pts3d is not None


def test_the_correction_chain_still_gets_the_smoother_2d_it_no_longer_stores(
    cameras, rng, tmp_path
):
    """The seam v3 leans on hardest, exercised through the real stage selector.

    ``select_postprocess_input`` asks the smoother for *both* layers of its pose, and
    insists they come from one stage -- "pairing a smoothed 3D with a triangulated 2D
    would make the output a pose that no stage ever produced". v3 stops storing that 2D,
    so this is the caller that would silently receive the wrong array, or ``None``, if the
    reconstruction were not wired into :meth:`StageStore.read_points`.
    """
    from deeperfly.config import STAGES
    from deeperfly.pipeline import stages as pipeline_stages

    store = StageStore(tmp_path / "results.h5")
    _write_base(store, cameras, rng)
    store.write_cameras("bundle_adjustment", cameras)
    pts2d, pts3d, proj = _stage_arrays(cameras, rng)
    store.write_points("triangulation", pts2d=pts2d, pts3d=pts3d, reproj_error=None)
    store.write_points("eks", pts2d=proj, pts3d=pts3d, reproj_error=None)

    enabled = dict.fromkeys(STAGES, True)
    got = pipeline_stages.select_postprocess_input(enabled, store)
    assert got is not None, "the chain would skip, reporting no 3D to correct"
    got2d, got3d = got
    # The smoother's, reconstructed -- not the triangulation 2D sitting next to it.
    np.testing.assert_allclose(got2d, proj, rtol=0, atol=1e-4)
    np.testing.assert_allclose(got3d, pts3d, rtol=0, atol=1e-6)
    assert np.abs(got2d - pts2d).max() > 1.0  # positively not the upstream array
