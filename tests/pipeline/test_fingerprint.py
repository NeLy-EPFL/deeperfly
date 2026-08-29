"""Unit tests for the per-stage config fingerprints and the run record."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.pipeline.fingerprint import (
    RunRecord,
    cameras_source,
    fingerprint_diff,
    model_source,
    pose_sources,
    pts2d_source,
    pts3d_source,
    stage_fingerprint,
    stage_valid,
)
from deeperfly.results import StageStore
from deeperfly.skeleton import Skeleton


def _cfg(extra: dict | None = None) -> Config:
    """A minimal two-view config with a full detection plan (geometry explicit)."""
    cameras = {
        name: {
            "focal_length_px": [100.0, 100.0],
            "principal_point_px": [7.5, 3.5],
            "azimuth_deg": az,
            "distance": 10.0,
        }
        for name, az in (("cam0", 0), ("cam1", 90))
    }
    for name in cameras:
        cameras[name]["video"] = f"{name}\\.mp4"
    data = {
        "cameras": cameras,
        "pose2d": {"class": "hrnet", "weights": "w.pth", "input_size": [256, 512]},
        "pipeline": {},
    }
    for key, value in (extra or {}).items():
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return Config.from_dict(data)


@pytest.fixture
def store(tmp_path):
    return StageStore(tmp_path / "results.h5")


def _seed_pose2d(store, cameras, *, candidates=None):
    v, t, n = len(cameras), 2, 38
    store.write_pose2d(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=np.zeros((v, t, n, 2)),
        conf=np.ones((v, t, n)),
        image_sizes={name: (8, 16) for name in cameras.names},
        candidates=candidates,
    )


# -- RunRecord ------------------------------------------------------------------


def test_record_roundtrip_and_truncation(tmp_path):
    record = RunRecord(tmp_path / "run.json")
    assert record.get("pose2d") is None
    record.set("pose2d", {"a": 1})
    record.set("triangulation", {"b": 2})
    # re-read from disk
    record = RunRecord(tmp_path / "run.json")
    assert record.get("pose2d") == {"a": 1}
    assert record.get("triangulation") == {"b": 2}
    # setting an upstream stage drops every later entry (their inputs changed)
    record.set("pose2d", {"a": 3})
    assert record.get("triangulation") is None
    assert RunRecord(tmp_path / "run.json").get("triangulation") is None


def test_record_unknown_version_resets(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"format_version": 99, "stages": {"pose2d": {}}}))
    assert RunRecord(path).get("pose2d") is None


def test_record_garbage_resets(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("not json {")
    assert RunRecord(path).get("pose2d") is None


# -- fingerprint_diff -------------------------------------------------------------


def test_diff_empty_on_match():
    fp = {"a": 1, "nested": {"x": [1, 2]}}
    assert fingerprint_diff(fp, fp) == []


def test_diff_reports_nested_paths():
    (line,) = fingerprint_diff({"nested": {"x": 1}}, {"nested": {"x": 2}})
    assert line.startswith("nested.x: 1 -> 2")


def test_diff_subset_semantics():
    # extra stored keys are ignored; missing expected keys are reported
    assert fingerprint_diff({"a": 1, "extra": 9}, {"a": 1}) == []
    (line,) = fingerprint_diff({"a": 1}, {"a": 1, "candidates": {"k": 5}})
    assert "candidates" in line and "(absent)" in line


# -- stage fingerprints ------------------------------------------------------------


def test_pose2d_fingerprint_excludes_perf_knobs(store):
    base = _cfg()
    perf = _cfg(
        {
            "pose2d.batch_size": 2,
            "pose2d.decode_buffer": 99,
            "io.image.workers": 3,
        }
    )
    enabled = base.stage_flags()
    assert stage_fingerprint("pose2d", base, enabled, store) == stage_fingerprint(
        "pose2d", perf, enabled, store
    )


def test_pose2d_fingerprint_tracks_result_affecting_keys(store):
    base = _cfg()
    enabled = base.stage_flags()
    fp = stage_fingerprint("pose2d", base, enabled, store)
    # precision is the [pose2d] fallback: changing it flows into every inheriting
    # model's resolved precision, so the fingerprint still changes.
    assert fingerprint_diff(
        fp,
        stage_fingerprint(
            "pose2d", _cfg({"pose2d.precision": "float32"}), enabled, store
        ),
    )
    # The synthesized plan, through the three keys that still describe it: the footage
    # pattern, the detection window, and the model's input.
    src = _cfg()
    src.data["cameras"]["cam0"]["video"] = "other.mp4"
    window = _cfg({"pose2d.crops": {"cam0": [1, 2, 5, 4]}})
    model = _cfg({"pose2d.input_size": [128, 256]})
    for changed in (src, window, model):
        assert fingerprint_diff(
            fp, stage_fingerprint("pose2d", changed, enabled, store)
        )


def test_pose2d_fingerprint_records_the_resolved_precision_per_model(store):
    base = _cfg()
    enabled = base.stage_flags()
    fp = stage_fingerprint("pose2d", base, enabled, store)
    # The resolved value lives inside the model's dict -- not as a top-level key, which
    # is what let a per-model override go unnoticed when there could be several.
    over = _cfg({"pose2d.precision": "float32"})
    fp_over = stage_fingerprint("pose2d", over, enabled, store)
    assert fingerprint_diff(fp, fp_over)
    assert fp_over["models"]["hrnet"]["precision"] == "float32"
    assert "precision" not in fp_over  # no stale top-level key


def test_pose2d_fingerprint_candidates_iff_pictorial_enabled(store):
    config = _cfg({"pictorial_structures.k": 7})
    off = stage_fingerprint("pose2d", config, config.stage_flags(), store)
    assert "candidates" not in off
    enabled = dict(config.stage_flags(), pictorial_structures=True)
    on = stage_fingerprint("pose2d", config, enabled, store)
    assert on["candidates"] == {
        "k": 7,
        "peak_threshold": 5e-2,
        "peak_threshold_rel": 0.0,
    }
    # subset rule: disabling pictorial again does not invalidate the stored fp
    assert fingerprint_diff(on, off) == []
    assert fingerprint_diff(off, on)  # but enabling it does
    # The peak gate prunes during EXTRACTION, so a set pruned too hard cannot be repaired
    # downstream: moving it has to re-detect, exactly as moving `k` does.
    looser = _cfg(
        {"pictorial_structures.k": 7, "pictorial_structures.peak_threshold_rel": 0.1}
    )
    assert fingerprint_diff(stage_fingerprint("pose2d", looser, enabled, store), on)


def test_bundle_adjustment_fingerprint_is_geometry_only(store):
    """BA depends on the rig geometry, not the footage sources feeding the views."""
    base = _cfg()
    enabled = base.stage_flags()
    # changing a footage pattern does not touch BA (footage lives in the plan, not here)
    moved = _cfg()
    moved.data["cameras"]["cam0"]["video"] = "elsewhere.mp4"
    assert stage_fingerprint(
        "bundle_adjustment", base, enabled, store
    ) == stage_fingerprint("bundle_adjustment", moved, enabled, store)
    # a view geometry edit does invalidate BA
    geom = _cfg({"cameras.cam0.distance": 11.0})
    assert fingerprint_diff(
        stage_fingerprint("bundle_adjustment", base, enabled, store),
        stage_fingerprint("bundle_adjustment", geom, enabled, store),
    )


def test_the_calibration_a_config_points_at_is_fingerprinted_by_content(
    store, tmp_path
):
    """A solved rig must invalidate its consumers -- by CONTENT, not merely by path.

    ``camera_table()`` never sees the calibration -- it is a path in its own table, not a
    camera -- so fingerprinting the tables alone left the rig a run *actually builds from*
    invisible to the cache. Re-solving a calibration rewrites it under the same name, which
    is the common case and the one a path cannot see -- so cached bundle_adjustment /
    pictorial_structures / triangulation / visualization were reused against a rig that no
    longer existed.
    """
    rig = tmp_path / "solved.toml"
    rig.write_text("# solved rig, pass 1\n")
    enabled = _cfg().stage_flags()
    before = stage_fingerprint(
        "bundle_adjustment", _cfg({"calibration.path": str(rig)}), enabled, store
    )

    # Re-solved IN PLACE: same path, different numbers.
    rig.write_text("# solved rig, pass 2 -- different numbers\n")
    after = stage_fingerprint(
        "bundle_adjustment", _cfg({"calibration.path": str(rig)}), enabled, store
    )
    assert fingerprint_diff(before, after), (
        "re-solving in place must invalidate the cache"
    )

    # And pointing at a different file is a different rig too.
    other = tmp_path / "other.toml"
    other.write_text(
        "# solved rig, pass 2 -- different numbers\n"
    )  # same bytes, new name
    elsewhere = stage_fingerprint(
        "bundle_adjustment", _cfg({"calibration.path": str(other)}), enabled, store
    )
    assert fingerprint_diff(after, elsewhere)

    # A calibration that vanishes is a distinct state, not a silently unchanged one.
    rig.unlink()
    gone = stage_fingerprint(
        "bundle_adjustment", _cfg({"calibration.path": str(rig)}), enabled, store
    )
    assert fingerprint_diff(after, gone)


def test_an_orbit_only_config_is_unaffected_by_the_calibration_key(store):
    """The default path must not gain a fingerprint entry it never had."""
    fp = stage_fingerprint("bundle_adjustment", _cfg(), _cfg().stage_flags(), store)
    assert "calibration" not in json.dumps(fp)


def test_source_selectors_follow_enabled_and_present(store, cameras):
    config = _cfg()
    enabled = {n: True for n in config.stage_flags()}
    # nothing in the store yet -> config rig / pose2d points
    assert cameras_source(enabled, store) == "config"
    assert pts2d_source(enabled, store) == "pose2d"
    assert pose_sources(enabled, store) == {"pts2d": "pose2d", "pts3d": None}

    _seed_pose2d(store, cameras)
    store.write_cameras("bundle_adjustment", cameras)
    v, t, n = len(cameras), 2, 38
    store.write_points(
        "pictorial_structures",
        pts2d=np.zeros((v, t, n, 2)),
        pts3d=np.zeros((t, n, 3)),
        reproj_error=None,
    )
    assert cameras_source(enabled, store) == "bundle_adjustment"
    assert pts2d_source(enabled, store) == "pictorial_structures"
    assert pose_sources(enabled, store)["pts3d"] == "pictorial_structures"

    # a disabled stage's output is not selected, even though it is present
    disabled = dict(enabled, bundle_adjustment=False, pictorial_structures=False)
    assert cameras_source(disabled, store) == "config"
    assert pts2d_source(disabled, store) == "pose2d"
    assert pose_sources(disabled, store) == {"pts2d": "pose2d", "pts3d": None}


def test_triangulation_fingerprint_embeds_config_rig_only_without_ba(store, cameras):
    base = _cfg()
    geom = _cfg({"cameras.cam0.distance": 11.0})
    enabled = base.stage_flags()  # bundle_adjustment on, but nothing stored yet
    # no BA output stored -> the config geometry is embedded -> edits invalidate
    assert fingerprint_diff(
        stage_fingerprint("triangulation", base, enabled, store),
        stage_fingerprint("triangulation", geom, enabled, store),
    )
    # with a BA output stored, the rig source is the BA stage; geometry edits
    # flow through BA's own fingerprint (and cascade) instead
    _seed_pose2d(store, cameras)
    store.write_cameras("bundle_adjustment", cameras)
    assert stage_fingerprint(
        "triangulation", base, enabled, store
    ) == stage_fingerprint("triangulation", geom, enabled, store)


def test_pts3d_and_model_source_selectors(store, cameras):
    config = _cfg()
    enabled = {n: True for n in config.stage_flags()}
    assert pts3d_source(enabled, store) is None  # nothing stored
    assert model_source(enabled, store) is None

    _seed_pose2d(store, cameras)
    v, t, n = len(cameras), 2, 38
    store.write_points(
        "triangulation",
        pts2d=np.zeros((v, t, n, 2)),
        pts3d=np.zeros((t, n, 3)),
        reproj_error=None,
    )
    assert pts3d_source(enabled, store) == "triangulation"
    # a disabled triangulation is not selected even when present
    assert pts3d_source(dict(enabled, triangulation=False), store) is None

    store.write_ik(
        angles=np.zeros((t, 4)), angle_names=["a"] * 4, model_pts3d=np.zeros((t, n, 3))
    )
    assert model_source(enabled, store) == "inverse_kinematics"
    assert model_source(dict(enabled, inverse_kinematics=False), store) is None


def test_inverse_kinematics_fingerprint_tracks_template_and_bounds(store, cameras):
    base = _cfg()
    enabled = base.stage_flags()
    bounds = _cfg(
        {"inverse_kinematics.bounds.rf_trochanterfemur-rf_tibia-pitch": [10, 160]}
    )
    legs = _cfg({"inverse_kinematics.legs": ["rf", "lf"]})
    # A retarget: the head chain's markers, redeclared at a different offset. It has to
    # invalidate for the same reason a bounds override does -- it changes what the fit is
    # fitting -- and it did not until the marker placement joined the digest.
    retarget = _cfg(
        {
            "inverse_kinematics.markers.head": {
                "neck": {"body": "c_head", "offset": [0.0, 0.0, 0.0], "base": True},
                "l_antenna": {"body": "l_pedicel", "offset": [0.0, 0.0, 0.05]},
                "r_antenna": {"body": "r_pedicel", "offset": [0.0, 0.0, 0.05]},
            }
        }
    )
    # a bounds override, a leg restriction and a marker retarget each change it
    assert fingerprint_diff(
        stage_fingerprint("inverse_kinematics", base, enabled, store),
        stage_fingerprint("inverse_kinematics", bounds, enabled, store),
    )
    assert fingerprint_diff(
        stage_fingerprint("inverse_kinematics", base, enabled, store),
        stage_fingerprint("inverse_kinematics", legs, enabled, store),
    )
    assert fingerprint_diff(
        stage_fingerprint("inverse_kinematics", base, enabled, store),
        stage_fingerprint("inverse_kinematics", retarget, enabled, store),
    )
    # an identical config is cache-valid
    assert not fingerprint_diff(
        stage_fingerprint("inverse_kinematics", base, enabled, store),
        stage_fingerprint("inverse_kinematics", _cfg(), enabled, store),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"inverse_kinematics.n_iterations": 5},
        {"inverse_kinematics.neutral_weight": 0.5},
        {"inverse_kinematics.damping": 1e-4},
        {"inverse_kinematics.position_tolerance": 1e-5},
        {"inverse_kinematics.angle_tolerance": 1e-5},
        {"inverse_kinematics.fixed_body": False},
        {"inverse_kinematics.symmetric_segments": False},
        {"inverse_kinematics.weigh_by_confidence": True},
        {"inverse_kinematics.parallel": True},
        {"inverse_kinematics.segment_len": 64},
        {"inverse_kinematics.overlap_len": 3},
    ],
)
def test_inverse_kinematics_fingerprint_tracks_every_solver_knob(
    store, cameras, override
):
    """Each QuickIK knob changes the fit, so each must invalidate the cached angles.

    ``parallel``/``segment_len``/``overlap_len`` read like performance knobs (which this
    module deliberately excludes) but are not: a segmented solve restarts every segment
    from the neutral pose, so the angles it produces genuinely differ.
    """
    base = _cfg()
    enabled = base.stage_flags()
    assert fingerprint_diff(
        stage_fingerprint("inverse_kinematics", base, enabled, store),
        stage_fingerprint("inverse_kinematics", _cfg(override), enabled, store),
    )


def test_inverse_kinematics_fingerprint_names_its_solver(store, cameras):
    """A record written by the previous solver must not pass as valid QuickIK output.

    Comparison is subset semantics -- a key that *drops out* of the expected fingerprint
    does not invalidate the cache -- so dropping the old scipy knobs was not enough on
    its own. The ``solver`` key is what makes an upgraded install recompute instead of
    silently serving scipy-era angles to a QuickIK-driven GUI.
    """
    base = _cfg()
    enabled = base.stage_flags()
    fresh = stage_fingerprint("inverse_kinematics", base, enabled, store)
    assert fresh["solver"] == "quickik"
    legacy = {k: v for k, v in fresh.items() if k not in {"solver", "solver_revision"}}
    legacy |= {"max_nfev": 100, "loss": "linear", "f_scale": 1.0}
    assert fingerprint_diff(fresh, legacy)
    # ... and a hand-bumped revision invalidates too, for a change no config key shows.
    assert fingerprint_diff(fresh, {**fresh, "solver_revision": 0})


# -- stage_valid -------------------------------------------------------------------


def test_stage_valid_needs_record_fingerprint_and_output(tmp_path, store, cameras):
    config = _cfg()
    enabled = config.stage_flags()
    record = RunRecord(tmp_path / "run.json")
    expected = stage_fingerprint("pose2d", config, enabled, store)

    ok, why = stage_valid("pose2d", config, expected, store, record, tmp_path)
    assert not ok and "no cached result" in why

    record.set("pose2d", expected)
    ok, why = stage_valid("pose2d", config, expected, store, record, tmp_path)
    assert not ok and "missing" in why  # fingerprint matches but no h5 output

    _seed_pose2d(store, cameras)
    ok, why = stage_valid("pose2d", config, expected, store, record, tmp_path)
    assert ok and why is None

    changed = stage_fingerprint(
        "pose2d", _cfg({"pose2d.precision": "float32"}), enabled, store
    )
    ok, why = stage_valid("pose2d", config, changed, store, record, tmp_path)
    assert not ok and "precision" in why


def test_stage_valid_visualization_checks_mp4s(tmp_path, store):
    config = _cfg(
        {
            "visualization.default_video": {"cell": [8, 8], "footage": False},
            "visualization.videos": {
                "demo": {"grid": [["cam0"]], "layers": [{"draw": "skeleton_3d"}]}
            },
        }
    )
    enabled = config.stage_flags()
    record = RunRecord(tmp_path / "run.json")
    expected = stage_fingerprint("visualization", config, enabled, store)
    record.set("visualization", expected)

    ok, why = stage_valid("visualization", config, expected, store, record, tmp_path)
    assert not ok and "demo" in why  # fingerprint fine, MP4 missing

    (tmp_path / "demo.mp4").write_bytes(b"rendered")
    ok, _ = stage_valid("visualization", config, expected, store, record, tmp_path)
    assert ok
