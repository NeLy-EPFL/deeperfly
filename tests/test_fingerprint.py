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
    pose_sources,
    pts2d_source,
    stage_fingerprint,
    stage_valid,
)
from deeperfly.results import StageStore
from deeperfly.skeleton import Skeleton


def _cfg(extra: dict | None = None) -> Config:
    """A minimal two-camera config (geometry explicit, so no image sizes needed)."""
    cameras = {
        name: {
            "focal_length_px": [100.0, 100.0],
            "principal_point_px": [7.5, 3.5],
            "azimuth_deg": az,
            "distance": 10.0,
        }
        for name, az in (("cam0", 0), ("cam1", 90))
    }
    data = {"cameras": cameras, "pipeline": {}}
    for key, value in (extra or {}).items():
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return Config.from_dict(data)


@pytest.fixture
def store(tmp_path):
    return StageStore(tmp_path / "poses.h5")


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
            "pipeline.pose2d.batch_size": 2,
            "pipeline.pose2d.decode_buffer": 99,
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
    for extra in (
        {"pipeline.pose2d.precision": "float32"},
        {"cameras.cam0.input": "other_pattern"},
        {"cameras.cam0.preprocess": {"rot90": 1}},
    ):
        other = stage_fingerprint("pose2d", _cfg(extra), enabled, store)
        assert fingerprint_diff(fp, other), extra


def test_pose2d_fingerprint_candidates_iff_pictorial_enabled(store):
    config = _cfg({"pipeline.pictorial_structures.k": 7})
    off = stage_fingerprint("pose2d", config, config.stage_flags(), store)
    assert "candidates" not in off
    enabled = dict(config.stage_flags(), pictorial_structures=True)
    on = stage_fingerprint("pose2d", config, enabled, store)
    assert on["candidates"] == {"k": 7}
    # subset rule: disabling pictorial again does not invalidate the stored fp
    assert fingerprint_diff(on, off) == []
    assert fingerprint_diff(off, on)  # but enabling it does


def test_camera_geometry_excludes_footage_keys(store):
    """[cameras] input/preprocess changes do not touch the BA fingerprint."""
    base = _cfg()
    moved = _cfg({"cameras.cam0.input": "elsewhere"})
    enabled = base.stage_flags()
    assert stage_fingerprint(
        "bundle_adjustment", base, enabled, store
    ) == stage_fingerprint("bundle_adjustment", moved, enabled, store)
    geom = _cfg({"cameras.cam0.distance": 11.0})
    assert fingerprint_diff(
        stage_fingerprint("bundle_adjustment", base, enabled, store),
        stage_fingerprint("bundle_adjustment", geom, enabled, store),
    )


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
        "pose2d", _cfg({"pipeline.pose2d.precision": "float32"}), enabled, store
    )
    ok, why = stage_valid("pose2d", config, changed, store, record, tmp_path)
    assert not ok and "precision" in why


def test_stage_valid_visualization_checks_mp4s(tmp_path, store):
    config = _cfg(
        {
            "pipeline.visualization.videos": [
                {"video_name": "demo", "panels": []},
            ]
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
