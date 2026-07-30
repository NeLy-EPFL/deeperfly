"""Tests for :mod:`deeperfly.acquisition` -- the active-learning frame ranking.

The tests that matter most here are the ones guarding the two ways this feature can
fail *silently*: scoring a substituted (reseeded) layer, whose residual is 0 by
construction, and returning an unspaced top-N of near-duplicate frames. Both are
asserted directly rather than inferred from the score's value.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from deeperfly.acquisition import (
    SUGGESTIONS_FORMAT_VERSION,
    Pick,
    _spacing_capacity,
    file_md5,
    frame_reason,
    glob_mask,
    prepare_inputs,
    read_labeled_frames,
    read_suggestions,
    score_frames,
    select_frames,
    stored_vs_pose2d,
    suggestions_staleness,
    write_suggestions,
)
from deeperfly.gui.labels import Labels, labels_identity, save_labels
from deeperfly.results import StageStore
from deeperfly.triangulation import reprojection_error, triangulate

# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def truth(cameras, rng):
    """A consistent synthetic sequence: ``(pts3d (T,P,3), pts2d (V,T,P,2))``."""
    pts3d = rng.uniform(-1.5, 1.5, size=(40, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    return pts3d, pts2d


def _write_results(path, cameras, skeleton, pts2d, *, pts3d=None, meta=None):
    """Write a minimal but complete synthetic ``results.h5`` (pose2d + BA + tri)."""
    store = StageStore(path)
    store.write_pose2d(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=np.full(pts2d.shape[:3], 0.9),
        image_sizes={name: (512, 1024) for name in cameras.names},
        meta=meta,
    )
    store.write_cameras("bundle_adjustment", cameras)
    if pts3d is not None:
        store.write_points(
            "triangulation",
            pts2d=pts2d,
            pts3d=pts3d,
            reproj_error=reprojection_error(cameras, pts3d, pts2d),
        )
    return path


def _reseed(path, cameras, contra_views=(0, 1, 2)):
    """Overwrite ``triangulation/points`` at some cells with the 3D's reprojection.

    Reproduces what ``dfpose.predict`` does to a directory: the *displayed* layer's
    contralateral cells become reprojection seeds (so their stored ``reproj_error``
    is exactly 0), while ``pose2d/points`` is untouched. Also stamps the file the way
    that tool does, so the detection path is exercised.
    """
    with h5py.File(path, "a") as f:
        pts3d = np.asarray(f["triangulation/points3d"][()])
        tri = np.asarray(f["triangulation/points"][()])
        proj = np.array(cameras.project(pts3d))
        seed = np.zeros(tri.shape[:3], dtype=np.uint8)  # 1 = detection
        seed[...] = 1
        for v in contra_views:
            tri[v] = proj[v]
            seed[v] = 7  # reprojection_trusted
        f["triangulation/points"][...] = tri
        f["triangulation/reproj_error"][...] = reprojection_error(cameras, pts3d, tri)
        g = f.create_group("dfpose_predict")
        ds = g.create_dataset("contra_seed_source", data=seed)
        ds.attrs["legend"] = json.dumps(
            {"not_predicted": 0, "detection": 1, "reprojection_trusted": 7}
        )
        meta = json.loads(f.attrs["meta"])
        meta["dfpose_predict"] = {
            "model": "hrnet_w18_small_v2",
            "checkpoint_md5": "eb090eae9b36e51483116e0be93e2b37",
            "contra_seed": {"strategy": "reproject-trusted"},
        }
        f.attrs["meta"] = json.dumps(meta)
    return path


# -- score_frames -------------------------------------------------------------


def test_consistent_2d_scores_zero(cameras, truth):
    """A perfectly multi-view-consistent pose has nothing for a human to fix."""
    _, pts2d = truth
    scores = score_frames(cameras, pts2d)
    assert scores.score.shape == (40,)
    assert np.allclose(scores.score, 0.0)
    assert np.nanmax(scores.resid) < 1e-6
    assert scores.coverage["scorable_joint_frac"] == 1.0


def test_perturbing_views_raises_that_frame(cameras, truth):
    """Moving one joint in several views is exactly what the score must catch."""
    _, pts2d = truth
    pts2d = pts2d.copy()
    pts2d[:4, 7, 3] += 200.0  # frame 7, point 3, four of seven views
    scores = score_frames(cameras, pts2d)
    assert scores.score[7] == pytest.approx(scores.score.max())
    assert scores.score[7] > 0.0
    others = np.delete(scores.score, 7)
    assert scores.score[7] > others.max()
    assert scores.percentile[7] == pytest.approx(100.0)


def test_cell_disagreement_saturates_at_cap(cameras, truth):
    """Clipping: a 2567 px blown view must not turn the ranking into a lottery."""
    _, pts2d = truth
    mild = pts2d.copy()
    mild[:4, 7, 3] += 100.0
    wild = pts2d.copy()
    wild[:4, 7, 3] += 4000.0
    a = score_frames(cameras, mild, cap=60.0).score[7]
    b = score_frames(cameras, wild, cap=60.0).score[7]
    assert a == pytest.approx(b)  # both saturate
    assert 0.0 < b <= 1.0


def test_two_view_joint_is_not_scorable(cameras, truth):
    """A joint seen by exactly two views reprojects onto both by construction.

    That structural zero is *uninformative*, not safe, so ``min_views`` must drop it
    rather than let it read as "the model is right here".
    """
    _, pts2d = truth
    pts2d = pts2d.copy()
    pts2d[2:, :, 5] = np.nan  # point 5 seen by two views only
    scores = score_frames(cameras, pts2d, min_views=3)
    assert np.isnan(scores.joint[:, 5]).all()
    assert not np.isnan(scores.joint[:, 4]).any()
    assert scores.coverage["scorable_joint_frac"] < 1.0
    # ...and relaxing min_views brings it back with its unearned ~0 disagreement.
    relaxed = score_frames(cameras, pts2d, min_views=2)
    assert np.isfinite(relaxed.joint[:, 5]).all()
    assert np.allclose(np.nan_to_num(relaxed.joint[:, 5]), 0.0)


def test_score_is_a_top_k_mean_not_a_max(cameras, truth):
    """Several wrong joints must beat one very wrong joint."""
    _, pts2d = truth
    one = pts2d.copy()
    one[:4, 1, 3] += 4000.0
    many = pts2d.copy()
    many[:4, 2, 3:9] += 100.0
    a = score_frames(cameras, one, top_k=8).score[1]
    b = score_frames(cameras, many, top_k=8).score[2]
    assert b > a


def test_score_is_continuous_not_quantized(cameras, rng):
    """A quantized score leaves ties that the spacing pass breaks toward frame 0."""
    pts3d = rng.uniform(-1.5, 1.5, size=(200, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    pts2d += rng.normal(scale=8.0, size=pts2d.shape)
    scores = score_frames(cameras, pts2d)
    assert scores.coverage["n_distinct_scores"] > 190


def test_masks_restrict_scoring_not_geometry(cameras, truth, fly):
    """``--points``/``--cameras`` narrow what is *scored*; the fit keeps every view."""
    _, pts2d = truth
    pts2d = pts2d.copy()
    pts2d[:4, 7, 3] += 200.0
    names = list(fly.point_names)
    everything = score_frames(cameras, pts2d)
    elsewhere = score_frames(cameras, pts2d, point_mask=glob_mask(names, ["*abdomen*"]))
    on_it = score_frames(cameras, pts2d, point_mask=glob_mask(names, [names[3]]))
    assert elsewhere.score[7] == pytest.approx(0.0)
    assert on_it.score[7] > 0.0
    # The residual (the geometry) is identical either way -- only the aggregate moved.
    assert np.allclose(everything.resid, on_it.resid, equal_nan=True)


def test_score_rejects_bad_shapes(cameras, truth):
    _, pts2d = truth
    with pytest.raises(ValueError, match="V, T, P, 2"):
        score_frames(cameras, pts2d[..., 0])
    with pytest.raises(ValueError, match="view"):
        score_frames(cameras, pts2d[:3])
    with pytest.raises(ValueError, match="min_views"):
        score_frames(cameras, pts2d, min_views=1)
    with pytest.raises(ValueError, match="top_k"):
        score_frames(cameras, pts2d, top_k=0)


# -- the reseeded trap --------------------------------------------------------


def test_reseeding_does_not_change_the_score(tmp_path, cameras, fly, truth):
    """The headline guard: a reseeded copy must score *bit-identically* to the original.

    If the score ever reached ``triangulation/*`` this test fails, because the
    reseeded copy's stored residual is exactly 0 on the substituted cells.
    """
    pts3d, pts2d = truth
    noisy = pts2d.copy()
    noisy[:3] += 40.0  # the three views the reseed will substitute
    plain = _write_results(tmp_path / "a.h5", cameras, fly, noisy, pts3d=pts3d)
    reseeded = _write_results(tmp_path / "b.h5", cameras, fly, noisy, pts3d=pts3d)
    _reseed(reseeded, cameras)

    a, b = prepare_inputs(plain), prepare_inputs(reseeded)
    assert not a.reseeded
    assert b.reseeded
    assert b.reseed["model"] == "hrnet_w18_small_v2"
    assert "dfpose_predict/contra_seed_source" in b.reseed["detected_by"]
    assert np.array_equal(a.pts2d, b.pts2d)  # pose2d is pristine in both

    sa, sb = score_frames(cameras, a.pts2d), score_frames(cameras, b.pts2d)
    assert np.array_equal(sa.score, sb.score)
    assert np.array_equal(sa.percentile, sb.percentile)

    # And the trap itself is measurable: the stored array says ~nothing there.
    trap = stored_vs_pose2d(b, sb, threshold=15.0)
    assert trap["far_cells_median_px"] == 0.0
    assert trap["far_cells_frac_over_thresh"] == 0.0
    assert trap["pose2d_far_median_px"] > 15.0
    assert trap["n_cells"] == 3 * 40 * 38


def test_prepare_inputs_reads_pose2d_not_the_derived_layer(
    tmp_path, cameras, fly, truth
):
    """``PoseResult.load`` would hand back the substituted layer; we must not."""
    from deeperfly.results import PoseResult

    pts3d, pts2d = truth
    noisy = pts2d.copy()
    noisy[:3] += 40.0
    path = _write_results(tmp_path / "r.h5", cameras, fly, noisy, pts3d=pts3d)
    _reseed(path, cameras)

    inputs = prepare_inputs(path)
    assert np.array_equal(inputs.pts2d, noisy)
    assert not np.array_equal(PoseResult.load(path).pts2d, noisy)  # the trap


def test_prepare_inputs_detects_reseeding_without_the_per_cell_array(
    tmp_path, cameras, fly, truth
):
    """An older reseeded file has no seed-source codes; measure the mask instead."""
    pts3d, pts2d = truth
    path = _write_results(tmp_path / "r.h5", cameras, fly, pts2d + 5.0, pts3d=pts3d)
    _reseed(path, cameras)
    with h5py.File(path, "a") as f:
        del f["dfpose_predict"]
    inputs = prepare_inputs(path)
    assert inputs.reseeded  # from meta.dfpose_predict alone
    assert inputs.reseed["detected_by"] == ["meta.dfpose_predict"]
    assert inputs.substituted.sum() == 3 * 40 * 38


def test_prepare_inputs_needs_pose2d(tmp_path, cameras, fly, truth):
    pts3d, pts2d = truth
    path = _write_results(tmp_path / "r.h5", cameras, fly, pts2d, pts3d=pts3d)
    with h5py.File(path, "a") as f:
        del f["pose2d/points"]
    with pytest.raises(ValueError, match="no pose2d/points"):
        prepare_inputs(path)


def test_prepare_inputs_falls_back_to_the_config_rig(tmp_path, cameras, fly, truth):
    pts3d, pts2d = truth
    path = _write_results(tmp_path / "r.h5", cameras, fly, pts2d, pts3d=pts3d)
    assert prepare_inputs(path).cameras_from == "bundle_adjustment"
    with h5py.File(path, "a") as f:
        del f["bundle_adjustment"]
    assert prepare_inputs(path).cameras_from == "pose2d"


def test_fps_prefers_the_stamp_then_the_default(tmp_path, cameras, fly, truth):
    pts3d, pts2d = truth
    plain = _write_results(tmp_path / "a.h5", cameras, fly, pts2d, pts3d=pts3d)
    stamped = _write_results(
        tmp_path / "b.h5", cameras, fly, pts2d, pts3d=pts3d, meta={"fps": 250.0}
    )
    assert prepare_inputs(plain).fps() == (100.0, False)
    assert prepare_inputs(stamped).fps() == (250.0, True)
    assert prepare_inputs(stamped).fps(30.0) == (30.0, True)


# -- select_frames ------------------------------------------------------------


@pytest.fixture
def ramp(cameras, rng):
    """Scores over 1000 frames whose worst frames are deliberately adjacent."""
    pts3d = rng.uniform(-1.5, 1.5, size=(1000, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    # Three hard *moments*, each smeared over neighbouring frames -- the shape that
    # makes an unspaced top-N return one moment N times.
    for center in (100, 101, 102, 500, 501, 900):
        pts2d[:4, center, 3:10] += 300.0
    return score_frames(cameras, pts2d)


def test_spacing_is_hard(ramp):
    picks, shortfall = select_frames(
        ramp, count=10, min_gap_frames=200, reserve_diversity=0.0
    )
    frames = sorted(p.frame for p in picks)
    gaps = np.diff(frames)
    assert (gaps >= 200).all(), frames
    assert shortfall["most_wrong"] == len(picks)


def test_without_spacing_the_top_n_is_one_moment(ramp):
    """The failure the hard gap exists to prevent, asserted so it cannot regress."""
    unspaced = select_frames(ramp, count=6, min_gap_frames=1, reserve_diversity=0.0)[0]
    frames = sorted(p.frame for p in unspaced)
    assert min(np.diff(frames)) < 200  # near-duplicates, as expected
    spaced = select_frames(ramp, count=6, min_gap_frames=200, reserve_diversity=0.0)[0]
    assert min(np.diff(sorted(p.frame for p in spaced))) >= 200


def test_labeled_frames_are_excluded_and_seed_the_spacing(ramp):
    """A suggestion may not land on -- or next to -- existing human work."""
    picks, _ = select_frames(
        ramp,
        count=10,
        min_gap_frames=200,
        reserve_diversity=0.0,
        exclude={100, 500},
    )
    frames = [p.frame for p in picks]
    assert 100 not in frames and 500 not in frames
    for t in frames:
        assert abs(t - 100) >= 200 and abs(t - 500) >= 200


def test_diversity_reserve_is_a_uniform_grid(ramp):
    picks, shortfall = select_frames(
        ramp, count=8, min_gap_frames=100, reserve_diversity=0.25
    )
    div = [p for p in picks if p.kind == "diversity"]
    assert shortfall["diversity"] == len(div) == 2
    assert {p.grid_slot for p in div} == {(0, 2), (1, 2)}
    assert sorted(p.frame for p in div) == [250, 750]
    assert 0 not in [p.frame for p in div]  # never the settling first frame


def test_diversity_slots_keep_their_original_index(ramp):
    """A dropped grid point must not renumber the surviving slots' ``i/n``."""
    picks, _ = select_frames(
        ramp, count=8, min_gap_frames=100, reserve_diversity=0.5, exclude={125}
    )
    div = sorted((p.frame, p.grid_slot) for p in picks if p.kind == "diversity")
    # Slots are the midpoints of 4 bins: 125, 375, 625, 875; 125 is excluded.
    assert div == [(375, (1, 4)), (625, (2, 4)), (875, (3, 4))]


def test_shortfall_is_reported_not_swallowed(ramp):
    picks, shortfall = select_frames(
        ramp, count=50, min_gap_frames=200, reserve_diversity=0.0
    )
    assert shortfall["requested"] == 50
    assert shortfall["selected"] == len(picks) < 50
    assert shortfall["spacing_slots"] == 5
    assert "ran out of room" in shortfall["reason"]


def test_selection_is_deterministic(ramp):
    kwargs = dict(count=12, min_gap_frames=150, reserve_diversity=0.25, exclude={7})
    a = select_frames(ramp, **kwargs)[0]
    b = select_frames(ramp, **kwargs)[0]
    assert [(p.frame, p.kind, p.grid_slot) for p in a] == [
        (p.frame, p.kind, p.grid_slot) for p in b
    ]


def test_picks_are_ranked_by_descending_score(ramp):
    picks, _ = select_frames(ramp, count=10, min_gap_frames=100)
    values = [ramp.score[p.frame] for p in picks]
    assert values == sorted(values, reverse=True)


def test_unscorable_frames_are_never_offered(cameras, truth):
    _, pts2d = truth
    pts2d = pts2d.copy()
    pts2d[:, 3] = np.nan  # frame 3: the detector fired nowhere
    scores = score_frames(cameras, pts2d)
    picks, shortfall = select_frames(scores, count=40, min_gap_frames=1)
    assert 3 not in [p.frame for p in picks]
    assert shortfall["n_unscorable_frames"] == 1


# -- per-pick reasons ---------------------------------------------------------


def test_reason_names_the_driving_joint_and_view(cameras, truth, fly):
    _, pts2d = truth
    pts2d = pts2d.copy()
    pts2d[5, 7, 3] += 400.0  # one view (index 5 = "lm") badly wrong
    pts2d[1, 7, 3] += 120.0
    scores = score_frames(cameras, pts2d)
    reason = frame_reason(
        Pick(frame=7, kind="most-wrong"),
        scores,
        cameras=cameras,
        point_names=list(fly.point_names),
    )
    top = reason["drivers"][0]
    assert top["point"] == 3
    assert top["point_name"] == fly.point_names[3]
    assert top["worst_camera"] == "lm"
    assert top["worst_px"] > 100
    assert top["views_over_threshold"] >= 2
    assert top["relation"] in ("far", "near")
    assert 0.0 <= top["disagreement"] <= 1.0
    assert fly.point_names[3] in reason["summary"]
    assert reason["n_joints_over_threshold"] >= 1


def test_far_relation_is_geometric_not_naming(cameras, truth, fly):
    """A joint beyond the body centroid from a camera is reported ``far`` for it."""
    pts3d, pts2d = truth
    pts2d = pts2d.copy()
    centroid = np.nanmean(pts3d[0], axis=0)
    # Push point 3 far along the "lm" camera's viewing axis, past the centroid.
    axis = centroid - cameras["lm"].position
    axis /= np.linalg.norm(axis)
    moved = pts3d[0].copy()
    moved[3] = centroid + 5.0 * axis
    frame = np.array(cameras.project(moved))
    pts2d[:, 0] = frame
    pts2d[5, 0, 3] += 400.0
    scores = score_frames(cameras, pts2d)
    reason = frame_reason(
        Pick(frame=0, kind="most-wrong"),
        scores,
        cameras=cameras,
        point_names=list(fly.point_names),
    )
    driver = next(d for d in reason["drivers"] if d["point"] == 3)
    assert driver["worst_camera"] == "lm"
    assert driver["relation"] == "far"


def test_diversity_reason_says_it_is_deliberate(cameras, truth, fly):
    _, pts2d = truth
    scores = score_frames(cameras, pts2d)
    reason = frame_reason(
        Pick(frame=10, kind="diversity", grid_slot=(1, 4)),
        scores,
        cameras=cameras,
        point_names=list(fly.point_names),
    )
    assert reason["grid_slot"] == [1, 4]
    assert "on purpose" in reason["summary"]
    assert "drivers" not in reason


# -- glob selectors -----------------------------------------------------------


def test_glob_mask(fly):
    names = list(fly.point_names)
    assert glob_mask(names, None).all()
    assert glob_mask(names, []).all()
    picked = [n for n, m in zip(names, glob_mask(names, ["l?_claw"])) if m]
    assert picked == ["lf_claw", "lm_claw", "lh_claw"]
    with pytest.raises(ValueError, match="none of"):
        glob_mask(names, ["nope*"])


# -- the labels sidecar -------------------------------------------------------


def _identity(cameras, fly, n_frames):
    return labels_identity(
        point_names=list(fly.point_names),
        camera_names=list(cameras.names),
        n_frames=n_frames,
    )


def test_read_labeled_frames_absent(tmp_path, cameras, fly):
    got = read_labeled_frames(
        tmp_path / "labels.h5", identity=_identity(cameras, fly, 40)
    )
    assert got is None


def test_read_labeled_frames_counts_occlusion_and_reviewed(tmp_path, cameras, fly):
    """A frame where the operator authored only occlusions is still *done*."""
    labels = Labels.empty(7, 40, 38)
    labels.set_gt(0, 5, 3, (10.0, 20.0))
    labels.set_occluded(1, 9, 4, True)  # occlusion-only frame
    labels.set_reviewed(12, True)  # reviewed, no point labels
    identity = _identity(cameras, fly, 40)
    path = tmp_path / "labels.h5"
    save_labels(path, labels, identity=identity)

    got = read_labeled_frames(path, identity=identity)
    assert got["labeled_frames"] == [5, 9]
    assert got["reviewed_frames"] == [12]
    assert got["n_gt"] == 1 and got["n_occluded"] == 1
    assert got["md5"] == file_md5(path)


def test_read_labeled_frames_refuses_another_recording(tmp_path, cameras, fly):
    identity = _identity(cameras, fly, 40)
    save_labels(tmp_path / "labels.h5", Labels.empty(7, 40, 38), identity=identity)
    other = _identity(cameras, fly, 41)
    with pytest.raises(ValueError, match="different result"):
        read_labeled_frames(tmp_path / "labels.h5", identity=other)


# -- the suggestions sidecar --------------------------------------------------


def test_write_read_roundtrip(tmp_path):
    doc = {
        "deeperfly_suggestions_format_version": SUGGESTIONS_FORMAT_VERSION,
        "frames": [{"rank": 1, "frame": 5}],
    }
    out = write_suggestions(tmp_path / "labels_suggest.json", doc)
    assert read_suggestions(out) == doc
    assert not list(tmp_path.glob("*.tmp"))  # atomic: no leftover


def test_read_suggestions_tolerates_absent_and_broken(tmp_path):
    assert read_suggestions(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_suggestions(bad) is None
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    assert read_suggestions(listy) is None


def test_unknown_format_version_reads_as_absent(tmp_path):
    p = tmp_path / "labels_suggest.json"
    p.write_text(json.dumps({"deeperfly_suggestions_format_version": 999}))
    assert read_suggestions(p) is None


# -- staleness ----------------------------------------------------------------


def _doc(**over):
    doc = {
        "deeperfly_suggestions_format_version": SUGGESTIONS_FORMAT_VERSION,
        "source": {
            "identity": {"n_frames": 40, "point_names": ["a"], "camera_names": ["c"]},
            "results_md5": "deadbeef",
            "results_size": 10,
            "results_mtime_ns": 1,
        },
        "labels": {"labeled_frames": [1]},
        "frames": [{"frame": 1}, {"frame": 2}, {"frame": 3}],
    }
    doc.update(over)
    return doc


def test_staleness_none():
    got = suggestions_staleness(_doc(), identity=_doc()["source"]["identity"])
    assert got["level"] == "none"
    assert got["reasons"] == []


def test_staleness_hard_on_a_different_recording():
    live = {"n_frames": 41, "point_names": ["a"], "camera_names": ["c"]}
    got = suggestions_staleness(_doc(), identity=live)
    assert got["level"] == "hard"
    assert "different recording" in got["reasons"][0]


def test_staleness_predictions_when_results_changed(tmp_path):
    results = tmp_path / "results.h5"
    results.write_bytes(b"different bytes entirely")
    got = suggestions_staleness(_doc(), results_path=results)
    assert got["level"] == "predictions"
    assert "superseded" in got["reasons"][0]


def test_staleness_ignores_a_mere_touch(tmp_path):
    results = tmp_path / "results.h5"
    results.write_bytes(b"payload")
    doc = _doc()
    doc["source"]["results_md5"] = file_md5(results)
    doc["source"]["results_size"] = results.stat().st_size
    doc["source"]["results_mtime_ns"] = 12345  # a touch: stat differs, bytes do not
    got = suggestions_staleness(doc, results_path=results)
    assert got["level"] == "none"


def test_staleness_progress_as_the_operator_works():
    got = suggestions_staleness(_doc(), labeled_frames={1, 2})
    assert got["level"] == "progress"
    assert got["done"] == 2
    assert got["total"] == 3
    assert "2 of 3" in got["reasons"][0]


def test_staleness_hard_wins_over_progress():
    live = {"n_frames": 41, "point_names": ["a"], "camera_names": ["c"]}
    got = suggestions_staleness(_doc(), identity=live, labeled_frames={1, 2, 3})
    assert got["level"] == "hard"


# -- end-to-end library use ---------------------------------------------------


def test_pipeline_end_to_end_on_a_synthetic_file(tmp_path, cameras, fly, rng):
    """The whole library path: file -> inputs -> scores -> picks -> sidecar."""
    pts3d = rng.uniform(-1.5, 1.5, size=(600, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    pts2d[:4, 150, 3:10] += 300.0
    pts2d[:4, 400, 3:10] += 200.0
    path = _write_results(tmp_path / "results.h5", cameras, fly, pts2d, pts3d=pts3d)

    inputs = prepare_inputs(path)
    scores = score_frames(inputs.cameras, inputs.pts2d)
    picks, shortfall = select_frames(scores, count=5, min_gap_frames=100)
    assert [p.frame for p in picks][:2] == [150, 400]

    from deeperfly.acquisition import build_suggestions

    doc = build_suggestions(
        inputs,
        scores,
        picks,
        params={
            "count": 5,
            "min_gap_s": 1.0,
            "fps": 100.0,
            "min_gap_frames": 100,
            "threshold_px": 15.0,
        },
        shortfall=shortfall,
        labels=None,
        output_dir=tmp_path,
    )
    out = write_suggestions(tmp_path / "labels_suggest.json", doc)
    back = read_suggestions(out)
    assert back["source"]["scored_array"] == "pose2d/points"
    assert back["source"]["never_scored"] == [
        "triangulation/points",
        "triangulation/reproj_error",
    ]
    assert back["frames"][0]["frame"] == 150
    assert back["frames"][0]["t_s"] == 1.5
    assert back["labels"]["exists"] is False
    assert back["source"]["results_md5"] == file_md5(path)
    # A triangulation-only check on the geometry: the un-perturbed frames really are
    # consistent, so the queue is not ranking noise.
    assert (
        np.nanmax(
            reprojection_error(cameras, triangulate(cameras, pts2d[:, 0]), pts2d[:, 0])
        )
        < 1e-6
    )


# -- regression tests for defects found by the verification pass ---------------


@pytest.mark.parametrize(
    "doc",
    [
        {"frames": ["junk"]},  # a row that is not a dict
        {"frames": {"a": 1}},  # frames is a dict, not a list
        {"frames": [{"frame": "one"}]},  # non-numeric frame index
        {"frames": [{"frame": None}]},
        {"frames": [{"frame": True}]},  # bool is not a frame index
        {"source": "nope", "frames": []},  # source is not a dict
        {},
    ],
)
def test_staleness_never_raises_on_a_hand_edited_sidecar(doc):
    """The sidecar is plain JSON an operator may edit, so every shape is untrusted.

    A malformed queue must degrade to "nothing to show" -- never raise into the editor
    the operator is labeling in. Each of these shapes crashed before.
    """
    out = suggestions_staleness(doc, labeled_frames=[0, 5])
    assert out["level"] in {"none", "progress", "predictions", "hard"}
    assert isinstance(out["reasons"], list)
    assert out["done"] >= 0 and out["total"] >= 0


def test_spacing_capacity_accounts_for_the_already_labelled_seeds():
    """``spacing_slots`` must be a real ceiling, not ``n_frames // gap``.

    The labeled frames seed the spacing constraint and carve the timeline up, so the
    naive bound overstates what is reachable -- 21 vs the true 15 on the real recording
    (4,016 frames, gap 200, labels at 0/1923/2244/2252/3279). A reader who trusts a
    loose bound lowers --min-gap-s for no reason.
    """
    assert _spacing_capacity(4016, 200, set()) == 21
    assert _spacing_capacity(4016, 200, {0, 1923, 2244, 2252, 3279}) == 15
    # a seed at every gap leaves no room at all
    assert _spacing_capacity(1000, 100, set(range(0, 1000, 100))) == 0
    assert _spacing_capacity(0, 200, set()) == 0


def test_staleness_hashes_when_a_sidecar_carries_only_an_md5(tmp_path):
    """Absent size/mtime must mean "hash it", not "assume unchanged".

    The writer always stamps size + mtime + md5, but a hand-written sidecar carrying
    only ``results_md5`` was never checked at all, reporting "none" against genuinely
    superseded predictions.
    """
    f = tmp_path / "results.h5"
    f.write_bytes(b"live bytes")
    doc = {"frames": [], "source": {"results_md5": "0" * 32}}  # deliberately wrong
    out = suggestions_staleness(doc, results_path=f)
    assert out["level"] == "predictions", out
    # ...and the matching md5 is still reported as fresh
    doc["source"]["results_md5"] = file_md5(f)
    assert suggestions_staleness(doc, results_path=f)["level"] == "none"
