"""Tests for the ``postprocess`` stage and its op chain.

The premise the stage rests on is physical, not statistical: on a tethered fly the
thorax-coxa joints and the neck sit on a sclerotized plate, so their position is a
constant of the recording and everything the estimate does over time is per-frame
noise. The synthetic recording here makes that literal -- a few points are held
*exactly* still and then given noise, so "the freeze should recover the constant" has
a ground truth, and the points that really move are checked to be left alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import postprocess as pp
from deeperfly.config import Config
from deeperfly.pipeline import fingerprint, stages
from deeperfly.results import PoseResult, StageStore
from deeperfly.skeleton import Skeleton

#: Points held exactly still in the synthetic truth -- the tethered-thorax set.
RIGID = ("lf_thorax_coxa", "rf_thorax_coxa", "lm_thorax_coxa")
#: A point that genuinely moves, to check the freeze does not reach past its list.
MOVING = "lf_claw"

N_FRAMES = 40
NOISE_PX = 1.5


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


@pytest.fixture
def scene(cameras, fly, rng):
    """``(truth3d, pts3d, pts2d)``: RIGID points exactly still, everything else moving.

    ``pts2d`` is the noisy per-view observation and ``pts3d`` its triangulation, so the
    arrays handed to the stage are the shape and the noise level a real upstream stage
    produces.
    """
    n_points = len(fly.point_names)
    base = rng.uniform(-1.0, 1.0, size=(1, n_points, 3))
    phase = np.linspace(0.0, 2.0 * np.pi, N_FRAMES)[:, None, None]
    truth = base + 0.3 * np.sin(phase + rng.uniform(0, 6.28, (1, n_points, 3)))
    rigid_cols = [fly.point_names.index(n) for n in RIGID]
    truth[:, rigid_cols] = base[:, rigid_cols]  # exactly constant over time
    pts2d = np.asarray(cameras.project(truth)) + rng.normal(
        scale=NOISE_PX, size=(len(cameras), N_FRAMES, n_points, 2)
    )
    return truth, np.asarray(cameras.triangulate(pts2d)), pts2d


def _config(points, **opts) -> Config:
    """A config whose chain is a single ``static`` op over ``points``."""
    return Config.from_dict(
        {"postprocess": {"ops": [{"op": "static", "points": list(points), **opts}]}}
    )


def _chain(*ops) -> Config:
    """A config with an explicit ordered op chain."""
    return Config.from_dict({"postprocess": {"ops": list(ops)}})


def _validate_method(method: str) -> str:
    """The op's method check, reached the way a config reaches it."""
    return pp.op_static(
        np.zeros((1, 1, 1, 2)),
        np.zeros((1, 1, 3)),
        skeleton=Skeleton.fly(),
        spec={"op": "static", "method": method, "points": ["lf_claw"]},
    )


# -- the center estimators -----------------------------------------------------


def test_every_method_recovers_a_clean_constant(rng):
    """The floor: with symmetric noise about a known value, all five land on it.

    A test that only pinned the median would let a broken alternative ship, since the
    stage's *shape* is identical whichever it uses -- the output is constant over time
    either way, so every structural assertion in this file passes on a wrong center.
    """
    truth = np.array([3.0, -7.0, 11.0])
    samples = truth + rng.normal(scale=0.05, size=(400, 3))
    pts3d = np.tile(samples[:, None, :], (1, 2, 1))  # (T, 2, 3), point 1 unused
    for method in pp.STATIC_METHODS:
        got = pp.freeze_3d(pts3d, [0], method=method)[0, 0]
        np.testing.assert_allclose(got, truth, atol=0.02, err_msg=method)


def test_the_mean_follows_an_outlier_and_the_robust_centers_do_not(rng):
    """What actually separates the five: how each answers a contaminated sample.

    The mean is *supposed* to move -- it is the minimum-variance choice only when the
    noise is clean, and offering it is how a user checks that assumption. The point of
    the flag is that the others do not have to move with it.
    """
    values = np.zeros((100, 1, 3))
    values += rng.normal(scale=0.01, size=values.shape)
    values[:10] += 1000.0  # 10% of frames blown, all in one direction
    got = {m: pp.freeze_3d(values, [0], method=m)[0, 0, 0] for m in pp.STATIC_METHODS}
    assert got["mean"] > 50.0  # dragged by a tenth of the sample
    for robust in ("median", "trimmed_mean", "mode", "geometric_median"):
        assert abs(got[robust]) < 0.1, f"{robust} moved to {got[robust]}"


def test_the_mode_follows_density_where_the_median_follows_count(rng):
    """The case that justifies shipping ``mode`` at all.

    The median is the 50th percentile, so it follows the *count*: a minority
    contaminant, however gross, cannot move it -- which is exactly why it is the
    default. What it cannot survive is a contaminant that is the **majority**. Here the
    correct detections are tight and outnumbered, and the wrong ones are scattered:
    the median goes with the crowd and lands at ~1.1, a position the point never
    occupied, while the half-sample mode follows the *density* onto the true peak.

    That asymmetry -- tight signal, diffuse error -- is what a detector that sometimes
    locks onto a nearby wrong feature actually produces, and it is the one shape a
    breakdown-point argument does not cover. Note the premise this test does *not*
    make: at a merely 60/40 split the median is still inside the true cluster, so
    ``mode`` buys nothing there.
    """
    tight = 0.02  # the correct cluster's spread; the scale everything is judged on
    values = np.zeros((100, 1, 3))
    values[:40] = rng.normal(0.0, tight, size=(40, 1, 3))  # correct, and outnumbered
    values[40:] = rng.normal(5.0, 4.0, size=(60, 1, 3))  # scattered, wrong, majority
    got = {
        m: pp.freeze_3d(values, [0], method=m)[0, 0, 0]
        for m in ("median", "mean", "mode")
    }
    # Thresholds in units of the true cluster's own spread, not magic numbers: the mode
    # lands inside it, the median lands an order of magnitude outside it.
    assert abs(got["mode"]) < 5 * tight
    assert got["median"] > 10 * tight
    assert got["mean"] > got["median"]  # and the mean further still


def test_the_geometric_median_is_rotation_equivariant(rng):
    """Its whole reason for existing: the per-axis centers are not.

    Rotate the world frame, freeze, rotate back -- the geometric median returns the same
    point, because it minimizes Euclidean distance rather than each coordinate's
    absolute deviations separately. A camera rig's world axes are an arbitrary choice,
    so an estimator that depends on them is reporting partly on the rig.
    """
    from scipy.spatial.transform import Rotation

    pts = rng.normal(size=(80, 3))
    pts[:8] += np.array([20.0, 5.0, -3.0])  # directional outliers: where they differ
    block = pts[:, None, :]
    rot = Rotation.from_rotvec([0.4, -0.9, 0.2]).as_matrix()
    rotated = (block @ rot.T).copy()

    def frozen(arr, method):
        return pp.freeze_3d(arr, [0], method=method)[0, 0]

    geo = frozen(block, "geometric_median")
    geo_rot = frozen(rotated, "geometric_median") @ rot  # undo the rotation
    np.testing.assert_allclose(geo, geo_rot, atol=1e-6)
    # The per-axis median is not: same data, same rotation, a different point.
    med = frozen(block, "median")
    med_rot = frozen(rotated, "median") @ rot
    assert np.linalg.norm(med - med_rot) > 1e-3


def test_trim_is_validated_and_bounds_the_two_extremes(rng):
    """``trim = 0`` is the mean; a trim outside [0, 0.5) is an error, not a silent clip."""
    values = np.zeros((100, 1, 3)) + rng.normal(scale=0.01, size=(100, 1, 3))
    values[:10] += 1000.0
    as_mean = pp.freeze_3d(values, [0], method="trimmed_mean", trim=0.0)
    plain = pp.freeze_3d(values, [0], method="mean")
    np.testing.assert_allclose(as_mean, plain)
    with pytest.raises(ValueError, match=r'op = "static".*trim must be in'):
        pp.freeze_3d(values, [0], method="trimmed_mean", trim=0.5)


def test_an_unknown_method_names_the_choices():
    with pytest.raises(ValueError, match=r'unknown .*op = "static".*method'):
        _validate_method("mediun")


def test_the_half_sample_mode_handles_degenerate_samples():
    """The sizes the recursion cannot recurse on, and the all-NaN column."""
    assert np.isnan(pp._half_sample_mode(np.array([np.nan, np.nan])))
    assert pp._half_sample_mode(np.array([4.0])) == 4.0
    assert pp._half_sample_mode(np.array([2.0, 6.0])) == 4.0
    assert pp._half_sample_mode(np.array([1.0, np.nan, 1.0, 1.0, 9.0])) == 1.0


def test_the_geometric_median_survives_landing_on_a_sample_point():
    """Weiszfeld's classic division-by-zero: an iterate exactly on an input.

    Made certain here by a sample whose own median *is* one of its points. Returning
    that point is correct as well as safe -- it is the minimizer of a term that can only
    grow by moving away from it.
    """
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    np.testing.assert_allclose(pp._geometric_median(pts), [0.0, 0.0], atol=1e-9)
    assert np.isnan(pp._geometric_median(np.full((4, 3), np.nan))).all()


def test_the_method_reaches_the_stage_and_is_recorded(cameras, fly, scene):
    """The config key has to actually drive the arrays *and* land in the metadata."""
    _, pts3d, pts2d = scene
    col = fly.point_names.index(RIGID[0])
    frozen = {}
    for method in pp.STATIC_METHODS:
        _, out3d, _, (report,) = stages.stage_postprocess(
            _config(RIGID, method=method), cameras, fly, pts2d, pts3d
        )
        assert report["method"] == method
        frozen[method] = out3d[0, col]
    # Different estimators on real noisy data give different (but nearby) answers --
    # so the flag is not silently collapsing to one code path.
    assert not np.allclose(frozen["mean"], frozen["mode"])
    for method, value in frozen.items():
        assert np.linalg.norm(value - frozen["median"]) < 0.5, method


def test_the_drift_report_is_measured_against_the_chosen_center(cameras, fly, scene):
    """The report must describe the method the run used, not a median it did not.

    Taken as ``|before - after|``, so this holds by construction -- which is the point:
    the alternative, recomputing a median to compare against, would silently report on
    an estimator the run never applied.
    """
    _, pts3d, pts2d = scene
    cols = [fly.point_names.index(n) for n in RIGID]
    for method in ("mean", "mode", "geometric_median"):
        out2d, _, _, (report,) = stages.stage_postprocess(
            _config(RIGID, method=method), cameras, fly, pts2d, pts3d
        )
        expect = float(
            np.nanmedian(np.linalg.norm(pts2d[:, :, cols] - out2d[:, :, cols], axis=-1))
        )
        assert report["moved_2d_median_px"] == pytest.approx(expect)


# -- the array helpers ---------------------------------------------------------


def test_the_2d_freeze_uses_each_view_s_own_median(rng):
    """Every view keeps its own center: a freeze is not one pixel shared across views."""
    pts2d = rng.normal(size=(3, 20, 4, 2)) + np.arange(3)[:, None, None, None] * 100.0
    out = pp.freeze_2d(pts2d, [1])
    for v in range(3):
        assert np.allclose(out[v, :, 1], np.median(pts2d[v, :, 1], axis=0))
    assert (out[:, :, 1] == out[:, :1, 1]).all()  # constant over time
    np.testing.assert_array_equal(out[:, :, [0, 2, 3]], pts2d[:, :, [0, 2, 3]])


def test_a_view_that_never_sees_the_point_stays_nan():
    """The freeze must not invent an observation in a camera that cannot see it.

    The distinction that matters downstream: deeperfly reads NaN as "not observed", so
    a whole-column fill would hand the next stage a prediction dressed as a
    measurement. Within a view that *does* see the point, filling the frames where the
    detection dropped out is the whole premise of calling the point static, so that
    case is asserted here too rather than left ambiguous.
    """
    pts2d = np.zeros((2, 10, 2, 2))
    pts2d[0, :, 0] = np.nan  # view 0 never sees point 0
    pts2d[1, :, 0] = 7.0
    pts2d[1, 3:6, 0] = np.nan  # view 1 sees it, but drops three frames
    out = pp.freeze_2d(pts2d, [0])
    assert np.isnan(out[0, :, 0]).all()
    assert np.allclose(out[1, :, 0], 7.0)  # including the dropped frames


def test_absent_keypoints_are_not_resurrected(cameras, fly, scene):
    """A limb lost part-way through must stay lost.

    The trap this pins: absence is applied upstream, so the incoming arrays are already
    NaN for the frames the limb was gone -- but the median is finite (taken over the
    frames it was still there) and broadcasting it over *all* frames would put the limb
    back. The re-erase has to happen after the freeze, not before.
    """
    truth, pts3d, pts2d = scene
    col = fly.point_names.index(RIGID[0])
    absent = np.zeros((N_FRAMES, len(fly.point_names)), dtype=bool)
    absent[N_FRAMES // 2 :, col] = True  # amputated half-way through
    pts3d = pts3d.copy()
    pts2d = pts2d.copy()
    pts3d[absent] = np.nan
    pts2d[:, absent] = np.nan

    out2d, out3d, _, _ = stages.stage_postprocess(
        _config(RIGID), cameras, fly, pts2d, pts3d, absent=absent
    )
    assert np.isnan(out3d[N_FRAMES // 2 :, col]).all()
    assert np.isnan(out2d[:, N_FRAMES // 2 :, col]).all()
    assert np.isfinite(out3d[: N_FRAMES // 2, col]).all()  # still frozen where present


# -- the stage -----------------------------------------------------------------


def test_the_freeze_recovers_the_constant_and_spares_the_rest(cameras, fly, scene):
    """The claim in one test: listed points get closer to truth, others are untouched."""
    truth, pts3d, pts2d = scene
    rigid_cols = [fly.point_names.index(n) for n in RIGID]
    moving_col = fly.point_names.index(MOVING)

    out2d, out3d, reproj, (report,) = stages.stage_postprocess(
        _config(RIGID), cameras, fly, pts2d, pts3d
    )
    assert out3d.shape == pts3d.shape
    assert out2d.shape == pts2d.shape
    assert reproj.shape == pts2d.shape[:3]

    # Frozen: identical in every frame, in both spaces.
    assert np.allclose(out3d[:, rigid_cols], out3d[:1, rigid_cols])
    assert np.allclose(out2d[:, :, rigid_cols], out2d[:, :1, rigid_cols])
    # And closer to the truth than the per-frame estimate was -- averaging out the
    # noise is the point, so this is the assertion that would fail if the stage froze
    # to something other than a central value.
    before = np.linalg.norm(pts3d[:, rigid_cols] - truth[:, rigid_cols], axis=-1)
    after = np.linalg.norm(out3d[:, rigid_cols] - truth[:, rigid_cols], axis=-1)
    assert np.nanmedian(after) < np.nanmedian(before)
    # A point that really moves is left exactly alone.
    np.testing.assert_array_equal(out3d[:, moving_col], pts3d[:, moving_col])
    np.testing.assert_array_equal(out2d[:, :, moving_col], pts2d[:, :, moving_col])


def test_the_report_measures_what_was_frozen(cameras, fly, scene):
    """The log line is the only place the premise is checkable, so it is pinned.

    A point drifting a fraction of a pixel really was static; one drifting tens of
    pixels was moving and does not belong in the list. The report has to separate them,
    which means naming the worst offender rather than only reporting an aggregate.
    """
    _, pts3d, pts2d = scene
    points = [*RIGID, MOVING]
    _, _, _, (report,) = stages.stage_postprocess(
        _config(points), cameras, fly, pts2d, pts3d
    )
    assert report["points"] == points
    per_point = report["moved_2d_median_px_per_point"]
    assert set(per_point) == set(points)
    # The genuinely moving point drifted furthest, and is the one named.
    assert per_point[MOVING] == max(per_point.values())
    assert report["worst_point"].startswith(MOVING)
    assert report["moved_2d_p90_px"] >= report["moved_2d_median_px"] > 0.0


def test_an_empty_list_passes_the_pose_through(cameras, fly, scene):
    """Off is a pass-through, not a skip.

    A downstream stage's input must not depend on whether the list happened to be
    filled in, so the stage still writes its group -- with its input unchanged.
    """
    _, pts3d, pts2d = scene
    out2d, out3d, _, (report,) = stages.stage_postprocess(
        _config([]), cameras, fly, pts2d, pts3d
    )
    np.testing.assert_array_equal(out3d, pts3d)
    np.testing.assert_array_equal(out2d, pts2d)
    assert report == {"op": "static", "points": [], "method": "median"}


def test_an_unknown_point_names_its_own_table(cameras, fly, scene):
    """Two tables name a set of held-still points; the error has to say which one."""
    _, pts3d, pts2d = scene
    with pytest.raises(ValueError, match=r'op = "static".*points references'):
        stages.stage_postprocess(_config(["not_a_point"]), cameras, fly, pts2d, pts3d)


def test_the_residual_is_measured_against_the_detections(cameras, fly, scene):
    """``reproj_error`` reads "how far the frozen point sits from the raw detection".

    Measured against the stage's own 2D it would be near-zero by construction for the
    frozen points and say nothing; against the observations it is the number worth
    looking at.
    """
    _, pts3d, pts2d = scene
    _, _, reproj, _ = stages.stage_postprocess(
        _config(RIGID), cameras, fly, pts2d, pts3d, obs2d=pts2d
    )
    assert np.nanmedian(reproj) > 0.0


# -- store, selectors and cache ------------------------------------------------


def _store_with(tmp_path, cameras, fly, pts2d, pts3d, *, stage_names=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = StageStore(tmp_path / "results.h5")
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.full(pts2d.shape[:3], 0.9),
        image_sizes={name: (480, 960) for name in cameras.names},
    )
    for name in stage_names:
        store.write_points(
            name, pts2d=pts2d, pts3d=pts3d, reproj_error=np.zeros(pts2d.shape[:3])
        )
    return store


def test_store_round_trips_the_postprocess_group(tmp_path, cameras, fly, scene):
    """Without a ``_STAGE_MARKER`` entry the stage would recompute forever."""
    _, pts3d, pts2d = scene
    store = _store_with(tmp_path, cameras, fly, pts2d, pts3d)
    assert not store.has("postprocess")
    store.write_points(
        "postprocess",
        pts2d=pts2d,
        pts3d=pts3d,
        reproj_error=np.zeros(pts2d.shape[:3]),
        meta={"algorithm": "temporal_center"},
    )
    assert store.has("postprocess")
    assert store.read_point_meta("postprocess")["algorithm"] == "temporal_center"
    store.truncate_from("postprocess")  # must not raise: it has to be in STAGES
    assert not store.has("postprocess")


def test_the_input_selector_never_resolves_to_the_stage_itself(
    tmp_path, cameras, fly, scene
):
    """The cache-killing trap, pinned.

    If this stage's fingerprint named a selector that could resolve to
    ``postprocess``, the recorded value (``triangulation``, written before any
    frozen output existed) would never match the expected one (``postprocess``,
    once it does), so the stage would recompute on every run forever.
    """
    _, pts3d, pts2d = scene
    on = {name: True for name in Config.default().stage_flags()}
    before = _store_with(
        tmp_path / "a", cameras, fly, pts2d, pts3d, stage_names=("triangulation",)
    )
    after = _store_with(
        tmp_path / "b",
        cameras,
        fly,
        pts2d,
        pts3d,
        stage_names=("triangulation", "postprocess"),
    )
    assert fingerprint.postprocess_source(on, before) == "triangulation"
    assert fingerprint.postprocess_source(on, after) == "triangulation"

    config = _config(RIGID)
    assert fingerprint.stage_fingerprint("postprocess", config, on, before) == (
        fingerprint.stage_fingerprint("postprocess", config, on, after)
    )
    # ... while the downstream selectors *do* prefer it, which is the whole point.
    assert fingerprint.pts3d_source(on, after) == "postprocess"
    assert fingerprint.pose_sources(on, after)["pts2d"] == "postprocess"


def test_the_fingerprint_notices_an_edited_point_list(tmp_path, cameras, fly, scene):
    _, pts3d, pts2d = scene
    store = _store_with(
        tmp_path, cameras, fly, pts2d, pts3d, stage_names=("triangulation",)
    )
    on = {name: True for name in Config.default().stage_flags()}
    base = fingerprint.stage_fingerprint("postprocess", _config(RIGID), on, store)
    diff = fingerprint.fingerprint_diff(
        base,
        fingerprint.stage_fingerprint(
            "postprocess", _config([*RIGID, MOVING]), on, store
        ),
    )
    assert any("ops" in line for line in diff)
    # ... and so does appending an op, which a flat section could not have expressed.
    longer = fingerprint.fingerprint_diff(
        base,
        fingerprint.stage_fingerprint(
            "postprocess",
            _chain(
                {"op": "static", "points": list(RIGID)},
                {"op": "symmetrize", "pairs": [[RIGID[0], RIGID[1]]]},
            ),
            on,
            store,
        ),
    )
    assert any("ops" in line for line in longer)


def test_run_recording_freezes_and_then_reuses_the_cache(
    tmp_path, cameras, fly, scene, caplog
):
    """End to end through the CLI: the stage runs, lands in the file, and caches."""
    from deeperfly import cli

    truth, _, pts2d = scene
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(cameras, fly, pts2d, conf=np.full(pts2d.shape[:3], 0.9)).save(
        outdir / "results.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_postprocess = true\n"
        "do_visualization = false\n"
        '[[postprocess.ops]]\nop = "static"\npoints = '
        + repr(list(RIGID)).replace("'", '"')
        + "\n"
    )
    argv = ["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)]
    cli.main([*argv, "--log-level", "error"])

    store = StageStore(outdir / "results.h5")
    assert store.has("postprocess")
    frozen2d, frozen3d, reproj = store.read_points("postprocess")
    cols = [fly.point_names.index(n) for n in RIGID]
    assert np.allclose(frozen3d[:, cols], frozen3d[:1, cols])
    assert np.allclose(frozen2d[:, :, cols], frozen2d[:, :1, cols])
    assert reproj.shape == pts2d.shape[:3]
    meta = store.read_point_meta("postprocess")
    assert meta["pose_from"] == "triangulation"
    # One report per op, in order -- the metadata mirrors the configured chain.
    assert [o["op"] for o in meta["ops"]] == ["static"]
    assert meta["ops"][0]["method"] == "median"
    assert meta["ops"][0]["points"] == list(RIGID)
    # The most-derived pose is what a reader of the file now gets.
    loaded = PoseResult.load(outdir / "results.h5")
    np.testing.assert_allclose(loaded.pts3d, frozen3d)

    caplog.clear()
    with caplog.at_level("INFO", logger="deeperfly"):
        cli.main([*argv, "--log-level", "info"])
    assert any("reusing cached postprocess" in r.message for r in caplog.records)


# -- op: symmetrize ------------------------------------------------------------


def _mirrored_scene(rng, n_frames=30, tilt=0.0):
    """A synthetic body: 3 exactly-symmetric pairs + a midline point, plus noise.

    ``tilt`` rotates the whole thing out of the world axes, which is the case that
    separates a real plane fit from one that only works when the sagittal plane happens
    to be a coordinate plane.
    """
    left = np.array([[1.0, 0.6, 0.0], [0.0, 0.6, 0.2], [-1.0, 0.6, 0.1]])
    right = left * np.array([1.0, -1.0, 1.0])  # mirrored in y
    mid = np.array([[1.6, 0.0, 0.05]])
    body = np.concatenate([left, right, mid])  # (7, 3): L0 L1 L2 R0 R1 R2 M
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        body = body @ rot.T
    truth = np.tile(body[None], (n_frames, 1, 1))
    return truth, truth + rng.normal(scale=0.01, size=truth.shape)


PAIRS = [(0, 3), (1, 4), (2, 5)]
MID = [6]


def test_symmetrize_recovers_an_exactly_mirrored_body(rng):
    """The floor: noise about a symmetric truth, and the pairs come back symmetric."""
    truth, noisy = _mirrored_scene(rng)
    out, normal, offset = pp.symmetrize_3d(noisy, PAIRS, MID)
    for a, b in PAIRS:
        np.testing.assert_allclose(
            out[:, a], pp._mirror(out[:, b], normal, offset), atol=1e-9
        )
    # And the midline point is ON the plane.
    assert np.abs(out[:, MID[0]] @ normal - offset).max() < 1e-9
    # Closer to the truth than the noisy input was -- the constraint is information.
    before = np.linalg.norm(noisy - truth, axis=-1)[:, [a for a, _ in PAIRS]]
    after = np.linalg.norm(out - truth, axis=-1)[:, [a for a, _ in PAIRS]]
    assert np.nanmedian(after) < np.nanmedian(before)


def test_the_plane_fit_is_not_axis_aligned(rng):
    """A real fit, not one that assumes the sagittal plane is a coordinate plane."""
    _, noisy = _mirrored_scene(rng, tilt=0.7)
    normal, offset = pp.sagittal_plane(noisy, PAIRS)
    assert normal is not None
    # The true normal is the rotated y axis; sign is arbitrary, so compare |cos|.
    truth_n = np.array([-np.sin(0.7), np.cos(0.7), 0.0])
    assert abs(float(normal @ truth_n)) > 0.999
    out, n2, o2 = pp.symmetrize_3d(noisy, PAIRS, MID)
    for a, b in PAIRS:
        np.testing.assert_allclose(out[:, a], pp._mirror(out[:, b], n2, o2), atol=1e-9)


def test_the_plane_is_fitted_from_pairs_only(rng):
    """A midline point about to be moved must not vote on where to move it to.

    Dragging the midline point far off-plane leaves the fitted plane untouched -- which
    is what keeps a laterally-bending abdomen from dragging the plane it is measured
    against.
    """
    _, noisy = _mirrored_scene(rng)
    base_n, base_o = pp.sagittal_plane(noisy, PAIRS)
    moved = noisy.copy()
    moved[:, MID[0]] += np.array([0.0, 5.0, 0.0])  # far off the midline
    got_n, got_o = pp.sagittal_plane(moved, PAIRS)
    np.testing.assert_allclose(np.abs(base_n), np.abs(got_n), atol=1e-12)
    assert abs(base_o - got_o) < 1e-12


def test_symmetrize_moves_both_sides_not_one(rng):
    """Neither side is authoritative, so each moves half-way to the other's mirror.

    Averaging into one side would import that side's error into both, which is the
    failure this asserts against: the two displacements must be equal and opposite in
    the plane-normal sense, not one zero.
    """
    _, noisy = _mirrored_scene(rng)
    noisy = noisy.copy()
    noisy[:, 0] += np.array([0.0, 0.30, 0.0])  # break one pair, one side only
    out, _, _ = pp.symmetrize_3d(noisy, PAIRS, MID)
    moved_l = np.linalg.norm(out[:, 0] - noisy[:, 0], axis=-1).mean()
    moved_r = np.linalg.norm(out[:, 3] - noisy[:, 3], axis=-1).mean()
    assert moved_l > 1e-3 and moved_r > 1e-3
    assert moved_l == pytest.approx(moved_r, rel=0.05)


def test_strength_scales_the_correction(rng):
    _, noisy = _mirrored_scene(rng)
    noisy = noisy.copy()
    noisy[:, 0] += np.array([0.0, 0.30, 0.0])
    full, _, _ = pp.symmetrize_3d(noisy, PAIRS, MID, strength=1.0)
    half, _, _ = pp.symmetrize_3d(noisy, PAIRS, MID, strength=0.5)
    none, _, _ = pp.symmetrize_3d(noisy, PAIRS, MID, strength=0.0)
    np.testing.assert_array_equal(none, noisy)
    d_full = np.linalg.norm(full[:, 0] - noisy[:, 0], axis=-1).mean()
    d_half = np.linalg.norm(half[:, 0] - noisy[:, 0], axis=-1).mean()
    assert d_half == pytest.approx(0.5 * d_full, rel=1e-6)


def test_a_single_plane_leaves_a_static_point_static(rng):
    """Why ``per_frame`` defaults off, and why ``static`` comes first in the chain.

    One fitted plane is a *fixed* map, so it maps a constant to a constant: static then
    symmetrize satisfies both properties exactly. Fitted per frame the plane wobbles with
    the estimate, and symmetrizing after a freeze un-freezes it -- asserted here so the
    default cannot be flipped without someone noticing what it costs.
    """
    _, noisy = _mirrored_scene(rng)
    frozen = pp.freeze_3d(noisy, list(range(noisy.shape[1])))
    once, _, _ = pp.symmetrize_3d(frozen, PAIRS, MID, per_frame=False)
    assert np.allclose(once, once[:1])  # still static
    per_frame, _, _ = pp.symmetrize_3d(noisy, PAIRS, MID, per_frame=True)
    frozen_then_pf, _, _ = pp.symmetrize_3d(frozen, PAIRS, MID, per_frame=True)
    assert np.allclose(frozen_then_pf, frozen_then_pf[:1])  # frozen input: no wobble
    assert not np.allclose(per_frame, per_frame[:1])  # unfrozen input: does wobble


def test_symmetrize_leaves_the_2d_alone(cameras, fly, scene):
    """A per-view 2D detection is a measurement in that camera's pixels.

    There is no sense in which two *different cameras'* pixels mirror each other, so the
    symmetry is imposed where the animal is and the 2D is passed through untouched.
    """
    _, pts3d, pts2d = scene
    cfg = _chain(
        {
            "op": "symmetrize",
            "pairs": [
                ["lf_thorax_coxa", "rf_thorax_coxa"],
                ["lm_thorax_coxa", "rm_thorax_coxa"],
            ],
            "midline": ["neck"] if "neck" in fly.point_names else [],
        }
    )
    out2d, out3d, _, (report,) = stages.stage_postprocess(
        cfg, cameras, fly, pts2d, pts3d
    )
    np.testing.assert_array_equal(out2d, pts2d)
    assert not np.allclose(out3d, pts3d)
    assert report["op"] == "symmetrize" and report["fitted"]


def test_too_few_pairs_is_reported_not_guessed(rng, cameras, fly, scene):
    """One pair fixes a normal but no offset; that is under-determined, not an error."""
    _, pts3d, pts2d = scene
    cfg = _chain({"op": "symmetrize", "pairs": [["lf_thorax_coxa", "rf_thorax_coxa"]]})
    all_nan = np.full_like(pts3d, np.nan)
    out2d, out3d, _, (report,) = stages.stage_postprocess(
        cfg, cameras, fly, pts2d, all_nan
    )
    assert report["fitted"] is False
    np.testing.assert_array_equal(out3d, all_nan)


# -- the chain -----------------------------------------------------------------


def test_the_chain_runs_in_order_and_reports_each_op(cameras, fly, scene):
    """Ops see each other's output, so the reports are an ordered list, not a dict."""
    _, pts3d, pts2d = scene
    cfg = _chain(
        {"op": "static", "points": list(RIGID)},
        {"op": "symmetrize", "pairs": [["lf_thorax_coxa", "rf_thorax_coxa"]]},
    )
    out2d, out3d, _, reports = stages.stage_postprocess(cfg, cameras, fly, pts2d, pts3d)
    assert [r["op"] for r in reports] == ["static", "symmetrize"]
    # Both properties hold at the end: the static points are still static ...
    cols = [fly.point_names.index(n) for n in RIGID]
    assert np.allclose(out3d[:, cols], out3d[:1, cols])
    # ... and the symmetrized pair is symmetric.
    a, b = (
        fly.point_names.index("lf_thorax_coxa"),
        fly.point_names.index("rf_thorax_coxa"),
    )
    normal = np.asarray(reports[1]["plane_normal"])
    offset = float(reports[1]["plane_offset"])
    np.testing.assert_allclose(
        out3d[:, a], pp._mirror(out3d[:, b], normal, offset), atol=1e-9
    )


def test_an_unknown_op_or_key_is_a_hard_error(cameras, fly, scene):
    _, pts3d, pts2d = scene
    with pytest.raises(ValueError, match="unknown op 'stati'"):
        stages.stage_postprocess(
            _chain({"op": "stati", "points": []}), cameras, fly, pts2d, pts3d
        )
    with pytest.raises(ValueError, match="unknown key"):
        stages.stage_postprocess(
            _chain({"op": "static", "pointz": []}), cameras, fly, pts2d, pts3d
        )
    with pytest.raises(ValueError, match="must be a table with an `op` key"):
        stages.stage_postprocess(_chain({"points": []}), cameras, fly, pts2d, pts3d)
