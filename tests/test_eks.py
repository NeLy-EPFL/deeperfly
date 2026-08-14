"""Tests for the ensemble Kalman smoother and its pipeline stage.

The synthetic recording is a slow sinusoid per keypoint seen through the canonical
seven-camera rig, so "the smoother should recover this" has a ground truth to check
against, and "the smoother should not follow that" has a planted outlier to ignore.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.eks import ensemble_statistics, inflate_variances, smooth
from deeperfly.eks.core import project_all
from deeperfly.pipeline import fingerprint
from deeperfly.results import StageStore
from deeperfly.skeleton import Skeleton

# -- fixtures -----------------------------------------------------------------


def _trajectory(rng, n_frames=200, n_points=4, sweep=2.0 * np.pi):
    """A smooth 3D trajectory: each point a sinusoid about its own center.

    ``sweep`` sets the speed, and the speed is what decides whether temporal
    smoothing can help at all. At the default the per-frame step is comparable to
    the noise-limited localization error, which is the regime a tethered fly's
    body and proximal joints live in. Doubling it puts the target well above that
    floor, where there is nothing left for a temporal prior to recover -- see
    :func:`test_smoothing_does_not_hurt_a_fast_target`.
    """
    phase = np.linspace(0.0, sweep, n_frames)[:, None, None]
    center = rng.uniform(-1.0, 1.0, size=(1, n_points, 3))
    amplitude = rng.uniform(0.1, 0.4, size=(1, n_points, 3))
    offset = rng.uniform(0.0, 2.0 * np.pi, size=(1, n_points, 3))
    return center + amplitude * np.sin(phase + offset)


#: Per-observation 2D noise, in pixels -- a plausible detector localization error.
CAMERAS_NOISE = 1.5


@pytest.fixture
def cameras_noise() -> float:
    return CAMERAS_NOISE


@pytest.fixture
def truth3d(rng):
    return _trajectory(rng)


@pytest.fixture
def noisy(cameras, truth3d, rng):
    """``(clean2d, obs2d, conf)`` with 1.5 px noise and no missing data."""
    clean = np.asarray(cameras.project(truth3d))  # (V, T, P, 2)
    obs = clean + rng.normal(scale=CAMERAS_NOISE, size=clean.shape)
    return clean, obs, np.full(obs.shape[:3], 0.9)


def _err(a, b):
    d = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    return np.nanmedian(d), np.nanpercentile(d, 90), np.nanmax(d)


# -- the observation model ----------------------------------------------------


def test_project_all_matches_the_camera_group(cameras, rng):
    """``h(z)`` is the rig's own projection, flattened view-major."""
    pt = rng.normal(scale=1.0, size=3)
    stacked = np.asarray(
        project_all(pt, cameras.rvecs, cameras.tvecs, cameras.intrs, cameras.dists)
    )
    np.testing.assert_allclose(
        stacked.reshape(len(cameras), 2), np.asarray(cameras.project(pt)), atol=1e-10
    )


# -- the smoother -------------------------------------------------------------


def test_smoothing_beats_raw_triangulation(cameras, truth3d, noisy):
    """Per-frame noise is what a temporal prior is for: the 3D error should drop."""
    _, obs, conf = noisy
    raw = _err(cameras.triangulate(obs), truth3d)
    got = _err(smooth(cameras, obs, conf).pts3d, truth3d)
    assert got[0] < raw[0]  # median
    assert got[1] < raw[1]  # p90


def test_a_fast_target_costs_little(cameras, cameras_noise, rng):
    """The honest limit of the temporal half of the method, pinned.

    When the animal moves much further between frames than the detector can
    localize it, a position-only random walk has nothing left to recover. Two
    things then cost accuracy: the prior has no velocity term, so the prediction
    is systematically behind, and the measurement update is a *single*
    Gauss-Newton step, so a prediction that far from the truth is a poor
    linearization point. Both are properties of the published method, not of this
    wiring -- raising the smoothing parameter to its ceiling does not remove them.

    What the fitted per-keypoint parameter does buy is that the loss stays small:
    at four times the default step -- roughly seven times the noise-limited
    localization error -- the median is within about 10% of raw triangulation
    rather than diverging. This is why on a real recording the accuracy gain is
    modest next to the jitter and outlier gains, and it is the number to re-check
    if anyone ever swaps in an iterated update or a constant-velocity latent.
    """
    truth = _trajectory(rng, sweep=8.0 * np.pi)  # 4x the default step
    clean = np.asarray(cameras.project(truth))
    obs = clean + rng.normal(scale=cameras_noise, size=clean.shape)
    conf = np.full(obs.shape[:3], 0.9)
    raw = _err(cameras.triangulate(obs), truth)
    got = _err(smooth(cameras, obs, conf).pts3d, truth)
    assert got[0] <= raw[0] * 1.15


def test_inflation_repairs_a_blown_detection(cameras, truth3d, noisy, rng):
    """One view's gross outliers must not drag the 3D point off the animal.

    This is the mechanism Lightning Pose 3D adds, and the one that works with a
    single detector -- so it is checked against the smoother with it turned off,
    not just against raw triangulation.
    """
    _, obs, conf = noisy
    obs = obs.copy()
    bad = rng.random(obs.shape[:3]) < 0.02
    obs[bad] += rng.normal(scale=150.0, size=(int(bad.sum()), 2))

    without = _err(smooth(cameras, obs, conf, inflate_vars=False).pts3d, truth3d)
    with_ = smooth(cameras, obs, conf, inflate_vars=True)
    assert with_.n_inflated > 0
    assert _err(with_.pts3d, truth3d)[2] < without[2] / 2  # worst case, halved at least


def test_inflation_is_quiet_when_the_variance_is_honest(cameras, noisy):
    """A clean recording whose stated variance matches its noise flags few cells."""
    _, obs, _ = noisy
    honest = np.full(obs.shape[:3], 1.0 / 1.5**2)  # 1/conf == the true variance
    res = smooth(cameras, obs, honest)
    assert res.n_testable > 0
    assert res.n_inflated / res.n_testable < 0.10


def test_missing_views_are_marginalized_exactly(cameras, truth3d, noisy):
    """A NaN view must be *dropped*, not imputed: same answer as a rig without it.

    The masked dimensions get a zeroed Jacobian row, a zeroed residual and unit
    variance, which should make the recursion identical to the one for the observed
    dimensions alone -- to floating-point, not approximately.
    """
    from deeperfly.cameras import CameraGroup

    _, obs, conf = noisy
    keep = cameras.names[:-2]
    smaller = CameraGroup({name: cameras[name] for name in keep})

    blanked = obs.copy()
    blanked[-2:] = np.nan
    init = cameras.triangulate(blanked)

    full = smooth(cameras, blanked, conf, init3d=init, smooth_param=0.5)
    subset = smooth(
        smaller, obs[: len(keep)], conf[: len(keep)], init3d=init, smooth_param=0.5
    )
    np.testing.assert_allclose(full.pts3d, subset.pts3d, atol=1e-8)


def test_a_single_view_keypoint_is_nan_not_invented(cameras, noisy):
    """One view can never determine a 3D point, so the smoother must not report one.

    The filter would otherwise happily coast the prior along that view's ray and
    emit a finite, plausible-looking trajectory -- a prediction dressed as a
    measurement, which is exactly what the NaN convention exists to prevent.
    """
    _, obs, conf = noisy
    obs = obs.copy()
    obs[1:, :, 0] = np.nan  # only camera 0 ever sees keypoint 0
    res = smooth(cameras, obs, conf)
    assert np.isnan(res.pts3d[:, 0]).all()
    assert np.isfinite(res.pts3d[:, 1:]).all()


def test_a_keypoint_determined_only_briefly_is_kept(cameras, noisy):
    """Determined in *some* frames is enough -- interpolating the rest is the point."""
    _, obs, conf = noisy
    obs = obs.copy()
    obs[1:, 20:, 0] = np.nan  # two views for 20 frames, then one view only
    res = smooth(cameras, obs, conf)
    assert np.isfinite(res.pts3d[:, 0]).all()
    # ...and the smoother says so: uncertainty grows once the second view drops out.
    assert np.mean(res.posterior_var[150:, 0]) > np.mean(res.posterior_var[:20, 0])


def test_a_keypoint_no_view_ever_saw_is_nan(cameras, noisy):
    _, obs, conf = noisy
    obs = obs.copy()
    obs[:, :, 1] = np.nan  # keypoint 1 is not on this animal
    res = smooth(cameras, obs, conf)
    assert np.isnan(res.pts3d[:, 1]).all()
    assert np.isnan(res.posterior_var[:, 1]).all()
    assert np.isnan(res.smooth_param[1])
    assert np.isfinite(res.pts3d[:, 0]).all()  # its neighbors are unaffected


def test_unobserved_cells_stay_nan_unless_asked_to_fill(cameras, noisy):
    """NaN in deeperfly's 2D means "not observed"; filling it is opt-in."""
    _, obs, conf = noisy
    obs = obs.copy()
    obs[0, :, 0] = np.nan  # camera 0 never sees point 0
    kept = smooth(cameras, obs, conf)
    assert np.isnan(kept.pts2d[0, :, 0]).all()
    filled = smooth(cameras, obs, conf, fill_unobserved=True)
    assert np.isfinite(filled.pts2d).all()
    # Filling changes only which cells are reported, never the 3D behind them.
    np.testing.assert_allclose(kept.pts3d, filled.pts3d, atol=1e-12)


def test_reported_2d_is_the_reprojection_of_the_3d(cameras, noisy):
    _, obs, conf = noisy
    res = smooth(cameras, obs, conf)
    np.testing.assert_allclose(
        res.pts2d, np.asarray(cameras.project(res.pts3d)), atol=1e-9
    )


def test_a_smaller_smooth_param_smooths_harder(cameras, noisy):
    """The knob has to point the documented way round."""
    _, obs, conf = noisy
    step = {
        s: np.nanmedian(
            np.linalg.norm(
                np.diff(smooth(cameras, obs, conf, smooth_param=s).pts3d, axis=0),
                axis=-1,
            )
        )
        for s in (1e-3, 1e3)
    }
    assert step[1e-3] < step[1e3]


def test_fitted_smooth_params_are_positive_and_per_keypoint(cameras, noisy):
    _, obs, conf = noisy
    res = smooth(cameras, obs, conf)
    assert res.smooth_param.shape == (obs.shape[2],)
    assert np.all(res.smooth_param > 0)


def test_posterior_variance_grows_where_the_views_go_blind(cameras, noisy):
    """The uncertainty output has to be an uncertainty, not decoration."""
    _, obs, conf = noisy
    obs = obs.copy()
    obs[:, 80:120, 0] = np.nan  # point 0 unobserved for a stretch
    res = smooth(cameras, obs, conf)
    blind = np.nanmean(res.posterior_var[90:110, 0])
    seen = np.nanmean(res.posterior_var[:60, 0])
    assert blind > seen


def test_an_ensemble_of_independent_members_beats_one(cameras, truth3d, rng):
    """M > 1 turns the observation noise into a measurement instead of a prior."""
    clean = np.asarray(cameras.project(truth3d))
    members, confs = [], []
    for _ in range(3):
        obs = clean + rng.normal(scale=1.5, size=clean.shape)
        bad = rng.random(clean.shape[:3]) < 0.02  # independent blunders per model
        obs[bad] += rng.normal(scale=150.0, size=(int(bad.sum()), 2))
        members.append(obs)
        confs.append(np.full(obs.shape[:3], 0.9))
    one = _err(smooth(cameras, members[0], confs[0]).pts3d, truth3d)
    many = _err(smooth(cameras, np.stack(members), np.stack(confs)).pts3d, truth3d)
    assert many[0] < one[0]
    assert many[1] < one[1]


def test_shape_and_option_errors(cameras, noisy):
    _, obs, conf = noisy
    with pytest.raises(ValueError, match="must be"):
        smooth(cameras, obs[..., 0], conf)
    with pytest.raises(ValueError, match="views but the rig"):
        smooth(cameras, obs[:-1], conf[:-1])
    with pytest.raises(ValueError, match="at least two frames"):
        smooth(cameras, obs[:, :1], conf[:, :1])
    with pytest.raises(ValueError, match="conf must be"):
        smooth(cameras, obs, conf[:, :5])
    with pytest.raises(ValueError, match="init3d must be"):
        smooth(cameras, obs, conf, init3d=np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="smooth_param must be positive"):
        smooth(cameras, obs, conf, smooth_param=-1.0)
    with pytest.raises(ValueError, match="avg_mode"):
        smooth(cameras, obs, conf, avg_mode="mode")
    with pytest.raises(ValueError, match="var_mode"):
        smooth(cameras, obs, conf, var_mode="stdev")


# -- ensembling and inflation, on their own ------------------------------------


def test_single_member_variance_is_inverse_confidence(cameras, noisy):
    _, obs, conf = noisy
    _, var = ensemble_statistics(obs[None], conf[None])
    np.testing.assert_allclose(var, np.repeat((1.0 / conf)[..., None], 2, axis=-1))


def test_median_center_ignores_one_member_blowing_up(cameras, noisy):
    _, obs, _ = noisy
    members = np.stack([obs, obs, obs.copy()])
    members[2] += 500.0
    center, _ = ensemble_statistics(members, None, avg_mode="median")
    np.testing.assert_allclose(center, obs, atol=1e-9)


def test_inflation_needs_two_views(cameras, noisy):
    """With fewer than two observed views there is no consensus to disagree with."""
    _, obs, conf = noisy
    obs = obs.copy()
    obs[2:] = np.nan  # a single remaining view... plus one
    obs[1] = np.nan
    center, var = ensemble_statistics(obs[None], conf[None])
    mask = np.isfinite(center).all(axis=-1)
    out, n_inflated = inflate_variances(
        center, var, mask, cameras.triangulate(obs), cameras
    )
    assert n_inflated == 0
    np.testing.assert_allclose(out, var)


def test_inflation_skips_cells_with_no_finite_3d(cameras, noisy):
    """A non-finite linearization point is skipped, not turned into NaN variance."""
    _, obs, conf = noisy
    center, var = ensemble_statistics(obs[None], conf[None])
    mask = np.isfinite(center).all(axis=-1)
    init = np.asarray(cameras.triangulate(obs)).copy()
    init[10:20] = np.nan
    out, _ = inflate_variances(center, var, mask, init, cameras)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 10:20], var[:, 10:20])


# -- pipeline wiring -----------------------------------------------------------


def _store_with(tmp_path, cameras, pts2d, conf, *, stages=()):
    """A results.h5 holding a pose2d group, plus the named points stages."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = StageStore(tmp_path / "results.h5")
    store.write_pose2d(
        cameras=cameras,
        skeleton=Skeleton.fly(),
        pts2d=pts2d,
        conf=conf,
        image_sizes={name: (480, 960) for name in cameras.names},
    )
    for stage in stages:
        store.write_points(
            stage,
            pts2d=pts2d,
            pts3d=np.zeros(pts2d.shape[1:3] + (3,)),
            reproj_error=np.zeros(pts2d.shape[:3]),
        )
    return store


def test_store_round_trips_the_eks_group(tmp_path, cameras, noisy):
    """Without a _STAGE_MARKER entry the stage would recompute forever."""
    _, obs, conf = noisy
    store = _store_with(tmp_path, cameras, obs, conf)
    assert not store.has("eks")
    store.write_points(
        "eks",
        pts2d=obs,
        pts3d=np.zeros(obs.shape[1:3] + (3,)),
        reproj_error=np.zeros(obs.shape[:3]),
        extra={"posterior_var": np.ones(obs.shape[1:3] + (3,))},
        meta={"method": "eks_multiview_nonlinear"},
    )
    assert store.has("eks")
    assert store.read_points("eks")[1].shape == obs.shape[1:3] + (3,)
    assert store.read_point_extra("eks", "posterior_var").shape == obs.shape[1:3] + (3,)
    assert store.read_point_meta("eks")["method"] == "eks_multiview_nonlinear"
    store.truncate_from("eks")  # must not raise: eks has to be a member of STAGES
    assert not store.has("eks")


def test_downstream_prefers_the_smoothed_pose(tmp_path, cameras, noisy):
    _, obs, conf = noisy
    store = _store_with(tmp_path, cameras, obs, conf, stages=("triangulation", "eks"))
    on = {name: True for name in Config.default().stage_flags()}
    assert fingerprint.pts3d_source(on, store) == "eks"
    assert fingerprint.pose_sources(on, store) == {"pts2d": "eks", "pts3d": "eks"}
    off = {**on, "eks": False}
    assert fingerprint.pts3d_source(off, store) == "triangulation"


def test_the_smoother_never_seeds_itself(tmp_path, cameras, noisy):
    """The fingerprint's init source must not name the stage's own output.

    If it could, the value would be "triangulation" on the first run and "eks" on
    the second, so the recorded fingerprint would never match the expected one and
    the stage would recompute on every run.
    """
    _, obs, conf = noisy
    on = {name: True for name in Config.default().stage_flags()}
    before = _store_with(tmp_path / "a", cameras, obs, conf, stages=("triangulation",))
    after = _store_with(
        tmp_path / "b", cameras, obs, conf, stages=("triangulation", "eks")
    )
    assert fingerprint.eks_init_source(on, before) == "triangulation"
    assert fingerprint.eks_init_source(on, after) == "triangulation"

    config = Config.default()
    assert fingerprint.stage_fingerprint("eks", config, on, before) == (
        fingerprint.stage_fingerprint("eks", config, on, after)
    )


def test_fingerprint_notices_an_eks_config_change(tmp_path, cameras, noisy):
    _, obs, conf = noisy
    store = _store_with(tmp_path, cameras, obs, conf)
    on = {name: True for name in Config.default().stage_flags()}
    base = fingerprint.stage_fingerprint("eks", Config.default(), on, store)
    changed = Config.from_dict(
        {**Config.default().data, "eks": {"inflate_threshold": 2.0}}
    )
    diff = fingerprint.fingerprint_diff(
        base, fingerprint.stage_fingerprint("eks", changed, on, store)
    )
    assert any("inflate_threshold" in line for line in diff)


def test_stage_eks_end_to_end(cameras, truth3d, noisy, rng):
    """The stage wrapper: config in, arrays out, absent keypoints erased."""
    from deeperfly.pipeline import stages

    _, obs, conf = noisy
    obs = obs.copy()
    bad = rng.random(obs.shape[:3]) < 0.02
    obs[bad] += rng.normal(scale=150.0, size=(int(bad.sum()), 2))
    config = Config.from_dict({"eks": {"fit_frames": 50}})
    absent = np.zeros(obs.shape[2], dtype=bool)
    absent[2] = True  # keypoint 2 is not on this animal

    pts2d, pts3d, reproj, result = stages.stage_eks(
        config,
        cameras,
        obs,
        conf,
        init3d=np.asarray(cameras.triangulate(obs)),
        absent=absent,
    )
    assert pts2d.shape == obs.shape
    assert pts3d.shape == (obs.shape[1], obs.shape[2], 3)
    assert reproj.shape == obs.shape[:3]
    assert np.isnan(pts3d[:, 2]).all()  # the declared-absent keypoint
    assert result.n_inflated > 0
    # The residual is measured against the observations, so it is not identically
    # zero the way it would be against the stage's own reprojected 2D.
    assert np.nanmedian(reproj) > 0.0


def test_run_recording_caches_the_eks_stage(result, tmp_path, caplog):
    """A second run must *reuse* the smoother's output, not recompute it forever.

    The trap: if the stage's fingerprint named an input selector that could
    resolve to ``eks`` itself, the recorded value (``triangulation``, written
    before any smoothed output existed) would never match the expected one
    (``eks``, once it does), so the stage would recompute on every run and the
    cache would be dead weight. That failure is invisible except in a log line,
    which is why it is asserted here rather than left to inspection.
    """
    from deeperfly import cli
    from deeperfly.results import PoseResult

    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "results.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_eks = true\ndo_visualization = false\n"
        "[eks]\nfit_frames = 4\nfit_iterations = 4\n"
    )
    argv = ["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)]
    cli.main([*argv, "--log-level", "error"])

    store = StageStore(outdir / "results.h5")
    assert store.has("eks")
    pts2d, pts3d, reproj = store.read_points("eks")
    assert pts3d.shape == (result.pts2d.shape[1], result.pts2d.shape[2], 3)
    assert pts2d.shape == result.pts2d.shape
    assert reproj.shape == result.pts2d.shape[:3]
    assert store.read_point_extra("eks", "posterior_var") is not None
    assert store.read_point_meta("eks")["method"] == "eks_multiview_nonlinear"
    # The most-derived pose is what a reader of the file now gets.
    assert PoseResult.load(outdir / "results.h5").pts3d is not None

    caplog.clear()
    with caplog.at_level("INFO", logger="deeperfly"):
        cli.main([*argv, "--log-level", "info"])
    assert any("reusing cached eks" in r.message for r in caplog.records)


def test_stage_eks_rejects_a_mismatched_ensemble_member(cameras, noisy):
    from deeperfly.pipeline import stages

    _, obs, conf = noisy
    with pytest.raises(ValueError, match="ensemble member 0"):
        stages.stage_eks(
            Config.default(),
            cameras,
            obs,
            conf,
            members=[(obs[:, :10], conf[:, :10])],
        )
