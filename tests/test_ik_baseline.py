"""The inverse-kinematics numerical baseline.

``data/ik_baseline_scipy.npz`` records what the pre-QuickIK solver (scipy
``least_squares`` + JAX Jacobians) produced for two inputs: the example recording's
triangulated pose, and a fixed-seed synthetic pose built by forward kinematics from
known angles. It exists so the QuickIK migration can be judged numerically rather
than by eye.

Both *input* poses are stored in the fixture alongside the outputs, because
``examples/data/**/results.h5`` is git-ignored (``*.h5``) -- a fixture that read the
pose from there would only work on the machine that generated it.

The fixture was validated while both solvers coexisted, by asserting the old one
reproduced it bit-for-bit; that check went with the old solver. What remains is the
comparison that matters going forward: QuickIK against the recorded numbers, at
tolerances chosen from measurement and stated in each test.

The headline: on the example recording the fitted model lands a mean 0.049 world units
from the triangulated keypoints where the old solver managed 0.037. The gap is not a
convergence failure -- it is entirely in the hind legs, which press against joint limits
the packaged template only ever *guessed* (it gives the middle and hind legs the front
leg's ranges). Widen the pressed limits and QuickIK reaches 0.032, i.e. better than the
solver it replaced. See ``test_widening_the_pressed_limits_beats_the_old_solver``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import IK_BASELINE_PATH

from deeperfly.config import Config
from deeperfly.pipeline import stages
from deeperfly.skeleton import Skeleton

#: Leg DOFs come first in the angle columns: 6 legs x (ThC 3 + CTr 2 + FTi 1 + TiTa 1).
N_LEG_DOFS = 42


@pytest.fixture(scope="module")
def baseline() -> dict:
    """The recorded fixture, with its two JSON-encoded scalars decoded."""
    z = np.load(IK_BASELINE_PATH, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    for case in ("real", "synth"):
        out[f"{case}_chain_scales"] = json.loads(str(out[f"{case}_chain_scales"]))
        out[f"{case}_body_scale"] = float(out[f"{case}_body_scale"])
    out["synth_truth"] = {
        k: np.asarray(v) for k, v in json.loads(str(out["synth_truth"])).items()
    }
    out["real_angle_names"] = [str(n) for n in out["real_angle_names"]]
    out["synth_angle_names"] = [str(n) for n in out["synth_angle_names"]]
    return out


def solve_with_defaults(pts3d: np.ndarray, overrides: dict | None = None):
    """Solve ``pts3d`` exactly as a default `deeperfly run` would.

    Goes through the stage rather than the library entry point so the comparison covers
    the ``constant_points`` pin and the config plumbing too.
    """
    cfg = Config.from_dict({"inverse_kinematics": dict(overrides or {})})
    return stages.stage_inverse_kinematics(cfg, Skeleton.fly(), pts3d)


def keypoint_residual(model_pts3d: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """``(T, P)`` distance from each fitted model joint to its measured keypoint.

    The metric that decides whether the overlay sits on the fly, which is what the fit
    is *for* -- unlike a per-DOF angle difference, which a redundant chain can change
    freely without moving a single keypoint.
    """
    return np.linalg.norm(model_pts3d - pts3d, axis=-1)


def test_baseline_fixture_is_self_describing(baseline):
    """The fixture holds both cases with matching shapes and the synthetic truth."""
    fly = Skeleton.fly()
    assert baseline["real_angles"].shape == (64, 50)
    assert baseline["synth_angles"].shape == (3, 50)
    for case, n_frames in (("real", 64), ("synth", 3)):
        names = baseline[f"{case}_angle_names"]
        assert len(names) == baseline[f"{case}_angles"].shape[1]
        assert baseline[f"{case}_model_pts3d"].shape == (n_frames, fly.n_points, 3)
        assert baseline[f"{case}_pts3d"].shape == (n_frames, fly.n_points, 3)
    assert set(baseline["synth_truth"]) == {"lf", "lm", "lh", "rf", "rm", "rh"}
    # The legs are fully solved in both cases; the head/abdomen chains only in "real"
    # (the synthetic pose fills leg points only, so their markers are NaN).
    real_full = np.isfinite(baseline["real_angles"]).all(axis=0)
    synth_full = np.isfinite(baseline["synth_angles"]).all(axis=0)
    assert real_full.all()
    assert synth_full[:N_LEG_DOFS].all() and not synth_full[N_LEG_DOFS:].any()


def test_synthetic_baseline_records_the_generating_angles(baseline):
    """The synthetic case's recorded leg angles are the angles that generated it.

    Solver-independent: it pins what the fixture *means*, so QuickIK is held to ground
    truth rather than to the old solver's opinion.
    """
    template = Config.from_dict({}).ik_template()
    col = {n: i for i, n in enumerate(baseline["synth_angle_names"])}
    for leg in template.legs:
        truth = baseline["synth_truth"][leg.name]
        got = np.array([baseline["synth_angles"][0, col[n]] for n in leg.dof_names])
        np.testing.assert_allclose(got, truth, atol=1e-5)


def test_quickik_reproduces_the_recorded_registration(baseline):
    """The measured quantities are unchanged by the solver swap.

    The chain sizes and the body scale come from measurement -- the coxa registration and
    the chains' contour lengths -- not from the fit, so they must match the old solver's
    to within the float32 rounding in the two baked copies of the neutral coxae.
    """
    pytest.importorskip("quickik", reason="needs the deeperfly[ik] extra")
    res = solve_with_defaults(baseline["real_pts3d"])
    assert res.angle_names == baseline["real_angle_names"]
    assert res.chain_scales == pytest.approx(baseline["real_chain_scales"], abs=1e-6)
    assert res.body_scale == pytest.approx(baseline["real_body_scale"], abs=1e-6)


def test_quickik_fit_is_close_to_the_old_solver(baseline):
    """The QuickIK fit lands near the old one, and the difference is bound-limited.

    Not equality: this is a different solver on a differently-posed problem (one
    whole-body fit instead of eight independent ones). The tolerances are measured, and
    the point of the test is to catch a *regression* against them, not to bless the
    numbers as ideal.
    """
    pytest.importorskip("quickik", reason="needs the deeperfly[ik] extra")
    pts3d = baseline["real_pts3d"]
    res = solve_with_defaults(pts3d)
    got = keypoint_residual(res.model_pts3d, pts3d)
    was = keypoint_residual(baseline["real_model_pts3d"], pts3d)
    assert np.nanmean(was) < 0.04  # the recorded fixture, for context
    assert np.nanmean(got) < 0.06
    assert np.nanpercentile(got, 95) < 0.25
    assert np.isfinite(res.angles).all()  # every track solved on this recording


def test_widening_the_pressed_limits_beats_the_old_solver(baseline):
    """With the guessed hind/middle-leg limits widened, QuickIK fits *better*.

    This is what localises the gap above. QuickIK enforces limits by clamping the step,
    so a binding limit costs more accuracy than it did under the old trust-region solver
    -- and the packaged template's middle and hind legs carry the **front** leg's ranges
    as an admitted placeholder, which the hind legs press against on real data. Relax
    only those and the fit overtakes the solver it replaced, which says the residual is
    the bounds' doing rather than the solver's.
    """
    pytest.importorskip("quickik", reason="needs the deeperfly[ik] extra")
    pts3d = baseline["real_pts3d"]
    template = Config.from_dict({}).ik_template()
    wide = {
        name: [-180.0, 180.0]
        for leg in template.legs
        for name in leg.dof_names
        if leg.name[1] in "mh"  # middle and hind: the placeholder ranges
    }
    res = solve_with_defaults(pts3d, {"bounds": wide})
    got = keypoint_residual(res.model_pts3d, pts3d)
    was = keypoint_residual(baseline["real_model_pts3d"], pts3d)
    assert np.nanmean(got) < np.nanmean(was)
