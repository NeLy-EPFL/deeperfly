"""Tests for left/right swap detection.

A swapped symmetry pair is invisible to every metric that looks at a point cloud -- both
pixels sit on a real joint and only the identity is wrong -- so the tests here are about
two things the module claims and one it explicitly does not:

* it finds a planted swap, and it does so through the pair displacements rather than a
  midline (fly38 has no unpaired points to fit one from);
* it says "undecided" instead of "clean" whenever it cannot judge;
* it cannot catch a *wholesale* flip, and that limit is asserted rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import AZIMUTHS_DEG, CAMERA_NAMES, DISTANCE_MM, FOCAL_PX, HEIGHT, WIDTH

from deeperfly import chirality
from deeperfly.cameras import CameraGroup
from deeperfly.geometry import rmat_to_rvec


@pytest.fixture
def rig_cameras():
    """The reference 7-camera orbit rig (see ``conftest``'s ``rig``)."""
    from conftest import reference_rmat

    rmats = np.array([reference_rmat(t) for t in np.deg2rad(AZIMUTHS_DEG)])
    return CameraGroup.from_arrays(
        CAMERA_NAMES,
        np.asarray(rmat_to_rvec(rmats)),
        np.array([[0.0, 0.0, DISTANCE_MM]] * 7),
        np.tile([FOCAL_PX, FOCAL_PX, (WIDTH - 1) / 2, (HEIGHT - 1) / 2], (7, 1)),
        np.zeros((7, 0)),
    )


def _asymmetric_fly(rng):
    """A plausible 38-point pose with the two sides posed **independently**.

    Not a mirrored pair of halves: a walking fly's left and right legs are in different
    places, and a mirror-symmetric synthetic pose makes this test far easier than reality
    (it hides exactly the projection crossings that make per-view 2D unsound).
    """
    left = np.stack(
        [
            rng.uniform(-1.2, 1.2, 19),
            rng.uniform(0.35, 1.1, 19),  # left side: +y
            rng.uniform(-0.6, 0.6, 19),
        ],
        axis=1,
    )
    right = np.stack(
        [
            rng.uniform(-1.2, 1.2, 19),
            -rng.uniform(0.35, 1.1, 19),  # right side: -y
            rng.uniform(-0.6, 0.6, 19),
        ],
        axis=1,
    )
    return np.concatenate([left, right])


def _mirrored_fly(rng):
    """A pose whose right side is the exact mirror of its left -- the degenerate case.

    Only used to reach the collapsed-axis branch: in a camera looking down the left-right
    axis, an exactly-symmetric pose puts a pair's two points on the same pixel.
    """
    half = np.stack(
        [
            rng.uniform(-1.2, 1.2, 19),
            rng.uniform(0.35, 1.1, 19),
            rng.uniform(-0.6, 0.6, 19),
        ],
        axis=1,
    )
    return np.concatenate([half, half * [1, -1, 1]])


def _swap(points, pairs, rows):
    out = np.array(points, dtype=float, copy=True)
    for i, j in np.asarray(pairs)[rows]:
        out[[i, j]] = out[[j, i]]
    return out


# -- the sound case: 3D -------------------------------------------------------


def test_a_clean_3d_pose_is_decided_and_quiet(fly):
    rng = np.random.default_rng(0)
    flags = 0
    for _ in range(200):
        v = chirality.check(_asymmetric_fly(rng), fly.symmetries)
        assert v.decided, v.reason
        flags += len(v.swapped)
    # Measured at 1 in 500 poses; 200 poses must stay in single digits or the false-positive
    # rate has regressed into "an operator learns to ignore this".
    assert flags <= 3, f"{flags} false swap flags over 200 clean poses"


@pytest.mark.parametrize("n_swaps", [1, 2, 3, 4])
def test_planted_swaps_are_recovered_exactly(fly, n_swaps):
    rng = np.random.default_rng(1)
    hits = 0
    trials = 100
    for _ in range(trials):
        pose = _asymmetric_fly(rng)
        rows = rng.choice(fly.n_symmetries, n_swaps, replace=False)
        v = chirality.check(_swap(pose, fly.symmetries, rows), fly.symmetries)
        want = {tuple(int(x) for x in p) for p in np.asarray(fly.symmetries)[rows]}
        if v.decided and {c.points for c in v.swapped} == want:
            hits += 1
    assert hits / trials >= 0.95, f"recovered {hits}/{trials} exactly"


def test_candidates_are_ranked_worst_first(fly):
    rng = np.random.default_rng(2)
    pose = _asymmetric_fly(rng)
    v = chirality.check(_swap(pose, fly.symmetries, [0, 5, 11]), fly.symmetries)
    margins = [c.relative_margin for c in v.swapped]
    assert margins == sorted(margins, reverse=True)
    assert all(c.margin > 0 for c in v.swapped)


def test_truthiness_reads_as_intended(fly):
    rng = np.random.default_rng(3)
    pose = _asymmetric_fly(rng)
    assert not chirality.check(pose, fly.symmetries)
    assert chirality.check(_swap(pose, fly.symmetries, [7]), fly.symmetries)


# -- the axis fit -------------------------------------------------------------


def test_the_axis_is_recovered_from_partly_swapped_data(fly):
    """The premise of the whole method: the second-moment fit is sign-blind.

    If the axis needed correct pairs to be found, it could not then be used to find the
    incorrect ones. So swapping any subset of pairs must leave the axis unchanged up to
    sign.
    """
    rng = np.random.default_rng(4)
    pose = _asymmetric_fly(rng)
    pairs = np.asarray(fly.symmetries)
    base, _ = chirality.sagittal_axis(pose[pairs[:, 1]] - pose[pairs[:, 0]])
    for rows in ([0], [0, 1, 2], list(range(10)), list(range(19))):
        swapped = _swap(pose, pairs, rows)
        axis, _ = chirality.sagittal_axis(swapped[pairs[:, 1]] - swapped[pairs[:, 0]])
        assert abs(abs(float(base @ axis)) - 1.0) < 1e-6, rows


def test_the_reported_axis_points_at_the_majority_side(fly):
    """The rig puts the left side at +y, and most pairs are (left, right) -> -y."""
    rng = np.random.default_rng(5)
    v = chirality.check(_asymmetric_fly(rng), fly.symmetries)
    assert v.axis is not None
    assert abs(float(v.axis @ [0, 1, 0])) > 0.99  # essentially the y axis


# -- refusing to judge --------------------------------------------------------


def test_no_declared_pairs_is_undecided_not_clean(fly):
    rng = np.random.default_rng(6)
    v = chirality.check(_asymmetric_fly(rng), np.empty((0, 2), np.int64))
    assert not v.decided and not v.swapped
    assert "no symmetry pairs" in v.reason


def test_too_few_co_visible_pairs_is_undecided(fly):
    """Unobserved points are ``NaN``; a pair with either member missing cannot vote."""
    rng = np.random.default_rng(7)
    pose = _asymmetric_fly(rng)
    pose[2:] = np.nan  # leave one pair's worth of points at most
    v = chirality.check(pose, fly.symmetries)
    assert not v.decided
    assert "co-visible" in v.reason


def test_a_collapsed_left_right_axis_is_refused(fly, rig_cameras):
    """A camera looking down the left-right axis sees a symmetric pose's pairs coincide.

    That view must decline to judge rather than reporting the order of two points that are
    1.5 px apart in a 460 px animal.
    """
    rng = np.random.default_rng(8)
    pts2d = np.asarray(rig_cameras.project(_mirrored_fly(rng)[None]))[:, 0]
    lateral = [CAMERA_NAMES.index("rm"), CAMERA_NAMES.index("lm")]
    verdicts = chirality.check_views(pts2d, fly.symmetries)
    for v in (verdicts[i] for i in lateral):
        assert not v.decided
        assert "collapsed" in v.reason
        assert v.separation_frac < chirality.MIN_SEPARATION_FRAC
    # The front camera looks ALONG the midline, so it resolves the axis best of all.
    front = verdicts[CAMERA_NAMES.index("f")]
    assert front.decided and front.separation_frac > 0.4


def test_an_undecided_verdict_never_carries_candidates(fly):
    """``decided=False`` with a non-empty ``swapped`` would be read as a real finding."""
    for pairs in (np.empty((0, 2), np.int64), fly.symmetries):
        pose = np.full((38, 3), np.nan)
        v = chirality.check(pose, pairs)
        assert not v.decided and v.swapped == ()


# -- the limit ----------------------------------------------------------------


def test_a_wholesale_flip_is_self_consistent_and_reported_clean(fly):
    """Swapping EVERY pair is undetectable here, and that is correct, not a bug.

    With no unpaired landmark and no outside reference, nothing inside the sample says which
    side is which -- the mirrored labeling is a valid labeling of a mirrored animal. Pinning
    it stops a future "improvement" from claiming an ability the geometry does not grant.
    """
    rng = np.random.default_rng(9)
    pose = _asymmetric_fly(rng)
    flipped = _swap(pose, fly.symmetries, list(range(fly.n_symmetries)))
    v = chirality.check(flipped, fly.symmetries)
    assert v.decided
    assert v.swapped == ()


def test_a_majority_swap_is_reported_as_the_minority_being_wrong(fly):
    """The vote is a majority, so swapping most pairs inverts what "wrong side" means.

    Worth pinning because it is the honest consequence of having no external reference: the
    check reports disagreement with the rest of the sample, not with a global convention.
    """
    rng = np.random.default_rng(10)
    pose = _asymmetric_fly(rng)
    swapped_rows = list(range(15))  # 15 of 19
    v = chirality.check(_swap(pose, fly.symmetries, swapped_rows), fly.symmetries)
    assert v.decided
    flagged = {c.points for c in v.swapped}
    untouched = {tuple(int(x) for x in p) for p in np.asarray(fly.symmetries)[15:]}
    assert flagged == untouched


# -- 2D, the documented diagnostic --------------------------------------------


def test_per_view_2d_is_offered_but_noisy_on_side_views(fly, rig_cameras):
    """The measurement behind the module's "run this on 3D" instruction.

    The front camera is sound; the oblique side cameras flag real posture crossings. This
    asserts the *shape* of that result so the docstring's claim stays true -- if a change
    ever made side views clean, the guidance should change with it.
    """
    rng = np.random.default_rng(11)
    front_flags = side_flags = 0
    poses = 60
    for _ in range(poses):
        pts2d = np.asarray(rig_cameras.project(_asymmetric_fly(rng)[None]))[:, 0]
        verdicts = chirality.check_views(pts2d, fly.symmetries)
        front_flags += len(verdicts[CAMERA_NAMES.index("f")].swapped)
        for name in ("rh", "rf", "lf", "lh"):
            side_flags += len(verdicts[CAMERA_NAMES.index(name)].swapped)
    assert front_flags == 0, "the front view should be as clean as 3D"
    assert side_flags > poses, (
        "side views are expected to be noisy -- see the docstring"
    )


def test_check_works_the_same_on_2d(fly, rig_cameras):
    """A planted swap is found in the front view too: the function is dimension-agnostic."""
    rng = np.random.default_rng(12)
    pose = _asymmetric_fly(rng)
    pts2d = np.asarray(rig_cameras.project(pose[None]))[:, 0]
    f = CAMERA_NAMES.index("f")
    v = chirality.check(_swap(pts2d[f], fly.symmetries, [3]), fly.symmetries)
    assert v.decided
    assert {c.points for c in v.swapped} == {
        tuple(int(x) for x in np.asarray(fly.symmetries)[3])
    }


def test_the_gate_is_scale_free(fly, rig_cameras):
    """Scaling the whole sample must not change the verdict -- the gate is a ratio.

    Otherwise the same frame would judge differently at 1024 px and at 480 px, and world
    millimetres would need their own threshold.
    """
    rng = np.random.default_rng(13)
    pose = _asymmetric_fly(rng)
    swapped = _swap(pose, fly.symmetries, [2, 8])
    base = chirality.check(swapped, fly.symmetries)
    for scale in (0.01, 100.0):
        other = chirality.check(swapped * scale, fly.symmetries)
        assert other.decided == base.decided
        assert {c.points for c in other.swapped} == {c.points for c in base.swapped}
        assert other.separation_frac == pytest.approx(base.separation_frac)
