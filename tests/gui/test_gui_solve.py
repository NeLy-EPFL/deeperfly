"""Tests for the per-point 3D solve (labels + predictions -> 3D)."""

from __future__ import annotations

import numpy as np

from deeperfly.config import AnnotationParams, TriangulationParams
from deeperfly.gui.solve import (
    _dlt,
    solve_depth_on_ray,
    solve_point_3d,
    solve_point_3d_drag,
    solve_point_3d_stabilized,
)

FRAME, POINT = 0, 5


def _obs(result, views):
    """Exact projected pixels ``(V, 2)`` for the test point, NaN outside ``views``."""
    full = result.pts2d[:, FRAME, POINT].astype(float)  # (V, 2), exact projections
    out = np.full_like(full, np.nan)
    for v in views:
        out[v] = full[v]
    return out


def _nan(n_views):
    return np.full((n_views, 2), np.nan)


def test_gt_wins_zero_gt_recovers_run_3d(result):
    # No GT: predictions-only through the configured RANSAC recovers the true point.
    tri = TriangulationParams()
    ann = AnnotationParams()  # gt_wins
    true = result.pts3d[FRAME, POINT]
    x = solve_point_3d(
        result.cameras,
        _nan(result.n_views),
        _obs(result, range(result.n_views)),
        result.conf[:, FRAME, POINT],
        ann,
        tri,
    )
    assert np.allclose(x, true, atol=1e-6)


def test_gt_wins_two_gt_views_solve_from_gt_alone(result):
    tri = TriangulationParams()
    ann = AnnotationParams()
    true = result.pts3d[FRAME, POINT]
    # GT at views 0 and 3 only; predictions all NaN -> must still solve from GT.
    x = solve_point_3d(
        result.cameras,
        _obs(result, [0, 3]),
        _nan(result.n_views),
        None,
        ann,
        tri,
    )
    assert np.allclose(x, true, atol=1e-6)


# -- two GT views: the stabilized solve ---------------------------------------
#
# Two GT views do not determine a point equally well in every direction: a camera says
# nothing about distance along its own optical axis, so two cameras that face each other
# leave that distance almost free. The rig's `rm` (-90 deg) and `lm` (+90 deg) are exactly
# anti-parallel, which is what these tests use. The bar is the same one the one-GT depth
# solve is held to: fix the free direction, keep the operator's pixels, ignore an outlier,
# and do nothing where the geometry is already good.

RM, LM, FRONT = 1, 5, 3  # AZIMUTHS_DEG = [-120, -90, -45, 0, 45, 90, 120]


def _perturbed(result, gt_views, rng, sigma_gt=0.5, sigma_pred=3.0):
    """GT with click noise, predictions with detector noise."""
    exact = result.pts2d[:, FRAME, POINT].astype(float)
    gt = np.full_like(exact, np.nan)
    pred = np.full_like(exact, np.nan)
    for v in range(exact.shape[0]):
        if v in gt_views:
            gt[v] = exact[v] + rng.normal(0, sigma_gt, 2)
        else:
            pred[v] = exact[v] + rng.normal(0, sigma_pred, 2)
    return gt, pred


def _err(result, gt, pred, ann, tri):
    """Distance from the solved point to the true 3D, in mm."""
    x = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    return float(np.linalg.norm(x - result.pts3d[FRAME, POINT]))


def _gt_resid(result, gt, pred, ann, tri, gt_views):
    """Mean distance from the solved point's reprojection to the operator's own pixels."""
    x = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    proj = np.asarray(result.cameras.project(x[None, None, :]), dtype=float)
    proj = proj[:, 0, 0]
    return float(np.mean([np.linalg.norm(proj[v] - gt[v]) for v in gt_views]))


def test_two_gt_on_facing_cameras_is_rescued_by_the_other_views(result):
    """The complaint this solve exists for: two anti-parallel GT views leave depth free."""
    tri, ann = TriangulationParams(), AnnotationParams()
    alone = AnnotationParams(gt_wins_keep_stabilizers=False)
    rng = np.random.default_rng(0)
    err_alone, err_stab = [], []
    for _ in range(40):
        gt, pred = _perturbed(result, (RM, LM), rng)
        err_alone.append(_err(result, gt, pred, alone, tri))
        err_stab.append(_err(result, gt, pred, ann, tri))
    # The GT-only answer is dominated by the unconstrained direction; the stabilized one
    # is not. The measured ratio on this rig is ~30x; assert an order of magnitude.
    assert np.mean(err_stab) < np.mean(err_alone) / 10.0


def test_the_stabilized_solve_keeps_the_operators_pixels(result):
    """Fixing the free direction may not walk the point off the clicked pixels.

    Two GT views are four equations in three unknowns, so even the GT-only solve does not
    reproject exactly onto both clicks -- that residual is the floor. The stabilized solve
    must stay near it, and far below the reprojection warning's 8 px threshold.
    """
    tri, ann = TriangulationParams(), AnnotationParams()
    alone = AnnotationParams(gt_wins_keep_stabilizers=False)
    rng = np.random.default_rng(1)
    floor, got = [], []
    for _ in range(40):
        gt, pred = _perturbed(result, (RM, LM), rng)
        floor.append(_gt_resid(result, gt, pred, alone, tri, (RM, LM)))
        got.append(_gt_resid(result, gt, pred, ann, tri, (RM, LM)))
    assert np.mean(got) < np.mean(floor) + 0.5
    # poseView.js warnThreshold is 8 px: no new rings on the operator's own labels.
    assert np.mean(got) < 8.0


def test_the_stabilized_solve_ignores_an_outlier_stabilizer(result):
    """One grossly wrong prediction is down-weighted, not averaged in."""
    tri, ann = TriangulationParams(), AnnotationParams()
    rng = np.random.default_rng(2)
    clean, spoiled = [], []
    for _ in range(40):
        gt, pred = _perturbed(result, (RM, LM), rng)
        bad = np.array(pred, copy=True)
        bad[0] = bad[0] + np.array([160.0, 0.0])
        clean.append(_err(result, gt, pred, ann, tri))
        spoiled.append(_err(result, gt, bad, ann, tri))
    assert np.mean(spoiled) < 2.0 * np.mean(clean)


def test_the_stabilized_solve_is_a_no_op_on_good_geometry(result):
    """Where the GT pair already determines the point, the predictions must not move it."""
    tri, ann = TriangulationParams(), AnnotationParams()
    alone = AnnotationParams(gt_wins_keep_stabilizers=False)
    rng = np.random.default_rng(3)
    err_alone, err_stab = [], []
    for _ in range(40):
        gt, pred = _perturbed(result, (RM, FRONT), rng)  # 90 degrees apart
        err_alone.append(_err(result, gt, pred, alone, tri))
        err_stab.append(_err(result, gt, pred, ann, tri))
    assert np.mean(err_stab) <= 1.15 * np.mean(err_alone)


def test_three_gt_views_leave_nothing_for_the_stabilizers_to_do(result):
    """The estimator stops using predictions on its own, with no branch to say so."""
    tri, ann = TriangulationParams(), AnnotationParams()
    alone = AnnotationParams(gt_wins_keep_stabilizers=False)
    rng = np.random.default_rng(4)
    gaps = []
    for _ in range(20):
        gt, pred = _perturbed(result, (RM, LM, FRONT), rng)
        a = solve_point_3d(result.cameras, gt, pred, None, alone, tri)
        b = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
        gaps.append(np.linalg.norm(a - b))
    assert max(gaps) < 5e-3  # mm


def test_the_stabilized_solve_is_deterministic(result):
    """Bit-for-bit re-derivation: the derived-3D cache and undo history require it."""
    tri, ann = TriangulationParams(), AnnotationParams()
    rng = np.random.default_rng(5)
    gt, pred = _perturbed(result, (RM, LM), rng)
    a = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    b = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    assert np.array_equal(a, b)


def test_the_stabilized_solve_returns_the_anchor_with_no_stabilizers(result):
    """No usable prediction: the GT-only answer, unchanged."""
    tri, ann = TriangulationParams(), AnnotationParams()
    gt = _obs(result, [RM, LM])
    x0 = _dlt(result.cameras, gt)
    nan = _nan(result.n_views)
    x = solve_point_3d_stabilized(result.cameras, x0, gt, nan, ann, tri)
    assert np.array_equal(x, x0)


def test_a_separate_stabilizer_array_overrides_the_seeds(result):
    """``stab_obs`` is what fixes the free direction, not ``pred_obs``.

    The default seeds are reprojections of the current 3D, so the solve must be able to
    take a different, independent array for the job -- see ``EditorState._point_obs``.
    """
    tri, ann = TriangulationParams(), AnnotationParams()
    true = result.pts3d[FRAME, POINT]
    rng = np.random.default_rng(6)
    gt, pred = _perturbed(result, (RM, LM), rng)
    junk = np.where(np.isfinite(pred), pred + 300.0, np.nan)  # a wrong "seed" array
    with_stab = solve_point_3d(result.cameras, gt, junk, None, ann, tri, pred)
    without = solve_point_3d(result.cameras, gt, junk, None, ann, tri)
    assert np.linalg.norm(with_stab - true) < np.linalg.norm(without - true)


def test_gt_wins_single_gt_plus_predictions_fills(result):
    tri = TriangulationParams()
    ann = AnnotationParams()
    true = result.pts3d[FRAME, POINT]
    # One GT view + predictions in the rest (all consistent) -> recovers the point,
    # and the GT view reprojects onto its pixel.
    gt = _obs(result, [0])
    pred = _obs(result, range(1, result.n_views))
    x = solve_point_3d(result.cameras, gt, pred, result.conf[:, FRAME, POINT], ann, tri)
    assert np.allclose(x, true, atol=1e-5)
    reproj = np.asarray(result.cameras.project(x))
    assert np.allclose(reproj[0], gt[0], atol=1e-3)


# -- one GT: a depth, not a triangulation -------------------------------------
#
# With a single GT the pixel fixes the viewing ray and contributes nothing about depth, so
# the depth comes entirely from the detections -- which is why this branch, and only this
# branch, needs a robust estimator. `solve_depth_on_ray` uses Huber IRLS along the ray.


def test_one_gt_puts_the_point_exactly_on_its_ray(result):
    """The operator's pixel is honored exactly, not to within a weight-1000 compromise.

    A weighted DLT leaves the GT view reprojecting ~0.1 px off the pixel that was clicked;
    solving for depth *along* the ray is exact by construction.
    """
    ann, tri = AnnotationParams(), TriangulationParams()
    gt = _obs(result, [0])
    pred = _obs(result, range(1, result.n_views))
    x = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    reproj = np.asarray(result.cameras.project(x[None, None, :]), dtype=float)[0, 0]
    assert np.linalg.norm(reproj - gt[0]) < 1e-6


def test_one_gt_depth_ignores_an_outlier_detection(result):
    """A bad peak in one view must not drag the depth -- the old weighted DLT let it."""
    ann, tri = AnnotationParams(), TriangulationParams()
    true = result.pts3d[FRAME, POINT]
    gt = _obs(result, [0])
    pred = _obs(result, range(1, result.n_views))
    pred[3] = pred[3] + 160.0  # a contralateral-scale mislocalization

    x = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    clean = solve_point_3d(
        result.cameras, gt, _obs(result, range(1, result.n_views)), None, ann, tri
    )
    # the outlier is suppressed to a fraction of its unweighted influence
    assert np.linalg.norm(x - true) < 0.3 * np.linalg.norm(clean - true) + 0.05


def test_one_gt_with_clean_detections_does_not_lose_accuracy(result):
    """Huber, not a hard consensus, precisely so this holds.

    With nothing to reject the weights are all 1 and the estimate is the least-squares fit.
    An enumerate-and-score consensus measured 62-77% worse here once the detection noise
    approached ``ransac_threshold``, because truncation discarded the information that tells
    "every view mildly wrong" from "two views lucky".
    """
    ann, tri = AnnotationParams(), TriangulationParams()
    rng = np.random.default_rng(0)
    gt = _obs(result, [0])
    errs = []
    for _ in range(40):
        pred = _obs(result, range(1, result.n_views))
        pred[1:] += rng.normal(0, 6.0, pred[1:].shape)  # noisy but no outliers
        x = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
        errs.append(np.linalg.norm(x - result.pts3d[FRAME, POINT]))
    assert np.mean(errs) < 0.05  # comfortably better than the 6 px noise floor implies


def test_one_gt_and_no_usable_detection_has_no_depth(result):
    """Nothing fixes the depth, so there is nothing to return -- the caller falls back."""
    tri = TriangulationParams()
    gt = _obs(result, [0])
    assert (
        solve_depth_on_ray(result.cameras, 0, gt[0], _nan(result.n_views), tri) is None
    )


def test_one_gt_depth_is_deterministic(result):
    """Re-deriving the same labels twice must give the same answer, bit for bit.

    The derived-3D cache depends on it: an undo restores a snapshot and the frame is
    re-derived around it, so a solve that wandered would look like points moving on their
    own. Enumeration + a fixed reweighting schedule, no sampling.
    """
    ann, tri = AnnotationParams(), TriangulationParams()
    gt = _obs(result, [0])
    pred = _obs(result, range(1, result.n_views))
    pred[2] += 80.0
    a = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    b = solve_point_3d(result.cameras, gt, pred, None, ann, tri)
    np.testing.assert_array_equal(a, b)


def test_equal_weight_recovers_and_protects_gt(result):
    tri = TriangulationParams()
    ann = AnnotationParams(solve_policy="equal_weight")
    true = result.pts3d[FRAME, POINT]
    x = solve_point_3d(
        result.cameras,
        _obs(result, [0]),
        _obs(result, range(1, result.n_views)),
        result.conf[:, FRAME, POINT],
        ann,
        tri,
    )
    assert np.allclose(x, true, atol=1e-5)


def test_weighted_blend_recovers_point(result):
    tri = TriangulationParams()
    ann = AnnotationParams(solve_policy="weighted_blend")
    true = result.pts3d[FRAME, POINT]
    x = solve_point_3d(
        result.cameras,
        _obs(result, [0, 1]),
        _obs(result, range(2, result.n_views)),
        None,
        ann,
        tri,
    )
    assert np.allclose(x, true, atol=1e-5)


def test_fewer_than_two_views_returns_nan(result):
    ann = AnnotationParams()
    tri = TriangulationParams()
    x = solve_point_3d(
        result.cameras, _obs(result, [0]), _nan(result.n_views), None, ann, tri
    )
    assert not np.all(np.isfinite(x))


# -- drag solve ---------------------------------------------------------------


def test_drag_ray_fallback_lands_under_cursor(result):
    ann = AnnotationParams()
    prior = result.pts3d[FRAME, POINT]
    dragged_view = 2
    base = np.asarray(result.cameras.project(prior))[dragged_view]
    cursor = base + np.array([10.0, -8.0])
    # Only the dragged view is usable -> ray fallback, lands under the cursor.
    x = solve_point_3d_drag(
        result.cameras,
        _nan(result.n_views),
        _nan(result.n_views),
        None,
        dragged_view,
        cursor,
        prior,
        ann,
    )
    assert x is not None
    reproj = np.asarray(result.cameras.project(x))[dragged_view]
    assert np.allclose(reproj, cursor, atol=1e-3)


def test_drag_with_a_second_view_triangulates(result):
    ann = AnnotationParams()
    true = result.pts3d[FRAME, POINT]
    dragged_view = 2
    cursor = np.asarray(result.cameras.project(true))[dragged_view]
    # A GT constraint in view 0 + dragging view 2 to the true pixel -> two usable
    # views -> DLT recovers the true point.
    x = solve_point_3d_drag(
        result.cameras,
        _obs(result, [0]),
        _nan(result.n_views),
        None,
        dragged_view,
        cursor,
        true,
        ann,
    )
    assert x is not None
    assert np.allclose(x, true, atol=1e-4)


def test_drag_no_prior_and_no_views_returns_none(result):
    ann = AnnotationParams()
    x = solve_point_3d_drag(
        result.cameras,
        _nan(result.n_views),
        _nan(result.n_views),
        None,
        2,
        (1.0, 2.0),
        None,
        ann,
    )
    assert x is None
