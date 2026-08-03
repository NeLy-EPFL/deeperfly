"""Tests for the per-point 3D solve (labels + predictions -> 3D)."""

from __future__ import annotations

import numpy as np

from deeperfly.config import AnnotationParams, TriangulationParams
from deeperfly.gui.solve import (
    solve_depth_on_ray,
    solve_point_3d,
    solve_point_3d_drag,
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
