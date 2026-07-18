"""Tests for the per-point 3D solve (labels + predictions -> 3D)."""

from __future__ import annotations

import numpy as np

from deeperfly.config import AnnotationParams, TriangulationParams
from deeperfly.gui.solve import solve_point_3d, solve_point_3d_drag

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
