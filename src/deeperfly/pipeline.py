"""End-to-end orchestration: 2D points -> calibration -> 3D -> correction.

Pure functions over arrays, so every stage is testable in isolation and the 2D
detector stays pluggable (a callable producing ``(pts2d, conf)``):

- :func:`calibrate` -- treat the animal as the calibration target and refine the
  cameras with bundle adjustment (confidence weights, Huber loss, bone-length
  prior).
- :func:`reconstruct` -- triangulate a 2D sequence and *greedily* reject
  high-reprojection-error observations, re-triangulating from the survivors.
- :func:`reconstruct_ransac` -- the default: triangulate each point from its
  largest multi-view consensus set (RANSAC) instead of a contaminated fit.
- :func:`run_from_points2d` -- the whole pipeline from a 2D sequence to a saved
  :class:`PoseResult`.

All 2D arrays use the view-leading layout ``(V, T, N, 2)`` with NaN for missing
observations; 3D points come out as ``(T, N, 3)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float
from scipy.optimize import OptimizeResult

from . import pictorial
from .cameras import CameraGroup
from .correction import smooth_gaussian, smooth_one_euro
from .io import PoseResult
from .skeleton import Skeleton
from .triangulate import (
    apply_visibility,
    reprojection_error,
    triangulate,
    triangulate_ransac,
)
from .bundle_adjustment import bundle_adjust


def _subsample(n_frames: int, max_frames: int | None) -> np.ndarray:
    """Evenly spaced frame indices (all of them if ``max_frames`` is None)."""
    if max_frames is None or n_frames <= max_frames:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, max_frames).round().astype(int)


def _bone_prior(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V F N 2"],
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Bone-pair indices and per-bone target lengths for flattened frames.

    Targets are the median bone length across frames from an initial
    triangulation -- a soft, robust prior (shared with the pictorial-structures
    corrector via :func:`deeperfly.pictorial.bone_length_targets`). ``pts2d`` has
    ``F`` frames; the returned pairs index the flattened ``F * N`` point axis.
    """
    n_frames, n_pts = pts2d.shape[1], pts2d.shape[2]
    i, j, targets = pictorial.bone_length_targets(cameras, pts2d, skeleton)
    offsets = (np.arange(n_frames) * n_pts)[:, None]
    pairs = np.stack([(offsets + i).ravel(), (offsets + j).ravel()], axis=1)
    return pairs, np.tile(targets, n_frames)


def calibrate(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None = None,
    skeleton: Skeleton | None = None,
    *,
    ba_keypoints: Sequence[int] | None = None,
    fixed: list[str] = (),
    shared: list[list[str]] = (),
    bone_prior: bool = True,
    bone_weight: float = 1.0,
    loss: str = "huber",
    f_scale: float = 40.0,
    max_frames: int | None = 100,
    max_nfev: int = 300,
    **solver_kwargs,
) -> tuple[CameraGroup, OptimizeResult]:
    """Refine ``cameras`` from a 2D sequence (fly-as-calibration-target BA).

    Frames are flattened into one point cloud; detector confidences become
    per-observation weights; a robust loss and an optional bone-length prior
    stabilize the fit. ``fixed`` / ``shared`` anchor the gauge as in
    :func:`deeperfly.bundle_adjustment.bundle_adjust`.

    ``ba_keypoints`` (skeleton point indices) restricts which keypoints drive the
    refinement -- observations of unselected points are masked out. Defaults to
    every point; pass e.g. the leg-joint indices to calibrate on the sharp limb
    corners alone. Only the camera fit is restricted -- all points are still
    triangulated afterward by :func:`reconstruct`.

    Returns the refined :class:`CameraGroup` and the raw scipy result.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    n_views, n_frames, n_pts = pts2d.shape[:3]
    sel = _subsample(n_frames, max_frames)
    p = pts2d[:, sel]  # (V, F, N, 2)
    n_sel = len(sel)

    masked = False
    if ba_keypoints is not None:
        drop = np.ones(n_pts, dtype=bool)
        drop[np.asarray(ba_keypoints, dtype=int)] = False
        if drop.any():
            p = p.copy()
            p[:, :, drop] = np.nan  # masked-out observations are ignored by the solver
            masked = True

    bone_pairs = bone_targets = None
    if bone_prior and skeleton is not None and skeleton.bones.size:
        bone_pairs, bone_targets = _bone_prior(cameras, p, skeleton)
        finite = np.isfinite(bone_targets)  # drop bones with no selected triangulation
        bone_pairs, bone_targets = bone_pairs[finite], bone_targets[finite]

    weights = None
    if conf is not None:
        weights = np.asarray(conf, dtype=float)[:, sel].reshape(n_views, n_sel * n_pts)

    p_flat = p.reshape(n_views, n_sel * n_pts, 2)
    # Masked-out points have no observations, so their 3D triangulates to NaN;
    # seed a finite placeholder for a valid initial guess. They carry no
    # residuals, so they never move and don't affect the recovered cameras.
    init_pts3d = np.nan_to_num(triangulate(cameras, p_flat)) if masked else None

    result, optimized, _ = bundle_adjust(
        cameras,
        p_flat,
        fixed=fixed,
        shared=shared,
        pts3d=init_pts3d,
        weights=weights,
        bone_pairs=bone_pairs,
        bone_targets=bone_targets,
        bone_weight=bone_weight,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        **solver_kwargs,
    )
    return optimized, result


def reconstruct(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T N 2"],
    *,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
) -> tuple[
    Float[np.ndarray, "T N 3"], Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]
]:
    """Triangulate a sequence and greedily reject reprojection outliers.

    A gross 2D outlier inflates *every* view's reprojection error for that point,
    so thresholding all views would discard the good ones too. Instead each pass
    drops only the **single worst** view of each still-offending point (keeping at
    least two) and re-triangulates, removing outliers one at a time.

    Returns ``(pts3d, cleaned_pts2d, reproj_error)``.
    """
    pts2d = np.array(pts2d, dtype=float)
    n_views = pts2d.shape[0]
    for _ in range(max_drops):
        pts3d = triangulate(cameras, pts2d)
        err = reprojection_error(cameras, pts3d, pts2d).reshape(n_views, -1)  # (V, M)
        flat = pts2d.reshape(n_views, -1, 2)
        n_valid = np.isfinite(flat).all(-1).sum(0)  # (M,)
        filled = np.where(np.isfinite(err), err, -np.inf)
        worst_view = filled.argmax(0)  # (M,)
        worst_err = filled.max(0)  # (M,)
        drop = (worst_err > reproj_threshold) & (n_valid > 2)
        if not drop.any():
            break
        cols = np.flatnonzero(drop)
        flat[worst_view[cols], cols, :] = np.nan  # view of pts2d (in place)
    pts3d = triangulate(cameras, pts2d)
    err = reprojection_error(cameras, pts3d, pts2d)
    return pts3d, pts2d, err


def reconstruct_ransac(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T N 2"],
    *,
    threshold: float = 15.0,
    min_inliers: int = 2,
) -> tuple[
    Float[np.ndarray, "T N 3"], Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]
]:
    """Triangulate a sequence robustly via per-point RANSAC consensus.

    Unlike :func:`reconstruct`, which deletes the worst view from a contaminated
    fit, this builds each point from the *largest set of mutually consistent
    views* (:func:`deeperfly.triangulate.triangulate_ransac`), so a badly
    mislocated detection never enters the fit. NaN views never count as inliers.

    The returned 2D array is the input with every non-inlier observation set to
    ``NaN``, matching :func:`reconstruct`'s ``(pts3d, cleaned_pts2d,
    reproj_error)`` contract.
    """
    pts2d = np.array(pts2d, dtype=float)
    pts3d, inliers = triangulate_ransac(
        cameras, pts2d, threshold=threshold, min_inliers=min_inliers
    )
    cleaned = np.where(inliers[..., None], pts2d, np.nan)
    err = reprojection_error(cameras, pts3d, cleaned)
    return pts3d, cleaned, err


#: Triangulation strategies for the reconstruction step, plus legacy aliases.
_TRIANGULATORS = ("ransac", "greedy", "dlt")
_TRIANGULATION_ALIASES = {"reproject": "greedy", "none": "dlt"}


def _resolve_triangulation(triangulation: str) -> str:
    """Normalize a ``triangulation`` choice to one of :data:`_TRIANGULATORS`.

    Canonical names ``"ransac"`` / ``"greedy"`` / ``"dlt"``; ``"reproject"`` and
    ``"none"`` are aliases for ``"greedy"`` and ``"dlt"``.
    """
    method = _TRIANGULATION_ALIASES.get(triangulation, triangulation)
    if method not in _TRIANGULATORS:
        raise ValueError(f"unknown triangulation {triangulation!r} (ransac|greedy|dlt)")
    return method


def run_from_points2d(
    cameras: CameraGroup,
    skeleton: Skeleton,
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None = None,
    *,
    do_calibrate: bool = True,
    calibrate_kwargs: dict | None = None,
    triangulation: str = "ransac",
    do_pictorial: bool = False,
    candidates: pictorial.Candidates | None = None,
    ps_kwargs: dict | None = None,
    ransac_threshold: float = 15.0,
    min_inliers: int = 2,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
    smooth: str | None = None,
    fps: float = 100.0,
    smooth_kwargs: dict | None = None,
    meta: dict | None = None,
) -> PoseResult:
    """Run the full 2D-to-3D pipeline and return a :class:`PoseResult`.

    Steps: apply skeleton visibility -> (optional) calibrate cameras ->
    reconstruct 3D -> (optional) temporal smoothing.

    The 2D->3D reconstruction is two orthogonal choices:

    * ``triangulation`` (:func:`_resolve_triangulation`):

      * ``"ransac"`` (default) -- each point from its largest multi-view
        consensus set (:func:`reconstruct_ransac`); ``ransac_threshold`` /
        ``min_inliers``.
      * ``"greedy"`` -- DLT, greedily dropping the worst-reprojecting view of each
        offending point (:func:`reconstruct`); ``reproj_threshold`` /
        ``max_drops``. (``"reproject"`` alias.)
      * ``"dlt"`` -- plain least-squares, no outlier handling (``"none"`` alias).

    * ``do_pictorial`` -- when ``True``, first run pictorial-structures peak
      recovery over the detector's top-K candidates
      (:func:`deeperfly.pictorial.reconstruct`; requires ``candidates``, accepts
      ``ps_kwargs`` like ``temporal`` / ``lam`` / ``max_hyp``), then feed its
      committed 2D into ``triangulation`` (``"dlt"`` keeps the PS estimate).
      Calibration always uses the arg-max ``pts2d``.

    ``smooth`` is ``None``, ``"gaussian"`` or ``"one_euro"``; ``smooth_kwargs`` is
    forwarded to the matching :mod:`deeperfly.correction` function (``one_euro``
    also receives ``fps``).
    """
    method = _resolve_triangulation(triangulation)  # validate before calibrating
    names = cameras.names
    pts2d = apply_visibility(np.asarray(pts2d, dtype=float), skeleton, names)

    if do_calibrate:
        cameras, _ = calibrate(
            cameras, pts2d, conf, skeleton, **(calibrate_kwargs or {})
        )

    if do_pictorial:
        if candidates is None:
            raise ValueError("do_pictorial=True requires candidates=...")
        # PS recovers the right peaks; its committed per-view 2D then feeds the
        # triangulator below (a plain "dlt" pass reproduces the PS estimate).
        pts3d, pts2d, reproj = pictorial.reconstruct(
            cameras, skeleton, candidates, pts2d, **(ps_kwargs or {})
        )

    if method == "ransac":
        pts3d, pts2d, reproj = reconstruct_ransac(
            cameras, pts2d, threshold=ransac_threshold, min_inliers=min_inliers
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            cameras,
            pts2d,
            reproj_threshold=reproj_threshold,
            max_drops=max_drops,
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts3d = triangulate(cameras, pts2d)
        reproj = reprojection_error(cameras, pts3d, pts2d)

    pts3d_smoothed = None
    sk = smooth_kwargs or {}
    if smooth == "gaussian":
        pts3d_smoothed = smooth_gaussian(pts3d, **sk)
    elif smooth == "one_euro":
        pts3d_smoothed = smooth_one_euro(pts3d, fps, **sk)
    elif smooth is not None:
        raise ValueError(f"unknown smooth mode {smooth!r}")

    return PoseResult(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        pts3d_smoothed=pts3d_smoothed,
        reproj_error=reproj,
        meta={
            "fps": fps,
            "triangulation": method,
            "pictorial": do_pictorial,
            **(meta or {}),
        },
    )
