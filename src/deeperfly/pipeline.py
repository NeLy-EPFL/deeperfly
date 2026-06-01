"""End-to-end orchestration: 2D points -> calibration -> 3D -> correction.

This is the modern analog of DeepFly3D's ``Core``. It is written as pure
functions over arrays so every stage is testable in isolation and the heavy 2D
detector stays pluggable (a callable producing ``(pts2d, conf)``):

- :func:`calibrate` -- treat the animal itself as the calibration target and
  refine the cameras with bundle adjustment, using detector confidences as
  per-observation weights, a robust (Huber) loss and a soft bone-length prior.
- :func:`reconstruct` -- triangulate a 2D sequence and iteratively reject
  high-reprojection-error observations, re-triangulating from the survivors.
- :func:`run_from_points2d` -- the whole pipeline from a 2D sequence to a saved
  :class:`PoseResult` (calibration, reconstruction, optional template alignment
  and temporal smoothing).

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
from .correction import align_to_template, smooth_gaussian, smooth_one_euro
from .io import PoseResult
from .skeleton import Skeleton
from .triangulate import (
    apply_visibility,
    merge_points,
    merge_sources,
    reprojection_error,
    triangulate,
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

    The targets are the median bone length across frames from an initial
    triangulation -- a soft, robust length prior. ``pts2d`` has ``F`` frames;
    the returned pairs index into the flattened ``F * N`` point axis. The per-bone
    target estimation is shared with the pictorial-structures corrector via
    :func:`deeperfly.pictorial.bone_length_targets`.
    """
    n_frames, n_pts = pts2d.shape[1], pts2d.shape[2]
    i, j, targets = pictorial.bone_length_targets(cameras, pts2d, skeleton)
    offsets = (np.arange(n_frames) * n_pts)[:, None]
    pairs = np.stack([(offsets + i).ravel(), (offsets + j).ravel()], axis=1)
    return pairs, np.tile(targets, n_frames)


def _merge_stripes(
    skeleton: Skeleton,
    camera_names: list[str],
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None,
    candidates: pictorial.Candidates | None,
) -> tuple[
    Skeleton,
    np.ndarray,
    np.ndarray | None,
    pictorial.Candidates | None,
]:
    """Fuse the left/right abdominal stripes into shared points.

    Returns the merged skeleton and the 2D arrays remapped onto its (smaller)
    point layout. ``pts2d`` is assumed already visibility-masked so each stripe
    side is NaN outside the cameras that see it; the per-view source selection
    (:func:`deeperfly.triangulate.merge_sources`) then routes each camera to the
    side it observes, so a merged stripe is triangulated from all four cameras
    that see either side. If the skeleton has nothing to merge this is a no-op.
    """
    merged, remap = skeleton.merge_lr_stripes()
    if merged is skeleton:
        return skeleton, pts2d, conf, candidates
    src = merge_sources(remap, skeleton.visibility_mask(camera_names), merged.n_points)
    pts2d = merge_points(pts2d, src, axis=2)
    if conf is not None:
        conf = merge_points(conf, src, axis=2)
    if candidates is not None:
        candidates = pictorial.Candidates(
            merge_points(candidates.xy, src, axis=2),
            merge_points(candidates.score, src, axis=2),
        )
    return merged, pts2d, conf, candidates


def calibrate(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None = None,
    skeleton: Skeleton | None = None,
    *,
    ba_keypoints: Sequence[str] | None = ("legs",),
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

    Frames are flattened into one big point cloud; detector confidences become
    per-observation weights; a robust loss and an optional bone-length prior
    stabilize the fit. ``fixed`` / ``shared`` anchor the gauge exactly as in
    :func:`deeperfly.bundle_adjustment.bundle_adjust`.

    ``ba_keypoints`` restricts which keypoint categories drive the camera
    refinement -- a subset of ``{"legs", "antennae", "stripes"}`` resolved via
    :meth:`Skeleton.points_in_category`. It defaults to ``("legs",)`` because the
    leg joints sit at sharp limb corners and are the most trustworthy
    detections; observations of unselected points are masked out (the solver
    drops NaNs). Pass ``None`` (or omit a skeleton) to use every point. Only the
    camera fit is restricted -- all points are still triangulated afterward by
    :func:`reconstruct` with the refined cameras.

    Returns the refined :class:`CameraGroup` and the raw scipy result.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    n_views, n_frames, n_pts = pts2d.shape[:3]
    sel = _subsample(n_frames, max_frames)
    p = pts2d[:, sel]  # (V, F, N, 2)
    n_sel = len(sel)

    masked = False
    if ba_keypoints is not None and skeleton is not None:
        keep = skeleton.points_in_category(ba_keypoints)
        drop = np.ones(n_pts, dtype=bool)
        drop[keep] = False
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
    # Masked-out points have no observations, so their (unused) 3D triangulates
    # to NaN; seed them with a finite placeholder so the solver's initial guess
    # is valid. They carry no residuals, so they never move and do not affect the
    # recovered cameras (which are all that calibration returns).
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

    A gross 2D outlier pulls the least-squares triangulation, inflating *every*
    view's reprojection error for that point -- so flagging all views above a
    threshold would wrongly discard the good ones too. Instead each pass drops
    only the **single worst** view of each still-offending point (keeping at
    least two views) and re-triangulates, removing outliers one at a time.

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


def run_from_points2d(
    cameras: CameraGroup,
    skeleton: Skeleton,
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None = None,
    *,
    merge_stripes: bool = True,
    do_calibrate: bool = True,
    calibrate_kwargs: dict | None = None,
    correct: str = "reproject",
    candidates: pictorial.Candidates | None = None,
    ps_kwargs: dict | None = None,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
    template: Float[np.ndarray, "N 3"] | None = None,
    smooth: str | None = None,
    fps: float = 100.0,
    smooth_kwargs: dict | None = None,
    meta: dict | None = None,
) -> PoseResult:
    """Run the full 2D-to-3D pipeline and return a :class:`PoseResult`.

    Steps: apply skeleton visibility -> (optional) merge L/R stripes ->
    (optional) calibrate cameras -> reconstruct 3D -> (optional) Procrustes
    alignment to a template -> (optional) temporal smoothing.

    ``merge_stripes`` (default ``True``) fuses the left/right abdominal stripe
    markers into shared ``stripe0/1/2`` points (:meth:`Skeleton.merge_lr_stripes`),
    so each is triangulated from all four cameras that see either side and the
    result carries 35 instead of 38 points. The returned :class:`PoseResult`
    then uses the merged skeleton and point layout throughout.

    ``correct`` selects the 2D->3D reconstruction:

    * ``"reproject"`` (default) -- triangulate the arg-max detections and greedily
      reject reprojection outliers (:func:`reconstruct`).
    * ``"pictorial"`` -- DeepFly3D-style pictorial-structures correction over the
      detector's top-K candidate peaks (:func:`deeperfly.pictorial.reconstruct`),
      which requires ``candidates`` and accepts ``ps_kwargs`` (e.g. ``temporal``,
      ``lam``, ``max_hyp``). Calibration still uses the arg-max ``pts2d``.

    ``smooth`` is ``None``, ``"gaussian"`` or ``"one_euro"``; ``smooth_kwargs``
    is forwarded to the corresponding :mod:`deeperfly.correction` function
    (``one_euro`` also receives ``fps``).
    """
    names = cameras.names
    pts2d = apply_visibility(np.asarray(pts2d, dtype=float), skeleton, names)

    if merge_stripes:
        skeleton, pts2d, conf, candidates = _merge_stripes(
            skeleton, names, pts2d, conf, candidates
        )

    if do_calibrate:
        cameras, _ = calibrate(
            cameras, pts2d, conf, skeleton, **(calibrate_kwargs or {})
        )

    if correct == "pictorial":
        if candidates is None:
            raise ValueError("correct='pictorial' requires candidates=...")
        pts3d, pts2d, reproj = pictorial.reconstruct(
            cameras, skeleton, candidates, pts2d, **(ps_kwargs or {})
        )
    elif correct == "reproject":
        pts3d, pts2d, reproj = reconstruct(
            cameras,
            pts2d,
            reproj_threshold=reproj_threshold,
            max_drops=max_drops,
        )
    else:
        raise ValueError(f"unknown correct mode {correct!r} (reproject|pictorial)")

    if template is not None:
        pts3d = align_to_template(pts3d, template, skeleton)

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
        meta={"fps": fps, "correct": correct, **(meta or {})},
    )
