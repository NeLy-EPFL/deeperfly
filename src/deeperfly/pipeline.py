"""End-to-end orchestration: 2D points -> calibration -> 3D -> correction.

This is the modern analogue of DeepFly3D's ``Core``. It is written as pure
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

import numpy as np
from jaxtyping import Float
from scipy.optimize import OptimizeResult

from .cameras import CameraGroup
from .correction import align_to_template, smooth_gaussian, smooth_one_euro
from .io import PoseResult
from .skeleton import Skeleton
from .triangulate import apply_visibility, reprojection_error, triangulate
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
    the returned pairs index into the flattened ``F * N`` point axis.
    """
    n_frames, n_pts = pts2d.shape[1], pts2d.shape[2]
    pts3d0 = triangulate(cameras, pts2d)  # (F, N, 3)
    i, j = skeleton.bone_index_pairs()
    lengths = np.linalg.norm(pts3d0[:, i] - pts3d0[:, j], axis=-1)  # (F, B)
    with np.errstate(invalid="ignore"):
        targets = np.nanmedian(lengths, axis=0)  # (B,)
    offsets = (np.arange(n_frames) * n_pts)[:, None]
    pairs = np.stack([(offsets + i).ravel(), (offsets + j).ravel()], axis=1)
    return pairs, np.tile(targets, n_frames)


def calibrate(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T N 2"],
    conf: Float[np.ndarray, "V T N"] | None = None,
    skeleton: Skeleton | None = None,
    *,
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
    stabilise the fit. ``fixed`` / ``shared`` anchor the gauge exactly as in
    :func:`deeperfly.bundle_adjustment.bundle_adjust`.

    Returns the refined :class:`CameraGroup` and the raw scipy result.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    n_views, n_frames, n_pts = pts2d.shape[:3]
    sel = _subsample(n_frames, max_frames)
    p = pts2d[:, sel]  # (V, F, N, 2)
    n_sel = len(sel)

    bone_pairs = bone_targets = None
    if bone_prior and skeleton is not None and skeleton.bones.size:
        bone_pairs, bone_targets = _bone_prior(cameras, p, skeleton)

    weights = None
    if conf is not None:
        weights = np.asarray(conf, dtype=float)[:, sel].reshape(n_views, n_sel * n_pts)

    result, optimized, _ = bundle_adjust(
        cameras,
        p.reshape(n_views, n_sel * n_pts, 2),
        fixed=fixed,
        shared=shared,
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
    do_calibrate: bool = True,
    calibrate_kwargs: dict | None = None,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
    template: Float[np.ndarray, "N 3"] | None = None,
    smooth: str | None = None,
    fps: float = 100.0,
    smooth_kwargs: dict | None = None,
    meta: dict | None = None,
) -> PoseResult:
    """Run the full 2D-to-3D pipeline and return a :class:`PoseResult`.

    Steps: apply skeleton visibility -> (optional) calibrate cameras ->
    triangulate + outlier rejection -> (optional) Procrustes alignment to a
    template -> (optional) temporal smoothing.

    ``smooth`` is ``None``, ``"gaussian"`` or ``"one_euro"``; ``smooth_kwargs``
    is forwarded to the corresponding :mod:`deeperfly.correction` function
    (``one_euro`` also receives ``fps``).
    """
    names = cameras.names
    pts2d = apply_visibility(np.asarray(pts2d, dtype=float), skeleton, names)

    if do_calibrate:
        cameras, _ = calibrate(
            cameras, pts2d, conf, skeleton, **(calibrate_kwargs or {})
        )

    pts3d, pts2d, reproj = reconstruct(
        cameras,
        pts2d,
        reproj_threshold=reproj_threshold,
        max_drops=max_drops,
    )

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
        meta={"fps": fps, **(meta or {})},
    )
