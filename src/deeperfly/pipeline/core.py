"""End-to-end orchestration: 2D points -> bundle adjustment -> 3D.

Pure functions over arrays, so every stage is testable in isolation and the 2D
detector stays pluggable (a callable producing ``(pts2d, conf)``):

- :func:`bundle_adjust_cameras` -- treat the animal as the bundle-adjustment
  target and refine the cameras (confidence weights, Huber loss, bone-length
  prior).
- :func:`reconstruct` -- triangulate a 2D sequence and *greedily* reject
  high-reprojection-error observations, re-triangulating from the survivors.
- :func:`reconstruct_ransac` -- the default: triangulate each point from its
  largest multi-view consensus set (RANSAC) instead of a contaminated fit.
- :func:`run_from_points2d` -- the whole pipeline from a 2D sequence to a saved
  :class:`PoseResult`.

All 2D arrays use the view-leading layout ``(V, T, P, 2)`` with NaN for missing
observations; 3D points come out as ``(T, P, 3)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import OptimizeResult

from .. import pictorial
from ..bundle_adjustment import bundle_adjust
from ..cameras import CameraGroup
from ..results import PoseResult
from ..skeleton import Skeleton
from ..triangulation import (
    reprojection_error,
    triangulate,
    triangulate_ransac,
)

#: Frame-subsampling strategies understood by :func:`_subsample`.
FRAME_SAMPLERS = ("even", "confidence", "coverage", "diversity")


def _frame_scores(
    strategy: str,
    pts2d: Float[np.ndarray, "V T P 2"] | None,
    conf: Float[np.ndarray, "V T P"] | None,
) -> Float[np.ndarray, "T"]:
    """Per-frame bundle-adjustment-quality score (higher = a better frame to keep).

    ``"confidence"`` scores each frame by the mean of its (finite) detector
    confidences across views and points -- frames the detector is surest about.
    ``"coverage"`` scores each frame by how many keypoints are seen by at least
    two views -- i.e. how many points that frame can actually triangulate, a
    proxy for how well it conditions the solve.
    """
    if strategy == "confidence":
        assert conf is not None  # _subsample requires conf for this strategy
        c = np.where(np.isfinite(conf), np.asarray(conf, dtype=float), np.nan)
        with warnings.catch_warnings():  # all-NaN frames -> -inf (rank last)
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nan_to_num(np.nanmean(c, axis=(0, 2)), nan=-np.inf)
    # "coverage": number of points observed by >= 2 views (triangulable) per frame.
    observed = np.isfinite(np.asarray(pts2d, dtype=float)).all(axis=-1)  # (V, T, P)
    return (observed.sum(axis=0) >= 2).sum(axis=1).astype(float)  # (T,)


def _best_per_bin(scores: Float[np.ndarray, "T"], k: int) -> Int[np.ndarray, "F"]:
    """Index of the highest-scoring frame in each of ``k`` contiguous time bins.

    Binning keeps the temporal spread that makes ``"even"`` robust while letting
    each bin contribute its best frame; ties take the earliest. Returns the sorted
    unique picks (at most ``k``).
    """
    edges = np.linspace(0, len(scores), k + 1).round().astype(int)
    picks = {
        lo + int(np.argmax(scores[lo:hi]))
        for lo, hi in zip(edges[:-1], edges[1:])
        if hi > lo
    }
    return np.array(sorted(picks))


def _frame_features(pts2d: Float[np.ndarray, "V T P 2"]) -> Float[np.ndarray, "T D"]:
    """Per-frame posture descriptor: standardized, mean-imputed 2D keypoints.

    Each frame becomes its flattened ``(view, point, coord)`` vector. Missing
    observations are imputed with that coordinate's across-frame mean (so a
    dropout reads as "average posture", not spurious novelty), and every column
    is scaled to unit variance so all views/joints weigh in comparably despite
    differing pixel ranges. ``diversity`` sampling measures frame-to-frame
    distance in this space.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    n_frames = pts2d.shape[1]
    feats = np.moveaxis(pts2d, 1, 0).reshape(n_frames, -1)  # (T, V*P*2)
    with warnings.catch_warnings():  # all-NaN columns -> mean NaN -> imputed to 0
        warnings.simplefilter("ignore", RuntimeWarning)
        col_mean = np.nanmean(feats, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    feats = np.where(np.isfinite(feats), feats, col_mean)
    std = feats.std(axis=0)
    return feats / np.where(std > 0, std, 1.0)


def _farthest_first(feats: Float[np.ndarray, "T D"], k: int) -> Int[np.ndarray, "F"]:
    """Greedy farthest-point (k-center) selection: a maximally spread subset.

    Seeds on the most extreme frame (farthest from the centroid) and each step
    adds the frame farthest from everything already chosen, so the picks span the
    range of postures instead of clustering on whatever the animal did most
    often. Deterministic (ties take the earliest); returns sorted indices.
    """
    centroid = feats.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(feats - centroid, axis=1)))]
    dist = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        if dist[nxt] == 0:  # every remaining frame coincides with a chosen one
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
    return np.array(sorted(chosen))


def _subsample(
    n_frames: int,
    max_frames: int | None,
    strategy: str = "even",
    *,
    pts2d: Float[np.ndarray, "V T P 2"] | None = None,
    conf: Float[np.ndarray, "V T P"] | None = None,
) -> Int[np.ndarray, "F"]:
    """Pick at most ``max_frames`` frame indices to bundle-adjust on.

    ``max_frames=None`` (or a sequence already that short) keeps every frame.
    Otherwise ``strategy`` chooses which frames to keep:

    - ``"even"`` -- evenly spaced over the sequence (deterministic; the default).
      Guarantees temporal spread but is blind to detection quality.
    - ``"confidence"`` -- split the sequence into ``max_frames`` contiguous bins
      and keep the highest mean-confidence frame in each, trading a little spread
      for frames the detector is surest about. Needs ``conf`` (independent of
      whether bundle adjustment weighs residuals by confidence).
    - ``"coverage"`` -- same binning, but keep the frame with the most
      multi-view-observed (triangulable) keypoints, favouring well-conditioned
      frames (needs only ``pts2d``).
    - ``"diversity"`` -- pick frames whose 2D postures are maximally spread
      (farthest-point sampling over :func:`_frame_features`), so the target is
      seen in as many distinct configurations as possible rather than many
      near-duplicates of a common pose (needs only ``pts2d``).

    The binned ("stratified-best") ``confidence``/``coverage`` variants keep the
    temporal spread that makes ``"even"`` robust while preferring the better frame
    within each bin.
    """
    if strategy not in FRAME_SAMPLERS:
        raise ValueError(
            f"unknown frame_sampling {strategy!r}; choose from {list(FRAME_SAMPLERS)}"
        )
    if strategy == "confidence" and conf is None:
        raise ValueError("frame_sampling='confidence' needs detector confidences")
    if max_frames is None or n_frames <= max_frames:
        return np.arange(n_frames)
    if strategy == "even":
        return np.linspace(0, n_frames - 1, max_frames).round().astype(int)
    assert pts2d is not None  # the content-based scorers all read the 2D points
    if strategy == "diversity":
        return _farthest_first(_frame_features(pts2d), max_frames)
    return _best_per_bin(_frame_scores(strategy, pts2d, conf), max_frames)


def _bone_prior(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V F P 2"],
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Bone-pair indices and per-bone target lengths for flattened frames.

    Targets are the median bone length across frames from an initial
    triangulation -- a soft, robust prior (shared with the pictorial-structures
    corrector via :func:`deeperfly.pictorial.bone_length_targets`).

    Parameters
    ----------
    cameras
        Rig used for the initial triangulation that sets the target lengths.
    pts2d
        2D observations of shape ``(V, F, P, 2)`` (``F`` flattened frames).
    skeleton
        Skeleton supplying the bone (edge) list.

    Returns
    -------
    pairs : np.ndarray
        ``(B, 2)`` index pairs into the flattened ``F * P`` point axis.
    targets : np.ndarray
        Per-bone target lengths, tiled across the ``F`` frames.
    """
    n_frames, n_pts = pts2d.shape[1], pts2d.shape[2]
    i, j, targets = pictorial.bone_length_targets(cameras, pts2d, skeleton)
    offsets = (np.arange(n_frames) * n_pts)[:, None]
    pairs = np.stack([(offsets + i).ravel(), (offsets + j).ravel()], axis=1)
    return pairs, np.tile(targets, n_frames)


def bundle_adjust_cameras(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T P 2"],
    conf: Float[np.ndarray, "V T P"] | None = None,
    skeleton: Skeleton | None = None,
    *,
    ba_keypoints: Sequence[int] | None = None,
    fixed: list[str] = (),
    shared: list[list[str]] = (),
    weigh_by_confidence: bool = True,
    bone_prior: bool = True,
    bone_weight: float = 1.0,
    loss: str = "huber",
    f_scale: float = 40.0,
    max_frames: int | None = 100,
    frame_sampling: str = "even",
    max_nfev: int = 300,
    **solver_kwargs,
) -> tuple[CameraGroup, OptimizeResult]:
    """Refine ``cameras`` from a 2D sequence (the fly itself is the BA target).

    Frames are flattened into one point cloud; detector confidences become
    per-observation weights; a robust loss and an optional bone-length prior
    stabilize the fit.

    Parameters
    ----------
    cameras
        Initial camera rig to refine.
    pts2d
        2D observations of shape ``(V, T, P, 2)``, NaN for missing.
    conf
        Per-observation confidences ``(V, T, P)``, or ``None``. Used as residual
        weights when ``weigh_by_confidence`` and as the signal for
        ``frame_sampling="confidence"`` (the two uses are independent).
    skeleton
        Skeleton supplying the bone-length prior (used when ``bone_prior``).
    weigh_by_confidence
        Scale each reprojection residual by ``sqrt(confidence)`` when ``conf`` is
        given. Off still allows confidence-based *frame sampling*.
    ba_keypoints
        Skeleton point indices that drive the refinement -- observations of
        unselected points are masked out. ``None`` (default) uses every point;
        pass e.g. the leg-joint indices to bundle-adjust on the sharp limb corners
        alone. Only the camera fit is restricted; all points are still
        triangulated afterward by :func:`reconstruct`.
    fixed, shared
        Camera parameter groups held fixed or tied together to anchor the gauge,
        as in :func:`deeperfly.bundle_adjustment.bundle_adjust`.
    bone_prior
        Whether to add the soft bone-length prior to the residual.
    bone_weight, loss, f_scale, max_nfev
        Solver knobs forwarded to
        :func:`deeperfly.bundle_adjustment.bundle_adjust`.
    max_frames
        Subsample to at most this many frames before fitting; ``None`` uses
        every frame.
    frame_sampling
        Which frames to keep when subsampling (see :func:`_subsample`):
        ``"even"`` (evenly spaced, the default), ``"confidence"`` (the surest
        frame per time bin), ``"coverage"`` (the most multi-view-observed frame
        per time bin) or ``"diversity"`` (postures maximally spread apart).
    **solver_kwargs
        Extra keyword arguments forwarded to ``scipy.optimize.least_squares``.

    Returns
    -------
    cameras : CameraGroup
        The refined rig.
    result : scipy.optimize.OptimizeResult
        The raw scipy least-squares result.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    n_views, n_frames, n_pts = pts2d.shape[:3]
    sel = _subsample(n_frames, max_frames, frame_sampling, pts2d=pts2d, conf=conf)
    p = pts2d[:, sel]  # (V, F, P, 2)
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
    if conf is not None and weigh_by_confidence:
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
    pts2d: Float[np.ndarray, "V T P 2"],
    *,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
    weights: Float[np.ndarray, "V T P"] | None = None,
) -> tuple[
    Float[np.ndarray, "T P 3"], Float[np.ndarray, "V T P 2"], Float[np.ndarray, "V T P"]
]:
    """Triangulate a sequence and greedily reject reprojection outliers.

    A gross 2D outlier inflates *every* view's reprojection error for that point,
    so thresholding all views would discard the good ones too. Instead each pass
    drops only the **single worst** view of each still-offending point (keeping at
    least two) and re-triangulates, removing outliers one at a time.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    pts2d
        2D observations of shape ``(V, T, P, 2)``, NaN for missing.
    reproj_threshold
        Per-view reprojection error (px) above which a view may be dropped.
    max_drops
        Maximum number of drop-and-retriangulate passes.
    weights
        Optional per-observation weights ``(V, T, P)`` for a confidence-weighted
        DLT; ``None`` (default) is plain DLT. The drop logic stays geometric
        (driven by reprojection error), so confidence only shapes how the kept
        views are combined.

    Returns
    -------
    pts3d : np.ndarray
        Triangulated points ``(T, P, 3)``.
    cleaned_pts2d : np.ndarray
        ``pts2d`` with dropped observations set to NaN ``(V, T, P, 2)``.
    reproj_error : np.ndarray
        Per-observation reprojection error ``(V, T, P)``.
    """
    pts2d = np.array(pts2d, dtype=float)
    n_views = pts2d.shape[0]
    for _ in range(max_drops):
        pts3d = triangulate(cameras, pts2d, weights)
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
    pts3d = triangulate(cameras, pts2d, weights)
    err = reprojection_error(cameras, pts3d, pts2d)
    return pts3d, pts2d, err


def reconstruct_ransac(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T P 2"],
    *,
    threshold: float = 15.0,
    min_inliers: int = 2,
    weights: Float[np.ndarray, "V T P"] | None = None,
) -> tuple[
    Float[np.ndarray, "T P 3"], Float[np.ndarray, "V T P 2"], Float[np.ndarray, "V T P"]
]:
    """Triangulate a sequence robustly via per-point RANSAC consensus.

    Unlike :func:`reconstruct`, which deletes the worst view from a contaminated
    fit, this builds each point from the *largest set of mutually consistent
    views* (:func:`deeperfly.triangulation.triangulate_ransac`), so a badly
    mislocated detection never enters the fit. NaN views never count as inliers.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    pts2d
        2D observations of shape ``(V, T, P, 2)``, NaN for missing.
    threshold
        Inlier reprojection threshold (px) for the consensus set.
    min_inliers
        Minimum number of agreeing views for a point to be triangulated.
    weights
        Optional per-observation weights ``(V, T, P)``. Passed through to
        :func:`deeperfly.triangulation.triangulate_ransac` to weight the
        candidate fits and final refit; consensus scoring stays unweighted.

    Returns
    -------
    pts3d : np.ndarray
        Triangulated points ``(T, P, 3)``.
    cleaned_pts2d : np.ndarray
        ``pts2d`` with every non-inlier observation set to NaN, matching
        :func:`reconstruct`'s contract.
    reproj_error : np.ndarray
        Per-observation reprojection error ``(V, T, P)``.
    """
    pts2d = np.array(pts2d, dtype=float)
    pts3d, inliers = triangulate_ransac(
        cameras, pts2d, threshold=threshold, min_inliers=min_inliers, weights=weights
    )
    cleaned = np.where(inliers[..., None], pts2d, np.nan)
    err = reprojection_error(cameras, pts3d, cleaned)
    return pts3d, cleaned, err


#: Triangulation strategies for the reconstruction step.
_TRIANGULATORS = ("ransac", "greedy", "dlt")


def _validate_triangulation(triangulation: str) -> str:
    """Check that ``triangulation`` names a known strategy and return it.

    Parameters
    ----------
    triangulation
        One of :data:`_TRIANGULATORS` (``"ransac"``, ``"greedy"``, ``"dlt"``).

    Returns
    -------
    str
        The validated method name, unchanged.

    Raises
    ------
    ValueError
        If ``triangulation`` is not a known strategy.
    """
    if triangulation not in _TRIANGULATORS:
        raise ValueError(f"unknown triangulation {triangulation!r} (ransac|greedy|dlt)")
    return triangulation


def run_from_points2d(
    cameras: CameraGroup,
    skeleton: Skeleton,
    pts2d: Float[np.ndarray, "V T P 2"],
    conf: Float[np.ndarray, "V T P"] | None = None,
    *,
    do_bundle_adjust: bool = True,
    bundle_adjust_kwargs: dict | None = None,
    triangulation: str = "ransac",
    weigh_by_confidence: bool = False,
    do_pictorial: bool = False,
    candidates: pictorial.Candidates | None = None,
    ps_kwargs: dict | None = None,
    ransac_threshold: float = 15.0,
    min_inliers: int = 2,
    reproj_threshold: float = 40.0,
    max_drops: int = 5,
    fps: float = 100.0,
    meta: dict | None = None,
) -> PoseResult:
    """Run the full 2D-to-3D pipeline and return a :class:`PoseResult`.

    Steps: (optional) bundle-adjust cameras -> reconstruct 3D. Unobserved points are
    expected to already be NaN (the detector's pathway scatter leaves them so).

    Parameters
    ----------
    cameras
        The camera rig (refined in place when ``do_bundle_adjust``).
    skeleton
        Skeleton used for the bone-length prior.
    pts2d
        Detector 2D observations of shape ``(V, T, P, 2)``, NaN for missing.
    conf
        Per-observation confidences ``(V, T, P)``, or ``None``.
    do_bundle_adjust
        Whether to refine the cameras with bundle adjustment first.
    bundle_adjust_kwargs
        Extra keyword arguments forwarded to :func:`bundle_adjust_cameras`.
    triangulation
        Reconstruction strategy: ``"ransac"`` (default, largest multi-view
        consensus set; ``ransac_threshold`` / ``min_inliers``), ``"greedy"`` (DLT
        dropping the worst-reprojecting view; ``reproj_threshold`` / ``max_drops``),
        or ``"dlt"`` (plain least squares, no outlier handling).
    weigh_by_confidence
        When ``True`` and ``conf`` is given, the chosen triangulation uses a
        confidence-weighted DLT (each view's rows scaled by ``sqrt(conf)``). For
        ``"ransac"`` this weights the candidate fits and final refit but not the
        consensus vote. Default ``False`` (uniform weights).
    do_pictorial
        When ``True``, first run pictorial-structures peak recovery over the
        detector's top-K ``candidates`` (:func:`deeperfly.pictorial.reconstruct`,
        accepting ``ps_kwargs`` like ``temporal`` / ``lam`` / ``max_hyp``), then
        feed its committed 2D into ``triangulation`` (``"dlt"`` keeps the PS
        estimate). Bundle adjustment always uses the arg-max ``pts2d``.
    candidates
        The detector's top-K candidate peaks; required when ``do_pictorial``.
    ps_kwargs
        Extra keyword arguments forwarded to the pictorial-structures corrector.
    ransac_threshold, min_inliers, reproj_threshold, max_drops
        Per-strategy triangulation knobs (see ``triangulation`` above).
    fps
        The recording's frame rate, recorded in the result ``meta``.
    meta
        Extra key/value pairs merged into the result ``meta``.

    Returns
    -------
    PoseResult
        The bundle-adjusted cameras, committed 2D, triangulated 3D and diagnostics.

    Raises
    ------
    ValueError
        If ``do_pictorial`` is set but no ``candidates`` are given, or
        ``triangulation`` is unknown.
    """
    method = _validate_triangulation(triangulation)  # validate before bundle-adjusting
    # Unobserved (view, point) pairs are NaN (the detector's pathway scatter leaves
    # them so), which the bundle adjustment and triangulation below treat as "not seen".
    pts2d = np.asarray(pts2d, dtype=float)

    if do_bundle_adjust:
        cameras, _ = bundle_adjust_cameras(
            cameras, pts2d, conf, skeleton, **(bundle_adjust_kwargs or {})
        )

    if do_pictorial:
        if candidates is None:
            raise ValueError("do_pictorial=True requires candidates=...")
        # PS recovers the right peaks; its committed per-view 2D then feeds the
        # triangulator below (a plain "dlt" pass reproduces the PS estimate).
        pts3d, pts2d, reproj = pictorial.reconstruct(
            cameras, skeleton, candidates, pts2d, **(ps_kwargs or {})
        )

    weights = conf if (weigh_by_confidence and conf is not None) else None
    if method == "ransac":
        pts3d, pts2d, reproj = reconstruct_ransac(
            cameras,
            pts2d,
            threshold=ransac_threshold,
            min_inliers=min_inliers,
            weights=weights,
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            cameras,
            pts2d,
            reproj_threshold=reproj_threshold,
            max_drops=max_drops,
            weights=weights,
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts3d = triangulate(cameras, pts2d, weights)
        reproj = reprojection_error(cameras, pts3d, pts2d)

    return PoseResult(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        reproj_error=reproj,
        meta={
            "fps": fps,
            "triangulation": method,
            "pictorial": do_pictorial,
            **(meta or {}),
        },
    )
