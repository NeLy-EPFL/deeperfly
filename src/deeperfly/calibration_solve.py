"""Solve a camera rig from hand labels -- the from-scratch calibration.

The pipeline's normal order is *cameras -> detect -> triangulate -> label*. A project
started from scratch has to run it backwards: *label -> calibrate -> detect*. This module
is that inversion, in five steps.

.. code-block:: text

    1  assemble    hand labels          -> observations, one 3D unknown per "track"
    2  gate        conditioning check   -> refuse, with the reason, before solving
    3  initialize  orbit prior, else incremental SfM (essential matrix + PnP)
    4  bundle      deeperfly.bundle_adjustment, gauge-fixed, robust loss
    5  report      per-camera residuals, co-visibility, scatter -> accept or discard

**What a "track" is.** One 3D unknown: a skeleton keypoint at one frame, because the animal
moved between frames. Every track is one, which is why ``bundle_adjust``'s ``(V, N, 2)``
shape -- one observation per ``(view, track)`` -- needs no collapsing.

**The animal is the calibration target.** Dedicated static landmarks went in 0.3.0: no
project ever had any, so every rig solved here was already animal-as-target. What they
would have bought is conditioning -- a static point spread through the scene volume, where
keypoints sit in a ~3 mm blob near the middle of the field. What they were *also* claimed
to buy, equation efficiency, was never binding on this rig: a track seen in V views
contributes ``2V`` equations against 3 unknowns, which at ``V = 8`` is 5.33x against a
:data:`MIN_EQUATION_RATIO` of 1.5.

**What this refuses to do.** It will not invent intrinsics. Extrinsics are recoverable from
correspondences; focal length essentially is not, from a few hundred hand labels on a 3 mm
deforming animal -- bundle adjustment will happily trade focal error against depth and
report a beautiful residual. So intrinsics come in from outside (a board solve, or the lens
datasheet), the *source* is recorded in the calibration's provenance, and a guessed focal is
badged as such everywhere it surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float

from .cameras import CameraGroup

__all__ = [
    "Observations",
    "Track",
    "build_observations",
    "merge_observations",
    "conditioning",
    "covisibility",
    "initialize_extrinsics",
    "solve_rig",
    "SolveResult",
]

log = logging.getLogger("deeperfly")

#: A track needs at least this many observing views to constrain a 3D point at all.
MIN_VIEWS_PER_TRACK = 2

#: Observations-to-unknowns ratio below which the problem is under-determined enough that a
#: solution would be fitting noise. Not a guarantee at or above it -- a *necessary* check.
MIN_EQUATION_RATIO = 1.5

#: Pairwise baseline (degrees, at the scene centroid) below which two views are effectively
#: one: their essential matrix is near-degenerate and triangulation from them is unstable.
MIN_BASELINE_DEG = 5.0


@dataclass(frozen=True)
class Track:
    """One 3D unknown and where it came from.

    ``label`` is human-facing and appears in the report, so a bad residual names something
    the operator can navigate to. ``kind`` and ``static`` are gone with the landmarks:
    every track is one keypoint at one frame, so both were constants.
    """

    label: str
    recording: str | None = None
    frame: int | None = None
    n_observations: int = 0


@dataclass
class Observations:
    """The assembled solve input: ``(V, N, 2)`` pixels plus what each track is."""

    pts2d: Float[np.ndarray, "V N 2"]
    tracks: list[Track] = field(default_factory=list)
    view_names: list[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return self.pts2d.shape[0]

    @property
    def n_tracks(self) -> int:
        return self.pts2d.shape[1]

    @property
    def observed(self) -> Bool[np.ndarray, "V N"]:
        return np.isfinite(self.pts2d).all(axis=-1)

    @property
    def n_observations(self) -> int:
        return int(self.observed.sum())

    def summary(self) -> dict:
        """Counts for the provenance/report blocks."""
        obs = self.observed
        return {
            "views": self.n_views,
            "tracks": self.n_tracks,
            "observations": int(obs.sum()),
            "keypoint_tracks": self.n_tracks,
            "frames": len({t.frame for t in self.tracks if t.frame is not None}),
        }


# -- step 1: assemble ----------------------------------------------------------


def build_observations(
    *,
    view_names: list[str],
    keypoint_xy=None,
    keypoint_names=None,
    frames=None,
    recording: str | None = None,
) -> Observations:
    """Turn one recording's hand labels into tracks.

    Parameters
    ----------
    view_names
        The ``V``-axis camera names.
    keypoint_xy
        ``(V, T, P, 2)`` ground-truth keypoint pixels (NaN where unlabeled), or ``None``.
    keypoint_names
        The ``P``-axis point names (for track labels).
    frames
        Which frame indices are eligible. ``None`` = all. The caller restricts this to
        *reviewed* frames by default: a half-labeled frame contributes a systematically
        biased point, and no residual can reveal that.
    recording
        The recording's name, recorded on each track for the per-recording residual
        breakdown.

    Returns
    -------
    Observations
        The tracks with at least :data:`MIN_VIEWS_PER_TRACK` observing views. Tracks with
        fewer are dropped here rather than passed to the solver, where they would add
        three unknowns and constrain nothing.
    """
    n_views = len(view_names)
    columns: list[np.ndarray] = []
    tracks: list[Track] = []

    if keypoint_xy is not None:
        kp = np.asarray(keypoint_xy, dtype=float)
        names = list(keypoint_names or [f"point{i}" for i in range(kp.shape[2])])
        rows = np.nonzero(_frame_mask(kp.shape[1], frames))[0]
        for t in rows:
            for p, name in enumerate(names):
                col = kp[:, t, p, :]  # (V, 2)
                if np.isfinite(col).all(axis=-1).sum() < MIN_VIEWS_PER_TRACK:
                    continue
                columns.append(col)
                tracks.append(
                    Track(
                        label=f"{name}@{int(t)}",
                        recording=recording,
                        frame=int(t),
                        n_observations=int(np.isfinite(col).all(axis=-1).sum()),
                    )
                )

    if not columns:
        return Observations(np.zeros((n_views, 0, 2)), [], list(view_names))

    pts2d = np.stack(columns, axis=1)  # (V, N, 2)
    keep = np.isfinite(pts2d).all(axis=-1).sum(axis=0) >= MIN_VIEWS_PER_TRACK
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info(
            "dropped %d track(s) seen by fewer than %d views (they would add unknowns "
            "and constrain nothing)",
            n_dropped,
            MIN_VIEWS_PER_TRACK,
        )
    return Observations(
        pts2d[:, keep], [t for t, k in zip(tracks, keep) if k], list(view_names)
    )


def _frame_mask(n_frames: int, frames) -> np.ndarray:
    """``(T,)`` bool of eligible frames (``frames=None`` -> all)."""
    if frames is None:
        return np.ones(n_frames, dtype=bool)
    mask = np.zeros(n_frames, dtype=bool)
    idx = np.asarray(list(frames), dtype=int)
    mask[idx[(idx >= 0) & (idx < n_frames)]] = True
    return mask


def merge_observations(per_recording: list[Observations]) -> Observations:
    """Combine several recordings' observations into one solve, by concatenation.

    Plain concatenation, which is all a multi-recording solve is now. The rig-scoped
    cross-recording tie went with the static landmarks: it merged the recordings' views of
    one named static point into a single track whose per-view observation was the mean
    across sessions. That was the strongest constraint available -- it made several
    recordings one rigid problem -- and the most dangerous, because it is silently wrong the
    moment the rig is bumped between sessions; the merged track's cross-recording spread
    was what revealed that. With no static point to share, there is nothing to tie and
    nothing to be silently wrong about, and the per-recording residual breakdown in the
    report stays.

    Parameters
    ----------
    per_recording
        One :class:`Observations` per recording. All must share the same view order.

    Returns
    -------
    Observations
        The merged problem.

    Raises
    ------
    ValueError
        If the recordings disagree on their view names -- the ``V`` axis is positional, so
        merging mismatched orders would silently transpose cameras.
    """
    per_recording = [o for o in per_recording if o.n_tracks]
    if not per_recording:
        return Observations(np.zeros((0, 0, 2)), [], [])
    views = per_recording[0].view_names
    for other in per_recording[1:]:
        if other.view_names != views:
            raise ValueError(
                "cannot merge recordings with different view names/order "
                f"({views} vs {other.view_names}) -- the view axis is positional"
            )

    columns = [obs.pts2d[:, i] for obs in per_recording for i in range(obs.n_tracks)]
    tracks = [t for obs in per_recording for t in obs.tracks]
    pts2d = np.stack(columns, axis=1)
    keep = np.isfinite(pts2d).all(axis=-1).sum(axis=0) >= MIN_VIEWS_PER_TRACK
    return Observations(
        pts2d[:, keep], [t for t, k in zip(tracks, keep) if k], list(views)
    )


# -- step 2: the gate ----------------------------------------------------------


def covisibility(obs: Observations) -> np.ndarray:
    """``(V, V)`` count of tracks each pair of views both observe.

    The diagonal is each view's own track count. This is the matrix that reveals the
    failure mode of a camera ring: left and right views may share *no* tracks at all, with
    a single front view as the only bridge, so the rig is solvable only through it.
    """
    seen = obs.observed  # (V, N)
    return seen.astype(np.int64) @ seen.astype(np.int64).T


def conditioning(
    obs: Observations, *, free_focal: bool = False, free_k1: bool = False
) -> dict:
    """Whether this observation set can determine a rig -- and if not, why not.

    Runs *before* any solve, because the failure mode being guarded against is not a
    crash: an under-determined bundle adjustment converges to something plausible-looking
    with a small residual, and that rig then quietly misprojects every point downstream.

    The checks:

    - every view observes at least one track (a view with none cannot be placed);
    - the co-visibility graph is **connected** -- otherwise the rig is two rigs with no
      shared geometry, and their relative pose is unknowable no matter how many labels
      each half has;
    - ``observations >= MIN_EQUATION_RATIO x unknowns``.

    Parameters
    ----------
    obs
        The assembled observations.
    free_focal, free_k1
        Whether the solve will free each camera's focal length / first distortion
        coefficient (each adds unknowns, so each raises the bar).

    Returns
    -------
    dict
        ``ok`` plus ``reasons`` (why not), ``unknowns``, ``equations``, ``ratio``,
        ``components`` (the co-visibility components, as view-name lists), ``weakest_pair``
        and ``per_view_tracks``.
    """
    n_views, n_tracks = obs.n_views, obs.n_tracks
    seen = obs.observed
    per_view = seen.sum(axis=1)
    # 6 extrinsic DOF per camera except the one that defines the world frame.
    unknowns = 6 * max(0, n_views - 1) + 3 * n_tracks
    if free_focal:
        unknowns += n_views
    if free_k1:
        unknowns += n_views
    equations = 2 * obs.n_observations
    ratio = float(equations / unknowns) if unknowns else 0.0

    co = covisibility(obs)
    components = _components(co, n_views)
    reasons: list[str] = []

    blind = [obs.view_names[v] for v in range(n_views) if per_view[v] == 0]
    if blind:
        reasons.append(
            f"view(s) {blind} observe no tracks at all, so they cannot be placed -- "
            "label some points in them"
        )
    if len(components) > 1:
        groups = [
            "{" + ", ".join(obs.view_names[v] for v in comp) + "}"
            for comp in components
        ]
        reasons.append(
            "the co-visibility graph is disconnected: "
            + " vs ".join(groups)
            + " share no tracks, so their relative pose is unknowable. Label the same "
            "point in a view from each group -- one keypoint labeled in a camera from "
            "each is enough to connect them"
        )
    if ratio < MIN_EQUATION_RATIO:
        reasons.append(
            f"only {equations} equations for {unknowns} unknowns ({ratio:.2f}x, need "
            f"{MIN_EQUATION_RATIO}x) -- label more frames, or label more points per "
            "frame: every track seen in V views contributes 2V equations against 3 "
            "unknowns"
        )

    weakest = _weakest_pair(co, obs.view_names)
    return {
        "ok": not reasons,
        "reasons": reasons,
        "unknowns": int(unknowns),
        "equations": int(equations),
        "ratio": ratio,
        "components": [[obs.view_names[v] for v in comp] for comp in components],
        "covisibility": co.tolist(),
        "weakest_pair": weakest,
        "per_view_tracks": {
            obs.view_names[v]: int(per_view[v]) for v in range(n_views)
        },
    }


def _components(co: np.ndarray, n_views: int) -> list[list[int]]:
    """Connected components of the co-visibility graph (edge = >=1 shared track)."""
    adjacency = [
        {j for j in range(n_views) if j != i and co[i, j] > 0} for i in range(n_views)
    ]
    unseen = set(range(n_views))
    out: list[list[int]] = []
    while unseen:
        stack = [unseen.pop()]
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adjacency[v] & unseen:
                unseen.discard(w)
                stack.append(w)
        out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def _weakest_pair(co: np.ndarray, names: list[str]) -> dict | None:
    """The connected view pair sharing the fewest tracks -- where labeling helps most."""
    best = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = int(co[i, j])
            if n == 0:
                continue
            if best is None or n < best["shared"]:
                best = {"views": [names[i], names[j]], "shared": n}
    return best


# -- step 3: initialize --------------------------------------------------------


def initialize_extrinsics(
    obs: Observations, intrinsics: np.ndarray, dists: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Incremental structure-from-motion to seed the bundle adjustment.

    Only needed when there is no prior rig. Every rig deeperfly has shipped with is a
    hand-specified orbit, and that orbit *is* a perfectly good initialization -- so the
    caller passes it straight through and never reaches here. This path is for a genuinely
    new rig.

    The steps are textbook incremental SfM, with the choices that matter for a small
    animal rig called out:

    1. Seed pair: the pair maximizing ``shared_tracks * median_normalized_disparity``. The
       disparity factor matters -- a pair with 200 shared tracks and a 2 degree baseline is
       worthless, and picking by track count alone walks straight into it.
    2. Essential matrix (RANSAC) on normalized coordinates, then ``recoverPose`` for the
       chirality-correct decomposition. Translation is unit-norm; scale comes later.
    3. Triangulate the seed pair's shared tracks.
    4. Register each remaining view by PnP against the growing cloud, in descending order
       of available 2D-3D correspondences, re-triangulating as the cloud grows.

    Parameters
    ----------
    obs
        The observations.
    intrinsics
        ``(V, 4)`` packed ``[fx, fy, cx, cy]`` -- required, and never solved for here
        (see the module docstring).
    dists
        ``(V, K)`` distortion coefficients.

    Returns
    -------
    rvecs, tvecs : np.ndarray
        ``(V, 3)`` each. A view that could not be registered gets NaN, which the caller
        must treat as a failure rather than a starting point.
    report : dict
        ``seed_pair``, ``registration`` order with inlier counts, and ``failed`` views.
    """
    import cv2

    n_views = obs.n_views
    seen = obs.observed
    normalized = _normalize_pixels(obs.pts2d, intrinsics, dists)

    seed = _seed_pair(obs, normalized)
    if seed is None:
        return (
            np.full((n_views, 3), np.nan),
            np.full((n_views, 3), np.nan),
            {"seed_pair": None, "registration": [], "failed": list(obs.view_names)},
        )
    i, j, score = seed
    both = seen[i] & seen[j]
    pts_i = normalized[i, both].astype(np.float64)
    pts_j = normalized[j, both].astype(np.float64)

    essential, mask = cv2.findEssentialMat(
        pts_i, pts_j, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1e-3
    )
    if essential is None:
        return (
            np.full((n_views, 3), np.nan),
            np.full((n_views, 3), np.nan),
            {
                "seed_pair": [obs.view_names[i], obs.view_names[j]],
                "registration": [],
                "failed": list(obs.view_names),
                "error": "the essential matrix could not be estimated for the seed pair",
            },
        )
    _, rmat, tvec, pose_mask = cv2.recoverPose(
        essential, pts_i, pts_j, np.eye(3), mask=mask
    )

    rvecs = np.full((n_views, 3), np.nan)
    tvecs = np.full((n_views, 3), np.nan)
    rvecs[i] = 0.0  # the seed's first view IS the world frame
    tvecs[i] = 0.0
    rvecs[j] = cv2.Rodrigues(rmat)[0].reshape(3)
    tvecs[j] = np.asarray(tvec, dtype=float).reshape(3)

    registration = [
        {
            "view": obs.view_names[j],
            "via": "essential",
            "inliers": int(np.count_nonzero(pose_mask)),
            "shared": int(both.sum()),
        }
    ]
    cloud = _triangulate_pair(normalized, i, j, rvecs, tvecs, both)

    placed = {i, j}
    while len(placed) < n_views:
        candidate = _next_view(obs, cloud, placed)
        if candidate is None:
            break
        v, usable = candidate
        ok, rvec, tvec_v, inliers = _pnp(normalized[v], cloud, usable)
        if not ok:
            log.warning(
                "could not register view %s by PnP (%d correspondences); it will have no "
                "initial pose",
                obs.view_names[v],
                int(usable.sum()),
            )
            placed.add(v)  # do not retry forever
            registration.append(
                {
                    "view": obs.view_names[v],
                    "via": "pnp",
                    "inliers": 0,
                    "shared": int(usable.sum()),
                    "failed": True,
                }
            )
            continue
        rvecs[v], tvecs[v] = rvec, tvec_v
        placed.add(v)
        registration.append(
            {
                "view": obs.view_names[v],
                "via": "pnp",
                "inliers": int(inliers),
                "shared": int(usable.sum()),
            }
        )
        cloud = _grow_cloud(obs, normalized, rvecs, tvecs, cloud, placed)

    failed = [
        obs.view_names[v] for v in range(n_views) if not np.isfinite(rvecs[v]).all()
    ]
    return (
        rvecs,
        tvecs,
        {
            "seed_pair": [obs.view_names[i], obs.view_names[j]],
            "seed_score": float(score),
            "registration": registration,
            "failed": failed,
        },
    )


def _normalize_pixels(pts2d, intrinsics, dists) -> np.ndarray:
    """Pixels -> normalized camera coordinates, undistorted, per view.

    Undistorting *here* rather than inside the solve keeps the essential-matrix step on
    the pinhole model it assumes; the bundle adjustment afterwards works on raw pixels with
    the full forward model, so no approximation survives into the final rig.
    """
    import cv2

    out = np.full_like(np.asarray(pts2d, dtype=float), np.nan)
    for v in range(out.shape[0]):
        fx, fy, cx, cy = (float(x) for x in np.asarray(intrinsics)[v][:4])
        kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray(dists)[v].astype(np.float64).reshape(1, -1)
        finite = np.isfinite(pts2d[v]).all(axis=-1)
        if not finite.any():
            continue
        pts = np.asarray(pts2d[v][finite], dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, kmat, dist if dist.size else None)
        out[v, finite] = und.reshape(-1, 2)
    return out


def _seed_pair(obs: Observations, normalized: np.ndarray):
    """``(i, j, score)`` for the best-conditioned starting pair, or ``None``."""
    seen = obs.observed
    best = None
    for i in range(obs.n_views):
        for j in range(i + 1, obs.n_views):
            both = seen[i] & seen[j]
            n = int(both.sum())
            if n < 6:  # the essential matrix needs 5; 6 leaves room for one outlier
                continue
            # Median disparity in normalized coordinates: a proxy for baseline that needs
            # no pose. A wide pair separates the same points further.
            disparity = float(
                np.median(
                    np.linalg.norm(normalized[i, both] - normalized[j, both], axis=-1)
                )
            )
            score = n * disparity
            if best is None or score > best[2]:
                best = (i, j, score)
    return best


def _triangulate_pair(normalized, i, j, rvecs, tvecs, mask) -> np.ndarray:
    """Triangulate the tracks ``mask`` selects from views ``i`` and ``j``."""
    import cv2

    p_i = np.hstack([cv2.Rodrigues(rvecs[i])[0], tvecs[i].reshape(3, 1)])
    p_j = np.hstack([cv2.Rodrigues(rvecs[j])[0], tvecs[j].reshape(3, 1)])
    cloud = np.full((normalized.shape[1], 3), np.nan)
    idx = np.nonzero(mask)[0]
    if idx.size:
        homogeneous = cv2.triangulatePoints(
            p_i, p_j, normalized[i, idx].T, normalized[j, idx].T
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            cloud[idx] = (homogeneous[:3] / homogeneous[3]).T
    return cloud


def _next_view(obs: Observations, cloud: np.ndarray, placed: set):
    """The unplaced view with the most 2D-3D correspondences, or ``None``."""
    have3d = np.isfinite(cloud).all(axis=-1)
    best = None
    for v in range(obs.n_views):
        if v in placed:
            continue
        usable = obs.observed[v] & have3d
        n = int(usable.sum())
        if n < 6:  # solvePnPRansac needs at least 6 for a stable fit
            continue
        if best is None or n > int(best[1].sum()):
            best = (v, usable)
    return best


def _pnp(normalized_view, cloud, usable):
    """Register one view against the cloud. Returns ``(ok, rvec, tvec, n_inliers)``."""
    import cv2

    pts3d = cloud[usable].astype(np.float64)
    pts2d = normalized_view[usable].astype(np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d,
        pts2d,
        np.eye(3),  # already normalized coordinates
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=8e-3,
        confidence=0.999,
    )
    if not ok:
        return False, None, None, 0
    return (
        True,
        np.asarray(rvec, dtype=float).reshape(3),
        np.asarray(tvec, dtype=float).reshape(3),
        0 if inliers is None else len(inliers),
    )


def _grow_cloud(obs, normalized, rvecs, tvecs, cloud, placed) -> np.ndarray:
    """Triangulate tracks that two placed views now share but the cloud still lacks."""
    have3d = np.isfinite(cloud).all(axis=-1)
    placed_list = sorted(placed)
    for a_i in range(len(placed_list)):
        for b_i in range(a_i + 1, len(placed_list)):
            a, b = placed_list[a_i], placed_list[b_i]
            if not (np.isfinite(rvecs[a]).all() and np.isfinite(rvecs[b]).all()):
                continue
            fresh = obs.observed[a] & obs.observed[b] & ~have3d
            if not fresh.any():
                continue
            added = _triangulate_pair(normalized, a, b, rvecs, tvecs, fresh)
            cloud = np.where(np.isfinite(added), added, cloud)
            have3d = np.isfinite(cloud).all(axis=-1)
    return cloud


# -- steps 4 + 5: solve and report --------------------------------------------


@dataclass
class SolveResult:
    """A solved rig plus everything needed to decide whether to accept it."""

    cameras: CameraGroup
    pts3d: Float[np.ndarray, "N 3"]
    quality: dict
    report: dict
    ok: bool = True


def solve_rig(
    obs: Observations,
    *,
    intrinsics: np.ndarray,
    dists: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    free_focal: bool = False,
    free_k1: bool = False,
    cold_start: bool = False,
    loss: str = "cauchy",
    f_scale: float = 4.0,
    max_nfev: int = 800,
) -> SolveResult:
    """Bundle-adjust the rig, with the gauge fixed.

    Scale is NOT pinned here, and cannot be: images cannot determine it -- a rig twice as
    large viewing a fly twice as large produces pixel-identical images -- so a
    correspondence-only solve leaves it free and the result is honestly
    ``units = "arbitrary"``. Physical scale enters at inverse kinematics, where
    ``body_scale`` fits the point cloud to the fitted model's defined dimensions. The
    ``scale_pair`` / ``scale_distance`` pair that used to pin it from a measured
    landmark-to-landmark distance went with the landmarks.

    Reuses :func:`deeperfly.bundle_adjustment.bundle_adjust` unchanged.
    ``fixed=["<view0>.rvec", "<view0>.tvec"]`` nails six of the seven gauge freedoms by
    making the first view the world frame; the seventh -- scale -- is left to the solver,
    which has no reason to move along it, so the rig stays at whatever scale the
    initialization implied and the caller records ``units = "arbitrary"``.

    Parameters
    ----------
    obs
        The observations.
    intrinsics, dists
        ``(V, 4)`` and ``(V, K)`` starting intrinsics.
    rvecs, tvecs
        ``(V, 3)`` initial extrinsics (from an orbit prior or
        :func:`initialize_extrinsics`).
    free_focal, free_k1
        Whether to let the solver adjust focal length / ``k1``. Both default off: with
        sparse hand labels they are near-unidentifiable and absorb real extrinsic error.
    loss, f_scale, max_nfev
        Robust loss settings forwarded to the solver. ``cauchy`` at 4 px is what this lab
        measured as the setting that helps a hard rig rather than one that flatters it.

    Returns
    -------
    SolveResult
        The rig, the triangulated tracks, the quality block and the report.
    """
    # The low-level solver over a packed state, not the package's config-driven wrapper:
    # this caller owns the state (its "points" are tracks, not a skeleton) and needs the
    # bone-length prior as a scale bar rather than as an anatomical constraint.
    from .bundle_adjustment.core import bundle_adjust
    from .bundle_adjustment.state import build_state
    from .triangulation import reprojection_error

    names = list(obs.view_names)
    fixed = [f"{names[0]}.rvec", f"{names[0]}.tvec"]
    if not free_focal:
        fixed.append("*.intr")
    if not free_k1:
        fixed.append("*.dist")

    rvecs = np.asarray(rvecs, dtype=float)
    tvecs = np.asarray(tvecs, dtype=float)
    scale_applied = 1.0
    if cold_start:
        # Only for a reconstruction with no prior. Both steps change the *gauge*, which is
        # free there and meaningful otherwise: an orbit prior's scale is what a
        # `units = "config"` calibration records, so rescaling it would make that file lie.
        rvecs, tvecs = _rebase_to_first_view(rvecs, tvecs, names)
        rvecs, tvecs, pts3d_init, scale_applied = _normalize_initial_scale(
            rvecs, tvecs, obs, intrinsics, dists
        )
    else:
        seed = CameraGroup.from_arrays(names, rvecs, tvecs, intrinsics, dists)
        pts3d_init = np.nan_to_num(np.asarray(seed.triangulate(obs.pts2d), dtype=float))
    # Applied either way: a *free* camera parked exactly on the Rodrigues singularity
    # contributes a NaN Jacobian column and kills the first trust-region step, and a caller
    # can hand one in whatever the path (see _RVEC_SINGULARITY).
    rvecs = _nudge_singular_rvecs(rvecs)
    state = build_state(
        rvecs=np.asarray(rvecs, dtype=float),
        tvecs=np.asarray(tvecs, dtype=float),
        intrs=np.asarray(intrinsics, dtype=float),
        dists=np.asarray(dists, dtype=float),
        pts2d=np.asarray(obs.pts2d, dtype=float),
        names=names,
        fixed=fixed,
        pts3d=np.asarray(pts3d_init, dtype=float),
    )

    result, solution = bundle_adjust(
        state.values,
        state.fixed,
        state.rvecs_idx,
        state.tvecs_idx,
        state.intrs_idx,
        state.dists_idx,
        state.pts3d_idx,
        state.pts2d,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        # Jacobian-based variable scaling. Not optional here: a reprojection derivative
        # with respect to translation scales as fx/Z, so on a long-focal rig (this one is
        # ~22000 px) the translation columns are thousands of times larger than the
        # rotation ones, and an unscaled trust-region step overshoots into non-finite
        # values on the first iteration. Measured: keypoint-only solves diverged to NaN
        # without this and converge with it.
        x_scale="jac",
    )

    cameras = CameraGroup.from_arrays(
        names, solution.rvecs, solution.tvecs, solution.intrs, solution.dists
    )
    err = reprojection_error(cameras, solution.pts3d, obs.pts2d)
    from .calibration import quality_from_errors

    quality = quality_from_errors(err[:, None, :], names)
    quality["gauge"] = {
        "fixed": fixed,
        "scale_fixed_by": "none",  # images cannot determine scale; see the docstring
        "init_rescaled_by": scale_applied,
    }
    return SolveResult(
        cameras=cameras,
        pts3d=np.asarray(solution.pts3d, dtype=float),
        quality=quality,
        report={
            "solver": {
                "loss": loss,
                "f_scale": f_scale,
                "max_nfev": max_nfev,
                "nfev": int(result.nfev),
                "success": bool(result.success),
                "status": int(result.status),
                "cost": float(result.cost),
            },
            "observations": obs.summary(),
            "per_track": _per_track_report(obs, err),
        },
        ok=bool(result.success),
    )


#: Below this rotation magnitude an axis-angle vector sits on the Rodrigues singularity.
#:
#: ``R(rvec)``'s derivative carries ``sin(theta)/theta`` and ``(1 - cos theta)/theta**2``
#: terms, which autodiff evaluates as 0/0 at ``theta == 0`` -- so a *free* camera whose
#: rvec is exactly zero contributes a **NaN Jacobian column** and the whole solve dies on
#: the first trust-region step. It is not a small-angle accuracy issue; it is a hard hole in
#: the parameterization at one point.
#:
#: This never arose before because every rig deeperfly shipped with is a hand-specified
#: orbit, and no orbit camera has an identity rotation. A cold-start SfM initialization puts
#: its reference view at exactly identity, which lands straight in the hole.
#:
#: Two defences, both applied: :func:`_rebase_to_first_view` moves the identity rotation
#: onto the view whose rvec is *fixed* (so it is never differentiated), and
#: :func:`_nudge_singular_rvecs` perturbs any *other* near-zero rvec off the singularity by
#: an amount far below the solve's own precision.
_RVEC_SINGULARITY = 1e-8


def _rebase_to_first_view(rvecs, tvecs, view_names):
    """Re-express an SfM reconstruction so view 0 is the world frame.

    The reconstruction's frame is arbitrary (a gauge freedom), and
    :func:`initialize_extrinsics` happens to put it on whichever view seeded the essential
    matrix. Moving it to view 0 aligns it with the gauge the solve *fixes*, which means the
    one camera carrying an identity rotation is also the one never differentiated -- see
    :data:`_RVEC_SINGULARITY`.

    Returns ``(rvecs, tvecs)`` describing the identical geometry in the new frame.
    """
    from .geometry import rmat_to_rvec, rvec_to_rmat

    rvecs = np.asarray(rvecs, dtype=float).copy()
    tvecs = np.asarray(tvecs, dtype=float).copy()
    if not (np.isfinite(rvecs[0]).all() and np.isfinite(tvecs[0]).all()):
        # View 0 was never registered; leave the frame alone rather than rebase onto NaN.
        return rvecs, tvecs
    r0 = np.asarray(rvec_to_rmat(rvecs[0]), dtype=float)
    # .copy() is load-bearing: `tvecs[0]` is a VIEW into the array, and the loop below
    # zeroes row 0 first. Without the copy every later view is rebased against a t0 that
    # has already become zero -- which leaves the geometry silently wrong (measured: a
    # perfect rig came out at 112915 px rms) rather than raising.
    t0 = np.array(tvecs[0], dtype=float, copy=True)
    for v in range(len(view_names)):
        if not np.isfinite(rvecs[v]).all():
            continue
        rv = np.asarray(rvec_to_rmat(rvecs[v]), dtype=float)
        # world' = R0 @ world + t0, so R_v' = R_v @ R0^T and t_v' = t_v - R_v' @ t0.
        rmat = rv @ r0.T
        rvecs[v] = np.asarray(rmat_to_rvec(rmat), dtype=float)
        tvecs[v] = tvecs[v] - rmat @ t0
    return rvecs, tvecs


def _nudge_singular_rvecs(rvecs, fixed_first: bool = True):
    """Perturb any free near-identity rotation off the Rodrigues singularity.

    View 0's rvec is expected to be exactly zero *and fixed*, so it is skipped. Any other
    view sitting at identity -- two cameras that happen to share an orientation, or a PnP
    solution that landed there -- is nudged by :data:`_RVEC_SINGULARITY`, which is orders of
    magnitude below the solve's own convergence tolerance and cannot affect the result.
    """
    rvecs = np.asarray(rvecs, dtype=float).copy()
    start = 1 if fixed_first else 0
    for v in range(start, len(rvecs)):
        if np.isfinite(rvecs[v]).all() and np.linalg.norm(rvecs[v]) < _RVEC_SINGULARITY:
            rvecs[v] = np.array([_RVEC_SINGULARITY, 0.0, 0.0])
            log.debug("nudged view %d off the Rodrigues singularity", v)
    return rvecs


def _normalize_initial_scale(rvecs, tvecs, obs, intrinsics, dists):
    """Rescale a cold-start reconstruction to a sane working distance.

    ``cv2.recoverPose`` returns a **unit** baseline, so an SfM initialization lands at
    whatever scale that implies -- on this rig, ~1/180 of the truth. Scale is a gauge
    freedom, so this changes no reprojection at all; what it changes is conditioning and
    legibility. It is done on top of ``x_scale="jac"`` rather than instead of it: the
    scaling fixes the solver's step, this fixes the *numbers a human reads* in the report
    and in the resulting calibration file, which would otherwise describe a rig 180 times
    too small with no indication why.

    The target is the focal length in pixels, which is the only length scale available
    without an external measurement -- an arbitrary but *stable* choice, and one that puts
    the scene at a distance where depth derivatives are O(1).

    Returns ``(rvecs, tvecs, pts3d, scale)`` with the scale actually applied (1.0 when
    nothing could be measured).
    """
    seed = CameraGroup.from_arrays(
        list(obs.view_names), rvecs, tvecs, intrinsics, dists
    )
    pts3d = np.asarray(seed.triangulate(obs.pts2d), dtype=float)
    finite = np.isfinite(pts3d).all(axis=-1)
    centers = np.stack([seed[n].position for n in obs.view_names])
    if not finite.any() or not np.isfinite(centers).all():
        return rvecs, tvecs, np.nan_to_num(pts3d), 1.0
    centroid = pts3d[finite].mean(axis=0)
    distance = float(np.median(np.linalg.norm(centers - centroid, axis=1)))
    if not np.isfinite(distance) or distance <= 0:
        return rvecs, tvecs, np.nan_to_num(pts3d), 1.0
    target = float(np.median(np.asarray(intrinsics, dtype=float)[:, :2]))
    scale = target / distance
    if not np.isfinite(scale) or scale <= 0:
        return rvecs, tvecs, np.nan_to_num(pts3d), 1.0
    # Scaling the world by s means t -> s*t (rotation is scale-free) and X -> s*X.
    log.info(
        "initial reconstruction rescaled by %.4g (median camera-scene distance %.4g -> "
        "%.4g); scale is a gauge freedom, so no reprojection changes",
        scale,
        distance,
        target,
    )
    return rvecs, tvecs * scale, np.nan_to_num(pts3d * scale), scale


def _per_track_report(obs: Observations, err: np.ndarray) -> list[dict]:
    """The worst-fitting tracks, named, so a bad residual is navigable.

    Capped at 20 rows and the cap is *stated* in the payload, because a silently truncated
    list reads as "these are all the problems".
    """
    finite = np.isfinite(err)
    per_track = np.where(
        finite.any(axis=0), np.nanmax(np.where(finite, err, np.nan), axis=0), np.nan
    )
    order = np.argsort(-np.nan_to_num(per_track, nan=-1.0))
    rows = [
        {
            "label": obs.tracks[i].label,
            "recording": obs.tracks[i].recording,
            "max_reproj_px": float(per_track[i]),
            "views": int(obs.observed[:, i].sum()),
        }
        for i in order[:20]
        if np.isfinite(per_track[i])
    ]
    return rows
