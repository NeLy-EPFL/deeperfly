"""Active-learning acquisition: which frames are worth a human's next pass.

Ranks the frames of one recording by **multi-view disagreement of the detector's
own 2D**: triangulate each joint from ``pose2d/points`` and measure how far each
view's detection sits from the reprojection of that fit. Views cannot conspire, so
a large residual means the model is probably wrong -- which is exactly where a
human label buys the most.

Three things this module deliberately does *not* do, each because the obvious
alternative was measured to be wrong on this project:

**It never scores confidence.** Of the ground-truth points the detector missed by
more than 25 px, the median heatmap-peak confidence was 0.678 and only 4% fell
below 0.30: the model is confidently wrong precisely where it is wrong.

**It never scores a substituted layer.** In a directory produced by
``dfpose.predict``, the contralateral cells of ``triangulation/points`` have been
replaced by the reprojection of that frame's 3D (a *display* seed), so their
reprojection residual is ~0 **by construction** -- and ``triangulation/reproj_error``
is computed against that substituted layer. Measured on the real reseeded file, the
stored error on those cells is median 0.00 px with 0.000 of them over 15 px, versus
median 8.34 px and 0.160 over threshold for the pristine ``pose2d/points``. A queue
built on the stored array therefore ranks by *nothing* while looking entirely
plausible. Hence :func:`score_frames` takes 2D arrays the caller read from
``pose2d`` via :class:`~deeperfly.results.StageStore`, and :func:`prepare_inputs`
is the only supported way to get them from a file -- never
:meth:`~deeperfly.results.PoseResult.load`, which prefers the most-derived (i.e.
substituted) layer.

**It never returns the raw top-N.** At 100 fps adjacent frames are near-duplicates
(lag-1 autocorrelation of the score is ~+0.6 at 40 ms, ~+0.1 at 2 s), so an
unspaced top-20 is one hard moment sampled twenty times. :func:`select_frames`
enforces a *hard* minimum temporal gap, seeds that constraint with the frames a
human already labeled, and reserves a fraction of the list for a uniform temporal
grid so the round still sees typical poses and not only the tail.

The selection is written next to ``results.h5`` as ``labels_suggest.json``
(:func:`write_suggestions`) and read back by the GUI (:func:`read_suggestions`,
:func:`suggestions_staleness`). It is JSON because it is a handful of nested,
human-facing records; the only file this module ever writes is that sidecar, and
``results.h5`` / ``labels.h5`` are opened read-only.

Score magnitudes are **not comparable across recordings** -- the level tracks how
many cells the detector fired (measured median 0.51 on a dense 38-channel file vs
0.16 on a legacy 19-channel one, same formula), so every score is reported with its
within-recording percentile and never aggregated across files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float

from .cameras import CameraGroup
from .triangulation import reprojection_error, triangulate_ransac

__all__ = [
    "SUGGESTIONS_FILENAME",
    "SUGGESTIONS_FORMAT_VERSION",
    "AcquisitionInputs",
    "FrameScores",
    "Pick",
    "frame_reason",
    "prepare_inputs",
    "read_labeled_frames",
    "read_suggestions",
    "score_frames",
    "select_frames",
    "suggestions_staleness",
    "write_suggestions",
]

log = logging.getLogger("deeperfly")

#: Bumped when the sidecar schema changes incompatibly; an unknown version reads
#: as *absent* (see :func:`read_suggestions`) rather than being mis-parsed.
SUGGESTIONS_FORMAT_VERSION = 1

#: The sidecar's name, beside ``results.h5``/``labels.h5`` (all three share the
#: ``labels_*`` prefix of ``labels-export``'s ``labels_gt.npz``).
SUGGESTIONS_FILENAME = "labels_suggest.json"

#: Fallback capture rate when the result carries no ``meta["fps"]`` (warned).
DEFAULT_FPS = 100.0

#: The array the score is computed from. Recorded in the sidecar so a reader can
#: verify the queue was not built from a substituted layer.
SCORED_ARRAY = "pose2d/points"

#: One-line statement of the score, stamped into the sidecar's ``params``.
SCORE_DESCRIPTION = (
    "mean of the top_k joints' mean-over-views clipped reprojection residual of "
    "RANSAC-triangulated pose2d/points (within-recording rank only; not comparable "
    "across recordings)"
)


# -- scoring ------------------------------------------------------------------


@dataclass
class FrameScores:
    """Per-frame disagreement scores plus everything needed to explain them.

    Attributes
    ----------
    score
        ``(T,)`` frame score in ``[0, 1]``: the mean of the ``top_k`` largest
        per-joint disagreements. Frames with nothing scorable score 0.
    percentile
        ``(T,)`` the within-recording percentile of ``score`` (0-100).
    joint
        ``(T, P)`` per-joint disagreement in ``[0, 1]``, ``NaN`` where the joint
        is not scorable (fewer than ``min_views`` observing views, or no selected
        view observed it). **A structural zero is uninformative, not safe**: a
        joint triangulated from exactly two views reprojects onto both by
        construction, which is why ``min_views`` defaults to 3.
    joint_px
        ``(T, P)`` the same quantity in raw pixels (mean over the joint's selected
        observing views), for reporting.
    resid
        ``(V, T, P)`` raw reprojection residual in pixels, ``NaN`` where the
        detector did not fire or the joint did not triangulate.
    pts3d
        ``(T, P, 3)`` the RANSAC triangulation the residuals are measured against.
    n_obs
        ``(T, P)`` how many views observed each joint (geometry support).
    coverage
        Summary statistics of what was scorable (see :func:`score_frames`).
    params
        The knobs the score was computed with.
    """

    score: Float[np.ndarray, "T"]
    percentile: Float[np.ndarray, "T"]
    joint: Float[np.ndarray, "T P"]
    joint_px: Float[np.ndarray, "T P"]
    resid: Float[np.ndarray, "V T P"]
    pts3d: Float[np.ndarray, "T P 3"]
    n_obs: np.ndarray
    coverage: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.score.shape[0])


def score_frames(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T P 2"],
    *,
    threshold: float = 15.0,
    cap: float = 60.0,
    top_k: int = 8,
    min_views: int = 3,
    point_mask: Bool[np.ndarray, "P"] | None = None,
    camera_mask: Bool[np.ndarray, "V"] | None = None,
    absent_mask: Bool[np.ndarray, "T P"] | None = None,
) -> FrameScores:
    """Score every frame by the multi-view disagreement of ``pts2d``.

    ``pts2d`` **must** be the pristine detector output (``pose2d/points``); see the
    module docstring for why a triangulation layer would silently score ~0.

    The pipeline is: RANSAC-triangulate each joint (exhaustive ``C(V, 2)``
    two-view consensus, so one blown view cannot drag the fit and hide which view
    was wrong), reproject, then aggregate the residual twice --

    1. per cell, ``d = min(resid, cap) / cap`` in ``[0, 1]``. Clipping is not
       cosmetic: measured raw residuals reach 2567 px on a blown view, so any
       unclipped aggregate is a single-outlier lottery. Rank correlation against
       hand labels is insensitive to ``cap`` over 30-120 px.
    2. per **joint** (the unit a human actually fixes), the *mean* over that
       joint's observing views -- not the max, which is dominated by the one
       blown view and rewards a lone outlier over a genuinely ambiguous joint.
    3. per **frame**, the mean of the ``top_k`` largest joint disagreements: a
       frame earns a human pass when *several* joints are wrong. A fraction-over-
       threshold aggregate was measured to take only 36-51 distinct values across
       ~1000 frames, leaving hundreds of exact ties for the spacing pass to break
       toward the start of the recording; the top-K mean takes ~1000 distinct
       values of ~1000 frames.

    ``absent_mask`` marks keypoints that are **not on this animal** (an amputated leg).
    It is kept separate from ``point_mask`` on purpose: ``point_mask`` is the operator's
    choice of what to score, while absence is a fact about the animal that the sidecar must
    state. It removes those points from the residuals, from the top-k aggregate, and from
    every coverage *denominator* -- otherwise a recording whose amputated leg can never be
    scored reports a permanently depressed ``scorable_joint_frac`` and its frame scores are
    diluted by columns that can never carry signal.

    ``point_mask`` / ``camera_mask`` restrict which cells are *scored*; the
    triangulation always uses every view, so narrowing the score never degrades
    the geometry it is measured against.

    Parameters
    ----------
    cameras
        The rig -- prefer ``bundle_adjustment/cameras`` over the config rig.
    pts2d
        ``(V, T, P, 2)`` detections, ``NaN`` where the detector did not fire.
    threshold
        Pixel gate, used both as the RANSAC inlier cutoff and as the "this cell
        disagrees" cutoff in the reported reasons. A *ranking* knob, not an
        accuracy claim: a systematic rig or decode offset inflates every frame
        alike.
    cap
        Pixel saturation for one cell's disagreement.
    top_k
        How many joints are averaged into the frame score.
    min_views
        Observing views a joint needs before it is scorable at all.
    point_mask, camera_mask
        Optional boolean selections over points / views (``None`` = all).

    Returns
    -------
    FrameScores
        The scores plus the residuals, 3D and coverage used to explain them.

    Raises
    ------
    ValueError
        If ``pts2d`` is not ``(V, T, P, 2)`` with ``V`` matching ``cameras``, or a
        knob is out of range.
    """
    pts = np.asarray(pts2d, dtype=float)
    if pts.ndim != 4 or pts.shape[-1] != 2:
        raise ValueError(f"pts2d must be (V, T, P, 2), got {pts.shape}")
    n_views, n_frames, n_points = pts.shape[:3]
    if n_views != len(cameras):
        raise ValueError(f"pts2d has {n_views} view(s) but the rig has {len(cameras)}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if min_views < 2:
        raise ValueError(f"min_views must be >= 2, got {min_views}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")

    cam_sel = (
        np.ones(n_views, dtype=bool)
        if camera_mask is None
        else np.asarray(camera_mask, dtype=bool)
    )
    pt_sel = (
        np.ones(n_points, dtype=bool)
        if point_mask is None
        else np.asarray(point_mask, dtype=bool)
    )
    # `exists` is (T, P): absence can vary over the recording (a leg lost part-way), so a
    # frame where the joint was still there must keep scoring it.
    if absent_mask is None:
        exists = np.ones((n_frames, n_points), dtype=bool)
    else:
        a = np.asarray(absent_mask, dtype=bool)
        if a.ndim == 1:  # a whole-recording declaration broadcasts over time
            a = np.broadcast_to(a.reshape(1, -1), (n_frames, n_points))
        exists = ~a
    if (
        cam_sel.shape != (n_views,)
        or pt_sel.shape != (n_points,)
        or exists.shape != (n_frames, n_points)
    ):
        raise ValueError(
            "point_mask / camera_mask / absent_mask must match pts2d's T / P / V"
        )
    # A point that is not on the animal is never scored, in the frames it is missing.
    sel_tp = pt_sel[None, :] & exists  # (T, P)
    n_exist_t = exists.sum(axis=1)  # (T,) how many joints this frame could carry
    n_exist = int(exists.all(axis=0).sum())  # points that exist in EVERY frame

    pts3d, _ = triangulate_ransac(
        cameras, pts, threshold=float(threshold), min_inliers=2
    )
    # NaN out the joints that are not on this animal. The detector still emits a peak on an
    # amputated limb (an argmax decode always does) and RANSAC will happily triangulate those
    # peaks into a phantom 3D point, which would then bias the body centroid `_far_mask`
    # derives the ipsi/contra word from -- and that word appears in every driver line.
    if not exists.all():
        pts3d = np.asarray(pts3d, dtype=float).copy()
        pts3d[~exists] = np.nan
    resid = reprojection_error(cameras, pts3d, pts)  # (V, T, P), NaN where undefined
    observed = np.isfinite(resid)  # detector fired AND the joint triangulated
    n_obs = observed.sum(axis=0)  # (T, P) geometry support

    with np.errstate(invalid="ignore"):
        d = np.clip(resid, 0.0, float(cap)) / float(cap)  # (V, T, P) in [0, 1]
    selected = observed & cam_sel[:, None, None] & sel_tp[None, :, :]
    d_sel = np.where(selected, d, np.nan)
    px_sel = np.where(selected, resid, np.nan)

    scorable = (n_obs >= int(min_views)) & selected.any(axis=0)  # (T, P)
    with np.errstate(invalid="ignore"):
        joint = np.where(scorable, _nanmean0(d_sel), np.nan)  # (T, P)
        joint_px = np.where(scorable, _nanmean0(px_sel), np.nan)

    # Zero out the joints that do not exist in this frame before the top-k. An absent
    # column is never scorable, so leaving it in would contribute a padding zero to the
    # frame's mean -- diluting the score and shrinking the dynamic range that ranks frames
    # against each other. Sorting puts those zeros first, so taking the last k over a
    # per-frame k drops exactly them.
    filled = np.where(scorable & exists, np.nan_to_num(joint, nan=0.0), 0.0)
    ordered = np.sort(
        filled, axis=1
    )  # ascending; non-existent joints sort to the front
    k_t = np.minimum(int(top_k), np.maximum(n_exist_t, 1))  # (T,) per-frame top-k
    # Sort ascending and take the last k: frames with fewer than k scorable joints
    # keep the padding zeros, so a frame the detector barely covered cannot win on
    # one loud joint.
    cum = np.cumsum(ordered[:, ::-1], axis=1)  # running sum of the largest first
    idx = np.clip(k_t - 1, 0, n_points - 1)
    score = cum[np.arange(n_frames), idx] / np.maximum(k_t, 1)  # (T,)
    percentile = _percentile_of(score)

    obs_all = observed & cam_sel[:, None, None] & sel_tp[None, :, :]
    # Denominators count only cells that exist on this animal: a fraction over all 38 when
    # 3 can never be labeled reads as a coverage problem that no amount of work can fix.
    n_cells_exist = int(exists.sum()) * int(cam_sel.sum())
    n_joints_exist = int(exists.sum())
    coverage = {
        "scorable_cell_frac": (
            float(obs_all.sum() / n_cells_exist) if n_cells_exist else 0.0
        ),
        "scorable_joint_frac": (
            float((scorable & exists).sum() / n_joints_exist) if n_joints_exist else 0.0
        ),
        "n_existing_points": n_exist,
        "n_absent_points": int(n_points - n_exist),
        "median_observing_views": float(np.median(n_obs)) if n_obs.size else 0.0,
        "global_residual_median_px": (
            float(np.median(resid[obs_all])) if obs_all.any() else float("nan")
        ),
        "score_percentiles": {
            f"p{q}": float(np.percentile(score, q)) for q in (10, 50, 90, 99)
        }
        if score.size
        else {},
        "n_distinct_scores": int(np.unique(score).size),
    }
    params = {
        "threshold_px": float(threshold),
        "cap_px": float(cap),
        "top_k": int(top_k),
        "min_views": int(min_views),
        # Stamped so two sidecars for one recording are comparable: scores are a
        # within-recording ranking, and removing points from the top-k moves every level.
        "absent_points": [int(i) for i in np.nonzero(~exists.all(axis=0))[0]],
        "score": SCORE_DESCRIPTION,
    }
    return FrameScores(
        score=score,
        percentile=percentile,
        joint=joint,
        joint_px=joint_px,
        resid=resid,
        pts3d=pts3d,
        n_obs=n_obs,
        coverage=coverage,
        params=params,
    )


def _nanmean0(arr: np.ndarray) -> np.ndarray:
    """``np.nanmean`` over axis 0, returning 0 (not NaN + a warning) for all-NaN."""
    total = np.nansum(arr, axis=0)
    count = np.isfinite(arr).sum(axis=0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _percentile_of(score: np.ndarray) -> np.ndarray:
    """The within-array percentile (0-100) of every element, ties sharing a rank."""
    if score.size == 0:
        return score.astype(float)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=float)
    ranks[order] = np.arange(score.size, dtype=float)
    if score.size == 1:
        return np.zeros(1)
    return 100.0 * ranks / (score.size - 1)


# -- selection ----------------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """One suggested frame: which frame, and why it is in the list.

    ``kind`` is ``"most-wrong"`` (picked by score) or ``"diversity"`` (picked on a
    uniform temporal grid regardless of score). ``grid_slot`` is ``(i, n)`` for a
    diversity pick, ``None`` otherwise.
    """

    frame: int
    kind: str
    grid_slot: tuple[int, int] | None = None


def _spaced_take(order: list[int], n: int, min_gap: int, taken: list[int]) -> list[int]:
    """Greedily take from ``order`` while staying ``min_gap`` from everything taken."""
    out: list[int] = []
    chosen = list(taken)
    for t in order:
        if len(out) >= n:
            break
        if all(abs(t - c) >= min_gap for c in chosen):
            out.append(t)
            chosen.append(t)
    return out


def _spacing_capacity(n_frames: int, min_gap: int, seeded: "set[int]") -> int:
    """How many MORE frames can be picked at ``>= min_gap`` spacing, given ``seeded``.

    Leftmost-feasible packing: walk the timeline and take a frame whenever it clears
    every already-taken frame, seeds included. That is the exact optimum for a single
    minimum-distance constraint on a line, so the number is a real ceiling rather than
    the ``n_frames // min_gap`` upper bound (which ignores the seeds entirely).
    """
    if n_frames <= 0:
        return 0
    gap = max(int(min_gap), 1)
    taken = sorted(int(t) for t in seeded)
    n = 0
    t = 0
    while t < n_frames:
        if all(abs(t - o) >= gap for o in taken):
            taken.append(t)
            taken.sort()
            n += 1
            t += gap
        else:
            t += 1
    return n


def select_frames(
    scores: FrameScores,
    *,
    count: int = 20,
    min_gap_frames: int = 200,
    reserve_diversity: float = 0.25,
    exclude: "set[int] | list[int] | None" = None,
) -> tuple[list[Pick], dict]:
    """Pick ``count`` frames from ``scores``: most-wrong first, spaced, plus a grid.

    Fully deterministic (no randomness, stable tie-breaking by frame index). The
    order of operations matters:

    1. The **diversity reserve** (``reserve_diversity`` of ``count``) is taken
       first, on the midpoints of ``n`` equal temporal bins -- so the spread does
       not depend on the score at all, and never lands on frame 0 (where the fly
       is often still settling).
    2. The **most-wrong** picks fill in around it by descending score.
    3. Both passes are greedy under a **hard** ``min_gap_frames`` constraint whose
       ``taken`` list is seeded with ``exclude`` (the already-labeled frames), so
       a suggestion can never land a few frames from existing human work.

    Because the gap is hard, the list can come up short -- ``T / min_gap_frames``
    is an absolute upper bound (20 slots on a 4016-frame recording at 2 s / 100
    fps). That is reported, never swallowed.

    Parameters
    ----------
    scores
        The output of :func:`score_frames`.
    count
        How many frames to suggest.
    min_gap_frames
        Hard minimum spacing between any two picks (and between a pick and any
        excluded frame), in frames.
    reserve_diversity
        Fraction of ``count`` taken on the uniform grid instead of by score.
    exclude
        Frames never to suggest (typically the already-labeled ones).

    Returns
    -------
    picks : list of Pick
        Ranked by descending score (ties by frame index), both kinds interleaved.
    shortfall : dict
        ``requested`` / ``selected`` / ``most_wrong`` / ``diversity`` /
        ``spacing_slots`` and a human-readable ``reason`` when short.
    """
    # ``spacing_slots`` is a true capacity, not n_frames // gap: the already-labeled
    # frames seed the constraint and carve the timeline up, so the naive bound
    # overstates what is reachable (21 vs the real 15 on a 4,016-frame recording with
    # 5 labels). A reader who trusts a loose bound lowers --min-gap-s for no reason.
    n_frames = scores.n_frames
    excluded = {int(t) for t in (exclude or ())}
    count = max(int(count), 0)
    min_gap = max(int(min_gap_frames), 1)
    n_div = min(count, int(round(count * float(reserve_diversity))))
    n_hard = count - n_div
    seed = sorted(excluded)

    # Midpoints of n_div equal bins, so the grid never lands on frame 0 (where the
    # fly is often still settling) and each pick can report which slot it fills.
    slot_of: dict[int, tuple[int, int]] = {}
    grid: list[int] = []
    for i in range(n_div):
        t = min(max(int(round((i + 0.5) * n_frames / n_div)), 0), max(n_frames - 1, 0))
        if t not in excluded and t not in slot_of:
            slot_of[t] = (i, n_div)
            grid.append(t)
    div = _spaced_take(grid, n_div, min_gap, seed)

    # Stable descending sort: equal scores keep temporal order, and a frame with
    # nothing scorable (score 0 with no scorable joint) is not offered at all.
    has_signal = np.isfinite(scores.joint).any(axis=1)
    order = [
        int(t)
        for t in np.argsort(-scores.score, kind="stable")
        if int(t) not in excluded and bool(has_signal[t])
    ]
    hard = _spaced_take(order, n_hard, min_gap, seed + div)

    picks = [Pick(frame=t, kind="diversity", grid_slot=slot_of[t]) for t in div]
    picks += [Pick(frame=t, kind="most-wrong") for t in hard]
    picks.sort(key=lambda p: (-float(scores.score[p.frame]), p.frame))

    selected = len(picks)
    shortfall = {
        "requested": count,
        "selected": selected,
        "most_wrong": len(hard),
        "diversity": len(div),
        "spacing_slots": _spacing_capacity(n_frames, min_gap, excluded),
        "n_unscorable_frames": int((~has_signal).sum()),
        "reason": None,
    }
    if selected < count:
        shortfall["reason"] = (
            f"the >= {min_gap} frame spacing ran out of room "
            f"({n_frames} frames, {len(excluded)} already-labeled frame(s) seeded "
            f"the constraint); at most {shortfall['spacing_slots']} more pick(s) fit"
        )
    return picks, shortfall


# -- per-pick reasons ---------------------------------------------------------


def _far_mask(
    cameras: CameraGroup, pts3d_frame: Float[np.ndarray, "P 3"]
) -> Bool[np.ndarray, "V P"]:
    """``(V, P)``: is this joint on the *far* side of the body from this camera?

    Naming-free geometry: a joint is far iff it sits beyond the body centroid along
    the camera's viewing axis, ``dot(x - c, unit(c - camera_position)) > 0``. This
    reproduces the ipsi/contra split with no left/right convention hardcoded (which
    a :class:`~deeperfly.skeleton.Skeleton` does not model), and it is the label the
    operator needs: a far-side joint is the one whose pixel they may have to infer from the
    other views, and the one they may want held out of the training loss.
    """
    x = np.asarray(pts3d_frame, dtype=float)  # (P, 3)
    finite = np.isfinite(x).all(axis=-1)
    if not finite.any():
        return np.zeros((len(cameras), x.shape[0]), dtype=bool)
    centroid = np.nanmean(np.where(finite[:, None], x, np.nan), axis=0)  # (3,)
    out = np.zeros((len(cameras), x.shape[0]), dtype=bool)
    for vi, cam in enumerate(cameras):
        axis = centroid - cam.position
        norm = float(np.linalg.norm(axis))
        if norm == 0:
            continue
        out[vi] = finite & (((x - centroid) @ (axis / norm)) > 0)
    return out


def frame_reason(
    pick: Pick,
    scores: FrameScores,
    *,
    cameras: CameraGroup,
    point_names: "list[str]",
    threshold: float = 15.0,
    n_drivers: int = 3,
) -> dict:
    """Explain one pick: which joints and views drive it (or that it is a grid pick).

    A ``diversity`` pick reports its grid slot and its (usually low) score, so the
    operator can see the easy frame is deliberate rather than a bug.
    """
    t = int(pick.frame)
    if pick.kind == "diversity":
        i, n = pick.grid_slot or (0, 0)
        return {
            "grid_slot": [int(i), int(n)],
            "summary": (
                f"uniform temporal grid slot {i + 1}/{n} -- a typical pose on purpose"
            ),
        }

    joint = scores.joint[t]  # (P,) NaN where unscorable
    resid = scores.resid[:, t, :]  # (V, P)
    with np.errstate(invalid="ignore"):
        over = resid > float(threshold)
    n_over = int(over.any(axis=0).sum())
    far = _far_mask(cameras, scores.pts3d[t])
    cam_names = cameras.names

    ranked = [
        p for p in np.argsort(-np.nan_to_num(joint, nan=-1.0)) if np.isfinite(joint[p])
    ]
    drivers = []
    for p in ranked[:n_drivers]:
        col = resid[:, p]
        if not np.isfinite(col).any():
            continue
        worst = int(np.nanargmax(col))
        drivers.append(
            {
                "point": int(p),
                "point_name": str(point_names[p]) if p < len(point_names) else str(p),
                "disagreement": round(float(joint[p]), 4),
                "disagreement_px": round(float(scores.joint_px[t, p]), 2),
                "worst_camera": str(cam_names[worst]),
                "worst_px": round(float(col[worst]), 2),
                "views_over_threshold": int(np.nansum(over[:, p])),
                "n_observing_views": int(scores.n_obs[t, p]),
                "relation": "far" if bool(far[worst, p]) else "near",
            }
        )
    summary = "no joint is scorable in this frame"
    if drivers:
        d = drivers[0]
        summary = (
            f"{d['views_over_threshold']} view(s) disagree on {d['point_name']} by up "
            f"to {d['worst_px']:.0f} px ({d['relation']} side, worst in "
            f"{d['worst_camera']})"
        )
    return {
        "n_joints_over_threshold": n_over,
        "drivers": drivers,
        "summary": summary,
    }


# -- reading the inputs (the only supported path from a file) ------------------


@dataclass
class AcquisitionInputs:
    """Everything read out of one ``results.h5`` to score it.

    Built only by :func:`prepare_inputs`, which reads ``pose2d/points`` through
    :class:`~deeperfly.results.StageStore` -- deliberately *not*
    :meth:`~deeperfly.results.PoseResult.load`, whose "most-derived layer wins"
    assembly would hand back the substituted ``triangulation/points``.
    """

    results_path: Path
    cameras: CameraGroup
    cameras_from: str
    pts2d: Float[np.ndarray, "V T P 2"]
    conf: Float[np.ndarray, "V T P"] | None
    point_names: list[str]
    camera_names: list[str]
    meta: dict
    identity: dict
    reseed: dict | None
    stored_reproj_error: Float[np.ndarray, "V T P"] | None
    substituted: Bool[np.ndarray, "V T P"] | None

    @property
    def n_views(self) -> int:
        return int(self.pts2d.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.pts2d.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.pts2d.shape[2])

    @property
    def reseeded(self) -> bool:
        return self.reseed is not None

    def fps(self, override: float | None = None) -> tuple[float, bool]:
        """``(fps, was_stamped)``: the override, else ``meta["fps"]``, else 100."""
        if override is not None:
            return float(override), True
        stamped = self.meta.get("fps")
        if stamped:
            return float(stamped), True
        return DEFAULT_FPS, False


#: Seed codes of ``dfpose_predict/contra_seed_source`` that mean "this cell of
#: ``triangulation/points`` is a reprojected seed, not a detection".
_REPROJECTION_SEED_CODES = (
    "reprojection_ipsi",
    "reprojection_ipsi_corrected",
    "reprojection_trusted",
)


def prepare_inputs(results_path: str | Path) -> AcquisitionInputs:
    """Read one ``results.h5`` (read-only) into the arrays the score needs.

    Reads ``pose2d/points`` (the pristine detector output) and the
    ``bundle_adjustment`` rig when present, else the ``pose2d`` config rig. Also
    detects whether the file's *displayed* triangulation layer was reseeded by
    ``dfpose.predict`` -- from ``meta["dfpose_predict"]`` and/or the
    ``dfpose_predict/contra_seed_source`` dataset -- and loads the stored
    ``triangulation/reproj_error`` **only** so the report can show what scoring it
    would have (wrongly) said.

    Raises
    ------
    ValueError
        If the file has no ``pose2d`` output or no rig to score it with.
    """
    import h5py

    from .gui.labels import labels_identity
    from .results import StageStore

    path = Path(results_path)
    store = StageStore(path)
    pose2d = store.read_pose2d()
    if pose2d is None:
        raise ValueError(
            f"{path} has no pose2d/points -- acquisition scores the pristine detector "
            "output, so a file without it cannot be ranked (re-run 'deeperfly run')"
        )
    pts2d, conf = pose2d
    cameras = store.read_cameras("bundle_adjustment")
    cameras_from = "bundle_adjustment"
    if cameras is None:
        cameras = store.read_cameras("pose2d")
        cameras_from = "pose2d"
    if cameras is None:
        raise ValueError(f"{path} stores no cameras; cannot triangulate")
    skeleton = store.read_skeleton()
    point_names = (
        list(skeleton.point_names)
        if skeleton is not None
        else [str(i) for i in range(pts2d.shape[2])]
    )

    with h5py.File(path, "r") as f:
        try:
            meta = json.loads(f.attrs.get("meta", "{}"))
        except (TypeError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        stamp = meta.get("dfpose_predict")
        stamp = stamp if isinstance(stamp, dict) else None
        seed_source = legend = None
        if "dfpose_predict/contra_seed_source" in f:
            seed_source = np.asarray(f["dfpose_predict/contra_seed_source"][()])
            raw = f["dfpose_predict/contra_seed_source"].attrs.get("legend")
            try:
                legend = json.loads(raw) if raw else None
            except (TypeError, ValueError):
                legend = None
        stored_err = (
            np.asarray(f["triangulation/reproj_error"][()], dtype=float)
            if "triangulation/reproj_error" in f
            else None
        )
        tri_points = (
            np.asarray(f["triangulation/points"][()], dtype=float)
            if "triangulation/points" in f
            else None
        )

    substituted = _substituted_mask(seed_source, legend, tri_points, pts2d)
    reseed = None
    if stamp is not None or seed_source is not None:
        detected_by = []
        if stamp is not None:
            detected_by.append("meta.dfpose_predict")
        if seed_source is not None:
            detected_by.append("dfpose_predict/contra_seed_source")
        prov = _predict_provenance(stamp or {})
        reseed = {
            "detected_by": detected_by,
            "note": (
                "contralateral cells of triangulation/points are reprojection seeds "
                "(residual ~0 by construction); scored pose2d/points instead"
            ),
            "n_substituted_cells": (
                int(substituted.sum()) if substituted is not None else None
            ),
            **prov,
        }

    identity = labels_identity(
        point_names=point_names,
        camera_names=list(cameras.names),
        n_frames=int(pts2d.shape[1]),
        image_sizes=store.read_image_sizes(),
        footage=store.read_footage(),
    )
    return AcquisitionInputs(
        results_path=path,
        cameras=cameras,
        cameras_from=cameras_from,
        pts2d=np.asarray(pts2d, dtype=float),
        conf=None if conf is None else np.asarray(conf, dtype=float),
        point_names=point_names,
        camera_names=list(cameras.names),
        meta=meta,
        identity=identity,
        reseed=reseed,
        stored_reproj_error=stored_err,
        substituted=substituted,
    )


def _substituted_mask(
    seed_source: np.ndarray | None,
    legend: dict | None,
    tri_points: np.ndarray | None,
    pts2d: np.ndarray,
) -> np.ndarray | None:
    """``(V, T, P)`` cells of ``triangulation/points`` that are not the detection.

    Preferred source is the per-cell ``contra_seed_source`` codes; failing that (an
    older reseeded file, or one written by a different tool) the mask is measured
    directly, as the cells where the stored 2D differs from ``pose2d/points``.
    """
    if seed_source is not None and legend:
        codes = [legend[k] for k in _REPROJECTION_SEED_CODES if k in legend]
        if codes:
            return np.isin(seed_source, codes)
    if tri_points is None:
        return None
    same_shape = tri_points.shape == np.asarray(pts2d).shape
    if not same_shape:
        return None
    both = np.isfinite(tri_points).all(-1) & np.isfinite(pts2d).all(-1)
    moved = ~np.isclose(tri_points, pts2d, rtol=0, atol=1e-9, equal_nan=True).all(-1)
    return both & moved


def _predict_provenance(stamp: dict) -> dict:
    """Model / checkpoint / commit of a ``dfpose.predict`` stamp.

    A ``--from-results`` copy carries its 2D provenance one level down, under
    ``inherited_2d_provenance.dfpose_predict``, so look there too.
    """
    inherited = stamp.get("inherited_2d_provenance")
    inner = {}
    if isinstance(inherited, dict) and isinstance(
        inherited.get("dfpose_predict"), dict
    ):
        inner = inherited["dfpose_predict"]
    out = {}
    for key in ("model", "checkpoint_md5", "dfpose_commit", "created_utc"):
        value = stamp.get(key) or inner.get(key)
        if value is not None:
            out[key] = value
    strategy = (stamp.get("contra_seed") or {}).get("strategy")
    if strategy:
        out["contra_seed_strategy"] = strategy
    return out


def stored_vs_pose2d(
    inputs: AcquisitionInputs, scores: FrameScores, *, threshold: float = 15.0
) -> dict | None:
    """What the *stored* reprojection error would have said on the substituted cells.

    Printed by the CLI and stamped into the sidecar so the degenerate-signal trap is
    visible in the output and cannot be silently reintroduced. Returns ``None`` when
    the file stores no reprojection error / no substituted cells.
    """
    if inputs.stored_reproj_error is None or inputs.substituted is None:
        return None
    mask = inputs.substituted & np.isfinite(inputs.stored_reproj_error)
    if not mask.any():
        return None
    stored = inputs.stored_reproj_error[mask]
    ours = scores.resid[inputs.substituted & np.isfinite(scores.resid)]
    if ours.size == 0:
        return None
    return {
        "far_cells_median_px": round(float(np.median(stored)), 3),
        "far_cells_p90_px": round(float(np.percentile(stored, 90)), 3),
        "far_cells_frac_over_thresh": round(float((stored > threshold).mean()), 4),
        "pose2d_far_median_px": round(float(np.median(ours)), 3),
        "pose2d_far_p90_px": round(float(np.percentile(ours, 90)), 3),
        "pose2d_far_frac_over_thresh": round(float((ours > threshold).mean()), 4),
        "n_cells": int(mask.sum()),
        "cells_are": (
            "the cells where triangulation/points is a reprojection seed rather than "
            "the detection"
        ),
    }


# -- masks from glob selectors -------------------------------------------------


def glob_mask(names: "list[str]", patterns: "list[str] | None") -> np.ndarray:
    """Boolean mask over ``names`` of the entries matching any of ``patterns``.

    ``None`` (or an empty list) selects everything. Patterns are ``fnmatch`` globs
    matched case-sensitively against the whole name, so ``"*tibia*"`` and ``"l?_*"``
    both work.

    Raises
    ------
    ValueError
        If the patterns match nothing (a typo would otherwise silently score an
        empty selection).
    """
    if not patterns:
        return np.ones(len(names), dtype=bool)
    mask = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        mask[i] = any(fnmatch.fnmatchcase(str(name), p) for p in patterns)
    if not mask.any():
        raise ValueError(f"none of {list(patterns)} matches any of {list(names)}")
    return mask


# -- labels sidecar ------------------------------------------------------------


def read_labeled_frames(labels_path: str | Path, *, identity: dict) -> dict | None:
    """Which frames already carry human work, from the ``labels.h5`` sidecar.

    Read-only. Returns ``None`` when there is no sidecar (a first round). A frame
    counts as done when *any* view carries a GT pixel or a **hidden** mark, or the
    frame is marked reviewed -- the same rule the GUI's labeled-frames list uses
    (:meth:`~deeperfly.gui.state.EditorState.corrected_frames`), so the two lists
    can never disagree. Note a frame where the operator only marked cells hidden is still
    done: deciding what not to train on is work, and the queue must not re-offer it.

    Raises
    ------
    ValueError
        If the sidecar belongs to a different recording (identity mismatch).
    """
    from .gui.labels import absent_to_spans, load_labels

    path = Path(labels_path)
    labels = load_labels(path, identity=identity)
    if labels is None:
        return None
    # A recording-wide absence declaration is deliberately NOT folded into `decided`: it
    # would mark every frame labeled at a stroke, and `select_frames` (which excludes
    # labeled frames) would return nothing while the sidebar listed the whole recording.
    decided = labels.has_gt | labels.occluded_effective  # (V, T, P)
    labeled = np.nonzero(decided.any(axis=(0, 2)))[0]
    reviewed = np.nonzero(labels.reviewed)[0]
    absent = labels.absent_all_frames()
    return {
        "path": path.name,
        "exists": True,
        "md5": file_md5(path),
        "labeled_frames": [int(t) for t in labeled],
        "reviewed_frames": [int(t) for t in reviewed],
        "n_gt": int(labels.has_gt.sum()),
        "n_occluded": int(labels.occluded_effective.sum()),
        # `absent_points` is the whole-recording subset (what a structural consumer may
        # act on); `absent_spans` carries the full per-frame declaration compactly, so a
        # caller can rebuild the (T, P) mask without re-reading the file.
        "n_absent": int(labels.absent.sum()),
        "absent_points": [int(i) for i in np.nonzero(absent)[0]],
        "absent_spans": [
            [int(v) for v in row] for row in absent_to_spans(labels.absent)
        ],
    }


# -- the JSON sidecar ----------------------------------------------------------


def file_md5(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Hex MD5 of a file, read in chunks (read-only; ~67 ms on 55 MB)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_suggestions(
    inputs: AcquisitionInputs,
    scores: FrameScores,
    picks: "list[Pick]",
    *,
    params: dict,
    shortfall: dict,
    labels: dict | None,
    output_dir: Path,
) -> dict:
    """Assemble the sidecar document (see :func:`write_suggestions`).

    Every input's fingerprint travels with the queue -- identity, the results
    size/mtime/md5, the labels md5 and labeled-frame set -- so a reader can decide
    staleness without recomputing anything (:func:`suggestions_staleness`).
    """
    import importlib.metadata

    try:
        version = importlib.metadata.version("deeperfly")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover -- not installed
        version = "unknown"

    stat = inputs.results_path.stat()
    fps = float(params["fps"])
    source = {
        "results": os.path.relpath(inputs.results_path, output_dir),
        "results_md5": file_md5(inputs.results_path),
        "results_size": int(stat.st_size),
        "results_mtime_ns": int(stat.st_mtime_ns),
        "results_created_utc": inputs.meta.get("created_utc"),
        "cameras_from": inputs.cameras_from,
        "scored_array": SCORED_ARRAY,
        "never_scored": ["triangulation/points", "triangulation/reproj_error"],
        "n_views": inputs.n_views,
        "n_frames": inputs.n_frames,
        "n_points": inputs.n_points,
        "camera_names": list(inputs.camera_names),
        "identity": inputs.identity,
        "reseeded": inputs.reseeded,
    }
    if inputs.reseed is not None:
        reseed = dict(inputs.reseed)
        trap = stored_vs_pose2d(inputs, scores, threshold=params["threshold_px"])
        if trap is not None:
            reseed["stored_reproj_error_would_have_said"] = trap
        source["reseed"] = reseed

    frames = []
    for rank, pick in enumerate(picks, start=1):
        t = pick.frame
        frames.append(
            {
                "rank": rank,
                "frame": int(t),
                "t_s": round(t / fps, 4) if fps else None,
                "score": round(float(scores.score[t]), 4),
                "percentile": round(float(scores.percentile[t]), 2),
                "kind": pick.kind,
                "reason": frame_reason(
                    pick,
                    scores,
                    cameras=inputs.cameras,
                    point_names=inputs.point_names,
                    threshold=float(params["threshold_px"]),
                ),
            }
        )

    labeled = list((labels or {}).get("labeled_frames", []))
    reviewed = list((labels or {}).get("reviewed_frames", []))
    excluded_frames = sorted(set(labeled) | set(reviewed))
    return {
        "deeperfly_suggestions_format_version": SUGGESTIONS_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deeperfly_version": version,
        "params": params,
        "source": source,
        "labels": labels
        or {"path": "labels.h5", "exists": False, "labeled_frames": []},
        "coverage": scores.coverage,
        "shortfall": shortfall,
        "excluded": {
            "labeled": excluded_frames,
            "unscorable_frames": shortfall.get("n_unscorable_frames", 0),
        },
        "frames": frames,
    }


def write_suggestions(path: str | Path, doc: dict) -> Path:
    """Write the suggestions sidecar atomically (``tmp`` + :func:`os.replace`).

    The only file this module ever writes. JSON, in the output directory, never an
    HDF5 group -- so it can neither be mistaken for a pipeline stage nor truncate a
    ``results.h5`` (whose calibration may be the only copy).

    Returns
    -------
    Path
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, out)
    return out


def read_suggestions(path: str | Path) -> dict | None:
    """Read a ``labels_suggest.json`` sidecar, or ``None`` if it is not usable.

    The reader the GUI calls. ``None`` means "no queue": the file is absent,
    unreadable, not a JSON object, or stamped with a
    ``deeperfly_suggestions_format_version`` this build does not know (an unknown
    version is treated as *absent* rather than half-parsed). Never raises for a bad
    file, and never writes.

    Parameters
    ----------
    path
        Path to the sidecar (``<results_dir>/labels_suggest.json``).

    Returns
    -------
    dict or None
        The parsed document, or ``None``.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        log.warning("%s is not readable JSON; ignoring the suggestions sidecar", p)
        return None
    if not isinstance(doc, dict):
        return None
    version = doc.get("deeperfly_suggestions_format_version")
    if version != SUGGESTIONS_FORMAT_VERSION:
        log.warning(
            "%s has suggestions format version %r (expected %d); ignoring it -- "
            "re-run 'deeperfly labels-suggest'",
            p,
            version,
            SUGGESTIONS_FORMAT_VERSION,
        )
        return None
    return doc


def suggestions_staleness(
    doc: dict,
    *,
    identity: dict | None = None,
    results_path: str | Path | None = None,
    labeled_frames: "set[int] | list[int] | None" = None,
) -> dict:
    """How far a sidecar has drifted from the live state: ``{"level", "reasons"}``.

    Four tiers, cheapest-first, so the caller (the GUI route) never recomputes a
    score to find out:

    ``"hard"``
        ``identity`` differs -- a different recording, or a different point/camera
        set or frame count. The queue's frame indices mean something else; refuse
        to render it.
    ``"predictions"``
        ``results.h5``'s size/mtime differ *and* its md5 differs: the queue was
        computed from superseded predictions. A mere ``touch`` (same md5) is not
        stale, which is why both a stat and a hash are stored.
    ``"progress"``
        Frames in the queue are now labeled (or the labels file changed). Normal
        and expected -- the operator is working through it.
    ``"none"``
        Nothing drifted.

    Parameters
    ----------
    doc
        A document from :func:`read_suggestions`.
    identity
        The live recording fingerprint
        (:func:`~deeperfly.gui.labels.labels_identity`); skipped when ``None``.
    results_path
        The live ``results.h5``; skipped when ``None``.
    labeled_frames
        The frames that now carry human work; skipped when ``None``.

    Returns
    -------
    dict
        ``{"level": str, "reasons": list[str], "done": int, "total": int}``.
    """
    source = doc.get("source") or {}
    if not isinstance(source, dict):  # hand-edited sidecar
        source = {}
    # A sidecar is a plain JSON file an operator may edit, so every shape here is
    # untrusted: ``frames`` has been seen as a dict, and as a list holding strings.
    # Reducing to well-formed rows once means neither this function nor the GUI route
    # has to defend itself again -- and a malformed queue must degrade to "no
    # suggestions", never raise into the editor the operator is labeling in.
    raw = doc.get("frames") or []
    frames = [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []
    total = len(frames)
    reasons: list[str] = []
    level = "none"

    if identity is not None and source.get("identity") not in (None, identity):
        stored = source.get("identity") or {}
        differing = [
            k
            for k in (
                "point_names",
                "camera_names",
                "n_frames",
                "image_sizes",
                "footage",
            )
            if stored.get(k) and identity.get(k) and stored.get(k) != identity.get(k)
        ]
        if differing:
            return {
                "level": "hard",
                "reasons": [
                    "the queue was computed for a different recording "
                    f"({', '.join(differing)} differ) -- re-run "
                    "'deeperfly labels-suggest'"
                ],
                "done": 0,
                "total": total,
            }

    if results_path is not None:
        p = Path(results_path)
        if p.is_file():
            stat = p.stat()
            # A stat mismatch is the cheap trigger for the hash check, but a sidecar
            # carrying only ``results_md5`` (a hand-written one -- the writer always
            # stamps all three) would otherwise never be checked at all, silently
            # reporting "none" against genuinely superseded predictions. Absent stat
            # fields therefore mean "hash it", not "assume unchanged".
            have_stat = (
                source.get("results_size") is not None
                or source.get("results_mtime_ns") is not None
            )
            changed = (
                not have_stat
                or (
                    source.get("results_size") is not None
                    and int(source["results_size"]) != stat.st_size
                )
                or (
                    source.get("results_mtime_ns") is not None
                    and int(source["results_mtime_ns"]) != stat.st_mtime_ns
                )
            )
            if changed and source.get("results_md5"):
                if file_md5(p) != source["results_md5"]:
                    level = "predictions"
                    reasons.append(
                        "the queue was computed from superseded predictions "
                        "(results.h5 has changed) -- re-run "
                        "'deeperfly labels-suggest'"
                    )

    done = 0
    if labeled_frames is not None:
        live = {int(t) for t in labeled_frames}
        queued = {
            int(f["frame"])
            for f in frames
            if isinstance(f.get("frame"), (int, float))
            and not isinstance(f.get("frame"), bool)
        }
        done = len(queued & live)
        stored = {int(t) for t in (doc.get("labels") or {}).get("labeled_frames", [])}
        if done or live != stored:
            if level == "none":
                level = "progress"
            if done:
                reasons.append(f"{done} of {total} suggested frame(s) are now labeled")
            elif live - stored:
                reasons.append(
                    f"{len(live - stored)} frame(s) were labeled since the queue "
                    "was computed"
                )
    return {"level": level, "reasons": reasons, "done": done, "total": total}
