"""Pictorial-structures (PS) multi-view 2D->3D correction (DeepFly3D-style).

The optional, accuracy-oriented alternative to the default reprojection-outlier
rejection in :func:`deeperfly.pipeline.reconstruct`. Where that path can only
*veto* a bad detection, PS can *recover* the correct joint when the detector's
arg-max landed on the wrong heatmap peak (self-occlusion, crossing legs,
left/right confusion).

Following Gunel et al. (DeepFly3D, 2019):

1. Keep the **top-K candidate peaks** per (view, joint), not just the arg-max
   (:func:`extract_candidates`).
2. Per joint, build a pool of multi-view-consistent **3D hypotheses** by
   triangulating candidate pairs across views, refitting from inlier views, and
   scoring by summed heatmap confidence (batched per frame in :func:`solve_frame`).
3. Choose one hypothesis per joint by **exact dynamic programming** along each limb
   (:func:`_chain_dp`). The fly skeleton's 2D bones form a forest of simple chains,
   so the MAP over the bone-length-coupled model is exact -- no loopy belief
   propagation. An optional temporal term penalizes 3D jumps.

Everything is plain NumPy over a :class:`~deeperfly.rig.cameras.CameraGroup` and
:class:`~deeperfly.skeleton.Skeleton`. The detector forward and heatmap decode
happen upstream; this module consumes only candidate peaks + bundle-adjusted cameras.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from jaxtyping import Float

from ..rig.cameras import CameraGroup
from ..rig.triangulation import reprojection_error
from ..skeleton import Skeleton

__all__ = [
    "Candidates",
    "peak_candidates",
    "bone_length_targets",
    "skeleton_chains",
    "solve_frame",
    "elect_frame",
    "reconstruct",
]

# Internal defaults: knobs this module owns outright, because no config key exposes
# them. The knobs a run CAN set (`k`, `lam`, the peak gates) are deliberately absent --
# their defaults belong to `deeperfly.config.PictorialParams` and are written there
# once, so the functions below take them as required arguments rather than restating
# a number that would then be free to drift from the one a run actually gets.
DEFAULT_MAX_HYP = 10  # 3D hypotheses kept per joint after pruning
DEFAULT_INLIER_PX = 15.0  # a view supports a 3D hypothesis if a candidate is this close
DEFAULT_HUBER = 0.5  # Huber knee for the bone-length residual, in units of bone length
DEFAULT_MU = 5.0  # temporal weight (per unit squared 3D displacement / bone-scale^2)
DEFAULT_PEAK_RADIUS = 2  # NMS / sub-pixel-window half-width (heatmap pixels)
DEFAULT_SUBPIXEL = "weighted"  # peak refinement: "argmax" | "weighted" | "taylor"


@dataclass(frozen=True)
class Candidates:
    """Top-K detector peaks per (view, point) for a sequence, in image pixels.

    ``xy`` is ``(V, T, P, K, 2)`` and ``score`` is ``(V, T, P, K)``; padded /
    invisible / sub-threshold slots are ``NaN`` (``xy``) and ``0`` (``score``).
    The arg-max (``K = 0``) reproduces the single-peak detection, so bundle adjustment
    can still use the plain 2D path while PS consumes the full candidate set.
    """

    xy: Float[np.ndarray, "V T P K 2"]
    score: Float[np.ndarray, "V T P K"]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        v, t, n, k, _ = self.xy.shape
        return v, t, n, k

    def frame(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Candidate ``(xy, score)`` for one frame: ``(V, P, K, 2)`` and ``(V, P, K)``."""
        return self.xy[:, t], self.score[:, t]


# -- candidate extraction ----------------------------------------------------


def peak_candidates(
    heatmaps: Float[np.ndarray, "*chan H_out W_out"],
    k: int,
    *,
    radius: int = DEFAULT_PEAK_RADIUS,
    threshold: float,
    threshold_rel: float,
    method: str = DEFAULT_SUBPIXEL,
    normalize: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[Float[np.ndarray, "*chan K 2"], Float[np.ndarray, "*chan K"]]:
    """Top-``k`` local-maxima peaks per heatmap channel (normalized ``(x, y)`` + score).

    A pixel is a peak if it is the maximum of its ``(2*radius+1)`` neighborhood and
    exceeds ``threshold``; the strongest ``k`` are returned, score-ordered, padded
    with ``NaN`` / ``0`` when fewer exist. Each is refined to sub-pixel by
    ``method`` (the same :func:`~deeperfly.pose2d.inference.refine_peaks` the
    single-peak decoder uses), so candidates carry the arg-max's localization.

    Parameters
    ----------
    heatmaps
        Heatmaps of shape ``(*chan, H_out, W_out)``.
    k
        Number of peaks to keep per channel. Required: the shipped value is
        ``[pictorial_structures] k`` (:class:`deeperfly.config.PictorialParams`),
        which owns it so the accuracy/cost dial has exactly one default.
    radius
        NMS / sub-pixel-window half-width, in heatmap pixels.
    threshold
        Ignore peaks weaker than this, in RAW field units. Absolute, so it is a statement
        about a particular detector's output scale -- see ``threshold_rel``. Required,
        and deliberately: a default here would be a claim this module cannot make, and
        the one it used to carry (0.05) silently gated away every second candidate on a
        detector whose field peaks near 0.08. Pass 0.0 for no gate.
    threshold_rel
        Ignore peaks weaker than this fraction of the channel's OWN peak, which is the
        scale-free version of the same gate and the only one portable across detectors.
        The effective threshold is the larger of the two. It exists because the shipped
        absolute gate was set on a detector whose heatmaps peak near 1.0, and the multiview
        transformer's peak near 0.08: at ``threshold = 0.05`` an r28 field offers a second
        candidate in **0.04%** of cells, so recovery is choosing from a set of one almost
        everywhere and can only return its own input.
    method
        Sub-pixel refinement: ``"argmax"`` | ``"weighted"`` | ``"taylor"``.
    normalize
        Maps sub-pixel field cells ``(..., 2)`` to input-normalized ``(x, y)``. Pass the
        detector's own
        (:meth:`deeperfly.pose2d.models.LoadedModel.cells_to_normalized`) -- the default
        below is the half-pixel cell-center convention, which is correct only when the
        field spans the reported frame. On a PADDED field it is wrong by the margin and by
        the ``(w + 2m) / w`` scale, tens of model pixels at the frame's edge, which is
        exactly where a candidate is worth having.

    Returns
    -------
    xy : np.ndarray
        Peak coordinates of shape ``(*chan, K, 2)`` normalized to the model's reported
        frame (NaN-padded). Inside ``[0, 1]`` unless the field is padded, where a
        coordinate outside it is a peak outside that frame.
    score : np.ndarray
        Raw peak values of shape ``(*chan, K)`` (``0`` where padded).
    """
    from scipy.ndimage import maximum_filter

    from ..pose2d.inference import refine_peaks

    hm = np.asarray(heatmaps, dtype=float)
    hh, ww = hm.shape[-2:]
    chan = hm.shape[:-2]
    size = (1,) * (hm.ndim - 2) + (2 * radius + 1, 2 * radius + 1)
    thr = float(threshold)
    if threshold_rel > 0.0:
        # Per channel, so a weakly-detected joint is judged against its own peak rather
        # than against the frame's strongest one.
        chan_peak = hm.reshape(*chan, hh * ww).max(-1)[..., None, None]
        thr = np.maximum(thr, float(threshold_rel) * chan_peak)
    is_peak = (hm == maximum_filter(hm, size=size)) & (hm > thr)
    flat = np.where(is_peak, hm, -np.inf).reshape(*chan, hh * ww)

    k = min(k, flat.shape[-1])
    top = np.argpartition(-flat, k - 1, axis=-1)[..., :k]
    top_val = np.take_along_axis(flat, top, axis=-1)
    order = np.argsort(-top_val, axis=-1)  # strongest first
    idx = np.take_along_axis(top, order, axis=-1)
    val = np.take_along_axis(top_val, order, axis=-1)

    row, col = idx // ww, idx % ww
    m = int(np.prod(chan)) if chan else 1
    cx, cy = refine_peaks(
        hm.reshape(m, hh, ww),
        row.reshape(m, k),
        col.reshape(m, k),
        method=method,
        radius=radius,
    )
    cells = np.stack([cx.reshape(*chan, k), cy.reshape(*chan, k)], axis=-1)
    if normalize is not None:
        xy = np.asarray(normalize(cells), dtype=float)
    else:
        # +0.5: cell-center convention, matching inference.heatmap_to_points.
        xy = np.stack([(cells[..., 0] + 0.5) / ww, (cells[..., 1] + 0.5) / hh], axis=-1)
    valid = np.isfinite(val)
    xy = np.where(valid[..., None], xy, np.nan)
    score = np.where(valid, val, 0.0)
    return xy, score


# -- per-(view, joint) election (the cheap alternative to the hypothesis pool) ----


def elect_frame(
    cameras: CameraGroup,
    cand_xy: Float[np.ndarray, "V P K 2"],
    cand_score: Float[np.ndarray, "V P K"],
    *,
    threshold: float = 15.0,
    min_views: int = 3,
    max_reproj: float = 30.0,
    max_move: float = np.inf,
    min_margin: float = 0.0,
    min_conf_ratio: float = 0.0,
) -> Float[np.ndarray, "V P 2"]:
    """Per ``(view, joint)``, re-elect among that cell's own candidates. 2D only.

    **Not reachable from a config, on purpose.** It is the cheap research baseline the
    recovery numbers were measured against, and measurement is why it is not an option:
    over 12 held-out (detector, recording) pairs it swings 14.0-68.8% of the available
    gain where :func:`solve_frame` holds 53.3-66.2%, so it is the higher-variance
    estimator rather than a cheaper equivalent one. ``k`` is the production accuracy/cost
    dial (``k = 3`` keeps 88-100% of ``k = 5``'s improvement at 1.7x less time).

    A different estimator from :func:`solve_frame`, not a cheaper configuration of it, and
    the distinction is what makes it work: this holds view ``v`` out, triangulates the
    OTHER views' arg-max, reprojects into ``v``, and takes ``v``'s candidate nearest that
    reprojection. :func:`solve_frame` instead commits ONE 3D hypothesis per joint to every
    view at once. Measured on held-out animals, giving `solve_frame` this same
    leave-one-out pool (``M = V``) while keeping its single-commitment rule captures only
    24.7-46.8% of the available gain, where this rule captures 14.0-68.8% and the full
    ``C(V,2)*K**2`` pair pool 53.3-68.9% -- so the commitment rule, not the pool, is what
    separates them. That is why this is a function and not a parameter.

    Chosen for cost: no hypothesis pool, so ~5.8x cheaper than `solve_frame` at ``K = 5``
    (28.3 vs 163.6 ms/frame at V=8, P=38) and **flat in K** where the pair pool is
    quadratic. It also never abstains -- every cell returns a real detected peak -- so
    unlike `solve_frame` it cannot un-densify a dense run.

    **Selection, never substitution.** The result is always one of ``cand_xy``, so the
    +3.195 px this project measured for substituting a reprojection for a detection is
    structurally unreachable.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    cand_xy, cand_score
        Per-frame candidates, ``(V, P, K, 2)`` / ``(V, P, K)``, rank 0 the arg-max.
    threshold
        RANSAC inlier threshold (px) for the leave-one-out triangulation.
    min_views
        Inlier views the triangulation needs before its vote counts. Measured nearly
        inert (2/3/4 capture 93.4/93.4/91.6% of the gain); only 5-6 start costing.
    max_reproj
        Elect only if the winner lies within this many px of the reprojection. **The
        load-bearing guard.** It bounds how far the GEOMETRY may be off, not how far the
        detector's answer may move; 30 px chosen on a held-out animal.
    max_move
        Legacy guard on distance from the arg-max. Infinite by default because 40 px --
        the value this rule shipped with -- excluded the 150-300 px moves that carry the
        error mass, capturing 6.0% of the available gain instead of 93.4%.
    min_margin
        Elect only if the winner beats the arg-max by this margin in reprojection
        distance; breaks near-ties toward the incumbent.
    min_conf_ratio
        Elect only if the winner's peak value is at least this fraction of the arg-max's.
        Confidence here is a cliff rather than a gradient, so it is a gate, never a weight.

    Returns
    -------
    np.ndarray
        Elected 2D per view, ``(V, P, 2)``, in the same pixel frame as ``cand_xy``.
        Never NaN where the arg-max was finite.
    """
    from ..rig.triangulation import triangulate_ransac

    v, n, _k, _ = cand_xy.shape
    arg = cand_xy[:, :, 0, :]  # (V, P, 2) -- the incumbent
    out = arg.copy()
    for vid in range(v):
        held = arg.copy()
        held[vid] = np.nan  # a view must not vote for itself
        x3, inl = triangulate_ransac(cameras, held, threshold=threshold)
        n_in = np.asarray(inl).sum(axis=0)
        proj = np.asarray(cameras.project(x3))[vid]  # (P, 2)
        for p in range(n):
            if n_in[p] < min_views or not np.isfinite(proj[p]).all():
                continue
            cands = cand_xy[vid, p]  # (K, 2)
            d = np.linalg.norm(cands - proj[p], axis=-1)
            if not np.isfinite(d).any():
                continue
            c = int(np.nanargmin(d))
            if c == 0 or not np.isfinite(d[c]) or d[c] > max_reproj:
                continue
            if np.linalg.norm(cands[c] - cands[0]) > max_move:
                continue
            if (d[0] - d[c]) < min_margin:
                continue
            sc = cand_score[vid, p]
            if min_conf_ratio > 0.0 and sc[0] > 0 and sc[c] < min_conf_ratio * sc[0]:
                continue
            out[vid, p] = cands[c]
    return out


# -- bone-length prior (shared with bundle adjustment) -----------------------


def bone_length_targets(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V F P 2"],
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median bone length per skeleton bone, from an initial triangulation.

    Shared by bundle adjustment
    (:func:`deeperfly.pipeline._bone_prior`) and PS so the two agree on the
    anatomical prior.

    Parameters
    ----------
    cameras
        The rig used for the initial triangulation.
    pts2d
        2D observations of shape ``(V, F, P, 2)``.
    skeleton
        Skeleton supplying the bone (edge) list.

    Returns
    -------
    i, j : np.ndarray
        Bone endpoint index arrays (the columns of :attr:`Skeleton.edges`).
    targets : np.ndarray
        Per-bone median target length of shape ``(B,)`` (NaN for a bone never
        triangulated).
    """
    import warnings

    from ..rig.triangulation import triangulate

    pts3d0 = triangulate(cameras, pts2d)  # (F, P, 3)
    i, j = skeleton.edge_endpoints()
    lengths = np.linalg.norm(pts3d0[:, i] - pts3d0[:, j], axis=-1)  # (F, B)
    with warnings.catch_warnings():  # a never-triangulated bone -> NaN target (ok)
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = np.nanmedian(lengths, axis=0)  # (B,)
    return i, j, targets


# -- skeleton chains ---------------------------------------------------------


def skeleton_chains(skeleton: Skeleton) -> list[list[int]]:
    """Decompose the 2D bones into ordered simple chains (paths).

    Each connected component of :attr:`Skeleton.edges` is a path (max degree 2),
    returned as an ordered joint list walked from an endpoint; isolated points come
    back as singletons. :func:`_chain_dp` runs exact Viterbi over this ordering.

    Parameters
    ----------
    skeleton
        Skeleton whose bones are decomposed.

    Returns
    -------
    list of list of int
        Ordered joint-index chains (singletons for isolated points).
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in skeleton.edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    chains: list[list[int]] = []
    seen: set[int] = set()
    for start in range(skeleton.n_points):
        if start in seen:
            continue
        # Collect the connected component (BFS).
        comp, stack = [], [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            stack.extend(m for m in adj[n] if m not in seen)
        if len(comp) == 1:
            chains.append(comp)
            continue
        # Walk the path from one endpoint (a degree-1 node); the `walked` set
        # makes the walk robust to branches / cycles (the fly skeleton has none).
        ends = [n for n in comp if len(adj[n]) == 1]
        cur = ends[0] if ends else comp[0]
        order, prev, walked = [cur], None, {cur}
        while True:
            nxts = [m for m in adj[cur] if m != prev and m not in walked]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            walked.add(cur)
            order.append(cur)
        chains.append(order)
    return chains


# -- per-frame hypotheses (batched over joints) ------------------------------


def _combo_index(v: int, k: int):
    """View-pair and candidate index arrays for all ``C(V,2) * K*K`` hypotheses."""
    pairs = np.array(list(combinations(range(v), 2)), dtype=int)  # (C(V,2), 2)
    a = np.repeat(np.arange(k), k)  # (K*K,) slow index
    b = np.tile(np.arange(k), k)  # (K*K,) fast index
    vv = np.repeat(pairs[:, 0], k * k)
    ww = np.repeat(pairs[:, 1], k * k)
    aa = np.tile(a, len(pairs))
    bb = np.tile(b, len(pairs))
    return vv, ww, aa, bb


def _frame_hypotheses(
    cameras: CameraGroup,
    cand_xy: Float[np.ndarray, "V P K 2"],
    cand_score: Float[np.ndarray, "V P K"],
    *,
    inlier_px: float,
):
    """All multi-view 3D hypotheses for one frame's joints, scored by evidence.

    All geometry is two batched triangulate + project calls over the whole frame.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    cand_xy, cand_score
        Per-frame candidates of shape ``(V, P, K, 2)`` / ``(V, P, K)``.
    inlier_px
        A view supports a hypothesis if a candidate reprojects within this many
        pixels.

    Returns
    -------
    X : np.ndarray
        Refit 3D hypotheses of shape ``(P, M, 3)`` (``M = C(V, 2) K^2``).
    evidence : np.ndarray
        Summed heatmap confidence of supporting views ``(P, M)``.
    n_inlier : np.ndarray
        Supporting-view count per hypothesis ``(P, M)``.
    obs : np.ndarray
        Per-view chosen candidate observations ``(V, P, M, 2)`` (NaN for
        non-supporting views).
    """
    v, n, k, _ = cand_xy.shape
    vv, ww, aa, bb = _combo_index(v, k)
    m = len(vv)
    rng = np.arange(m)

    # Build (V, P, M, 2): each hypothesis activates its two views' chosen candidates.
    pts = np.full((v, n, m, 2), np.nan)
    pts[vv, :, rng] = cand_xy[vv, :, aa]  # (M, P, 2) -> view vv[m], hyp m
    pts[ww, :, rng] = cand_xy[ww, :, bb]
    x_pair = np.asarray(cameras.triangulate(pts.reshape(v, n * m, 2))).reshape(n, m, 3)

    chosen, evidence, n_in = _score_hypotheses(
        cameras, x_pair, cand_xy, cand_score, inlier_px
    )
    # Refit each hypothesis from all its inlier views, then re-score.
    x = np.asarray(cameras.triangulate(chosen.reshape(v, n * m, 2))).reshape(n, m, 3)
    obs, evidence, n_in = _score_hypotheses(cameras, x, cand_xy, cand_score, inlier_px)
    return x, evidence, n_in, obs


def _score_hypotheses(
    cameras: CameraGroup,
    x: Float[np.ndarray, "P M 3"],
    cand_xy: Float[np.ndarray, "V P K 2"],
    cand_score: Float[np.ndarray, "V P K"],
    inlier_px: float,
):
    """Reproject hypotheses and gather per-view nearest-candidate support.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    x
        3D hypotheses of shape ``(P, M, 3)``.
    cand_xy, cand_score
        Per-frame candidates of shape ``(V, P, K, 2)`` / ``(V, P, K)``.
    inlier_px
        Inlier reprojection threshold in pixels.

    Returns
    -------
    obs : np.ndarray
        Nearest in-threshold candidate per view ``(V, P, M, 2)`` (else NaN).
    evidence : np.ndarray
        Summed score of supporting views ``(P, M)``.
    n_inlier : np.ndarray
        Supporting-view count ``(P, M)``.
    """
    v, n, k, _ = cand_xy.shape
    proj = np.asarray(cameras.project(x))  # (V, P, M, 2)
    d = np.linalg.norm(proj[:, :, :, None, :] - cand_xy[:, :, None, :, :], axis=-1)
    valid_cand: np.ndarray = np.asarray(np.isfinite(cand_xy).all(-1))  # (V, P, K)
    d = np.where(valid_cand[:, :, None, :], d, np.inf)  # (V, P, M, K)
    nearest_k = np.argmin(d, axis=-1)  # (V, P, M)
    nearest_d = np.min(d, axis=-1)  # (V, P, M)

    vi = np.arange(v)[:, None, None]
    ni = np.arange(n)[None, :, None]
    nearest_xy = cand_xy[vi, ni, nearest_k]  # (V, P, M, 2)
    nearest_s = cand_score[vi, ni, nearest_k]  # (V, P, M)

    hyp_finite: np.ndarray = np.asarray(np.isfinite(x).all(-1))  # (P, M)
    inlier = (nearest_d < inlier_px) & hyp_finite[None]  # (V, P, M)
    obs = np.where(inlier[..., None], nearest_xy, np.nan)
    evidence = np.where(inlier, nearest_s, 0.0).sum(0)  # (P, M)
    n_in = inlier.sum(0)  # (P, M)
    return obs, evidence, n_in


def _prune_joint(
    x_n: Float[np.ndarray, "M 3"],
    evidence_n: Float[np.ndarray, "M"],
    n_in_n: Float[np.ndarray, "M"],
    *,
    max_hyp: int,
    nms_radius: float,
    max_pool: int = 64,
) -> np.ndarray:
    """Indices of up to ``max_hyp`` distinct, well-supported hypotheses for a joint.

    Keeps hypotheses with >= 2 supporting views, strongest evidence first, and
    suppresses any within ``nms_radius`` (3D) of an already-kept one. Only the
    ``max_pool`` strongest candidates are considered (the rest are near-duplicate
    triangulations of the same peaks), which bounds the greedy NMS cost per frame.

    Parameters
    ----------
    x_n
        Candidate 3D positions of shape ``(M, 3)`` for one joint.
    evidence_n, n_in_n
        Per-hypothesis evidence and supporting-view count of shape ``(M,)``.
    max_hyp
        Maximum hypotheses kept.
    nms_radius
        3D suppression radius.
    max_pool
        Cap on the strongest hypotheses considered.

    Returns
    -------
    np.ndarray
        The kept hypothesis indices.
    """
    valid = np.flatnonzero((n_in_n >= 2) & np.isfinite(x_n).all(-1))
    if valid.size == 0:
        return valid
    order = valid[np.argsort(-evidence_n[valid])][:max_pool]
    kept: list[int] = []
    for idx in order:
        p = x_n[idx]
        if all(np.linalg.norm(p - x_n[g]) > nms_radius for g in kept):
            kept.append(int(idx))
        if len(kept) >= max_hyp:
            break
    return np.array(kept, dtype=int)


def _huber(r: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss of residual ``r`` with knee ``delta`` (quadratic then linear)."""
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _chain_dp(
    chain: list[int],
    pos: dict[int, np.ndarray],
    unary: dict[int, np.ndarray],
    target_map: dict[tuple[int, int], float],
    *,
    lam: float,
    scale: float,
    huber: float,
) -> dict[int, int]:
    """Exact Viterbi over one chain: pick a hypothesis index per joint.

    Minimizes ``sum_j unary[j][c_j] + lam * sum_bones huber((len - target)/scale)``.
    Joints with no hypotheses are skipped (left for the caller to NaN), splitting
    the chain into independently-solved runs.

    Parameters
    ----------
    chain
        Ordered joint indices forming a simple path.
    pos
        ``joint -> (S, 3)`` candidate 3D positions.
    unary
        ``joint -> (S,)`` per-hypothesis unary cost.
    target_map
        ``(i, j) -> target bone length`` keyed by sorted endpoint pair.
    lam, scale, huber
        Bone-length prior weight, length scale and Huber knee.

    Returns
    -------
    dict of int to int
        ``{joint: chosen_index}`` for joints that had hypotheses.
    """
    present = [j for j in chain if unary[j].size > 0]
    if not present:
        return {}

    # Split into maximal runs of consecutive (in the chain) present joints.
    pos_in_chain = {j: idx for idx, j in enumerate(chain)}
    runs: list[list[int]] = []
    for j in present:
        if runs and pos_in_chain[j] == pos_in_chain[runs[-1][-1]] + 1:
            runs[-1].append(j)
        else:
            runs.append([j])

    choice: dict[int, int] = {}
    for run in runs:
        cost = unary[run[0]].astype(float).copy()  # (S0,)
        back: list[np.ndarray] = []
        for a, b in zip(run[:-1], run[1:]):
            target = target_map.get((min(a, b), max(a, b)))
            dist = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
            if target is None or not np.isfinite(target):
                pair = np.zeros_like(dist)
            else:
                pair = lam * _huber((dist - target) / max(scale, 1e-9), huber)
            total = cost[:, None] + pair  # (S_prev, S_cur)
            back.append(np.argmin(total, axis=0))
            cost = np.min(total, axis=0) + unary[b]
        c = int(np.argmin(cost))
        states = [c]
        for bp in reversed(back):
            c = int(bp[c])
            states.append(c)
        for j, s in zip(run, reversed(states)):
            choice[j] = s
    return choice


# -- public per-frame solve --------------------------------------------------


def solve_frame(
    cameras: CameraGroup,
    skeleton: Skeleton,
    cand_xy: Float[np.ndarray, "V P K 2"],
    cand_score: Float[np.ndarray, "V P K"],
    target_map: dict[tuple[int, int], float],
    chains: list[list[int]],
    *,
    scale: float,
    max_hyp: int = DEFAULT_MAX_HYP,
    inlier_px: float = DEFAULT_INLIER_PX,
    lam: float,
    huber: float = DEFAULT_HUBER,
    mu: float = DEFAULT_MU,
    prev_pts3d: Float[np.ndarray, "P 3"] | None = None,
) -> tuple[Float[np.ndarray, "P 3"], Float[np.ndarray, "V P 2"]]:
    """Pictorial-structures correction for one multi-camera frame.

    Generates per-joint 3D hypotheses, prunes them, and runs exact chain DP with
    the bone-length prior (and an optional temporal term against ``prev_pts3d``).

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    skeleton
        Skeleton (kept for symmetry with the sequence call).
    cand_xy, cand_score
        Per-frame candidates of shape ``(V, P, K, 2)`` / ``(V, P, K)``.
    target_map
        ``(i, j) -> target bone length`` for the prior.
    chains
        Pre-computed skeleton chains (:func:`skeleton_chains`).
    scale
        Characteristic bone length scaling the prior and NMS radius.
    lam
        Bone-length prior weight, relative to per-view evidence of order 1. Required:
        the shipped value is ``[pictorial_structures] lam``
        (:class:`deeperfly.config.PictorialParams`).
    max_hyp, inlier_px, huber, mu
        Pruning and cost knobs (see the module defaults).
    prev_pts3d
        Previous frame's 3D for the temporal term, or ``None``.

    Returns
    -------
    pts3d : np.ndarray
        Chosen 3D points of shape ``(P, 3)`` (NaN where unsolved).
    obs : np.ndarray
        Per-view 2D observations PS committed to ``(V, P, 2)`` (NaN where
        unsupported).
    """
    v, n, k, _ = cand_xy.shape
    x_all, evidence, n_in, obs_all = _frame_hypotheses(
        cameras, cand_xy, cand_score, inlier_px=inlier_px
    )

    pos: dict[int, np.ndarray] = {}
    unary: dict[int, np.ndarray] = {}
    kept_global: dict[int, np.ndarray] = {}
    for j in range(n):
        keep = _prune_joint(
            x_all[j], evidence[j], n_in[j], max_hyp=max_hyp, nms_radius=0.5 * scale
        )
        kept_global[j] = keep
        pos[j] = x_all[j, keep]  # (S, 3)
        u = -evidence[j, keep].astype(float)  # minimize -> negative evidence
        if (
            mu
            and prev_pts3d is not None
            and np.isfinite(prev_pts3d[j]).all()
            and keep.size
        ):
            jump = np.linalg.norm(pos[j] - prev_pts3d[j], axis=-1) / max(scale, 1e-9)
            u = u + mu * jump * jump
        unary[j] = u

    choice: dict[int, int] = {}
    for chain in chains:
        if len(chain) == 1:  # isolated joint: pick the strongest hypothesis
            j = chain[0]
            if unary[j].size:
                choice[j] = int(np.argmin(unary[j]))
        else:
            choice.update(
                _chain_dp(
                    chain, pos, unary, target_map, lam=lam, scale=scale, huber=huber
                )
            )

    pts3d = np.full((n, 3), np.nan)
    obs = np.full((v, n, 2), np.nan)
    for j, s in choice.items():
        g = int(kept_global[j][s])
        pts3d[j] = x_all[j, g]
        obs[:, j] = obs_all[:, j, g]
    return pts3d, obs


def reconstruct(
    cameras: CameraGroup,
    skeleton: Skeleton,
    candidates: Candidates,
    pts2d_argmax: Float[np.ndarray, "V T P 2"],
    *,
    bone_max_frames: int | None = 100,
    temporal: bool = False,
    max_hyp: int = DEFAULT_MAX_HYP,
    inlier_px: float = DEFAULT_INLIER_PX,
    lam: float,
    huber: float = DEFAULT_HUBER,
    mu: float = DEFAULT_MU,
    fallback_argmax: bool = False,
) -> tuple[
    Float[np.ndarray, "T P 3"], Float[np.ndarray, "V T P 2"], Float[np.ndarray, "V T P"]
]:
    """Run PS correction over a whole sequence.

    The bone-length prior is estimated once from an arg-max triangulation of up to
    ``bone_max_frames`` frames; PS then runs per frame (optionally threading the
    previous frame's 3D for the temporal term). Same shapes/contract as
    :func:`deeperfly.pipeline.reconstruct`.

    Parameters
    ----------
    cameras
        The bundle-adjusted rig.
    skeleton
        Skeleton supplying chains, visibility and the bone-length prior.
    candidates
        The detector's top-K candidate peaks for the sequence.
    pts2d_argmax
        Arg-max 2D of shape ``(V, T, P, 2)`` used to estimate the prior.
    bone_max_frames
        Frames subsampled to estimate the prior (``None`` uses all).
    fallback_argmax
        Where recovery committed no point, keep the arg-max instead of writing ``NaN``.
        The pipeline always passes ``True`` and exposes no knob for it (see
        ``deeperfly.config.PictorialParams``); this parameter stays here so the
        pre-fill array remains reachable, which is what lets a regression test assert
        recovery's own output bit-identically. Off by default for that reason, not as a
        recommendation.
    temporal
        Whether to add the inter-frame temporal term.
    lam
        Bone-length prior weight; see :func:`solve_frame`. Required, for the same
        reason.
    max_hyp, inlier_px, huber, mu
        Per-frame pruning and cost knobs.

    Returns
    -------
    pts3d : np.ndarray
        Corrected 3D of shape ``(T, P, 3)``.
    pts2d : np.ndarray
        Committed per-view 2D of shape ``(V, T, P, 2)``.
    reproj : np.ndarray
        Reprojection error of shape ``(V, T, P)``.
    """
    # Candidates already carry NaN where no pathway produced a (view, point), so
    # the visibility pattern is intrinsic to the detection -- no masking needed.
    cand_xy, cand_score = candidates.xy, candidates.score
    v, t, n, k = candidates.shape

    # Anatomical prior from a cheap arg-max triangulation (subsampled).
    sel = (
        np.arange(t)
        if bone_max_frames is None or t <= bone_max_frames
        else np.linspace(0, t - 1, bone_max_frames).round().astype(int)
    )
    i, j, targets = bone_length_targets(cameras, pts2d_argmax[:, sel], skeleton)
    target_map = {
        (min(int(a), int(b)), max(int(a), int(b))): float(tg)
        for a, b, tg in zip(i, j, targets)
        if np.isfinite(tg)
    }
    scale = float(np.nanmedian(targets)) if np.isfinite(targets).any() else 1.0
    chains = skeleton_chains(skeleton)

    pts3d = np.full((t, n, 3), np.nan)
    pts2d = np.full((v, t, n, 2), np.nan)
    prev = None
    for f in range(t):
        x3, x2 = solve_frame(
            cameras,
            skeleton,
            cand_xy[:, f],
            cand_score[:, f],
            target_map,
            chains,
            scale=scale,
            max_hyp=max_hyp,
            inlier_px=inlier_px,
            lam=lam,
            huber=huber,
            mu=mu,
            prev_pts3d=prev if temporal else None,
        )
        pts3d[f] = x3
        pts2d[:, f] = x2
        prev = x3
    if fallback_argmax:
        # An abstention is "no hypothesis had a candidate within inlier_px", which on a
        # dense detector is a cell to leave alone rather than a cell to delete.
        gap = ~np.isfinite(pts2d).all(-1) & np.isfinite(pts2d_argmax).all(-1)
        pts2d = np.where(gap[..., None], pts2d_argmax, pts2d)
    reproj = reprojection_error(cameras, pts3d, pts2d)
    return pts3d, pts2d, reproj
