"""Pictorial-structures (PS) multi-view 2D->3D correction (DeepFly3D-style).

This is the optional, accuracy-oriented alternative to the default
reprojection-outlier rejection in :func:`deeperfly.pipeline.reconstruct`. Where
that path can only *veto* a bad per-view detection (drop the worst-reprojecting
observation and re-triangulate), pictorial structures can *recover* the correct
joint when the detector's arg-max landed on the wrong heatmap peak -- the classic
Drosophila failure cases (self-occlusion, crossing legs, left/right confusion).

The idea, following Gunel et al. (DeepFly3D, 2019):

1. Keep the **top-K candidate peaks** per (view, joint), not just the arg-max
   (:func:`extract_candidates`). Full heatmaps are never retained for a sequence.
2. For each joint, build a pool of multi-view-consistent **3D hypotheses** by
   triangulating candidate pairs across views, refitting from inlier views, and
   scoring by summed heatmap confidence (:func:`_joint_hypotheses`, batched per
   frame in :func:`solve_frame`).
3. Choose one hypothesis per joint by **exact dynamic programming** along each
   limb (:func:`_chain_dp`). The fly skeleton's 2D bones form a forest of simple
   chains (each leg a 5-joint path, each stripe a 3-marker path), so the MAP over
   the bone-length-coupled graphical model is exact -- no loopy belief
   propagation. An optional causal temporal term penalises 3D jumps.

Everything is plain NumPy over a :class:`~deeperfly.cameras.CameraGroup` and
:class:`~deeperfly.skeleton.Skeleton`, mirroring :mod:`deeperfly.triangulate` /
:mod:`deeperfly.pipeline`. The detector forward and heatmap decode happen
upstream; this module only consumes candidate peaks + calibrated cameras.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from jaxtyping import Float

from .cameras import CameraGroup
from .skeleton import Skeleton
from .triangulate import reprojection_error

# Defaults (all overridable through the pipeline / CLI).
DEFAULT_K = 5  # candidate peaks kept per (view, joint)
DEFAULT_MAX_HYP = 10  # 3D hypotheses kept per joint after pruning
DEFAULT_INLIER_PX = 15.0  # a view supports a 3D hypothesis if a candidate is this close
DEFAULT_LAMBDA = 1.0  # bone-length prior weight (relative to per-view evidence ~O(1))
DEFAULT_HUBER = 0.5  # Huber knee for the bone-length residual, in units of bone length
DEFAULT_MU = 5.0  # temporal weight (per unit squared 3D displacement / bone-scale^2)
DEFAULT_PEAK_THRESHOLD = 0.05  # ignore heatmap peaks weaker than this
DEFAULT_PEAK_RADIUS = 2  # NMS neighbourhood half-width (heatmap pixels)


@dataclass(frozen=True)
class Candidates:
    """Top-K detector peaks per (view, point) for a sequence, in image pixels.

    ``xy`` is ``(V, T, N, K, 2)`` and ``score`` is ``(V, T, N, K)``; padded /
    invisible / sub-threshold slots are ``NaN`` (``xy``) and ``0`` (``score``).
    The arg-max (``K = 0``) reproduces the single-peak detection, so calibration
    can still use the plain 2D path while PS consumes the full candidate set.
    """

    xy: Float[np.ndarray, "V T N K 2"]
    score: Float[np.ndarray, "V T N K"]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        v, t, n, k, _ = self.xy.shape
        return v, t, n, k

    def frame(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Candidate ``(xy, score)`` for one frame: ``(V, N, K, 2)`` and ``(V, N, K)``."""
        return self.xy[:, t], self.score[:, t]


# -- candidate extraction ----------------------------------------------------


def peak_candidates(
    heatmaps: Float[np.ndarray, "*chan Hh Ww"],
    k: int = DEFAULT_K,
    *,
    radius: int = DEFAULT_PEAK_RADIUS,
    threshold: float = DEFAULT_PEAK_THRESHOLD,
) -> tuple[Float[np.ndarray, "*chan K 2"], Float[np.ndarray, "*chan K"]]:
    """Top-``k`` local-maxima peaks per heatmap channel (normalised ``(x, y)`` + score).

    A pixel is a peak if it is the maximum of its ``(2*radius+1)`` neighbourhood
    and exceeds ``threshold``; the strongest ``k`` peaks are returned, ordered by
    score and padded with ``NaN`` / ``0`` when fewer than ``k`` exist. Coordinates
    are normalised to ``[0, 1]`` like :func:`~deeperfly.pose2d.inference.heatmap_to_points`.
    """
    from scipy.ndimage import maximum_filter

    hm = np.asarray(heatmaps, dtype=float)
    hh, ww = hm.shape[-2:]
    size = (1,) * (hm.ndim - 2) + (2 * radius + 1, 2 * radius + 1)
    is_peak = (hm == maximum_filter(hm, size=size)) & (hm > threshold)
    flat = np.where(is_peak, hm, -np.inf).reshape(*hm.shape[:-2], hh * ww)

    k = min(k, flat.shape[-1])
    top = np.argpartition(-flat, k - 1, axis=-1)[..., :k]
    top_val = np.take_along_axis(flat, top, axis=-1)
    order = np.argsort(-top_val, axis=-1)  # strongest first
    idx = np.take_along_axis(top, order, axis=-1)
    val = np.take_along_axis(top_val, order, axis=-1)

    row, col = idx // ww, idx % ww
    xy = np.stack([col / ww, row / hh], axis=-1).astype(float)
    valid = np.isfinite(val)
    xy = np.where(valid[..., None], xy, np.nan)
    score = np.where(valid, val, 0.0)
    return xy, score


def extract_candidates(
    heatmaps: Float[np.ndarray, "V J Hh Ww"],
    sides: list[str],
    flips: list[bool],
    image_size: list[tuple[int, int]],
    *,
    k: int = DEFAULT_K,
    n_points: int = 38,
    n_side_joints: int = 19,
    **peak_kwargs,
) -> tuple[Float[np.ndarray, "V N K 2"], Float[np.ndarray, "V N K"]]:
    """Scatter per-view top-K single-side peaks into the full skeleton (pixels).

    The candidate analogue of :func:`deeperfly.pose2d.inference.assemble_skeleton`:
    extracts K peaks per detector channel, undoes the mirror flip (``x -> 1 - x``),
    scales to original pixels, and places a right camera's 19 channels into skeleton
    indices ``0..18`` and a left camera's into ``19..37``.
    """
    xy_norm, score = peak_candidates(heatmaps, k, **peak_kwargs)  # (V, J, K, 2/.)
    v = len(sides)
    cand_xy = np.full((v, n_points, k, 2), np.nan)
    cand_score = np.zeros((v, n_points, k))
    for i in range(v):
        p = xy_norm[i].copy()  # (J, K, 2)
        if flips[i]:
            p[..., 0] = 1.0 - p[..., 0]
        w, h = image_size[i]
        p = p * np.array([w, h])
        sl = (
            slice(0, n_side_joints)
            if sides[i] == "right"
            else slice(n_side_joints, 2 * n_side_joints)
        )
        cand_xy[i, sl] = p
        cand_score[i, sl] = score[i]
    return cand_xy, cand_score


def apply_visibility(
    cand_xy: Float[np.ndarray, "V *rest N K 2"],
    cand_score: Float[np.ndarray, "V *rest N K"],
    skeleton: Skeleton,
    camera_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """NaN out candidates for (camera, point) pairs the rig cannot see.

    Mirrors :func:`deeperfly.triangulate.apply_visibility` but for the candidate
    arrays (an extra trailing ``K`` axis). Broadcasts the ``(V, N)`` visibility
    mask over any middle (e.g. time) axes.
    """
    cand_xy = np.array(cand_xy, dtype=float)
    cand_score = np.array(cand_score, dtype=float)
    mask = skeleton.visibility_mask(camera_names)  # (V, N)
    n_mid = cand_xy.ndim - 4  # axes between V and N
    m = mask.reshape((mask.shape[0], *([1] * n_mid), mask.shape[1], 1))  # (V,*1,N,1)
    cand_xy = np.where(m[..., None], cand_xy, np.nan)
    cand_score = np.where(m, cand_score, 0.0)
    return cand_xy, cand_score


# -- bone-length prior (shared with calibration) -----------------------------


def bone_length_targets(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V F N 2"],
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median bone length per skeleton bone, from an initial triangulation.

    Returns the bone endpoint index arrays ``(i, j)`` (columns of
    :attr:`Skeleton.bones`) and the per-bone target length ``targets`` (``(B,)``).
    Shared by bundle-adjustment calibration (:func:`deeperfly.pipeline._bone_prior`)
    and PS so the two agree on the anatomical prior.
    """
    import warnings

    from .triangulate import triangulate

    pts3d0 = triangulate(cameras, pts2d)  # (F, N, 3)
    i, j = skeleton.bone_index_pairs()
    lengths = np.linalg.norm(pts3d0[:, i] - pts3d0[:, j], axis=-1)  # (F, B)
    with warnings.catch_warnings():  # a never-triangulated bone -> NaN target (ok)
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = np.nanmedian(lengths, axis=0)  # (B,)
    return i, j, targets


# -- skeleton chains ---------------------------------------------------------


def skeleton_chains(skeleton: Skeleton) -> list[list[int]]:
    """Decompose the 2D bones into ordered simple chains (paths).

    Each connected component of :attr:`Skeleton.bones` is a path (max degree 2),
    so it is returned as an ordered joint list walked from an endpoint; isolated
    points (no 2D bone, e.g. the antennae) come back as singletons. This ordering
    is what :func:`_chain_dp` runs exact Viterbi over.
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in skeleton.bones:
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
    pairs = np.array(list(combinations(range(v), 2)), dtype=int)  # (P, 2)
    a = np.repeat(np.arange(k), k)  # (K*K,) slow index
    b = np.tile(np.arange(k), k)  # (K*K,) fast index
    vv = np.repeat(pairs[:, 0], k * k)
    ww = np.repeat(pairs[:, 1], k * k)
    aa = np.tile(a, len(pairs))
    bb = np.tile(b, len(pairs))
    return vv, ww, aa, bb


def _frame_hypotheses(
    cameras: CameraGroup,
    cand_xy: Float[np.ndarray, "V N K 2"],
    cand_score: Float[np.ndarray, "V N K"],
    *,
    inlier_px: float,
):
    """All multi-view 3D hypotheses for one frame's joints, scored by evidence.

    Returns ``(X, evidence, n_inlier, obs)`` where ``X`` is ``(N, M, 3)`` refit
    3D hypotheses (``M = C(V,2) K^2``), ``evidence`` ``(N, M)`` is the summed
    heatmap confidence of supporting views, ``n_inlier`` ``(N, M)`` counts them,
    and ``obs`` ``(V, N, M, 2)`` are the per-view candidate observations chosen for
    each hypothesis (``NaN`` for non-supporting views). All geometry is two
    batched triangulate + project calls over the whole frame.
    """
    v, n, k, _ = cand_xy.shape
    vv, ww, aa, bb = _combo_index(v, k)
    m = len(vv)
    rng = np.arange(m)

    # Build (V, N, M, 2): each hypothesis activates its two views' chosen candidates.
    pts = np.full((v, n, m, 2), np.nan)
    pts[vv, :, rng] = cand_xy[vv, :, aa]  # (M, N, 2) -> view vv[m], hyp m
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
    x: Float[np.ndarray, "N M 3"],
    cand_xy: Float[np.ndarray, "V N K 2"],
    cand_score: Float[np.ndarray, "V N K"],
    inlier_px: float,
):
    """Reproject hypotheses and gather per-view nearest-candidate support.

    Returns ``(obs, evidence, n_inlier)``: ``obs`` ``(V, N, M, 2)`` is the nearest
    in-threshold candidate per view (else ``NaN``), ``evidence`` ``(N, M)`` the
    summed score of supporting views, ``n_inlier`` ``(N, M)`` their count.
    """
    v, n, k, _ = cand_xy.shape
    proj = np.asarray(cameras.project(x))  # (V, N, M, 2)
    d = np.linalg.norm(proj[:, :, :, None, :] - cand_xy[:, :, None, :, :], axis=-1)
    valid_cand = np.isfinite(cand_xy).all(-1)  # (V, N, K)
    d = np.where(valid_cand[:, :, None, :], d, np.inf)  # (V, N, M, K)
    nearest_k = np.argmin(d, axis=-1)  # (V, N, M)
    nearest_d = np.min(d, axis=-1)  # (V, N, M)

    vi = np.arange(v)[:, None, None]
    ni = np.arange(n)[None, :, None]
    nearest_xy = cand_xy[vi, ni, nearest_k]  # (V, N, M, 2)
    nearest_s = cand_score[vi, ni, nearest_k]  # (V, N, M)

    hyp_finite = np.isfinite(x).all(-1)  # (N, M)
    inlier = (nearest_d < inlier_px) & hyp_finite[None]  # (V, N, M)
    obs = np.where(inlier[..., None], nearest_xy, np.nan)
    evidence = np.where(inlier, nearest_s, 0.0).sum(0)  # (N, M)
    n_in = inlier.sum(0)  # (N, M)
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

    Minimises ``sum_j unary[j][c_j] + lam * sum_bones huber((len - target)/scale)``.
    Joints with no hypotheses are skipped (left for the caller to NaN), splitting
    the chain into independently-solved runs. Returns ``{joint: chosen_index}``
    only for joints that had hypotheses.
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
    cand_xy: Float[np.ndarray, "V N K 2"],
    cand_score: Float[np.ndarray, "V N K"],
    target_map: dict[tuple[int, int], float],
    chains: list[list[int]],
    *,
    scale: float,
    max_hyp: int = DEFAULT_MAX_HYP,
    inlier_px: float = DEFAULT_INLIER_PX,
    lam: float = DEFAULT_LAMBDA,
    huber: float = DEFAULT_HUBER,
    mu: float = DEFAULT_MU,
    prev_pts3d: Float[np.ndarray, "N 3"] | None = None,
) -> tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "V N 2"]]:
    """Pictorial-structures correction for one multi-camera frame.

    Generates per-joint 3D hypotheses, prunes them, and runs exact chain DP with
    the bone-length prior (and an optional temporal term against ``prev_pts3d``).
    Returns the chosen 3D points ``(N, 3)`` and the per-view 2D observations PS
    committed to ``(V, N, 2)`` (``NaN`` where unsolved / unsupported).
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
        u = -evidence[j, keep].astype(float)  # minimise -> negative evidence
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
    pts2d_argmax: Float[np.ndarray, "V T N 2"],
    *,
    bone_max_frames: int | None = 100,
    temporal: bool = False,
    max_hyp: int = DEFAULT_MAX_HYP,
    inlier_px: float = DEFAULT_INLIER_PX,
    lam: float = DEFAULT_LAMBDA,
    huber: float = DEFAULT_HUBER,
    mu: float = DEFAULT_MU,
) -> tuple[
    Float[np.ndarray, "T N 3"], Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]
]:
    """Run PS correction over a whole sequence (the drop-in for reproject mode).

    The bone-length prior is estimated once from an arg-max triangulation of up to
    ``bone_max_frames`` frames; PS then runs per frame (optionally threading the
    previous frame's 3D for the temporal term). Returns ``(pts3d, pts2d, reproj)``
    with the same shapes/contract as :func:`deeperfly.pipeline.reconstruct`.
    """
    names = cameras.names
    cand_xy, cand_score = apply_visibility(
        candidates.xy, candidates.score, skeleton, names
    )
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
    reproj = reprojection_error(cameras, pts3d, pts2d)
    return pts3d, pts2d, reproj
