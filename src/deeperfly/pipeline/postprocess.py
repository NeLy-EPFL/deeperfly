"""Corrections applied to a finished 3D pose, as an ordered chain of named ops.

Everything upstream estimates the pose from *pixels*: the detector localizes, the rig
triangulates, the smoother fits a trajectory against every view. This module applies
what is known about the **animal** instead -- that some keypoints do not move, that the
body is bilaterally symmetric. Those are priors rather than measurements, and keeping
them out of the estimating stages is deliberate: an estimator that has already been told
the answer cannot be checked against it.

Two consequences shape everything here.

**The ops run last, and that order is forced.** The smoother re-derives the 3D from the
2D observations, so a correction applied before it is followed straight back off by the
fit chasing its pixels. Applied after, it sticks. The honest cost is that these priors
*overwrite* the estimate rather than informing it -- a constraint inside the smoother
would use them to weigh evidence, which is better statistics and a larger change.

**They do not commute, so the chain is ordered.** ``static`` then ``symmetrize`` (with a
single fitted plane) satisfies both properties exactly: a fixed plane maps a constant to
a constant. The other order does not -- a per-axis median of two mirrored points is not
itself mirrored.

Each op is a pure ``(pts2d, pts3d) -> (pts2d, pts3d, report)`` function registered in
:data:`OPS`, and reports what it measurably did so the stage's metadata can say more than
"it ran". Adding a correction is a new entry here plus a line in a config's ``ops``, not
a new pipeline stage.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..config import StaticPointsParams, SymmetrizeParams

log = logging.getLogger("deeperfly")

__all__ = ["OPS", "apply_ops", "op_static", "op_symmetrize"]

#: The temporal-center estimators ``{ op = "static" }.method`` chooses between. A tuple
#: so the error message, the config schema's ``choices`` and this module cannot drift.
STATIC_METHODS = ("median", "mean", "trimmed_mean", "mode", "geometric_median")


# -- shared helpers ------------------------------------------------------------


def _columns(names, skeleton, *, where: str) -> list[int]:
    """Skeleton point names -> column indices, or a ``ValueError`` naming ``where``."""
    index = {name: i for i, name in enumerate(skeleton.point_names)}
    try:
        return [index[str(n)] for n in names]
    except KeyError as e:
        raise ValueError(
            f"{where} references unknown skeleton point {e.args[0]!r}"
        ) from None


def _erase_absent_3d(pts3d: np.ndarray, absent) -> np.ndarray:
    """NaN the ``(frame, point)`` cells declared not on this animal, in 3D.

    The 2D counterpart is :func:`~deeperfly.pipeline.core.apply_absent`. Needed after any
    op that broadcasts one value over time: a limb lost part-way through a recording is
    real for the earlier frames, so a center taken over those is finite, and writing it
    into every frame would resurrect the limb for the frames it was declared gone.
    """
    if absent is None:
        return pts3d
    mask = np.asarray(absent, dtype=bool)
    if mask.ndim == 1:  # a whole-recording (P,) declaration broadcasts over time
        mask = np.broadcast_to(mask.reshape(1, -1), np.shape(pts3d)[:2])
    if not mask.any():
        return pts3d
    pts3d = np.array(pts3d, dtype=float)
    pts3d[mask] = np.nan
    return pts3d


def _moved(before: np.ndarray, after: np.ndarray, cols) -> tuple[float, float]:
    """``(median, p90)`` distance an op moved the given columns.

    Taken as ``|before - after|`` rather than against a recomputed reference, so the
    number always describes what the op actually did. A cell the input left NaN
    contributes nothing: an op may fill it, but "how far it moved" is not a question an
    unobserved cell has an answer to.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.linalg.norm(
            np.take(before, cols, axis=-2) - np.take(after, cols, axis=-2), axis=-1
        )
        if not np.isfinite(d).any():
            return float("nan"), float("nan")
        return float(np.nanmedian(d)), float(np.nanpercentile(d, 90))


# -- op: static ----------------------------------------------------------------


def _half_sample_mode(values: np.ndarray) -> float:
    """The half-sample mode of a 1-D sample (Robertson-Cryer), NaNs dropped.

    Recursively keep the *half* of the sorted sample with the smallest range, until two
    values remain, and return their midpoint. Parameter-free -- unlike a histogram or
    KDE mode, which would make the answer depend on a bin width nobody can set from a
    config -- and it has the median's 50% breakdown point.

    What it buys over the median is a *majority* contaminant with the wrong shape. The
    median is the 50th percentile, so it follows the count; this follows the density. When
    the correct detections are tight and outnumbered and the wrong ones are scattered, the
    median lands out in the middle on a position the point never occupied and this lands
    on the true peak. On clean unimodal data it is simply noisier, which is why it is not
    the default.
    """
    v = np.sort(values[np.isfinite(values)])
    n = v.size
    if n == 0:
        return float("nan")
    while n > 2:
        half = n // 2  # the window's width in samples
        widths = v[half - 1 :] - v[: n - half + 1]
        start = int(np.argmin(widths))
        v = v[start : start + half]
        n = v.size
    return float(v.mean())


def _geometric_median(
    points: np.ndarray, *, iterations: int = 64, tol: float = 1e-9
) -> np.ndarray:
    """The L1 multivariate median of ``(N, D)`` samples, by Weiszfeld iteration.

    The point minimizing the sum of *Euclidean* distances to the sample, rather than the
    per-axis median, which minimizes each coordinate's absolute deviations separately.
    The difference that matters here is equivariance: rotate the world frame and this
    answer rotates with it, while a per-axis median gives a different point. A camera
    rig's world axes are an arbitrary choice, so an estimator that depends on them is
    reporting partly on the rig.

    Seeded from the per-axis median and guarded at the degenerate case Weiszfeld is
    famous for -- an iterate landing exactly on a sample point, where the update divides
    by zero -- by returning that point, which is then the exact minimizer of a term that
    can only grow if it moves. Rows with any non-finite coordinate are dropped.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=-1)]
    if pts.shape[0] == 0:
        return np.full(np.shape(points)[-1], np.nan)
    if pts.shape[0] <= 2:
        return pts.mean(axis=0)
    guess = np.median(pts, axis=0)
    for _ in range(iterations):
        dist = np.linalg.norm(pts - guess, axis=-1)
        if (dist < tol).any():
            return pts[int(np.argmin(dist))]
        weights = 1.0 / dist
        nxt = (weights[:, None] * pts).sum(axis=0) / weights.sum()
        if np.linalg.norm(nxt - guess) < tol:
            return nxt
        guess = nxt
    return guess


def _temporal_center(
    arr: np.ndarray, axis: int, *, method: str, trim: float
) -> np.ndarray:
    """Collapse ``arr`` along ``axis`` to one value per remaining cell.

    The shared reducer behind the 3D and per-view 2D freezes, so the two cannot end up
    using different estimators. ``median`` / ``mean`` / ``trimmed_mean`` / ``mode`` reduce
    one coordinate at a time; ``geometric_median`` treats the last axis as a vector and is
    dispatched by the caller, which knows which axis that is.

    An all-NaN cell (a point no frame observed, or a view that never sees it) reduces to
    NaN rather than raising -- that is the "not observed" case, and it must survive.
    """
    with warnings.catch_warnings():  # all-NaN cells are expected, not exceptional
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "median":
            return np.nanmedian(arr, axis=axis)
        if method == "mean":
            return np.nanmean(arr, axis=axis)
        if method == "trimmed_mean":
            if not 0.0 <= trim < 0.5:
                raise ValueError(
                    f'{{ op = "static" }}.trim must be in [0, 0.5), got {trim!r} -- it '
                    "is the fraction dropped from EACH tail, so 0.5 drops everything"
                )
            lo = np.nanpercentile(arr, 100.0 * trim, axis=axis, keepdims=True)
            hi = np.nanpercentile(arr, 100.0 * (1.0 - trim), axis=axis, keepdims=True)
            keep = np.where((arr >= lo) & (arr <= hi), arr, np.nan)
            return np.nanmean(keep, axis=axis)
        if method == "mode":
            return np.apply_along_axis(_half_sample_mode, axis, arr)
    raise ValueError(
        f'unknown {{ op = "static" }}.method {method!r} ({"|".join(STATIC_METHODS)})'
    )


def freeze_3d(
    pts3d: np.ndarray, cols: list[int], *, method: str = "median", trim: float = 0.1
) -> np.ndarray:
    """Replace the ``cols`` of ``pts3d`` ``(T, P, 3)`` with one temporal center each.

    The center is broadcast back over all frames, so the pose is steady and occluded
    (NaN) frames are filled in. A column that is never observed stays all-NaN. Returns a
    copy; the input is not mutated.
    """
    pts3d = np.array(pts3d, dtype=float)
    block = pts3d[:, cols, :]  # (T, R, 3)
    if method == "geometric_median":
        # Per point, over time: the coordinate axis is a vector, not another cell.
        center = np.stack([_geometric_median(block[:, i]) for i in range(len(cols))])
    else:
        center = _temporal_center(block, axis=0, method=method, trim=trim)
    pts3d[:, cols, :] = center
    return pts3d


def freeze_2d(
    pts2d: np.ndarray, cols: list[int], *, method: str = "median", trim: float = 0.1
) -> np.ndarray:
    """The 2D counterpart: freeze the ``cols`` of ``pts2d`` ``(V, T, P, 2)`` per view.

    Each view gets its *own* temporal center, taken in that view's pixels -- not the
    reprojection of the frozen 3D. The two are near-identical for a static point (a
    projection is locally linear, so a center commutes with it to first order), and the
    per-view one is what stays a *pixel measurement*: reprojecting instead would move
    these points by the rig's residual, a few px on a solved fly rig, away from where the
    detector actually sees them. The cost is that the frozen 2D is then not exactly the
    projection of the frozen 3D.

    A ``(view, point)`` pair the view never observes stays all-NaN, so this does not
    invent an observation in a camera that cannot see the point; within a view that does
    see it, the center fills the frames where the detection dropped out, which is the
    whole premise of calling the point static. Returns a copy.
    """
    pts2d = np.array(pts2d, dtype=float)
    block = pts2d[:, :, cols, :]  # (V, T, R, 2)
    if method == "geometric_median":
        center = np.stack(
            [
                np.stack([_geometric_median(block[v, :, i]) for i in range(len(cols))])
                for v in range(block.shape[0])
            ]
        )
    else:
        center = _temporal_center(block, axis=1, method=method, trim=trim)
    pts2d[:, :, cols, :] = center[:, None]
    return pts2d


def op_static(pts2d, pts3d, *, skeleton, spec: dict):
    """``{ op = "static" }`` -- collapse the listed points to one position each.

    See :class:`~deeperfly.config.StaticPointsParams` for the options and for what each
    ``method`` assumes about the contamination.
    """
    p = _spec_params(spec, StaticPointsParams, op="static")
    if p.method not in STATIC_METHODS:
        raise ValueError(
            f'unknown {{ op = "static" }}.method {p.method!r} '
            f"({'|'.join(STATIC_METHODS)})"
        )
    report: dict = {"op": "static", "points": list(p.points), "method": p.method}
    if p.method == "trimmed_mean":
        report["trim"] = float(p.trim)
    if not p.points:
        return pts2d, pts3d, report
    cols = _columns(p.points, skeleton, where='{ op = "static" }.points')
    out2d = freeze_2d(pts2d, cols, method=p.method, trim=p.trim)
    out3d = freeze_3d(pts3d, cols, method=p.method, trim=p.trim)
    report["moved_2d_median_px"], report["moved_2d_p90_px"] = _moved(pts2d, out2d, cols)
    report["moved_3d_median"], report["moved_3d_p90"] = _moved(pts3d, out3d, cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per = np.nanmedian(
            np.linalg.norm(pts2d[:, :, cols] - out2d[:, :, cols], axis=-1), axis=(0, 1)
        )
    report["moved_2d_median_px_per_point"] = {
        n: float(per[i]) for i, n in enumerate(p.points)
    }
    finite = np.isfinite(per)
    report["worst_point"] = (
        f"{p.points[int(np.nanargmax(per))]} ({np.nanmax(per):.2f} px)"
        if finite.any()
        else None
    )
    return out2d, out3d, report


# -- op: symmetrize ------------------------------------------------------------


def sagittal_plane(pts3d: np.ndarray, pairs: list[tuple[int, int]]):
    """Fit the animal's mirror plane from left/right pairs: ``(normal, offset)``.

    Each pair contributes two independent facts, and the fit uses both: the **midpoint**
    of a pair lies on the plane, and the **difference** vector between the two sides is
    normal to it. So the normal is the dominant direction of the difference vectors --
    taken as the leading singular vector rather than a mean, since a pair's sign is
    arbitrary and averaging signed differences can cancel them to nothing -- and the
    offset places the plane through the centroid of the midpoints.

    Fitting from pairs *only* is what keeps the op honest: a point named in ``midline``
    is about to be moved onto this plane, so letting it vote on where the plane is would
    be circular, and a laterally-bent abdomen would drag the plane with it.

    Returns ``(None, None)`` when fewer than two pairs are usable, which is the
    under-determined case rather than an error: one pair fixes a normal but no offset
    along it.
    """
    left = pts3d[..., [a for a, _ in pairs], :]
    right = pts3d[..., [b for _, b in pairs], :]
    ok = np.isfinite(left).all(-1) & np.isfinite(right).all(-1)
    if ok.sum() < 2:
        return None, None
    diff = (left - right)[ok]  # (N, 3) normal directions, sign arbitrary
    mid = ((left + right) / 2.0)[ok]  # (N, 3) points on the plane
    # Leading right singular vector of the (sign-free) difference scatter.
    _, _, vt = np.linalg.svd(diff, full_matrices=False)
    normal = vt[0] / np.linalg.norm(vt[0])
    return normal, float(normal @ mid.mean(axis=0))


def _mirror(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Reflect ``points`` through the plane ``normal . x = offset``."""
    signed = points @ normal - offset
    return points - 2.0 * signed[..., None] * normal


def symmetrize_3d(
    pts3d: np.ndarray,
    pairs: list[tuple[int, int]],
    midline: list[int],
    *,
    per_frame: bool = False,
    strength: float = 1.0,
):
    """Make ``pairs`` mirror-symmetric and put ``midline`` on the plane.

    Each side is moved half-way toward the mirror of the other (scaled by ``strength``),
    which is the correction that does not privilege one side -- averaging into one side
    would import that side's error into both. Midline points are moved onto the plane
    along its normal, the shortest move that satisfies the constraint.

    With ``per_frame=False`` the plane is fitted once from the whole recording, which
    makes this a *fixed* map: an already-static point stays static through it, so a
    ``static`` op followed by this one satisfies both properties exactly. Returns
    ``(pts3d, normal, offset)``; ``normal`` is ``None`` when no plane could be fitted.
    """
    out = np.array(pts3d, dtype=float)
    if strength <= 0.0 or (not pairs and not midline):
        return out, None, None

    def apply(block: np.ndarray, normal, offset):
        if normal is None:
            return
        for a, b in pairs:
            la, rb = block[..., a, :], block[..., b, :]
            target_a = _mirror(rb, normal, offset)
            target_b = _mirror(la, normal, offset)
            # Both reads happen before either write: sequential assignment would
            # symmetrize the second point against an already-moved first one.
            block[..., a, :] = la + 0.5 * strength * (target_a - la)
            block[..., b, :] = rb + 0.5 * strength * (target_b - rb)
        for m in midline:
            pt = block[..., m, :]
            signed = (pt @ normal - offset)[..., None]
            block[..., m, :] = pt - strength * signed * normal

    if per_frame:
        normal = offset = None
        for t in range(out.shape[0]):
            n_t, o_t = sagittal_plane(out[t : t + 1], pairs)
            apply(out[t], n_t, o_t)
            normal, offset = n_t, o_t  # the last frame's, for the report
    else:
        normal, offset = sagittal_plane(out, pairs)
        apply(out, normal, offset)
    return out, normal, offset


def op_symmetrize(pts2d, pts3d, *, skeleton, spec: dict):
    """``{ op = "symmetrize" }`` -- impose bilateral symmetry on the body-fixed points.

    3D only: the 2D is left exactly as it was. A per-view 2D detection is a measurement in
    that camera's pixels, and there is no sense in which two *different cameras'* pixels
    mirror each other -- the symmetry is a fact about the animal in the world, so it is
    imposed where the animal is. See :class:`~deeperfly.config.SymmetrizeParams`.
    """
    p = _spec_params(spec, SymmetrizeParams, op="symmetrize")
    report: dict = {
        "op": "symmetrize",
        "pairs": [list(pair) for pair in p.pairs],
        "midline": list(p.midline),
        "per_frame": bool(p.per_frame),
        "strength": float(p.strength),
    }
    for pair in p.pairs:
        if len(pair) != 2:
            raise ValueError(
                f'{{ op = "symmetrize" }}.pairs entries must be [left, right]; got {pair!r}'
            )
    flat = [n for pair in p.pairs for n in pair]
    cols = _columns(flat, skeleton, where='{ op = "symmetrize" }.pairs')
    pair_cols = list(zip(cols[0::2], cols[1::2]))
    mid_cols = _columns(p.midline, skeleton, where='{ op = "symmetrize" }.midline')
    if not pair_cols and not mid_cols:
        return pts2d, pts3d, report
    out3d, normal, offset = symmetrize_3d(
        pts3d, pair_cols, mid_cols, per_frame=p.per_frame, strength=p.strength
    )
    if normal is None:
        log.warning(
            "symmetrize: could not fit a sagittal plane (fewer than two usable "
            "left/right pairs); the pose is unchanged"
        )
        report["fitted"] = False
        return pts2d, pts3d, report
    report["fitted"] = True
    report["plane_normal"] = [float(v) for v in normal]
    report["plane_offset"] = float(offset)
    touched = [c for pair in pair_cols for c in pair] + mid_cols
    report["moved_3d_median"], report["moved_3d_p90"] = _moved(pts3d, out3d, touched)
    # How far the pairs were from symmetric BEFORE, which is what says whether the
    # premise held: a pair already symmetric to within the noise had nothing to fix.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        asym = [
            float(
                np.nanmedian(
                    np.linalg.norm(
                        pts3d[:, a] - _mirror(pts3d[:, b], normal, offset), axis=-1
                    )
                )
            )
            for a, b in pair_cols
        ]
    report["asymmetry_before"] = {
        f"{pair[0]}|{pair[1]}": v for pair, v in zip(p.pairs, asym)
    }
    return pts2d, out3d, report


# -- the chain -----------------------------------------------------------------

#: ``op`` name -> implementation. Adding a correction is an entry here plus a line in a
#: config's ``ops``; it is deliberately not a new pipeline stage.
OPS = {"static": op_static, "symmetrize": op_symmetrize}


def _spec_params(spec: dict, cls, *, op: str):
    """Build an op's frozen params from its inline table, rejecting unknown keys.

    Mirrors ``Config``'s own strict ``_params`` loader rather than reusing it, because an
    op's table is an element of a list rather than a named section -- but the contract is
    the same: a key the dataclass does not define is a typo, and a typo is an error that
    names what was allowed.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(spec) - known - {"op"})
    if unknown:
        raise ValueError(
            f'{{ op = "{op}" }} has unknown key(s) {unknown}; allowed: {sorted(known)}'
        )
    return cls(**{k: v for k, v in spec.items() if k in known})


def apply_ops(pts2d, pts3d, *, ops, skeleton, absent=None):
    """Run an ordered ``ops`` chain over a pose, returning ``(pts2d, pts3d, reports)``.

    Each op sees the previous op's output, which is why the list is ordered and why the
    reports are a list rather than a dict: the same op may appear twice, and what the
    second one measures depends on what the first one did.

    Parameters
    ----------
    pts2d, pts3d
        The pose to correct, ``(V, T, P, 2)`` and ``(T, P, 3)``.
    ops
        The ``[postprocess].ops`` list: inline tables, each with an ``op`` key.
    skeleton
        Resolves point names to columns.
    absent
        The operator's declaration of which keypoints are not on this animal, re-applied
        once after the chain: an op that broadcasts a value over time would otherwise
        resurrect a limb lost part-way through the recording.

    Raises
    ------
    ValueError
        If an entry names no ``op``, names an unknown one, or carries an unknown key.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    pts3d = np.asarray(pts3d, dtype=float)
    reports: list[dict] = []
    for i, spec in enumerate(ops or []):
        if not isinstance(spec, dict) or "op" not in spec:
            raise ValueError(
                f"[postprocess].ops[{i}] must be a table with an `op` key, "
                f'e.g. {{ op = "static", points = [...] }}; got {spec!r}'
            )
        name = str(spec["op"])
        if name not in OPS:
            raise ValueError(
                f"[postprocess].ops[{i}] has unknown op {name!r}; "
                f"choose from {', '.join(sorted(OPS))}"
            )
        pts2d, pts3d, report = OPS[name](pts2d, pts3d, skeleton=skeleton, spec=spec)
        reports.append(report)
    # Once, at the end: any op that broadcasts a value over time can resurrect a limb
    # the operator declared gone part-way through, and re-erasing after each op would
    # only differ if a later op read the resurrected cell -- which for a *static* point
    # is the same constant either way.
    from ..pipeline.core import apply_absent

    pts2d, _ = apply_absent(pts2d, None, absent)
    return pts2d, _erase_absent_3d(pts3d, absent), reports
