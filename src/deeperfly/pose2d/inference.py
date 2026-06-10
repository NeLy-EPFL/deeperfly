"""Orchestration: run the detector(s) and assemble 2D skeletons from a plan.

Sits above the detector seam (:mod:`deeperfly.pose2d.detector`) and the model
registry (:mod:`deeperfly.pose2d.models`). A :class:`~deeperfly.pose2d.pathways.DetectionPlan`
says *what* to detect -- which footage source feeds which preprocessor + model,
and how each model output channel maps to a ``(view, skeleton-point)``. Pipeline
for one recording:

1. Decode each **source** once into a ``(T, H, W, 3)`` window.
2. For each **pathway**: orient the frame with the pathway's preprocessor (a
   mirror/crop/...), let its **model** resize + normalize + forward + decode to
   normalized peaks, then map those peaks back into the view frame by inverting
   the pathway's preprocessing (:func:`~deeperfly.pose2d.pathways.map_to_view`)
   and scatter them into the ``(V, N)`` skeleton
   (:func:`~deeperfly.pose2d.pathways.scatter_pathway`).

A ``(view, point)`` pair that no pathway writes stays ``NaN`` -- that is how
visibility is encoded, with no separate table. A source can feed several
pathways: the front camera, for instance, is one source feeding two pathways
(one mirrored) that both map into view ``f``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from jaxtyping import Float, Int

from .pathways import DetectionPlan, map_to_view, scatter_pathway


def _to_torch_image(image):
    """Image -> torch tensor, staying on-device for GPU inputs (zero-copy).

    A ``torch.Tensor`` passes through untouched. A GPU-decoded frame from another
    array library arrives DLPack-capable already on the device; bridge it zero-copy
    so it never round-trips through host memory. NumPy is copied to a writable
    tensor (``torch.from_numpy`` warns on the immutable array ``np.array`` makes).
    """
    import torch

    if isinstance(image, torch.Tensor):
        return image
    if hasattr(image, "__dlpack__"):  # most array libs -- same-device, zero-copy
        return torch.from_dlpack(image)
    return torch.from_numpy(np.array(image))


def _window_to_device(window, device):
    """Move one source's ``(T, H, W, 3)`` window onto the detector ``device`` once.

    Without this, each pathway reading the source would re-upload a frame -- many
    tiny synchronous host->device copies. Uploading the whole window in one
    transfer collapses that to a single copy; per-frame preparation then just
    re-slices on-device. An already-on-device window (a GPU-decoded tensor) is
    moved only if needed.
    """
    import torch

    if isinstance(window, torch.Tensor):
        return window.to(device)
    if hasattr(window, "__dlpack__"):
        return torch.from_dlpack(window).to(device)
    return torch.from_numpy(np.ascontiguousarray(window)).to(device)


def _image_wh(image) -> tuple[int, int]:
    """``(width, height)`` of an ``(H, W, ...)`` frame without copying to host."""
    shape = getattr(image, "shape", None)
    if shape is None:
        shape = np.asarray(image).shape
    return int(shape[1]), int(shape[0])


def _image_hw(image) -> tuple[int, int]:
    """``(height, width)`` of an ``(H, W, ...)`` frame without copying to host."""
    w, h = _image_wh(image)
    return h, w


SubpixelMethod = str  # "argmax" | "weighted" | "taylor"


def refine_peaks(
    hm: Float[np.ndarray, "M Hh Ww"],
    row: Int[np.ndarray, "M P"],
    col: Int[np.ndarray, "M P"],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[np.ndarray, "M P"], Float[np.ndarray, "M P"]]:
    """Refine integer peak cells ``(row, col)`` to sub-pixel ``(cx, cy)`` (heatmap px).

    Shared by :func:`heatmap_to_points` and
    :func:`deeperfly.pictorial.peak_candidates` so the single arg-max peak and the
    top-K candidate peaks are localized the same way. ``hm`` holds ``M`` heatmaps;
    each carries ``P`` peaks to refine (``P = 1`` for the arg-max, ``P = K`` for
    top-K). All three estimators are seeded by the arg-max cell:

    - ``"argmax"`` -- no refinement (the integer cell); the original behaviour.
    - ``"weighted"`` -- intensity-weighted centroid of the ``(2*radius+1)`` window
      around the cell (a localized soft-arg-max). The detector emits Gaussian
      blobs, so the blob centroid is its true centre; negatives are clipped
      (``relu``), out-of-bounds taps excluded, and an empty window keeps the cell.
    - ``"taylor"`` -- DARK-style (Zhang et al. 2020) one Newton step on the
      log-heatmap: ``offset = -H^{-1} g`` from the local gradient ``g`` and
      Hessian ``H``. Most accurate on clean Gaussians; needs ``radius >= 2`` and
      falls back to the cell at borders or where ``H`` is not negative-definite.

    Costs a handful of small gathers/reductions over the arg-max neighbourhood --
    negligible next to the forward pass.

    Parameters
    ----------
    hm
        ``M`` heatmaps of shape ``(M, Hh, Ww)``.
    row, col
        Integer peak cells of shape ``(M, P)`` to refine.
    method
        ``"argmax"`` | ``"weighted"`` | ``"taylor"`` (see above).
    radius
        Half-width of the refinement window in heatmap pixels.

    Returns
    -------
    cx, cy : np.ndarray
        The sub-pixel peak coordinates of shape ``(M, P)`` (heatmap pixels).

    Raises
    ------
    ValueError
        On an unknown ``method``, or ``"taylor"`` with ``radius < 2``.
    """
    hm = np.asarray(hm, dtype=float)
    m, hh, ww = hm.shape
    row, col = np.asarray(row), np.asarray(col)
    fcol, frow = col.astype(float), row.astype(float)
    if method == "argmax":
        return fcol, frow
    if method not in ("weighted", "taylor"):
        raise ValueError(f"unknown sub-pixel method {method!r}")
    if method == "taylor" and radius < 2:
        raise ValueError("taylor refinement needs radius >= 2")

    off = np.arange(-radius, radius + 1)
    dr, dc = (a.ravel() for a in np.meshgrid(off, off, indexing="ij"))  # (PP,)
    nr, nc = row[..., None] + dr, col[..., None] + dc  # (M, P, PP) window cells
    inb = (nr >= 0) & (nr < hh) & (nc >= 0) & (nc < ww)
    flat = hm.reshape(m, hh * ww)
    gidx = np.clip(nr, 0, hh - 1) * ww + np.clip(nc, 0, ww - 1)
    patch = flat[np.arange(m)[:, None, None], gidx]  # (M, P, PP) values around peak

    if method == "weighted":
        w = np.where(inb, np.maximum(patch, 0.0), 0.0)
        mass = w.sum(-1)
        ok = mass > 0
        denom = np.where(ok, mass, 1.0)  # avoid 0/0; offset is 0 when ok is False
        offc = np.where(ok, (w * dc).sum(-1) / denom, 0.0)
        offr = np.where(ok, (w * dr).sum(-1) / denom, 0.0)
        return fcol + offc, frow + offr

    # "taylor": fit the log-heatmap's local quadratic and Newton-step to its peak.
    p = 2 * radius + 1
    b = np.log(np.maximum(patch, 1e-10)).reshape(m, -1, p, p)  # (M, P, P_, P_)
    ib = inb.reshape(m, -1, p, p)
    c = radius  # centre tap; b[..., c + i, c + j] is row+i, col+j
    dx = 0.5 * (b[..., c, c + 1] - b[..., c, c - 1])
    dy = 0.5 * (b[..., c + 1, c] - b[..., c - 1, c])
    dxx = 0.25 * (b[..., c, c + 2] - 2 * b[..., c, c] + b[..., c, c - 2])
    dyy = 0.25 * (b[..., c + 2, c] - 2 * b[..., c, c] + b[..., c - 2, c])
    dxy = 0.25 * (
        b[..., c + 1, c + 1]
        - b[..., c + 1, c - 1]
        - b[..., c - 1, c + 1]
        + b[..., c - 1, c - 1]
    )
    det = dxx * dyy - dxy * dxy
    taps = (  # every tap the derivatives touch must be in-bounds
        ib[..., c, c + 1]
        & ib[..., c, c - 1]
        & ib[..., c + 1, c]
        & ib[..., c - 1, c]
        & ib[..., c, c + 2]
        & ib[..., c, c - 2]
        & ib[..., c + 2, c]
        & ib[..., c - 2, c]
        & ib[..., c + 1, c + 1]
        & ib[..., c + 1, c - 1]
        & ib[..., c - 1, c + 1]
        & ib[..., c - 1, c - 1]
    )
    good = taps & (det > 0) & (dxx < 0)  # a real local maximum
    denom = np.where(good, det, 1.0)
    ox = np.where(good, np.clip(-(dyy * dx - dxy * dy) / denom, -radius, radius), 0.0)
    oy = np.where(good, np.clip(-(-dxy * dx + dxx * dy) / denom, -radius, radius), 0.0)
    return fcol + ox, frow + oy


def heatmap_to_points(
    heatmaps: Float[np.ndarray, "*batch J Hh Ww"],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[np.ndarray, "*batch J 2"], Float[np.ndarray, "*batch J"]]:
    """Peak location (normalized ``(x, y)`` in [0, 1]) and confidence per joint.

    The heatmaps are ~8x smaller than the source image, so a plain arg-max
    quantizes every joint onto an 8-pixel grid. We take the arg-max cell and, by
    default, refine it to sub-pixel via :func:`refine_peaks` (``method``):
    ``"argmax"`` keeps the raw cell, ``"weighted"`` (default) is the windowed
    centroid, ``"taylor"`` is the DARK Newton step. Confidence stays the raw peak
    value.

    Coordinates keep the same ``col / Ww``, ``row / Hh`` normalization as a plain
    arg-max (DeepFly2D's ``heatmap2points``) and ``(x, y)`` ordering for the
    geometry layer, so a single-pixel spike still decodes to exactly its cell.

    Parameters
    ----------
    heatmaps
        Heatmaps of shape ``(*batch, J, Hh, Ww)``.
    method
        Sub-pixel refinement (see :func:`refine_peaks`).
    radius
        Refinement window half-width.

    Returns
    -------
    points : np.ndarray
        Normalized ``(x, y)`` peaks in ``[0, 1]`` of shape ``(*batch, J, 2)``.
    conf : np.ndarray
        The raw peak value per joint of shape ``(*batch, J)``.
    """
    hm = np.asarray(heatmaps, dtype=float)
    *lead, hh, ww = hm.shape  # lead = (*batch, J)
    flat = hm.reshape(-1, hh * ww)  # (M, Hh*Ww)
    idx = np.argmax(flat, axis=-1)  # (M,)
    conf = flat.max(axis=-1)
    row, col = idx // ww, idx % ww
    cx, cy = refine_peaks(
        flat.reshape(-1, hh, ww),
        row[:, None],
        col[:, None],
        method=method,
        radius=radius,
    )
    points = np.stack([cx[:, 0] / ww, cy[:, 0] / hh], axis=-1)  # (M, 2)
    return points.reshape(*lead, 2), conf.reshape(*lead)


# -- plan-driven detection ---------------------------------------------------


def _plan_device(models) -> str:
    """The device the plan's models live on (they share one)."""
    return next(iter(models.values())).device()


def _prepare_pathways(plan, models, windows):
    """Batched input prep: ``pathway -> (T, 3, Hh, Ww)`` model input.

    Each pathway's *whole* window is oriented (mirror/crop) and resized +
    normalized in one shot, so the heavy resize runs once per pathway over all
    ``T`` frames rather than per frame. Returns the per-pathway prepared inputs.
    """
    return [
        models[pw.model].prepare(pw.transform.apply(windows[pw.source]))
        for pw in plan.pathways
    ]


def detect_sequence(
    plan: DetectionPlan,
    models: dict,
    windows: dict,
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
    batch_size: int | None = None,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> tuple[Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]]:
    """Detect a multi-source sequence -> ``(V, T, N, 2)`` pixels and ``(V, T, N)`` conf.

    Fully batched: every pathway's window is preprocessed in one shot, then the
    forward runs **per model** over one big time-major batch (all pathways of a
    model for every frame), in chunks of ``batch_size``. Pathways sharing a model
    are never looped one-at-a-time through the network. ``batch_size`` ``None``
    forwards one frame's pathways at a time; a larger value flattens across frames
    (numerically identical, only dispatch differs).

    Parameters
    ----------
    plan
        The detection plan (views, pathways, models).
    models
        ``name -> LoadedModel`` for every model the plan references.
    windows
        ``source name -> (T, H, W, 3)`` window (NumPy or on-device tensor).
    method, radius
        Heatmap decode options (see :func:`heatmap_to_points`).
    batch_size
        Forward inputs per model forward, or ``None`` for one frame at a time.
    progress
        Optional wrapper of the per-frame iterator, advanced once per completed
        frame; defaults to the identity.

    Returns
    -------
    pts : np.ndarray
        2D pixels of shape ``(V, T, N, 2)`` (NaN where unobserved).
    conf : np.ndarray
        Per-point confidence of shape ``(V, T, N)``.
    """
    import torch

    device = _plan_device(models)
    windows = {name: _window_to_device(w, device) for name, w in windows.items()}
    source_sizes = {name: _image_hw(w[0]) for name, w in windows.items()}
    n_frames = len(next(iter(windows.values())))
    pathways = plan.pathways
    n_pass = len(pathways)

    out_pts = np.full((plan.n_views, n_frames, plan.n_points, 2), np.nan)
    out_conf = np.zeros((plan.n_views, n_frames, plan.n_points))
    prepared = _prepare_pathways(plan, models, windows)  # [(T, 3, Hh, Ww)] per pathway

    # results[t][pw_idx] = (points_norm (J, 2), conf (J,))
    results: list[list] = [[None] * n_pass for _ in range(n_frames)]
    steps = progress(range(n_frames)) if progress is not None else range(n_frames)
    step_iter = iter(steps)
    remaining = [n_pass] * n_frames  # pathway results still owed per frame

    def landed(t: int, pw_idx: int, pn, cc) -> None:
        results[t][pw_idx] = (pn, cc)
        remaining[t] -= 1
        if remaining[t] == 0:
            next(step_iter, None)

    # Forward each model's pathways as one time-major batch (t outer, pathway
    # inner), in chunks of whole frames so progress ticks land cleanly.
    by_model: dict[str, list[int]] = {}
    for pw_idx, pw in enumerate(pathways):
        by_model.setdefault(pw.model, []).append(pw_idx)

    for model_name, pw_idxs in by_model.items():
        model = models[model_name]
        stacked = torch.stack([prepared[p] for p in pw_idxs], dim=1)  # (T, Pm, 3, H, W)
        p_m = stacked.shape[1]
        # The model's standard input is (B, V, 3, H, W): hand it whole frames (B) of
        # this model's Pm pathways (V), in chunks of bs_t frames.
        bs_t = 1 if batch_size is None else max(1, int(batch_size) // p_m)
        for i in range(0, n_frames, bs_t):
            pn, cc = model.predict_points(
                stacked[i : i + bs_t], method=method, radius=radius
            )  # (b, Pm, J, 2), (b, Pm, J)
            for tt in range(pn.shape[0]):
                for local in range(p_m):
                    landed(i + tt, pw_idxs[local], pn[tt, local], cc[tt, local])
    for _ in step_iter:  # drain any trailing progress ticks
        pass

    # Scatter each frame's pathway results into the skeleton (out_pts[:, t] is a
    # view, so the in-place scatter writes through).
    for t in range(n_frames):
        for pw_idx, pw in enumerate(pathways):
            pn, cc = results[t][pw_idx]
            raw_xy = map_to_view(
                pn, pw.transform, models[pw.model].input_size, source_sizes[pw.source]
            )
            scatter_pathway(raw_xy, cc, pw.mapping, out_pts[:, t], out_conf[:, t])
    return out_pts, out_conf


def detect(
    plan: DetectionPlan,
    models: dict,
    images: dict,
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[np.ndarray, "V N 2"], Float[np.ndarray, "V N"]]:
    """Detect one multi-source frame -> ``(V, N, 2)`` pixels and ``(V, N)`` conf.

    Parameters
    ----------
    plan
        The detection plan.
    models
        ``name -> LoadedModel``.
    images
        ``source name -> (H, W, 3)`` frame.
    method, radius
        Heatmap decode options.

    Returns
    -------
    pts, conf : np.ndarray
        ``(V, N, 2)`` pixels and ``(V, N)`` confidence.
    """
    windows = {name: img[None] for name, img in images.items()}
    pts, conf = detect_sequence(plan, models, windows, method=method, radius=radius)
    return pts[:, 0], conf[:, 0]


def detect_candidates_sequence(
    plan: DetectionPlan,
    models: dict,
    windows: dict,
    *,
    k: int = 5,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
):
    """Detect a sequence, returning both arg-max poses and top-K candidate peaks.

    The same forward yields the single-peak ``(pts2d, conf)`` -- used by
    calibration and triangulation -- and a
    :class:`deeperfly.pictorial.Candidates` set of the top-``k`` peaks per
    (view, joint), consumed by the pictorial-structures corrector. Candidates a
    pathway does not map (or that no pathway produces) stay ``NaN``.

    Parameters
    ----------
    plan
        The detection plan.
    models
        ``name -> LoadedModel``.
    windows
        ``source name -> (T, H, W, 3)`` window.
    k
        Number of candidate peaks kept per (view, joint).
    method, radius
        Heatmap decode options.
    progress
        Optional wrapper of the per-frame iterator.

    Returns
    -------
    pts2d : np.ndarray
        Arg-max 2D pixels of shape ``(V, T, N, 2)``.
    conf : np.ndarray
        Per-point confidence of shape ``(V, T, N)``.
    candidates : deeperfly.pictorial.Candidates
        The top-``k`` candidate peak set.
    """
    import torch

    from .. import pictorial

    device = _plan_device(models)
    windows = {name: _window_to_device(w, device) for name, w in windows.items()}
    source_sizes = {name: _image_hw(w[0]) for name, w in windows.items()}
    n_frames = len(next(iter(windows.values())))
    pathways = plan.pathways

    V, N = plan.n_views, plan.n_points
    pts = np.full((V, n_frames, N, 2), np.nan)
    conf = np.zeros((V, n_frames, N))
    cand_xy = np.full((V, n_frames, N, k, 2), np.nan)
    cand_score = np.zeros((V, n_frames, N, k))

    prepared = _prepare_pathways(plan, models, windows)  # [(T, 3, Hh, Ww)] per pathway
    by_model: dict[str, list[int]] = {}
    for pw_idx, pw in enumerate(pathways):
        by_model.setdefault(pw.model, []).append(pw_idx)

    steps = progress(range(n_frames)) if progress is not None else range(n_frames)
    for t in steps:
        for model_name, pw_idxs in by_model.items():
            model = models[model_name]
            # One forward over all this model's pathways for frame t -> whole
            # heatmaps decoded/peaked in a single batched call.
            batch = torch.stack([prepared[p][t] for p in pw_idxs])  # (Pm, 3, H, W)
            # Standard (B, V, ...) input: this frame is B=1 over Pm views; strip B back.
            heatmaps = model.predict_heatmaps(batch.unsqueeze(0))[0]  # (Pm, J, Hh, Ww)
            pn, c = heatmap_to_points(heatmaps, method=method, radius=radius)
            cxy, csc = pictorial.peak_candidates(
                heatmaps, k, radius=radius, method=method
            )
            for local, pw_idx in enumerate(pw_idxs):
                pw = pathways[pw_idx]
                src_size = source_sizes[pw.source]
                raw_pn = map_to_view(
                    pn[local], pw.transform, model.input_size, src_size
                )
                scatter_pathway(raw_pn, c[local], pw.mapping, pts[:, t], conf[:, t])
                raw_cxy = map_to_view(
                    cxy[local], pw.transform, model.input_size, src_size
                )
                scatter_pathway(
                    raw_cxy, csc[local], pw.mapping, cand_xy[:, t], cand_score[:, t]
                )
    return pts, conf, pictorial.Candidates(xy=cand_xy, score=cand_score)
