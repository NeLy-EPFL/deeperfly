"""Backend-agnostic orchestration: run the detector and assemble 2D skeletons.

This layer is shared by both backends (:mod:`deeperfly.pose2d.backends`) -- it
preprocesses images, decodes heatmaps and scatters per-camera detections into the
full skeleton, dispatching the actual forward pass to whichever backend owns the
model. Pipeline for one recording:

1. :func:`preprocess` each camera image (mirror-flip the far-side cameras,
   resize to 256x512, subtract the training mean) -- matching DeepFly2D.
2. :func:`deeperfly.pose2d.backends.predict_heatmaps` (dispatched, batched) ->
   per-joint heatmaps as NumPy.
3. :func:`heatmap_to_points` -> normalized sub-pixel peak locations + confidence.
4. :func:`assemble_skeleton` -- place each camera's 19 single-side joints into
   the 38-point skeleton (right cameras -> indices 0..18, mirrored left cameras
   -> 19..37 with the x flip undone) and scale to original-image pixels.

The single-side ordering of the 19 detector channels matches the skeleton's
per-side ordering, so the mapping is a direct slice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

IMG_SIZE = (256, 512)  # (H, W) network input
MEAN = 0.22  # DeepFly2D subtracts this scalar from the [0, 1] image
N_SIDE_JOINTS = 19  # detector channels (one body side)


def preprocess(
    image: Float[np.ndarray, "H W 3"],
    *,
    flip: bool = False,
    img_size: tuple[int, int] = IMG_SIZE,
    mean: float = MEAN,
) -> Float[Array, "3 Hh Ww"]:
    """Image (HWC, uint8 or float[0,1]) -> normalized CHW network input.

    Mirror-side cameras are horizontally flipped so the fly faces the trained
    orientation. Uses bilinear (anti-aliased) resize; this is close to but not
    bit-identical with DeepFly2D's skimage resize -- argmax peak picking is
    robust to the difference.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    img = jnp.asarray(arr, dtype=jnp.float32)
    if img.ndim == 2:
        img = jnp.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if flip:
        img = img[:, ::-1]
    img = jax.image.resize(
        img, (img_size[0], img_size[1], 3), method="linear", antialias=True
    )
    return jnp.transpose(img, (2, 0, 1)) - mean


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


def refine_peaks_jax(
    hm: Float[Array, "*lead Hh Ww"],
    row: Int[Array, "*lead"],
    col: Int[Array, "*lead"],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[Array, "*lead"], Float[Array, "*lead"]]:
    """On-device (``jnp``) twin of :func:`refine_peaks` for one peak per heatmap.

    Same three estimators and conventions, but pure JAX so a fused
    forward+decode kernel can refine on the accelerator without shipping
    heatmaps to the host (``hm`` is ``(*lead, Hh, Ww)`` with one arg-max peak
    ``(row, col)`` per map, e.g. ``*lead = (J,)`` under :func:`jax.vmap`). Kept
    numerically equal to :func:`refine_peaks` (guarded by a test).
    """
    hm = jnp.asarray(hm)
    hh, ww = hm.shape[-2:]
    fcol, frow = col.astype(hm.dtype), row.astype(hm.dtype)
    if method == "argmax":
        return fcol, frow
    if method not in ("weighted", "taylor"):
        raise ValueError(f"unknown sub-pixel method {method!r}")
    if method == "taylor" and radius < 2:
        raise ValueError("taylor refinement needs radius >= 2")
    ys, xs = jnp.arange(hh), jnp.arange(ww)

    if method == "weighted":
        near = (jnp.abs(ys[:, None] - row[..., None, None]) <= radius) & (
            jnp.abs(xs - col[..., None, None]) <= radius
        )  # (*lead, Hh, Ww) window around each peak
        w = jnp.where(near, jnp.maximum(hm, 0.0), 0.0)
        mass = w.sum((-2, -1))
        ok = mass > 0
        denom = jnp.where(ok, mass, 1.0)  # 0/0 guard; the cell is kept below
        cx = jnp.where(ok, (w * xs).sum((-2, -1)) / denom, fcol)
        cy = jnp.where(ok, (w * ys[:, None]).sum((-2, -1)) / denom, frow)
        return cx, cy

    # "taylor": one Newton step on the log-heatmap from gathered ring taps.
    flat = hm.reshape(*hm.shape[:-2], hh * ww)

    def tap(dr, dc):  # log-heatmap at (row+dr, col+dc), edges clamped
        r, cc = jnp.clip(row + dr, 0, hh - 1), jnp.clip(col + dc, 0, ww - 1)
        v = jnp.take_along_axis(flat, (r * ww + cc)[..., None], axis=-1)[..., 0]
        return jnp.log(jnp.maximum(v, 1e-10))

    b0 = tap(0, 0)
    dx = 0.5 * (tap(0, 1) - tap(0, -1))
    dy = 0.5 * (tap(1, 0) - tap(-1, 0))
    dxx = 0.25 * (tap(0, 2) - 2 * b0 + tap(0, -2))
    dyy = 0.25 * (tap(2, 0) - 2 * b0 + tap(-2, 0))
    dxy = 0.25 * (tap(1, 1) - tap(1, -1) - tap(-1, 1) + tap(-1, -1))
    det = dxx * dyy - dxy * dxy
    inb = (row >= 2) & (row < hh - 2) & (col >= 2) & (col < ww - 2)
    good = inb & (det > 0) & (dxx < 0)
    denom = jnp.where(good, det, 1.0)
    ox = jnp.where(good, jnp.clip(-(dyy * dx - dxy * dy) / denom, -radius, radius), 0.0)
    oy = jnp.where(
        good, jnp.clip(-(-dxy * dx + dxx * dy) / denom, -radius, radius), 0.0
    )
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


def assemble_skeleton(
    points_norm: Float[np.ndarray, "V J2 2"],
    conf: Float[np.ndarray, "V J2"],
    sides: list[str],
    flips: list[bool],
    image_size: list[tuple[int, int]],
    *,
    n_points: int = 38,
) -> tuple[Float[np.ndarray, "V N 2"], Float[np.ndarray, "V N"]]:
    """Scatter per-camera single-side detections into the full skeleton (pixels).

    Parameters
    ----------
    points_norm, conf
        Per-camera detector output: ``(V, 19, 2)`` normalized ``(x, y)`` and
        ``(V, 19)`` confidence.
    sides
        ``"right"`` or ``"left"`` per camera -- which half of the skeleton the
        19 channels populate (right -> 0..18, left -> 19..37).
    flips
        Whether each camera's image was mirror-flipped in :func:`preprocess`
        (the x coordinate is then undone as ``1 - x``).
    image_size
        ``(W, H)`` original pixel size per camera to scale normalized coords.
    """
    points_norm = np.asarray(points_norm, dtype=float)
    conf = np.asarray(conf, dtype=float)
    n_views = len(sides)
    pts = np.full((n_views, n_points, 2), np.nan)
    cout = np.zeros((n_views, n_points))
    for v in range(n_views):
        p = points_norm[v].copy()
        if flips[v]:
            p[:, 0] = 1.0 - p[:, 0]
        w, h = image_size[v]
        p = p * np.array([w, h])
        sl = (
            slice(0, N_SIDE_JOINTS)
            if sides[v] == "right"
            else slice(N_SIDE_JOINTS, 2 * N_SIDE_JOINTS)
        )
        pts[v, sl] = p
        cout[v, sl] = conf[v]
    return pts, cout


def fly_camera_layout(camera_names: list[str]) -> tuple[list[str], list[bool]]:
    """Default ``(sides, flips)`` for the canonical 7-camera fly rig.

    Left cameras (names starting ``l``) image the left side and are mirror-
    flipped so the fly faces the trained orientation; right and front cameras
    image the right side un-flipped. The front camera's single-side assignment
    is approximate -- override for rigs where it should feed the other side.
    """
    sides, flips = [], []
    for name in camera_names:
        if name.lower().startswith("l"):
            sides.append("left")
            flips.append(True)
        else:
            sides.append("right")
            flips.append(False)
    return sides, flips


def detect(
    model,
    images: list[Float[np.ndarray, "H W 3"]],
    sides: list[str],
    flips: list[bool],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[np.ndarray, "V N 2"], Float[np.ndarray, "V N"]]:
    """Detect one multi-camera frame -> ``(V, 38, 2)`` pixels and ``(V, 38)`` conf.

    ``model`` is a detector from either backend (:mod:`deeperfly.pose2d.backends`):
    :func:`~deeperfly.pose2d.backends.predict_heatmaps` dispatches on its type, so
    this function is identical for the JAX and PyTorch paths. ``method`` / ``radius``
    pick the heatmap decode (see :func:`heatmap_to_points`).
    """
    from . import backends  # lazy: dispatch never imports the unused framework

    inputs = np.stack(
        [np.asarray(preprocess(images[v], flip=flips[v])) for v in range(len(images))]
    )
    points_norm, conf = heatmap_to_points(
        backends.predict_heatmaps(model, inputs), method=method, radius=radius
    )
    image_size = [(np.asarray(im).shape[1], np.asarray(im).shape[0]) for im in images]
    return assemble_skeleton(
        np.asarray(points_norm), np.asarray(conf), sides, flips, image_size
    )


def detect_sequence(
    model,
    frames: Float[np.ndarray, "V T H W 3"],
    sides: list[str],
    flips: list[bool],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
) -> tuple[Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]]:
    """Detect a multi-camera sequence -> ``(V, T, 38, 2)`` pixels and ``(V, T, 38)`` conf."""
    n_views, n_frames = len(frames), len(frames[0])
    pts = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS, 2))
    conf = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS))
    for t in range(n_frames):
        pts[:, t], conf[:, t] = detect(
            model,
            [frames[v][t] for v in range(n_views)],
            sides,
            flips,
            method=method,
            radius=radius,
        )
    return pts, conf


def detect_candidates_sequence(
    model,
    frames: Float[np.ndarray, "V T H W 3"],
    sides: list[str],
    flips: list[bool],
    *,
    k: int = 5,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
):
    """Detect a sequence, returning both arg-max poses and top-K candidate peaks.

    Runs the detector via the full-heatmap path (not the fused arg-max fast path)
    so the same forward yields the single-peak ``(pts2d, conf)`` -- used by
    calibration and the reproject reconstructor -- and a
    :class:`deeperfly.pictorial.Candidates` set of the top-``k`` peaks per
    (view, joint), consumed by the pictorial-structures corrector. Returns
    ``(pts2d (V, T, 38, 2), conf (V, T, 38), candidates)``.
    """
    from .. import pictorial
    from . import backends

    n_views, n_frames = len(frames), len(frames[0])
    n_pts = 2 * N_SIDE_JOINTS
    pts = np.empty((n_views, n_frames, n_pts, 2))
    conf = np.empty((n_views, n_frames, n_pts))
    cand_xy = np.empty((n_views, n_frames, n_pts, k, 2))
    cand_score = np.empty((n_views, n_frames, n_pts, k))
    for t in range(n_frames):
        images = [frames[v][t] for v in range(n_views)]
        inputs = np.stack(
            [np.asarray(preprocess(images[v], flip=flips[v])) for v in range(n_views)]
        )
        heatmaps = np.asarray(backends.predict_heatmaps(model, inputs))  # (V,J,Hh,Ww)
        image_size = [
            (np.asarray(im).shape[1], np.asarray(im).shape[0]) for im in images
        ]
        points_norm, c = heatmap_to_points(heatmaps, method=method, radius=radius)
        pts[:, t], conf[:, t] = assemble_skeleton(
            np.asarray(points_norm), np.asarray(c), sides, flips, image_size
        )
        cand_xy[:, t], cand_score[:, t] = pictorial.extract_candidates(
            heatmaps, sides, flips, image_size, k=k, method=method, radius=radius
        )
    return pts, conf, pictorial.Candidates(xy=cand_xy, score=cand_score)
