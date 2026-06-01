"""Backend-agnostic orchestration: run the detector and assemble 2D skeletons.

This layer is shared by both backends (:mod:`deeperfly.pose2d.backends`) -- it
preprocesses images, decodes heatmaps and scatters per-camera detections into the
full skeleton, dispatching the actual forward pass to whichever backend owns the
model. Pipeline for one recording:

1. :func:`expand_passes` -- turn the per-camera ``(side, flip)`` layout into a
   flat list of forward *passes*. A side-camera is one pass; the **front camera
   is two passes** (un-flipped -> right legs, mirror-flipped -> left legs) that
   share one physical view, so the front image bridges the two body sides.
2. :func:`preprocess` each pass (mirror-flip where required, resize to 256x512,
   subtract the training mean) -- matching DeepFly2D.
3. :func:`deeperfly.pose2d.backends.predict_heatmaps` (dispatched, batched) ->
   per-joint heatmaps as NumPy.
4. :func:`heatmap_to_points` -> normalized sub-pixel peak locations + confidence.
5. :func:`assemble_skeleton` -- place each pass's 19 single-side joints into the
   38-point skeleton (right pass -> indices 0..18, mirrored left pass -> 19..37
   with the x flip undone) and scale to original-image pixels. The front camera's
   two passes fill *both* halves of its row, so it observes left and right joints.

The single-side ordering of the 19 detector channels matches the skeleton's
per-side ordering, so the mapping is a direct slice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

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


def expand_passes(
    sides: list[str], flips: list[bool]
) -> tuple[list[int], list[str], list[bool]]:
    """Expand a per-camera ``(side, flip)`` layout into per-*pass* lists.

    A *pass* is one detector forward run. Most cameras are a single pass; a camera
    whose side is ``"both"`` (the front camera) becomes **two** passes that share
    its physical view index: ``("right", flip=False)`` populating skeleton indices
    ``0..18`` and ``("left", flip=True)`` -- the mirror-flipped image -- populating
    ``19..37``. So the one front image yields detections for both body sides,
    making it the cross-side bridge the rig calibration relies on.

    Returns ``(views, pass_sides, pass_flips)`` -- the physical view index, side
    and flip for each pass, ready for :func:`assemble_skeleton` (``views=...``).
    """
    views: list[int] = []
    pass_sides: list[str] = []
    pass_flips: list[bool] = []
    for v, (side, flip) in enumerate(zip(sides, flips)):
        if side == "both":
            views += [v, v]
            pass_sides += ["right", "left"]
            pass_flips += [False, True]
        else:
            views.append(v)
            pass_sides.append(side)
            pass_flips.append(flip)
    return views, pass_sides, pass_flips


def assemble_skeleton(
    points_norm: Float[np.ndarray, "P J2 2"],
    conf: Float[np.ndarray, "P J2"],
    sides: list[str],
    flips: list[bool],
    image_size: list[tuple[int, int]],
    *,
    views: list[int] | None = None,
    n_views: int | None = None,
    n_points: int = 38,
) -> tuple[Float[np.ndarray, "V N 2"], Float[np.ndarray, "V N"]]:
    """Scatter per-*pass* single-side detections into the full skeleton (pixels).

    Parameters
    ----------
    points_norm, conf
        Per-pass detector output: ``(P, 19, 2)`` normalized ``(x, y)`` and
        ``(P, 19)`` confidence (``P`` passes, see :func:`expand_passes`).
    sides
        ``"right"`` or ``"left"`` per pass -- which half of the skeleton the 19
        channels populate (right -> 0..18, left -> 19..37).
    flips
        Whether each pass's image was mirror-flipped in :func:`preprocess`
        (the x coordinate is then undone as ``1 - x``).
    image_size
        ``(W, H)`` original pixel size per *physical view* to scale normalized
        coords (indexed by ``views``).
    views
        Physical view index per pass (default: identity, one pass per view). Two
        passes may share a view -- that is how the front camera fills both halves.
    n_views
        Number of physical views in the output (default: ``max(views) + 1``).
    """
    points_norm = np.asarray(points_norm, dtype=float)
    conf = np.asarray(conf, dtype=float)
    n_passes = len(sides)
    if views is None:
        views = list(range(n_passes))
    if n_views is None:
        n_views = (max(views) + 1) if views else 0
    pts = np.full((n_views, n_points, 2), np.nan)
    cout = np.zeros((n_views, n_points))
    for i in range(n_passes):
        v = views[i]
        p = points_norm[i].copy()
        if flips[i]:
            p[:, 0] = 1.0 - p[:, 0]
        w, h = image_size[v]
        p = p * np.array([w, h])
        sl = (
            slice(0, N_SIDE_JOINTS)
            if sides[i] == "right"
            else slice(N_SIDE_JOINTS, 2 * N_SIDE_JOINTS)
        )
        pts[v, sl] = p
        cout[v, sl] = conf[i]
    return pts, cout


def fly_camera_layout(camera_names: list[str]) -> tuple[list[str], list[bool]]:
    """Default ``(sides, flips)`` for the canonical 7-camera fly rig.

    Left cameras (names starting ``l``) image the left side and are mirror-
    flipped so the fly faces the trained orientation; right cameras image the
    right side un-flipped. The **front camera** (name starting ``f``) gets side
    ``"both"``: :func:`expand_passes` runs it twice (un-flipped -> right legs,
    flipped -> left legs) so it observes joints on both sides and bridges them in
    one world frame. Override for rigs whose front camera should feed a single
    side only.
    """
    sides, flips = [], []
    for name in camera_names:
        n = name.lower()
        if n.startswith("l"):
            sides.append("left")
            flips.append(True)
        elif n.startswith("f"):
            sides.append("both")
            flips.append(False)  # ignored: expand_passes sets both passes' flips
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
    pick the heatmap decode (see :func:`heatmap_to_points`). A ``"both"`` camera is
    run twice (:func:`expand_passes`) so the front image fills both body sides.
    """
    from . import backends  # lazy: dispatch never imports the unused framework

    views, pass_sides, pass_flips = expand_passes(sides, flips)
    inputs = np.stack(
        [
            np.asarray(preprocess(images[views[i]], flip=pass_flips[i]))
            for i in range(len(views))
        ]
    )
    points_norm, conf = heatmap_to_points(
        backends.predict_heatmaps(model, inputs), method=method, radius=radius
    )
    image_size = [(np.asarray(im).shape[1], np.asarray(im).shape[0]) for im in images]
    return assemble_skeleton(
        np.asarray(points_norm),
        np.asarray(conf),
        pass_sides,
        pass_flips,
        image_size,
        views=views,
        n_views=len(images),
    )


def detect_sequence(
    model,
    frames: Float[np.ndarray, "V T H W 3"],
    sides: list[str],
    flips: list[bool],
    *,
    method: SubpixelMethod = "weighted",
    radius: int = 2,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> tuple[Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]]:
    """Detect a multi-camera sequence -> ``(V, T, 38, 2)`` pixels and ``(V, T, 38)`` conf.

    ``progress`` optionally wraps the per-frame iterator (e.g. ``tqdm``) so callers
    can show a progress bar; it defaults to the identity, keeping the library
    UI-free.
    """
    n_views, n_frames = len(frames), len(frames[0])
    pts = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS, 2))
    conf = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS))
    steps = progress(range(n_frames)) if progress is not None else range(n_frames)
    for t in steps:
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
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
):
    """Detect a sequence, returning both arg-max poses and top-K candidate peaks.

    Runs the detector via the full-heatmap path (not the fused arg-max fast path)
    so the same forward yields the single-peak ``(pts2d, conf)`` -- used by
    calibration and the reproject reconstructor -- and a
    :class:`deeperfly.pictorial.Candidates` set of the top-``k`` peaks per
    (view, joint), consumed by the pictorial-structures corrector. The front
    camera is run as two passes (:func:`expand_passes`), so both its arg-max pose
    and its candidates cover both body sides. Returns
    ``(pts2d (V, T, 38, 2), conf (V, T, 38), candidates)``.
    """
    from .. import pictorial
    from . import backends

    n_views, n_frames = len(frames), len(frames[0])
    n_pts = 2 * N_SIDE_JOINTS
    views, pass_sides, pass_flips = expand_passes(sides, flips)
    pts = np.empty((n_views, n_frames, n_pts, 2))
    conf = np.empty((n_views, n_frames, n_pts))
    cand_xy = np.empty((n_views, n_frames, n_pts, k, 2))
    cand_score = np.empty((n_views, n_frames, n_pts, k))
    steps = progress(range(n_frames)) if progress is not None else range(n_frames)
    for t in steps:
        images = [frames[v][t] for v in range(n_views)]
        inputs = np.stack(
            [
                np.asarray(preprocess(images[views[i]], flip=pass_flips[i]))
                for i in range(len(views))
            ]
        )
        heatmaps = np.asarray(backends.predict_heatmaps(model, inputs))  # (P,J,Hh,Ww)
        image_size = [
            (np.asarray(im).shape[1], np.asarray(im).shape[0]) for im in images
        ]
        points_norm, c = heatmap_to_points(heatmaps, method=method, radius=radius)
        pts[:, t], conf[:, t] = assemble_skeleton(
            np.asarray(points_norm),
            np.asarray(c),
            pass_sides,
            pass_flips,
            image_size,
            views=views,
            n_views=n_views,
        )
        cand_xy[:, t], cand_score[:, t] = pictorial.extract_candidates(
            heatmaps,
            pass_sides,
            pass_flips,
            image_size,
            k=k,
            views=views,
            n_views=n_views,
            method=method,
            radius=radius,
        )
    return pts, conf, pictorial.Candidates(xy=cand_xy, score=cand_score)
