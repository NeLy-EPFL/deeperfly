"""Backend-agnostic orchestration: run the detector and assemble 2D skeletons.

This layer is shared by both backends (:mod:`deeperfly.pose2d.backends`) -- it
preprocesses images, decodes heatmaps and scatters per-camera detections into the
full skeleton, dispatching the actual forward pass to whichever backend owns the
model. Pipeline for one recording:

1. :func:`preprocess` each camera image (mirror-flip the far-side cameras,
   resize to 256x512, subtract the training mean) -- matching DeepFly2D.
2. :func:`deeperfly.pose2d.backends.predict_heatmaps` (dispatched, batched) ->
   per-joint heatmaps as NumPy.
3. :func:`heatmap_to_points` -> normalized peak locations + confidence.
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
from jaxtyping import Array, Float

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


def heatmap_to_points(
    heatmaps: Float[Array, "*batch J Hh Ww"],
) -> tuple[Float[Array, "*batch J 2"], Float[Array, "*batch J"]]:
    """Argmax peak (normalized ``(x, y)`` in [0, 1]) and confidence per joint.

    Matches DeepFly2D's ``heatmap2points`` (first global argmax, normalized by
    heatmap size), but returns ``(x, y)`` ordering for the geometry layer.
    """
    heatmaps = jnp.asarray(heatmaps)
    hh, ww = heatmaps.shape[-2:]
    flat = heatmaps.reshape(*heatmaps.shape[:-2], hh * ww)
    idx = jnp.argmax(flat, axis=-1)
    conf = jnp.max(flat, axis=-1)
    row, col = idx // ww, idx % ww
    points = jnp.stack([col / ww, row / hh], axis=-1)
    return points, conf


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
) -> tuple[Float[np.ndarray, "V N 2"], Float[np.ndarray, "V N"]]:
    """Detect one multi-camera frame -> ``(V, 38, 2)`` pixels and ``(V, 38)`` conf.

    ``model`` is a detector from either backend (:mod:`deeperfly.pose2d.backends`):
    :func:`~deeperfly.pose2d.backends.predict_heatmaps` dispatches on its type, so
    this function is identical for the JAX and PyTorch paths.
    """
    from . import backends  # lazy: dispatch never imports the unused framework

    inputs = np.stack(
        [np.asarray(preprocess(images[v], flip=flips[v])) for v in range(len(images))]
    )
    points_norm, conf = heatmap_to_points(backends.predict_heatmaps(model, inputs))
    image_size = [(np.asarray(im).shape[1], np.asarray(im).shape[0]) for im in images]
    return assemble_skeleton(
        np.asarray(points_norm), np.asarray(conf), sides, flips, image_size
    )


def detect_sequence(
    model,
    frames: Float[np.ndarray, "V T H W 3"],
    sides: list[str],
    flips: list[bool],
) -> tuple[Float[np.ndarray, "V T N 2"], Float[np.ndarray, "V T N"]]:
    """Detect a multi-camera sequence -> ``(V, T, 38, 2)`` pixels and ``(V, T, 38)`` conf."""
    n_views, n_frames = len(frames), len(frames[0])
    pts = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS, 2))
    conf = np.empty((n_views, n_frames, 2 * N_SIDE_JOINTS))
    for t in range(n_frames):
        pts[:, t], conf[:, t] = detect(
            model, [frames[v][t] for v in range(n_views)], sides, flips
        )
    return pts, conf


def detect_candidates_sequence(
    model,
    frames: Float[np.ndarray, "V T H W 3"],
    sides: list[str],
    flips: list[bool],
    *,
    k: int = 5,
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
        points_norm, c = heatmap_to_points(heatmaps)
        pts[:, t], conf[:, t] = assemble_skeleton(
            np.asarray(points_norm), np.asarray(c), sides, flips, image_size
        )
        cand_xy[:, t], cand_score[:, t] = pictorial.extract_candidates(
            heatmaps, sides, flips, image_size, k=k
        )
    return pts, conf, pictorial.Candidates(xy=cand_xy, score=cand_score)
