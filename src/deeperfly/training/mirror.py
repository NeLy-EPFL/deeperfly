"""The left-right mirror: the **one** place a training sample is flipped.

A horizontal flip is the cheapest real augmentation available for a bilaterally symmetric
animal -- and the easiest to get silently wrong, because mirroring the image turns a left
leg into a right leg. Get the channel permutation wrong and every left channel trains on a
right joint: the loss still falls, no warning fires, and the result looks exactly like a
model that will not converge. That failure has a history in this project's lineage, which
is why this module exists as the single implementation rather than as a few lines inside
whatever dataset needs it.

Four things have to move together, and only the first is obvious:

1. **The image** (``x -> (W - 1) - x``, not ``W - x``; see :func:`mirror_sample`).
2. **The x coordinates**, by the same map, so labels stay on their pixels.
3. **The point channels**, by :meth:`deeperfly.skeleton.Skeleton.flip_perm` -- along with
   every per-point array that rides with them: visibility, occlusion flags, confidence.
   Permuting the coordinates but not the visibility mask is the subtle version of the bug.
4. **The camera identity**, by ``[cameras.<name>].mirror`` (see
   :func:`mirror_view_names`). A flipped right-camera sample *is* a left-camera sample; if
   it keeps its original camera id, every metric that splits ipsilateral from
   contralateral error -- the split that matters most on this rig, where the far legs are
   the hard case -- reports the swapped channels under the wrong side.

And one thing that must **not** move per-sample: in a multi-view batch the flip decision
belongs to the *frame group*, not the sample. See :func:`mirror_decisions`.

**When the permutation is the identity, and why that is not a bug.** deeperfly's shipping
detector is *side-agnostic*: 19 output channels for "front/mid/hind leg joints, antenna,
abdomen" with no side, reached by a ``fliplr`` preprocessor that canonicalizes the
left-side views (:mod:`deeperfly.pose2d.pathways`). A model like that has no left channel
to confuse with a right one, so a mirror permutes nothing -- and it still maps canonical
images to canonical images, so the augmentation stays valid. That falls out for free: a
19-point skeleton declares no ``symmetries``, so ``flip_perm()`` is ``arange(19)``. The
permutation only does work for an **all-keypoint** model (the 38-channel one the in-tree
trainer targets per fork F3c), which is exactly where it is needed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mirror_decisions",
    "mirror_sample",
    "mirror_view_names",
]


def mirror_sample(
    image,
    points,
    *per_point,
    flip_perm,
    width: int | None = None,
):
    """Mirror one sample: the image, the x coordinates, and the point channels.

    Parameters
    ----------
    image
        ``(..., H, W)`` or ``(..., H, W, C)``. NumPy array or torch tensor; the type and
        device are preserved. ``None`` is allowed (a coordinates-only mirror) and returns
        ``None``, which is what makes this reusable for mirroring labels alone.
    points
        ``(..., P, 2)`` in this image's own pixels. ``NaN`` (this package's unobserved
        marker) rides through untouched -- ``(W - 1) - NaN`` is ``NaN``.
    *per_point
        Any number of ``(..., P)`` or ``(..., P, k)`` arrays that must follow the channels:
        visibility, occlusion flags, confidence, weights. Permuted on the point axis and
        otherwise untouched. Passing these is the difference between a correct mirror and
        one that trains the right joint against the left joint's visibility.
    flip_perm
        The point permutation, i.e.
        :meth:`deeperfly.skeleton.Skeleton.flip_perm`. Required and keyword-only: a caller
        that has to name it cannot forget that mirroring implies permuting.
    width
        The pixel width the coordinates live in. Inferred from ``image`` when it is given;
        required when it is not.

    Returns
    -------
    tuple
        ``(image, points, *per_point)``, each a new object (or ``None`` for a ``None``
        image). Never in-place: a dataset caches decoded frames, and mirroring one in place
        corrupts the cache for every later epoch.

    Notes
    -----
    ``x -> (W - 1) - x`` and **not** ``W - x``. Coordinates here are pixel *centres*, so
    column 0 must land on column ``W - 1``. The wrong form shifts every mirrored label by
    exactly one pixel -- a quarter of a heatmap cell at :data:`~deeperfly.training.STRIDE`
    4, far too small to notice in a loss curve and exactly the class of half-pixel error
    this package has already shipped once, in a heatmap decoder.

    Applying this twice returns the input: ``flip_perm`` is an involution (each point is in
    at most one symmetry pair) and the coordinate map is its own inverse. ``tests`` pin that
    round trip, because it is the cheapest total check on the whole operation.
    """
    perm = np.asarray(flip_perm, dtype=np.int64)
    pts = np.asarray(points) if not _is_torch(points) else points
    n_points = pts.shape[-2]
    if perm.shape != (n_points,):
        raise ValueError(
            f"flip_perm has {perm.shape} entries but points carry {n_points} points; "
            "the permutation must cover exactly the skeleton's points"
        )

    if image is not None:
        w = _width_of(image)
        if width is not None and int(width) != w:
            raise ValueError(
                f"width={width} contradicts the image's own width {w}; pass one or neither"
            )
        width = w
    elif width is None:
        raise ValueError("width is required when no image is given")

    out_image = None if image is None else _flip_x(image)
    out_points = _mirror_points(pts, int(width), perm)
    out_extra = tuple(_take_points(a, perm) for a in per_point)
    return (out_image, out_points, *out_extra)


def mirror_view_names(view_names, mirror: dict[str, str]):
    """``view index -> the index of its mirror view``, as an ``(V,)`` int array.

    Parameters
    ----------
    view_names
        The view order (the ``V`` axis of a points array).
    mirror
        ``view -> mirror view``, i.e. :meth:`deeperfly.config.Config.mirror_views`.

    Returns
    -------
    np.ndarray
        ``(V,)`` indices. A view that declares no mirror maps to **itself**, which is the
        conservative choice: it leaves the sample's camera id unchanged rather than
        inventing a pairing, so a partially-declared rig degrades to "this camera is not
        remapped" instead of to a wrong side label.

    Raises
    ------
    ValueError
        If a declared mirror is not one of ``view_names``. (Symmetry is checked where the
        mapping is read from config, in :meth:`~deeperfly.config.Config.mirror_views`.)
    """
    names = list(view_names)
    index = {n: i for i, n in enumerate(names)}
    out = np.arange(len(names), dtype=np.int64)
    for name, other in mirror.items():
        if name not in index:
            continue  # a mirror declared for a view this plan does not carry
        if other not in index:
            raise ValueError(
                f"camera {name!r} mirrors to unknown view {other!r}; views: {names}"
            )
        out[index[name]] = index[other]
    return out


def mirror_decisions(
    n_groups: int,
    p: float,
    *,
    rng,
    n_views: int = 1,
):
    """One mirror decision **per frame group**, broadcast to that group's views.

    Parameters
    ----------
    n_groups
        Number of frame groups in the batch (``1`` per sample for a single-view dataset).
    p
        Probability of mirroring a group.
    rng
        A :class:`numpy.random.Generator`. Supplied by the caller so a dataset can reseed
        it per DataLoader worker -- one shared stream would hand every worker the same
        sequence of decisions.
    n_views
        Views per group.

    Returns
    -------
    np.ndarray
        ``(n_groups, n_views)`` bool, constant along the view axis.

    Notes
    -----
    The per-group constraint is the part that is easy to miss and expensive to get wrong.
    A mirror permutes the L/R channels *and* remaps the camera id, so flipping the views of
    one moment independently leaves the group internally inconsistent: within a single
    frame, some views' channel 0 means the left front leg and others' means the right, with
    nothing in a cross-view fusion to disambiguate them. It also breaks the geometry --
    those views no longer describe one physical scene, so multi-view triangulation or an
    epipolar consistency term is being fed a chimera.

    Deciding per group costs nothing (mirrored and unmirrored groups still appear at rate
    ``p``) and is why this returns a ``(groups, views)`` array instead of a flat one.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability in [0, 1], got {p}")
    per_group = rng.random(int(n_groups)) < p
    return np.repeat(per_group[:, None], int(n_views), axis=1)


# -- array-library plumbing ---------------------------------------------------


def _is_torch(a) -> bool:
    """Whether ``a`` is a torch tensor, without importing torch to find out."""
    return type(a).__module__.startswith("torch")


def _width_of(image) -> int:
    """The pixel width of a ``(..., H, W)`` or ``(..., H, W, C)`` image.

    A trailing axis of 1, 3 or 4 is read as channels. That heuristic is only ever wrong for
    an image 1, 3 or 4 pixels wide, which is not a frame; pass ``width`` explicitly for the
    pathological case.
    """
    shape = tuple(image.shape)
    if len(shape) >= 3 and shape[-1] in (1, 3, 4):
        return int(shape[-2])
    return int(shape[-1])


def _channels_last(image) -> bool:
    shape = tuple(image.shape)
    return len(shape) >= 3 and shape[-1] in (1, 3, 4)


def _flip_x(image):
    """Mirror an image on its width axis, preserving array type and device."""
    axis = -2 if _channels_last(image) else -1
    if _is_torch(image):
        return image.flip(axis)
    # `np.flip` returns a reversed view; the copy is what keeps a cached frame intact and
    # keeps downstream torch.from_numpy (which rejects negative strides) working.
    return np.ascontiguousarray(np.flip(np.asarray(image), axis=axis))


def _mirror_points(points, width: int, perm):
    """``(..., P, 2)`` -> x-mirrored and channel-permuted, as a new array."""
    if _is_torch(points):
        import torch

        out = points.clone()
        out[..., 0] = (width - 1) - out[..., 0]
        return out.index_select(-2, torch.as_tensor(perm, device=out.device))
    out = np.array(points, dtype=np.asarray(points).dtype, copy=True)
    out[..., 0] = (width - 1) - out[..., 0]
    return np.take(out, perm, axis=-2)


def _take_points(a, perm):
    """Permute a per-point array on its point axis.

    The point axis is the last for a ``(..., P)`` array and the second-to-last for a
    ``(..., P, k)`` one. Which it is cannot be guessed from the shape alone, so the rule is
    declared: an array whose last axis length equals the permutation length is ``(..., P)``.
    A ``(..., P, k)`` array with ``k == P`` is ambiguous and rejected rather than guessed.
    """
    n = len(perm)
    shape = tuple(a.shape)
    trailing_is_points = shape[-1] == n
    inner_is_points = len(shape) >= 2 and shape[-2] == n
    if trailing_is_points and inner_is_points:
        raise ValueError(
            f"per-point array of shape {shape} is ambiguous: both of the last two axes "
            f"have length {n}, so which one indexes points cannot be determined. Reshape "
            "it, or pass the coordinates through the `points` argument instead."
        )
    if not trailing_is_points and not inner_is_points:
        raise ValueError(
            f"per-point array of shape {shape} has no axis of length {n} to permute"
        )
    axis = -1 if trailing_is_points else -2
    if _is_torch(a):
        import torch

        return a.index_select(axis % a.dim(), torch.as_tensor(perm, device=a.device))
    return np.take(np.asarray(a), perm, axis=axis)
