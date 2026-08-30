"""Frame geometry for detection: a detection WINDOW, and the resize into the model.

A :class:`FrameTransform` is an ordered sequence of image operations applied to a camera's
frames on the way to the detector and **inverted on the way back**, so a detection lands
in raw footage pixels however it was windowed to get there -- which is what lets a camera's
intrinsics go on describing the raw frame.

Two ops survive, and neither is spelled in a config any more:

* :class:`Crop` -- a detection window, from ``[pose2d.crops]``, keyed by camera.
* :class:`Resize` -- constructed internally to fit the detector's own ``input_size``
  (:mod:`deeperfly.pose2d.pathways`); never a config op, and it never had a caller as one.

:class:`AutoCrop` is a *placeholder* rather than a transform: it declares a window
**searched per recording** by :mod:`deeperfly.pose2d.autocrop` instead of written down (a
camera named in ``[pose2d] auto_crops``). Until the search fills it in, every geometric
method raises :class:`UnresolvedAutoCrop`.

:class:`FrameOp` is the contract those three satisfy. It is a :class:`typing.Protocol`
(structural) rather than a base class, because the ops share no implementation to put in
one -- :class:`AutoCrop` reaches its geometry by *holding* a :class:`Crop`, not by
inheriting from it. Each op applies to **any** array type through a single
:meth:`~FrameOp.apply`; only :class:`Resize` has to tell NumPy and torch apart, and it
does so behind that method.

What went with the op grammar in 0.3.0: ``fliplr``, ``flipud``, ``rot90``, and ``resize``
as a config op. The flips existed for the mirrored detection pathway -- one source detected
twice, once flipped, for a side-agnostic 19-channel checkpoint -- and that pathway is not
expressible under a dense one-detector-per-camera plan. With no reflection left in any
chain there is no handedness to reverse either, so ``reverses_handedness`` goes too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from jaxtyping import Float, Shaped

from .io.base import to_numpy

__all__ = [
    "FrameOp",
    "Crop",
    "AutoCrop",
    "UnresolvedAutoCrop",
    "Resize",
    "PadToAspect",
    "FrameTransform",
]

_INTERPOLATIONS = ("bilinear", "nearest")


def _is_torch(frames) -> bool:
    """Whether ``frames`` is a torch tensor, decided without importing torch.

    ``detach`` and ``device`` together are unique to torch among the array types that
    reach here: a NumPy array has neither, a JAX array has ``device`` but no ``detach``.
    """
    return hasattr(frames, "detach") and hasattr(frames, "device")


class FrameOp(Protocol):
    """One image operation in a :class:`FrameTransform` chain.

    The three implementations (:class:`Crop`, :class:`AutoCrop`, :class:`Resize`) are
    frozen dataclasses that satisfy this structurally -- nothing inherits from it, and
    nothing needs to: the protocol exists to write the contract down in one place, not
    to share code.

    The contract is a *pairing*, and it is the whole reason a detection can be carried
    back to raw footage pixels: :meth:`affine` must be the exact pixel map that
    :meth:`apply` performs, and :meth:`output_size` the size it produces. Every op maps
    an axis-aligned rectangle to an axis-aligned rectangle. An implementation that
    breaks the pairing breaks :meth:`FrameTransform.unmap_points` silently -- points
    land in the wrong place rather than raising.
    """

    def is_identity(self) -> bool:
        """Whether this op provably leaves *any* frame untouched (see the chain)."""
        ...

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """The ``(height, width)`` a frame of ``(height, width)`` ``size`` becomes."""
        ...

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        """The ``3x3`` homogeneous pixel map this op applies to a ``size`` frame."""
        ...

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        """Apply the op to a ``(..., H, W, C)`` batch of any array type.

        The input's array type and device are preserved: a NumPy array stays NumPy, a
        torch tensor stays a tensor on its device.
        """
        ...

    def to_json(self) -> dict:
        """The op as a canonical JSON-able dict (fingerprints, logs)."""
        ...


@dataclass(frozen=True)
class Crop:
    """Keep the ``width x height`` window with top-left corner ``(x, y)``."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"crop origin must be non-negative, got ({self.x}, {self.y})"
            )
        if self.width < 1 or self.height < 1:
            raise ValueError(
                f"crop size must be positive, got {self.width}x{self.height}"
            )

    def is_identity(self) -> bool:
        return False  # without the frame size, a crop is never provably a no-op

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        h, w = size
        if self.x + self.width > w or self.y + self.height > h:
            raise ValueError(
                f"crop x={self.x} y={self.y} width={self.width} height={self.height} "
                f"exceeds the {w}x{h} (WxH) frame"
            )
        return (self.height, self.width)

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        self.output_size(size)  # bounds check
        return np.array(
            [[1.0, 0.0, -float(self.x)], [0.0, 1.0, -float(self.y)], [0.0, 0.0, 1.0]]
        )

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        # Bounds-check first, so an out-of-frame window raises instead of truncating.
        self.output_size((frames.shape[-3], frames.shape[-2]))
        # One expression for every array type: NumPy and torch slice alike.
        return frames[
            ..., self.y : self.y + self.height, self.x : self.x + self.width, :
        ]

    def to_json(self) -> dict:
        return {
            "op": "crop",
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


class UnresolvedAutoCrop(ValueError):
    """An automatic crop was asked for its geometry before its box was decided.

    Every geometric method of :class:`AutoCrop` raises this while the window is unknown,
    so a plan that reaches the pixels unresolved fails loudly here instead of quietly
    detecting through the whole frame -- which is the very failure the automatic crop
    exists to prevent, and which looks like a bad detector rather than a bad config.
    """


@dataclass(frozen=True)
class AutoCrop:
    """A crop whose window is **searched per recording** instead of written down.

    Declared by naming the camera in ``[pose2d] auto_crops``, optionally with a box in
    ``[pose2d.crops]`` as a **seed**. The seed is not the box:
    it is the incumbent the search starts from, and it narrows the search domain to its
    neighbourhood. Without one the search covers the whole frame at the model's own
    aspect, which is what a brand-new rig needs and costs a few more probes.

    Two states, and the distinction is the whole design:

    * **unresolved** (``box is None``) -- every geometric method raises
      :class:`UnresolvedAutoCrop`. A :class:`FrameTransform` carrying one cannot transform
      a frame, size an output, or map a point.
    * **resolved** -- :meth:`resolve` returns a copy whose ``box`` is set, and every method
      then behaves exactly as the equivalent :class:`Crop`.

    :meth:`to_json` always reports the **declaration**, never the resolved window. That is
    deliberate: the JSON is what the pipeline fingerprints, so a run whose search picked a
    box does not then look like a config change and recompute detection forever. The box
    is a *deterministic function* of the footage, the weights and the seed -- all three
    already fingerprinted or fixed by the output directory -- so it needs no entry of its
    own, and it is recorded beside the results for provenance instead.

    Attributes
    ----------
    seed
        The starting ``(x, y, width, height)``, or ``None`` for a blind search.
    box
        The resolved ``(x, y, width, height)``, or ``None`` while unresolved.
    where
        A config location used in error messages; excluded from equality and from
        :meth:`to_json`, so two identical declarations from different files still compare
        equal (and fingerprint the same).
    """

    seed: tuple[int, int, int, int] | None = None
    box: tuple[int, int, int, int] | None = None
    where: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for name in ("seed", "box"):
            value = getattr(self, name)
            if value is None:
                continue
            box = tuple(int(v) for v in value)
            if len(box) != 4:
                raise ValueError(
                    f"an automatic crop's {name} must be (x, y, width, height), "
                    f"got {value!r}"
                )
            Crop(*box)  # the same non-negative-origin / positive-size checks
            object.__setattr__(self, name, box)

    @property
    def resolved(self) -> bool:
        """Whether the window has been decided (see :meth:`resolve`)."""
        return self.box is not None

    def resolve(self, box: tuple[int, int, int, int]) -> "AutoCrop":
        """A copy of this op with its window set to ``box`` (the seed is kept)."""
        return AutoCrop(seed=self.seed, box=box, where=self.where)

    def _crop(self) -> Crop:
        if self.box is None:
            raise UnresolvedAutoCrop(
                f"the automatic crop {self.where or '(unnamed)'} has no window yet, so "
                "this frame cannot be transformed. Its box is searched by the pose2d "
                "stage: run pose2d (or `deeperfly auto-crop`) before anything that needs "
                "the geometry, or replace `auto = true` with an explicit "
                "x/y/width/height."
            )
        return Crop(*self.box)

    def is_identity(self) -> bool:
        return False  # a searched window is never provably a no-op

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        return self._crop().output_size(size)

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        return self._crop().affine(size)

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        return self._crop().apply(frames)

    def to_json(self) -> dict:
        """The DECLARATION (``auto`` plus the seed), never the resolved window."""
        out: dict = {"op": "crop", "auto": True}
        if self.seed is not None:
            x, y, w, h = self.seed
            out.update(x=x, y=y, width=w, height=h)
        return out


@dataclass(frozen=True)
class PadToAspect:
    """Widen or heighten a frame to ``aspect`` (width/height) by padding, never cropping.

    The op that makes "a crop must not stretch the animal" enforceable. A camera does not
    stretch anything; every stretch in this pipeline is manufactured by resizing a window
    of one shape into a network input of another, per axis. Padding the window out to the
    network's aspect first makes that resize a pure scale, which is the only thing the
    detector was trained to undo.

    Padding rather than cropping to the aspect, and the difference matters more the worse
    the mismatch is. Cropping a 960x512 frame to 2:1 costs 32 rows, which on this rig is
    survivable; cropping a 1280x800 one costs 160, which can take a leg with it. Padding
    cannot clip anything, at the price of a border the network must be trained through --
    which is why this is not on by default, and why turning it on is a training-round
    decision rather than a config tweak.

    ``value`` is in raw frame units (0-255 for a decoded frame). It defaults to 0, but the
    caller that knows the model should pass the corpus mean: a border at the mean is
    exactly zero once the model subtracts its mean, where black is a hard edge the network
    can read as anatomy. Same argument as ``dfpose``'s patch masking, which masks to the
    corpus mean deliberately rather than to black.

    The padding is split evenly, with the odd pixel going right/bottom, so the content stays
    centred to within half a pixel and :meth:`affine` is an exact translation.
    """

    aspect: float
    value: float = 0.0

    def __post_init__(self) -> None:
        if not self.aspect > 0:
            raise ValueError(f"pad aspect must be positive, got {self.aspect}")

    def _insets(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` in pixels for a ``(height, width)`` frame."""
        h, w = int(size[0]), int(size[1])
        if w < h * self.aspect:  # too tall/narrow -> widen
            extra = int(round(h * self.aspect)) - w
            left = extra // 2
            return (left, 0, extra - left, 0)
        if h < w / self.aspect:  # too wide/short -> heighten
            extra = int(round(w / self.aspect)) - h
            top = extra // 2
            return (0, top, 0, extra - top)
        return (0, 0, 0, 0)

    def is_identity(self) -> bool:
        return False  # without the frame size, whether anything is added is unknown

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        left, top, right, bottom = self._insets(size)
        return (int(size[0]) + top + bottom, int(size[1]) + left + right)

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        left, top, _, _ = self._insets(size)
        return np.array(
            [[1.0, 0.0, float(left)], [0.0, 1.0, float(top)], [0.0, 0.0, 1.0]]
        )

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        left, top, right, bottom = self._insets((frames.shape[-3], frames.shape[-2]))
        if not (left or top or right or bottom):
            return frames
        if _is_torch(frames):
            import torch.nn.functional as F

            # F.pad's last-axis-first pad order, over (..., H, W, C): the channel axis
            # takes no padding, then W, then H.
            return F.pad(
                frames,
                (0, 0, left, right, top, bottom),
                mode="constant",
                value=self.value,
            )
        pad_width = [(0, 0)] * frames.ndim
        pad_width[-3] = (top, bottom)
        pad_width[-2] = (left, right)
        return np.pad(
            to_numpy(frames), pad_width, mode="constant", constant_values=self.value
        )

    def to_json(self) -> dict:
        return {"op": "pad", "aspect": self.aspect, "value": self.value}


def _nearest_indices(out_dim: int, in_dim: int) -> np.ndarray:
    """Half-pixel nearest-neighbor source index per output index.

    Matches torch's ``nearest-exact`` (cv2's ``INTER_NEAREST`` does *not* use
    the half-pixel convention, so the NumPy path gathers explicitly).
    """
    src = np.floor((np.arange(out_dim) + 0.5) * in_dim / out_dim)
    return np.clip(src.astype(int), 0, in_dim - 1)


@dataclass(frozen=True)
class Resize:
    """Resample to a target size (``width``/``height``) or by a uniform ``scale``."""

    width: int | None = None
    height: int | None = None
    scale: float | None = None
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        sized = self.width is not None or self.height is not None
        if self.scale is not None:
            if sized:
                raise ValueError("resize takes either scale or width/height, not both")
            if self.scale <= 0:
                raise ValueError(f"resize scale must be positive, got {self.scale}")
        else:
            if self.width is None or self.height is None:
                raise ValueError("resize needs either scale or both width and height")
            if self.width < 1 or self.height < 1:
                raise ValueError(
                    f"resize size must be positive, got {self.width}x{self.height}"
                )
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(
                f"unknown resize interpolation {self.interpolation!r}; "
                f"allowed: {list(_INTERPOLATIONS)}"
            )

    def is_identity(self) -> bool:
        return self.scale == 1.0  # a width/height no-op depends on the frame size

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        if self.scale is None:
            assert self.height is not None and self.width is not None
            return (self.height, self.width)
        h, w = size
        # Round half away from zero (cv2's saturate_cast), not banker's rounding.
        return (
            max(1, int(np.floor(h * self.scale + 0.5))),
            max(1, int(np.floor(w * self.scale + 0.5))),
        )

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        h, w = size
        oh, ow = self.output_size(size)
        # Half-pixel convention: x' = (x + 0.5) * sx - 0.5, with the *actual*
        # post-rounding ratios (what cv2/torch use when given a target size).
        sx, sy = ow / w, oh / h
        return np.array(
            [[sx, 0.0, (sx - 1.0) / 2.0], [0.0, sy, (sy - 1.0) / 2.0], [0.0, 0.0, 1.0]]
        )

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        h, w = frames.shape[-3], frames.shape[-2]
        oh, ow = self.output_size((h, w))
        if (oh, ow) == (h, w):
            return frames
        # The one op whose backends genuinely differ: cv2 (or an index gather) on the
        # host, F.interpolate on whatever device the tensor already lives on.
        if _is_torch(frames):
            return self._resize_torch(frames, (oh, ow))
        return self._resize_numpy(frames, (oh, ow))

    def _resize_numpy(self, arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        oh, ow = size
        h, w = arr.shape[-3], arr.shape[-2]
        if self.interpolation == "nearest":
            rows = _nearest_indices(oh, h)
            cols = _nearest_indices(ow, w)
            return arr[..., rows[:, None], cols, :]
        import cv2

        batch = arr.reshape((-1,) + arr.shape[-3:])
        out = np.empty(batch.shape[:1] + (oh, ow) + batch.shape[3:], dtype=arr.dtype)
        for i, frame in enumerate(batch):
            out[i] = cv2.resize(
                frame, (ow, oh), interpolation=cv2.INTER_LINEAR
            ).reshape(oh, ow, -1)
        return out.reshape(arr.shape[:-3] + (oh, ow) + arr.shape[-1:])

    def _resize_torch(self, frames, size: tuple[int, int]):
        oh, ow = size
        import torch
        import torch.nn.functional as F  # noqa: N812

        batch = frames.reshape((-1,) + frames.shape[-3:]).permute(0, 3, 1, 2)
        # Integer interpolate is not portable on CUDA: go through float32.
        x = batch if batch.is_floating_point() else batch.float()
        if self.interpolation == "bilinear":
            x = F.interpolate(
                x, size=(oh, ow), mode="bilinear", align_corners=False, antialias=False
            )
        else:
            x = F.interpolate(x, size=(oh, ow), mode="nearest-exact")
        if x.dtype != batch.dtype:
            info = torch.iinfo(batch.dtype)
            x = x.round_().clamp_(info.min, info.max).to(batch.dtype)
        out = x.permute(0, 2, 3, 1)
        return out.reshape(frames.shape[:-3] + (oh, ow, frames.shape[-1]))

    def to_json(self) -> dict:
        out: dict = {"op": "resize"}
        if self.scale is not None:
            out["scale"] = self.scale
        else:
            out["width"], out["height"] = self.width, self.height
        out["interpolation"] = self.interpolation  # changes pixels -> fingerprinted
        return out


def _apply_affine(
    a: Float[np.ndarray, "3 3"], pts: Float[np.ndarray, "*N 2"]
) -> Float[np.ndarray, "*N 2"]:
    """Apply a 3x3 homogeneous pixel map ``a`` to ``(..., 2)`` points ``(x, y)``."""
    pts = np.asarray(pts, dtype=float)
    return pts @ a[:2, :2].T + a[:2, 2]


@dataclass(frozen=True)
class FrameTransform:
    """An ordered op sequence for one camera's frames (default: identity).

    :meth:`apply` works on a ``(T, H, W, C)`` (or ``(H, W, C)``) batch and
    preserves the input's array type/device where it can -- a NumPy array stays
    NumPy, a torch tensor stays a torch tensor on its device (so a GPU-decoded
    window still feeds the detector without a host round-trip). Identity ops are
    dropped, so an empty chain and a chain of no-ops are the same object.
    """

    ops: tuple[FrameOp, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ops", tuple(op for op in self.ops if not op.is_identity())
        )
        if sum(isinstance(op, AutoCrop) for op in self.ops) > 1:
            raise ValueError(
                "a preprocessing chain may carry at most one automatic crop; two would "
                "make 'the searched window' ambiguous (which of them does a panel or a "
                "detector mean?). Write the fixed one as an explicit x/y/width/height."
            )

    def is_identity(self) -> bool:
        return not self.ops

    @property
    def auto_crop(self) -> AutoCrop | None:
        """This chain's :class:`AutoCrop`, or ``None`` (at most one, see the constructor)."""
        for op in self.ops:
            if isinstance(op, AutoCrop):
                return op
        return None

    @property
    def needs_auto_crop(self) -> bool:
        """Whether this chain carries an automatic crop whose window is still unknown."""
        auto = self.auto_crop
        return auto is not None and not auto.resolved

    def resolve_auto_crop(self, box: tuple[int, int, int, int]) -> "FrameTransform":
        """This chain with its automatic crop's window set to ``box``.

        The rest of the chain is untouched, so a mirrored pathway keeps its mirror and
        keeps applying it *after* the crop -- the order written in the config is the order
        the detector was trained through.

        Raises
        ------
        ValueError
            If the chain has no automatic crop to resolve.
        """
        if self.auto_crop is None:
            raise ValueError(
                "this preprocessing chain has no automatic crop to resolve"
            )
        return FrameTransform(
            tuple(
                op.resolve(box) if isinstance(op, AutoCrop) else op for op in self.ops
            )
        )

    def apply(self, frames: Shaped[Any, "*B H W C"]) -> Shaped[Any, "*B H2 W2 C"]:
        """Apply the op sequence to a frame batch, on the ``(H, W)`` axes (-3, -2).

        Parameters
        ----------
        frames
            A ``(T, H, W, C)`` or ``(H, W, C)`` batch (NumPy array or torch
            tensor; the array type/device is preserved where possible).

        Returns
        -------
        The transformed batch (the input unchanged for the identity transform).
        """
        if self.is_identity():
            return frames
        if _is_torch(frames):
            for op in self.ops:
                frames = op.apply(frames)
            return frames
        arr = to_numpy(frames)  # NumPy already (the CPU-decode path)
        for op in self.ops:
            arr = op.apply(arr)
        return np.ascontiguousarray(arr)

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """The ``(height, width)`` a frame of ``size`` has after the chain."""
        for op in self.ops:
            size = op.output_size(size)
        return (int(size[0]), int(size[1]))

    def affine(self, size: tuple[int, int]) -> Float[np.ndarray, "3 3"]:
        """The composed ``3x3`` raw-to-canonical pixel map for a ``size`` frame.

        Homogeneous pixel-center coordinates ``(x, y, 1)`` (x = column,
        y = row); a raw-frame point ``p`` lands at ``affine(size) @ p`` in the
        transformed frame.
        """
        a = np.eye(3)
        for op in self.ops:
            a = op.affine(size) @ a
            size = op.output_size(size)
        return a

    def map_points(
        self, pts: Float[np.ndarray, "*N 2"], size: tuple[int, int]
    ) -> Float[np.ndarray, "*N 2"]:
        """Map raw-frame pixel points ``(x, y)`` into the transformed frame.

        Parameters
        ----------
        pts
            Points of shape ``(..., 2)`` in raw-frame pixel-center coordinates.
        size
            The raw frame ``(height, width)`` the chain is anchored on.

        Returns
        -------
        np.ndarray
            The points of shape ``(..., 2)`` in the transformed frame.
        """
        return _apply_affine(self.affine(size), pts)

    def unmap_points(
        self, pts: Float[np.ndarray, "*N 2"], size: tuple[int, int]
    ) -> Float[np.ndarray, "*N 2"]:
        """Map transformed-frame pixel points ``(x, y)`` back to the raw frame.

        The inverse of :meth:`map_points`: a detector/model peak located in the
        transformed (preprocessed) frame is brought back into the raw frame the
        camera's intrinsics describe -- this is how a pathway's points return to
        their view (undoing a mirror, resize, crop, ...). ``size`` is the *raw*
        frame size the chain is anchored on (not the transformed size).

        Parameters
        ----------
        pts
            Points of shape ``(..., 2)`` in transformed-frame pixel-center
            coordinates.
        size
            The raw frame ``(height, width)`` the chain is anchored on.

        Returns
        -------
        np.ndarray
            The points of shape ``(..., 2)`` in the raw frame.
        """
        return _apply_affine(np.linalg.inv(self.affine(size)), pts)

    def raw_window(self, raw_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """The RAW-frame window ``(x, y, width, height)`` this chain's output covers.

        Every op in the grammar maps an axis-aligned rectangle to an axis-aligned
        rectangle -- a crop translates, a resize scales, a flip or quarter-turn permutes
        the corners -- so the transformed frame always has exactly one raw-pixel box
        behind it. Recovered by mapping the output frame's pixel *edges* back through
        :meth:`affine`, which is exact for every op (the half-pixel resize convention
        included: its edges are fixed points of the map).

        This is what lets a consumer that can only express a *window* borrow one from a
        preprocessing chain instead of restating the box -- a ``[[visualization.videos]]``
        panel showing the region its detection pathway looked through, say. Only the
        region transfers, not the chirality: a chain that mirrors or turns the picture
        has the same window as one that does not.

        Parameters
        ----------
        raw_size
            The raw frame ``(height, width)`` the chain is anchored on.

        Returns
        -------
        x, y, width, height : int
            The window in raw-frame pixels. The full frame for the identity.

        Raises
        ------
        ValueError
            If a crop in the chain does not fit inside ``raw_size`` (a stale box against
            differently-sized footage) -- via :meth:`output_size`.
        """
        height, width = self.output_size(raw_size)  # bounds-checks the crops
        inv = np.linalg.inv(self.affine(raw_size))
        # Pixel EDGES, not centers: the window's extent is the outer boundary of the
        # first and last pixel kept, and a center-to-center span would lose a pixel.
        edges = np.array(
            [
                [-0.5, -0.5],
                [width - 0.5, -0.5],
                [-0.5, height - 0.5],
                [width - 0.5, height - 0.5],
            ]
        )
        raw = _apply_affine(inv, edges)
        x0, y0 = raw.min(axis=0)
        x1, y1 = raw.max(axis=0)
        return (
            int(round(float(x0) + 0.5)),
            int(round(float(y0) + 0.5)),
            int(round(float(x1 - x0))),
            int(round(float(y1 - y0))),
        )

    def to_json(self) -> list[dict]:
        """The chain as a canonical JSON-able op list (fingerprints, logs)."""
        return [op.to_json() for op in self.ops]
