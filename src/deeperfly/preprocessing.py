"""Per-camera frame preprocessing applied once at decode time.

A :class:`FrameTransform` is an ordered sequence of standard image operations
-- left-right / up-down flip, quarter-turn rotation, crop and resize -- applied
to a camera's frames right after they are read, in the order written in the
config. The transformed frame is the canonical frame for the whole run
(detector, 2D points, calibration, overlays); nothing maps back to the raw
footage.

Camera intrinsics in the config refer to the *raw* footage frame: every op
carries an exact affine pixel map, and the composed chain transforms the
principal point (and scales/swaps the focal lengths) into the canonical frame
via :meth:`FrameTransform.map_intrinsics`.

Configured per camera as an ordered list under ``[cameras.<camera>]``::

    preprocess = [
        { op = "rot90", k = 1 },
        { op = "fliplr" },
        { op = "crop", x = 10, y = 10, width = 80, height = 80 },
        { op = "resize", scale = 0.5 },
    ]

(see :func:`parse_frame_transforms`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from .io.base import to_numpy

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    "Fliplr",
    "Flipud",
    "Rot90",
    "Crop",
    "Resize",
    "FrameTransform",
    "frame_transform_from_ops",
    "parse_frame_transforms",
]

_OP_NAMES = ("fliplr", "flipud", "rot90", "crop", "resize")
_INTERPOLATIONS = ("bilinear", "nearest")

# Distortion coefficients (OpenCV order, as in :mod:`deeperfly.geometry`) that
# are *not* radially symmetric: tangential p1/p2 and thin-prism s1..s4. These
# do not survive a mirror or rotation of the image, unlike the radial k terms.
_NON_RADIAL_DIST_IDX = (2, 3, 8, 9, 10, 11)


@dataclass(frozen=True)
class Fliplr:
    """Left-right mirror (:func:`numpy.fliplr` per frame)."""

    def is_identity(self) -> bool:
        return False

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        return size

    def affine(self, size: tuple[int, int]) -> np.ndarray:
        _, w = size
        return np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.flip(arr, axis=-2)

    def apply_torch(self, frames):
        return frames.flip(-2)

    def to_json(self) -> dict:
        return {"op": "fliplr"}


@dataclass(frozen=True)
class Flipud:
    """Up-down flip (:func:`numpy.flipud` per frame)."""

    def is_identity(self) -> bool:
        return False

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        return size

    def affine(self, size: tuple[int, int]) -> np.ndarray:
        h, _ = size
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]])

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.flip(arr, axis=-3)

    def apply_torch(self, frames):
        return frames.flip(-3)

    def to_json(self) -> dict:
        return {"op": "flipud"}


@dataclass(frozen=True)
class Rot90:
    """``k`` counter-clockwise quarter-turns (:func:`numpy.rot90` semantics)."""

    k: int = 1  # any sign, kept mod 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", int(self.k) % 4)

    def is_identity(self) -> bool:
        return self.k == 0

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        h, w = size
        return (w, h) if self.k % 2 else (h, w)

    def affine(self, size: tuple[int, int]) -> np.ndarray:
        # One CCW quarter-turn maps (x, y) -> (y, w-1-x); compose it k times,
        # threading the intermediate size (each turn swaps h and w).
        a = np.eye(3)
        h, w = size
        for _ in range(self.k):
            quarter = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0], [0.0, 0.0, 1.0]])
            a = quarter @ a
            h, w = w, h
        return a

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.rot90(arr, self.k, axes=(-3, -2))

    def apply_torch(self, frames):
        return frames.rot90(self.k, dims=(-3, -2))

    def to_json(self) -> dict:
        return {"op": "rot90", "k": self.k}


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

    def affine(self, size: tuple[int, int]) -> np.ndarray:
        self.output_size(size)  # bounds check
        return np.array(
            [[1.0, 0.0, -float(self.x)], [0.0, 1.0, -float(self.y)], [0.0, 0.0, 1.0]]
        )

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        self.output_size((arr.shape[-3], arr.shape[-2]))  # never truncate silently
        return arr[..., self.y : self.y + self.height, self.x : self.x + self.width, :]

    def apply_torch(self, frames):
        self.output_size((frames.shape[-3], frames.shape[-2]))
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
            return (self.height, self.width)
        h, w = size
        # Round half away from zero (cv2's saturate_cast), not banker's rounding.
        return (
            max(1, int(np.floor(h * self.scale + 0.5))),
            max(1, int(np.floor(w * self.scale + 0.5))),
        )

    def affine(self, size: tuple[int, int]) -> np.ndarray:
        h, w = size
        oh, ow = self.output_size(size)
        # Half-pixel convention: x' = (x + 0.5) * sx - 0.5, with the *actual*
        # post-rounding ratios (what cv2/torch use when given a target size).
        sx, sy = ow / w, oh / h
        return np.array(
            [[sx, 0.0, (sx - 1.0) / 2.0], [0.0, sy, (sy - 1.0) / 2.0], [0.0, 0.0, 1.0]]
        )

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[-3], arr.shape[-2]
        oh, ow = self.output_size((h, w))
        if (oh, ow) == (h, w):
            return arr
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

    def apply_torch(self, frames):
        h, w = frames.shape[-3], frames.shape[-2]
        oh, ow = self.output_size((h, w))
        if (oh, ow) == (h, w):
            return frames
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


FrameOp = Union[Fliplr, Flipud, Rot90, Crop, Resize]


def _normalize_ops(ops) -> tuple[FrameOp, ...]:
    """Drop no-ops and fold exact adjacent compositions (same-type flips cancel,
    consecutive rotations sum), so equivalent chains compare -- and
    fingerprint -- equal."""
    out: list[FrameOp] = []
    for op in ops:
        if op.is_identity():
            continue
        prev = out[-1] if out else None
        if isinstance(op, Rot90) and isinstance(prev, Rot90):
            out.pop()
            op = Rot90(k=prev.k + op.k)
            if op.is_identity():
                continue
        elif isinstance(op, (Fliplr, Flipud)) and type(prev) is type(op):
            out.pop()
            continue
        out.append(op)
    return tuple(out)


def _apply_affine(a: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous pixel map ``a`` to ``(..., 2)`` points ``(x, y)``."""
    pts = np.asarray(pts, dtype=float)
    return pts @ a[:2, :2].T + a[:2, 2]


@dataclass(frozen=True)
class FrameTransform:
    """An ordered op sequence for one camera's frames (default: identity).

    :meth:`apply` works on a ``(T, H, W, C)`` (or ``(H, W, C)``) batch and
    preserves the input's array type/device where it can -- a NumPy array stays
    NumPy, a torch tensor stays a torch tensor on its device (so a GPU-decoded
    window still feeds the detector without a host round-trip). The op list is
    normalized (see :func:`_normalize_ops`), so e.g. two half-turns are the
    identity and reuse the identity's cached results.
    """

    ops: tuple[FrameOp, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", _normalize_ops(self.ops))

    def is_identity(self) -> bool:
        return not self.ops

    def apply(self, frames):
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
        if hasattr(frames, "rot90") and hasattr(frames, "flip"):  # torch.Tensor
            for op in self.ops:
                frames = op.apply_torch(frames)
            return frames
        arr = to_numpy(frames)  # NumPy already (the CPU-decode path)
        for op in self.ops:
            arr = op.apply_numpy(arr)
        return np.ascontiguousarray(arr)

    def output_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """The ``(height, width)`` a frame of ``size`` has after the chain."""
        for op in self.ops:
            size = op.output_size(size)
        return tuple(int(d) for d in size)

    def affine(self, size: tuple[int, int]) -> np.ndarray:
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

    def map_points(self, pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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

    def unmap_points(self, pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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

    def map_intrinsics(
        self, intr: np.ndarray, dist: np.ndarray, raw_size: tuple[int, int]
    ) -> np.ndarray:
        """Map raw-frame intrinsics ``[fx, fy, cx, cy]`` into the canonical frame.

        The principal point maps as a pixel through :meth:`affine`; the focal
        lengths are *magnitudes*: they swap under an odd quarter-turn count and
        scale under a resize, but a mirror never makes them negative -- the
        config's orbit extrinsics describe the *canonical* (already corrected)
        view, so no reflection is folded into the camera model.

        Parameters
        ----------
        intr
            Packed raw-frame intrinsics ``[fx, fy, cx, cy]``.
        dist
            The camera's distortion coefficients (only inspected, never
            changed): radial terms are rotation/mirror-symmetric, but
            tangential/thin-prism terms are not.
        raw_size
            The raw footage ``(height, width)``.

        Returns
        -------
        np.ndarray
            Packed canonical-frame intrinsics ``[fx, fy, cx, cy]``.

        Raises
        ------
        ValueError
            If a nonzero tangential/thin-prism distortion coefficient is
            combined with a mirroring or rotating op (the coefficients would
            silently describe the wrong lens).
        """
        fx, fy, cx, cy = (float(v) for v in intr)
        a = self.affine(raw_size)
        lin, off = a[:2, :2], a[:2, 2]
        diag_positive = (
            lin[0, 1] == 0 and lin[1, 0] == 0 and lin[0, 0] > 0 and lin[1, 1] > 0
        )
        if not diag_positive:
            bad = [
                i
                for i in _NON_RADIAL_DIST_IDX
                if i < len(dist) and float(dist[i]) != 0.0
            ]
            if bad:
                raise ValueError(
                    f"non-radial distortion coefficients (index {bad}) do not "
                    f"survive a flip/rot90 preprocess op; calibrate for the "
                    f"flipped/rotated frame or drop the tangential/thin-prism terms"
                )
        new_c = lin @ np.array([cx, cy]) + off
        new_f = np.abs(lin) @ np.array([fx, fy])
        return np.array([new_f[0], new_f[1], new_c[0], new_c[1]])

    def to_json(self) -> list[dict]:
        """The chain as a canonical JSON-able op list (fingerprints, logs)."""
        return [op.to_json() for op in self.ops]


def _require_int(value, minimum: int, key: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{where}.{key} must be >= {minimum}, got {value}")
    return value


def _parse_op(step, where: str) -> FrameOp:
    """Parse one ``{ op = ... }`` table into a frame op (loud on typos)."""
    if not isinstance(step, dict) or not isinstance(step.get("op"), str):
        raise ValueError(
            f'{where} must be a table naming the op, like {{ op = "fliplr" }}; '
            f"got {step!r}"
        )
    name = step["op"]
    if name not in _OP_NAMES:
        raise ValueError(f"{where} has unknown op {name!r}; allowed: {list(_OP_NAMES)}")
    keys = set(step) - {"op"}
    allowed = {
        "fliplr": set(),
        "flipud": set(),
        "rot90": {"k"},
        "crop": {"x", "y", "width", "height"},
        "resize": {"width", "height", "scale", "interpolation"},
    }[name]
    if keys - allowed:
        raise ValueError(
            f"{where} ({name}) has unknown key(s) {sorted(keys - allowed)}; "
            f"allowed: {sorted(allowed)}"
        )
    try:
        if name == "fliplr":
            return Fliplr()
        if name == "flipud":
            return Flipud()
        if name == "rot90":
            k = step.get("k", 1)
            if isinstance(k, bool) or not isinstance(k, int):
                raise ValueError(
                    f"{where}.k must be an integer quarter-turn count "
                    f"(e.g. 1/2/3, any sign), got {k!r}"
                )
            return Rot90(k=k)
        if name == "crop":
            missing = allowed - set(step)
            if missing:
                raise ValueError(f"{where} (crop) missing key(s) {sorted(missing)}")
            return Crop(
                x=_require_int(step["x"], 0, "x", where),
                y=_require_int(step["y"], 0, "y", where),
                width=_require_int(step["width"], 1, "width", where),
                height=_require_int(step["height"], 1, "height", where),
            )
        scale = step.get("scale")
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise ValueError(f"{where}.scale must be a number, got {scale!r}")
            scale = float(scale)
        width = step.get("width")
        height = step.get("height")
        if width is not None:
            width = _require_int(width, 1, "width", where)
        if height is not None:
            height = _require_int(height, 1, "height", where)
        return Resize(
            width=width,
            height=height,
            scale=scale,
            interpolation=step.get("interpolation", "bilinear"),
        )
    except ValueError as exc:
        if str(exc).startswith(where):
            raise
        raise ValueError(f"{where}: {exc}") from exc


def frame_transform_from_ops(ops, where: str) -> FrameTransform:
    """Build a :class:`FrameTransform` from a list of ``{ op = ... }`` tables.

    The shared parser behind ``[cameras.<name>].preprocess`` and the named
    ``[[preprocessors]]`` of the detection plan, so both accept the exact same
    op grammar (and fail the same way on a typo).

    Parameters
    ----------
    ops
        An ordered list of op tables (or empty / ``None`` for the identity).
    where
        A label for error messages (e.g. ``"[[preprocessors]] 'mirror'"``).

    Returns
    -------
    FrameTransform
        The parsed, normalized transform.

    Raises
    ------
    ValueError
        If ``ops`` is not a list, or any op table is malformed.
    """
    if not ops:
        return FrameTransform(())
    if not isinstance(ops, list):
        raise ValueError(f"{where} must be a list of op tables, got {ops!r}")
    return FrameTransform(
        tuple(_parse_op(step, f"{where}[{i}]") for i, step in enumerate(ops))
    )


def parse_frame_transforms(
    config: "Config",
) -> dict[str, FrameTransform]:
    """Build ``camera name -> FrameTransform`` from the per-camera preprocess lists.

    Each ``[cameras.<camera>]`` table may carry ``preprocess``, an ordered list
    of op tables applied in the order written::

        preprocess = [
            { op = "rot90", k = 1 },          # CCW quarter-turns, any sign
            { op = "fliplr" },                  # also: flipud
            { op = "crop", x = 0, y = 0, width = 100, height = 100 },
            { op = "resize", scale = 0.5 },     # or width = .. , height = ..
        ]

    (equivalently ``[[cameras.<camera>.preprocess]]`` blocks).

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`.

    Returns
    -------
    dict of str to FrameTransform
        ``camera_name -> FrameTransform`` for cameras with a preprocess list;
        cameras without one are absent (callers treat them as the identity).

    Raises
    ------
    ValueError
        If ``preprocess`` is not an ordered list of op tables, or names an
        unknown op or op key, or carries a malformed op parameter (so config
        typos fail loudly).
    """
    defaults, cameras = config.camera_table()
    if "preprocess" in defaults:
        raise ValueError(
            "[cameras.defaults] does not take preprocess; frames are corrected "
            "per camera -- set preprocess on each camera that needs it"
        )
    out: dict[str, FrameTransform] = {}
    for name, cam in cameras.items():
        spec = cam.get("preprocess")
        if not spec:
            continue
        if not isinstance(spec, list):
            raise ValueError(
                f"[cameras.{name}].preprocess must be an ordered list of op "
                f'tables, e.g.\n  preprocess = [{{ op = "fliplr" }}, '
                f'{{ op = "rot90", k = 1 }}]\n'
                f"(or [[cameras.{name}.preprocess]] blocks), applied in the "
                f"order written; got {spec!r}"
            )
        ops = tuple(
            _parse_op(step, f"[cameras.{name}].preprocess[{i}]")
            for i, step in enumerate(spec)
        )
        out[name] = FrameTransform(ops)
    return out
