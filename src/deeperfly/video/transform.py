"""Per-camera frame preprocessing applied once at decode time.

A :class:`FrameTransform` is a small, lossless geometric correction --
left-right flip, up-down flip and/or rotation by whole quarter-turns -- applied
to a camera's frames *right after they are read*, before anything else sees them.
A camera physically mounted sideways/upside-down (or whose sensor is mirrored)
is brought into a sane orientation once, and the **transformed** frame then is
the canonical frame for the whole run: the 2D detector, the inferred principal
point, calibration, triangulation and the visualization overlays all operate on
it. Nothing maps coordinates back to the raw footage.

Configured per camera under ``[preprocess.<camera>]`` (see
:func:`parse_frame_transforms`); a camera with no entry is left untouched.

The forward op matches the NumPy functions of the same name and is applied in a
fixed order: :func:`numpy.fliplr` (left-right), then :func:`numpy.flipud`
(up-down), then :func:`numpy.rot90` ``rot90`` times (counter-clockwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import to_numpy

_ALLOWED_KEYS = ("fliplr", "flipud", "rot90")


@dataclass(frozen=True)
class FrameTransform:
    """A flip / rot90 correction for one camera's frames (default: identity).

    ``rot90`` is a counter-clockwise quarter-turn count (``np.rot90`` semantics),
    kept ``mod 4``. :meth:`apply` works on a ``(T, H, W, C)`` (or ``(H, W, C)``)
    batch and preserves the input's array type/device where it can -- a NumPy
    array stays NumPy, a torch tensor stays a torch tensor on its device (so a
    GPU-decoded window still feeds the detector without a host round-trip).
    """

    fliplr: bool = False
    flipud: bool = False
    rot90: int = 0  # counter-clockwise quarter-turns (k), taken mod 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "rot90", int(self.rot90) % 4)

    def is_identity(self) -> bool:
        return not self.fliplr and not self.flipud and self.rot90 == 0

    def apply(self, frames):
        """Apply the transform to a frame batch, on the ``(H, W)`` axes (-3, -2)."""
        if self.is_identity():
            return frames
        if hasattr(frames, "rot90") and hasattr(frames, "flip"):  # torch.Tensor
            if self.fliplr:
                frames = frames.flip(-2)
            if self.flipud:
                frames = frames.flip(-3)
            if self.rot90:
                frames = frames.rot90(self.rot90, dims=(-3, -2))
            return frames
        arr = to_numpy(frames)  # NumPy already; decord/DALI -> host (rare path)
        if self.fliplr:
            arr = np.flip(arr, axis=-2)
        if self.flipud:
            arr = np.flip(arr, axis=-3)
        if self.rot90:
            arr = np.rot90(arr, self.rot90, axes=(-3, -2))
        return np.ascontiguousarray(arr)


def parse_frame_transforms(config: dict) -> dict[str, FrameTransform]:
    """Build ``camera name -> FrameTransform`` from the ``[preprocess.*]`` tables.

    Each ``[preprocess.<camera>]`` table takes ``fliplr`` / ``flipud`` (bool) and
    ``rot90`` (int quarter-turns, CCW). Cameras with no table are absent from the
    dict (callers treat a missing camera as the identity). Raises on an unknown
    key or a non-integer ``rot90`` so config typos fail loudly.
    """
    section = config.get("preprocess", {})
    out: dict[str, FrameTransform] = {}
    for name, spec in section.items():
        extra = set(spec) - set(_ALLOWED_KEYS)
        if extra:
            raise ValueError(
                f"[preprocess.{name}] has unknown key(s) {sorted(extra)}; "
                f"allowed: {list(_ALLOWED_KEYS)}"
            )
        rot90 = spec.get("rot90", 0)
        if isinstance(rot90, bool) or not isinstance(rot90, int):
            raise ValueError(
                f"[preprocess.{name}].rot90 must be an integer quarter-turn count "
                f"(e.g. 0/1/2/3), got {rot90!r}"
            )
        out[name] = FrameTransform(
            fliplr=bool(spec.get("fliplr", False)),
            flipud=bool(spec.get("flipud", False)),
            rot90=rot90,
        )
    return out
