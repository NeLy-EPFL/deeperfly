"""Per-camera frame preprocessing applied once at decode time.

A :class:`FrameTransform` is a lossless geometric correction -- left-right flip,
up-down flip and/or quarter-turn rotation -- applied to a camera's frames right
after they are read, for a camera mounted sideways/upside-down or with a mirrored
sensor. The transformed frame is the canonical frame for the whole run (detector,
principal point, calibration, overlays); nothing maps back to the raw footage.

Configured per camera under ``[cameras.<camera>.preprocess]`` (see
:func:`parse_frame_transforms`). Applied in a fixed order: :func:`numpy.fliplr`,
then :func:`numpy.flipud`, then :func:`numpy.rot90` ``rot90`` times (CCW).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .io.base import to_numpy

if TYPE_CHECKING:
    from .config import Config

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
        """Apply the transform to a frame batch, on the ``(H, W)`` axes (-3, -2).

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
            if self.fliplr:
                frames = frames.flip(-2)
            if self.flipud:
                frames = frames.flip(-3)
            if self.rot90:
                frames = frames.rot90(self.rot90, dims=(-3, -2))
            return frames
        arr = to_numpy(frames)  # NumPy already (the CPU-decode path)
        if self.fliplr:
            arr = np.flip(arr, axis=-2)
        if self.flipud:
            arr = np.flip(arr, axis=-3)
        if self.rot90:
            arr = np.rot90(arr, self.rot90, axes=(-3, -2))
        return np.ascontiguousarray(arr)


def parse_frame_transforms(
    config: "Config",
) -> dict[str, FrameTransform]:
    """Build ``camera name -> FrameTransform`` from the per-camera preprocess tables.

    Each ``[cameras.<camera>.preprocess]`` table takes ``fliplr`` / ``flipud`` (bool)
    and ``rot90`` (int quarter-turns, CCW).

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`.

    Returns
    -------
    dict of str to FrameTransform
        ``camera_name -> FrameTransform`` for cameras with a preprocess table;
        cameras without one are absent (callers treat them as the identity).

    Raises
    ------
    ValueError
        On an unknown preprocess key or a non-integer ``rot90`` (so config typos
        fail loudly).
    """
    _, cameras = config.camera_table()
    out: dict[str, FrameTransform] = {}
    for name, cam in cameras.items():
        spec = cam.get("preprocess")
        if not spec:
            continue
        extra = set(spec) - set(_ALLOWED_KEYS)
        if extra:
            raise ValueError(
                f"[cameras.{name}.preprocess] has unknown key(s) {sorted(extra)}; "
                f"allowed: {list(_ALLOWED_KEYS)}"
            )
        rot90 = spec.get("rot90", 0)
        if isinstance(rot90, bool) or not isinstance(rot90, int):
            raise ValueError(
                f"[cameras.{name}.preprocess].rot90 must be an integer quarter-turn "
                f"count (e.g. 0/1/2/3), got {rot90!r}"
            )
        out[name] = FrameTransform(
            fliplr=bool(spec.get("fliplr", False)),
            flipud=bool(spec.get("flipud", False)),
            rot90=rot90,
        )
    return out
