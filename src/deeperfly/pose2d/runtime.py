"""Device, dtype and tensor plumbing shared by every detector class.

None of this is architecture-specific: where the forward runs, how an input array
becomes an on-device tensor, and what precision the convolutions are allowed to use are
the same questions for the multiview transformer, for HRNet and for anything a project
plugs in later. They live here rather than inside one network's module so that adding or
retiring a detector does not move them -- which is exactly what happened before: these
four helpers were written for a network since retired, both dense detectors imported
them out of it, and deleting that network meant relocating them first.

The only piece deliberately left out is ``torch.compile``: whether a forward is worth
compiling is a property of the network (its graph, its batch size, its warm-up cost), so
it belongs to whichever class wants it.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "USE_CHANNELS_LAST",
    "device",
    "set_precision",
]

#: Lay the CUDA conv batch out ``channels_last`` (NHWC in memory, same NCHW logical
#: shape) so cuDNN picks its faster Tensor-Core conv kernels. A CUDA-only win;
#: CPU/MPS keep the default contiguous layout regardless. Set ``False`` to force
#: the plain NCHW path (e.g. to A/B the speedup or work around a cuDNN regression).
USE_CHANNELS_LAST = True

#: Detector forward precision -> autocast dtype (``None`` = run in float32).
_PRECISIONS = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def device() -> str:
    """Best available torch device: NVIDIA CUDA, then Apple Metal (MPS), else CPU.

    On Apple Silicon ``"mps"`` runs the forward on the GPU via Metal Performance
    Shaders; output matches CPU to float32 epsilon, so the detector is accelerated on
    macOS with no setup.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def as_torch(inputs) -> "torch.Tensor":
    """Coerce network inputs to a torch tensor, on-device when possible.

    ``inputs`` is usually already a ``torch.Tensor`` on the detector device (so a
    GPU-decoded frame reaches the forward without leaving the GPU) and passes
    straight through. Any other DLPack-capable on-device array is bridged
    zero-copy; host NumPy is copied to a writable tensor.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs
    if hasattr(inputs, "__dlpack__"):  # other on-device array -- zero-copy
        return torch.from_dlpack(inputs)
    return torch.from_numpy(np.array(inputs))  # host array -> writable copy


def set_precision(model, precision: str = "float32") -> None:
    """Record the forward precision the detector should run in.

    ``"float32"`` (default, the reference), ``"float16"`` or ``"bfloat16"`` (CUDA
    autocast). float16 is fastest; bfloat16 trades a touch of that speed for the
    wider exponent range of float32, so it cannot overflow. Stored on the model so
    the forward picks it up without threading it through every call.

    Parameters
    ----------
    model
        The detector (the precision is stored on it).
    precision
        ``"float32"``, ``"float16"`` or ``"bfloat16"``.

    Raises
    ------
    ValueError
        On an unknown ``precision`` name.
    """
    precision = (precision or "float32").lower()
    if precision not in _PRECISIONS:
        opts = ", ".join(repr(p) for p in _PRECISIONS)
        raise ValueError(f"unknown detector precision {precision!r}; use one of {opts}")
    # A real detector (nn.Module) carries a __dict__; bare test stubs don't (and
    # never run the real forward), so there's nothing to set on them.
    if hasattr(model, "__dict__"):
        model._deeperfly_precision = precision  # type: ignore[assignment]


def autocast_dtype(model, dev: "torch.device"):
    """Autocast dtype for the forward, or ``None`` to run in float32.

    Honors the precision set by :func:`set_precision`, but only on CUDA: fp16
    autocast is where the win is, and CPU/MPS stay on the float32 reference path.
    """
    if dev.type != "cuda":
        return None
    return _PRECISIONS.get(getattr(model, "_deeperfly_precision", "float32"))
