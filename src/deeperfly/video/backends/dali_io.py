"""NVIDIA DALI backend: GPU (NVDEC) video decode, frame-accurate.

DALI (https://github.com/NVIDIA/DALI) builds a small one-shot pipeline that
decodes a video on the GPU via NVDEC. ``fn.decoders.video`` selects an exact
frame range (``start_frame`` / ``end_frame`` / ``stride``) or explicit ``frames``,
so a windowed read decodes **only** that window -- the streaming detector's reads
stay bounded, not "decode the whole clip then slice". On a CUDA ``device`` frames
come back as a GPU ``torch.Tensor`` (zero-copy via DLPack); otherwise they are
copied to the host as NumPy.

DALI's seeking is frame-accurate (its windows match
pyav/opencv/torchcodec exactly, bar a rounding bit in the YUV->RGB
conversion), so it is safe in ``backend="auto"``.

The CUDA-version-specific wheel is on PyPI -- install the one matching your CUDA
toolkit, e.g. ``pip install nvidia-dali-cuda130`` (CUDA 13) or the ``dali`` extra.
"""

from __future__ import annotations

import numpy as np

from ..base import ReaderBackend, _have, device_id, is_gpu_device, register_reader


def _decode(path, device, *, start_frame=0, end_frame=None, stride=1, frames=None):
    """Decode one window (or explicit ``frames``) -> ``(F, H, W, 3)`` on ``device``."""
    from nvidia.dali import fn, pipeline_def

    dev_id = device_id(device) if is_gpu_device(device) else 0

    @pipeline_def(batch_size=1, num_threads=2, device_id=dev_id)
    def _pipeline():
        data, _ = fn.readers.file(files=[str(path)])
        # "mixed" = NVDEC on the GPU. pad_mode="none" returns a short sequence at
        # the end of the clip instead of zero-padding, so the last streaming window
        # yields exactly the frames that remain.
        kwargs = {"device": "mixed", "pad_mode": "none"}
        if frames is not None:
            kwargs["frames"] = [int(i) for i in frames]
        else:
            kwargs["start_frame"] = int(start_frame)
            if end_frame is not None:
                kwargs["end_frame"] = int(end_frame)
            if stride != 1:
                kwargs["stride"] = int(stride)
        return fn.decoders.video(data, **kwargs)

    pipe = _pipeline()
    pipe.build()
    (out,) = pipe.run()  # one sample, FHWC
    if is_gpu_device(device):
        import torch

        return torch.from_dlpack(out[0])  # GPU tensor, zero-copy
    return np.asarray(out.as_cpu()[0])


@register_reader
class DALIReader(ReaderBackend):
    name = "dali"
    requires = ("nvidia.dali",)
    supports_gpu = True
    supports_seek = True  # start_frame/end_frame/frames decode only what's asked

    @classmethod
    def is_available(cls) -> bool:
        # ``nvidia.dali`` can resolve to a namespace-package stub (another
        # ``nvidia-*`` wheel created the ``nvidia`` namespace) without the compiled
        # decode API. Require the real ``fn`` submodule so a half-install reports
        # unavailable here instead of dying with an ImportError at decode time.
        return _have("nvidia.dali.fn")

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        return _decode(path, device, start_frame=start, end_frame=stop, stride=step)

    @classmethod
    def _read_indices(cls, path, device, indices):
        return _decode(path, device, frames=indices)
