"""Decord backend: random-access video reader (CPU or CUDA).

Decord (https://github.com/dmlc/decord) is built around frame-accurate random
access -- ``VideoReader.get_batch`` fetches arbitrary frames cheaply -- and can
decode straight onto a GPU context. Frames come back RGB ``(N, H, W, 3)``; on a
GPU context they stay a ``torch.Tensor`` (via DLPack), otherwise NumPy.

Decord is unmaintained upstream but still widely installed; it is an optional
extra here. CUDA decode requires a GPU-enabled build.
"""

from __future__ import annotations

from ..base import ReaderBackend, device_id, is_gpu_device, register_reader


def _context(decord, device):
    if is_gpu_device(device):
        return decord.gpu(device_id(device))
    return decord.cpu(0)


def _to_frames(batch, device):
    """Decord NDArray -> NumPy (CPU) or torch.Tensor (GPU, zero-copy)."""
    if is_gpu_device(device):
        import torch

        return torch.from_dlpack(batch.to_dlpack())
    return batch.asnumpy()


@register_reader
class DecordReader(ReaderBackend):
    name = "decord"
    requires = ("decord",)
    supports_gpu = True
    supports_seek = True

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        import decord

        reader = decord.VideoReader(str(path), ctx=_context(decord, device))
        n = len(reader)
        stop = n if stop is None else min(stop, n)
        return _to_frames(reader.get_batch(list(range(start, stop, step))), device)

    @classmethod
    def _read_indices(cls, path, device, indices):
        import decord

        reader = decord.VideoReader(str(path), ctx=_context(decord, device))
        return _to_frames(reader.get_batch(list(indices)), device)
