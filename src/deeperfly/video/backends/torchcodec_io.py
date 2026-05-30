"""TorchCodec backend: hardware/CPU decode straight into a torch tensor.

TorchCodec (https://github.com/pytorch/torchcodec) decodes via FFmpeg and can
place frames directly on a CUDA device (NVDEC), so the result never round-trips
through host memory. Frames stay a ``torch.Tensor`` on the requested device;
use :func:`deeperfly.video.to_numpy` to bring them to the host, or
:func:`deeperfly.video.to_jax` to hand them to JAX zero-copy via DLPack.

Random access uses TorchCodec's native ``get_frames_at`` (true seeking).
"""

from __future__ import annotations

from ..base import ReaderBackend, register_reader


def _to_nhwc(batch):
    if batch.ndim == 3:  # a single (C, H, W) frame
        batch = batch.unsqueeze(0)
    return batch.permute(0, 2, 3, 1).contiguous()  # -> (N, H, W, C)


@register_reader
class TorchCodecReader(ReaderBackend):
    name = "torchcodec"
    requires = ("torchcodec", "torch")
    supports_gpu = True
    supports_seek = True

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), device=str(device))
        n = len(decoder)
        stop = n if stop is None else min(stop, n)
        return _to_nhwc(decoder[start:stop:step])  # uint8 tensor on ``device``

    @classmethod
    def _read_indices(cls, path, device, indices):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), device=str(device))
        batch = decoder.get_frames_at(indices=list(indices))
        # FrameBatch.data is (N, C, H, W) uint8 on ``device``.
        data = getattr(batch, "data", batch)
        return _to_nhwc(data)
