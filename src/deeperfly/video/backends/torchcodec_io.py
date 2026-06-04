"""TorchCodec backend: CPU decode straight into a torch tensor.

TorchCodec (https://github.com/pytorch/torchcodec) decodes via FFmpeg into a
``torch.Tensor`` on the CPU; use :func:`deeperfly.video.to_numpy` to bring the
frames to a NumPy array (they are already torch tensors, so the detector can also
consume them directly via :func:`deeperfly.video.to_torch`).

Random access uses TorchCodec's native ``get_frames_at`` (true seeking).
"""

from __future__ import annotations

from ..base import ReaderBackend, register_reader


def _decoder(path):
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(str(path), device="cpu")


def _to_nhwc(batch):
    if batch.ndim == 3:  # a single (C, H, W) frame
        batch = batch.unsqueeze(0)
    return batch.permute(0, 2, 3, 1).contiguous()  # -> (N, H, W, C)


@register_reader
class TorchCodecReader(ReaderBackend):
    name = "torchcodec"
    requires = ("torchcodec", "torch")
    supports_seek = True

    @staticmethod
    def _read_sequential(path, start, stop, step):
        decoder = _decoder(path)
        n = len(decoder)
        stop = n if stop is None else min(stop, n)
        return _to_nhwc(decoder[start:stop:step])  # uint8 CPU tensor

    @classmethod
    def _read_indices(cls, path, indices):
        decoder = _decoder(path)
        batch = decoder.get_frames_at(indices=list(indices))
        # FrameBatch.data is (N, C, H, W) uint8 on the CPU.
        data = getattr(batch, "data", batch)
        return _to_nhwc(data)
