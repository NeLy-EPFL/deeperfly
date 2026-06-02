"""TorchCodec backend: hardware/CPU decode straight into a torch tensor.

TorchCodec (https://github.com/pytorch/torchcodec) decodes via FFmpeg and can
place frames directly on a CUDA device (NVDEC), so the result never round-trips
through host memory. Frames stay a ``torch.Tensor`` on the requested device;
use :func:`deeperfly.video.to_numpy` to bring them to the host, or
:func:`deeperfly.video.to_jax` to hand them to JAX zero-copy via DLPack.

Random access uses TorchCodec's native ``get_frames_at`` (true seeking).

CUDA decode needs a CUDA-enabled torchcodec build (from the PyTorch index, e.g.
``pip install torchcodec --index-url https://download.pytorch.org/whl/cu130``) and
NVIDIA NPP (``pip install nvidia-npp``). NPP's shared libraries ship under
``site-packages/nvidia/.../lib`` -- off the loader path -- so :func:`_preload_npp`
``dlopen``s them by absolute path before the first CUDA decode, sparing the user
an ``LD_LIBRARY_PATH`` tweak. (The plain PyPI ``torchcodec`` wheel is CPU-only and
fails over to the next GPU backend.)
"""

from __future__ import annotations

from ..base import ReaderBackend, is_gpu_device, register_reader

_npp_preloaded = False


def _preload_npp() -> None:
    """Load NVIDIA NPP libs (RTLD_GLOBAL) so torchcodec's CUDA decode can find them.

    No-op if NPP isn't installed (torchcodec then errors and ``auto`` falls back).
    """
    global _npp_preloaded
    if _npp_preloaded:
        return
    import ctypes
    import glob
    import os
    import sysconfig

    libs: list[str] = []
    for site in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
        libs += glob.glob(
            os.path.join(site, "nvidia", "**", "lib", "libnpp*.so*"), recursive=True
        )
    libs.sort(key=lambda p: 0 if "nppc" in os.path.basename(p) else 1)  # core first
    for lib in libs:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # a missing dependency just means this lib stays unavailable
    _npp_preloaded = True


def _decoder(path, device):
    from torchcodec.decoders import VideoDecoder

    if is_gpu_device(device):
        _preload_npp()
    return VideoDecoder(str(path), device=str(device))


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
        decoder = _decoder(path, device)
        n = len(decoder)
        stop = n if stop is None else min(stop, n)
        return _to_nhwc(decoder[start:stop:step])  # uint8 tensor on ``device``

    @classmethod
    def _read_indices(cls, path, device, indices):
        decoder = _decoder(path, device)
        batch = decoder.get_frames_at(indices=list(indices))
        # FrameBatch.data is (N, C, H, W) uint8 on ``device``.
        data = getattr(batch, "data", batch)
        return _to_nhwc(data)
