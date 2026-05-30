"""PyNvVideoCodec backend: NVIDIA's direct NVDEC bindings (GPU only).

PyNvVideoCodec (NVIDIA's officially supported NVDEC/NVENC bindings, the
successor to VPF) decodes entirely on the GPU. ``SimpleDecoder`` supports random
access by frame index, and decoded frames expose ``__dlpack__`` so they hand off
to torch (and onward to JAX) without leaving the device.

GPU-only and version-sensitive -- treat as experimental; validate on a CUDA box.
Frames are requested as RGB ``(N, H, W, 3)``; pass ``device="cpu"`` to copy the
result back to the host as NumPy.
"""

from __future__ import annotations

from ..base import ReaderBackend, device_id, is_gpu_device, register_reader


def _decoder(nvc, path, device):
    kwargs = {}
    if hasattr(nvc, "OutputColorType"):  # ask for RGB where the build supports it
        kwargs["output_color_type"] = nvc.OutputColorType.RGB
    if is_gpu_device(device):
        kwargs["gpu_id"] = device_id(device)
    try:
        return nvc.SimpleDecoder(str(path), **kwargs)
    except TypeError:  # older signature without those kwargs
        return nvc.SimpleDecoder(str(path))


def _frame_to_tensor(frame):
    import torch

    if hasattr(frame, "__dlpack__"):
        return torch.from_dlpack(frame)
    data = getattr(frame, "data", frame)  # some versions wrap the surface
    return torch.as_tensor(data)


def _stack(decoder, idx, device):
    import torch

    frames = torch.stack([_frame_to_tensor(decoder[i]) for i in idx])
    return frames if is_gpu_device(device) else frames.detach().cpu().numpy()


@register_reader
class PyNvVideoCodecReader(ReaderBackend):
    name = "pynvvideocodec"
    requires = ("PyNvVideoCodec", "torch")
    supports_gpu = True
    supports_seek = True

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        import PyNvVideoCodec as nvc

        decoder = _decoder(nvc, path, device)
        n = len(decoder)
        stop = n if stop is None else min(stop, n)
        return _stack(decoder, range(start, stop, step), device)

    @classmethod
    def _read_indices(cls, path, device, indices):
        import PyNvVideoCodec as nvc

        decoder = _decoder(nvc, path, device)
        return _stack(decoder, list(indices), device)
