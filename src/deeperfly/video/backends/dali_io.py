"""NVIDIA DALI backend: GPU (NVDEC) batch video decode.

DALI (https://github.com/NVIDIA/DALI) builds a small one-shot pipeline that
decodes the whole clip on the GPU. When a CUDA ``device`` is requested the
frames are handed back as a GPU ``torch.Tensor`` (zero-copy via DLPack where
possible); otherwise they are copied to the host as NumPy.

DALI ships on NVIDIA's package index rather than PyPI, e.g.::

    pip install nvidia-dali-cuda120

so it is intentionally absent from this project's extras and only used when the
package and a GPU are present. The whole clip is decoded, then ``start:stop:step``
(or ``indices`` via the base fallback) is applied; it needs enough device memory
to hold every frame.
"""

from __future__ import annotations

import numpy as np

from ..base import ReaderBackend, device_id, is_gpu_device, register_reader


@register_reader
class DALIReader(ReaderBackend):
    name = "dali"
    requires = ("nvidia.dali",)
    supports_gpu = True
    supports_seek = False

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        from nvidia.dali import fn, pipeline_def

        dev_id = device_id(device) if is_gpu_device(device) else 0

        @pipeline_def(batch_size=1, num_threads=2, device_id=dev_id)
        def _pipeline():
            data, _ = fn.readers.file(files=[str(path)])
            return fn.experimental.decoders.video(data, device="mixed")

        pipe = _pipeline()
        pipe.build()
        (out,) = pipe.run()  # TensorListGPU, one sample, layout FHWC
        sl = slice(start, stop, step)

        if is_gpu_device(device):
            try:  # keep frames on the GPU
                import torch

                return torch.from_dlpack(out[0])[sl]
            except Exception:  # noqa: BLE001 -- fall back to a host copy
                pass

        arr = np.asarray(out.as_cpu()[0])[sl]
        if is_gpu_device(device):
            import torch

            return torch.as_tensor(arr, device=str(device))
        return arr
