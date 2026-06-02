"""imageio (+ imageio-ffmpeg) backend -- the portable CPU default.

Sequential decode only; random access uses the base decode-once-and-gather
fallback (imageio-ffmpeg's per-index seeking is not reliably frame-accurate).
"""

from __future__ import annotations

import numpy as np

from ..base import (
    ReaderBackend,
    WriterBackend,
    register_reader,
    register_writer,
    require_cpu,
)


@register_reader
class ImageIOReader(ReaderBackend):
    name = "imageio"
    requires = ("imageio", "imageio_ffmpeg")
    supports_gpu = False
    supports_seek = False

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        require_cpu(device, "imageio")
        import imageio.v2 as imageio

        out = []
        with imageio.get_reader(str(path)) as reader:
            for i, frame in enumerate(reader):
                if i < start:
                    continue
                if stop is not None and i >= stop:
                    break
                if (i - start) % step == 0:
                    out.append(np.asarray(frame)[..., :3])
        if not out:
            raise ValueError(f"imageio decoded no frames from {path!r}")
        return np.stack(out)


@register_writer
class ImageIOWriter(WriterBackend):
    name = "imageio"
    requires = ("imageio", "imageio_ffmpeg")

    @staticmethod
    def write(frames, path, *, fps=30.0, codec=None, quality=None, **kwargs):
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            fps=fps,
            codec=codec or "libx264",
            quality=quality,
            macro_block_size=None,
            **kwargs,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
