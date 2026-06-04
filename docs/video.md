# Video I/O

`deeperfly.video` reads and writes frames through a pluggable backend registry.
The base install reads/writes via `pyav` — in-process libx264, with FFmpeg
bundled in the wheel (no system FFmpeg needed). Install an extra for an
alternative or faster decoder:

| Backend | Read | Write | Frames | Install |
| --- | :-: | :-: | --- | --- |
| `pyav` | ✓ | ✓ | NumPy | core (default) |
| `opencv` | ✓ | ✓ | NumPy | core |
| `video_reader_rs` | ✓ | – | NumPy | `video-reader-rs` |
| `torchcodec` | ✓ | – | `torch.Tensor` | `torchcodec` |

All decoding runs on the CPU. Image *sequences* (a directory or glob of
PNG/JPG/…) are read by a separate image reader, independent of the video backends
above:

| Image reader | Install | Notes |
| --- | --- | --- |
| `opencv` | core (default) | fast; ~1.6× quicker than imageio on JPEG |
| `imageio` | `imageio` extra | broad-format fallback for files OpenCV can't decode |

`image_backend="auto"` uses OpenCV and falls back to `imageio` (when the extra is
installed) only for files OpenCV cannot decode.

```python
from deeperfly import video

frames = video.read_frames(path)                        # video file or image dir; NumPy (host)
frames = video.read_video("clip.mp4", indices=[0, 50])  # random access
frames = video.read_video("clip.mp4", backend="torchcodec")  # torch tensor (CPU)
video.write_mp4(frames, "out.mp4", fps=30)
```

`backend="auto"` (the default) picks the fastest installed decoder. `deeperfly
run` decodes on the CPU and uploads each window to the detector device in one shot
— decode is not the bottleneck, the detector forward is. The backends are
configured once in the shared `[io]` section — `[io.video] reader` (input
decoder), `[io.video] writer` (output encoder) and `[io.image] reader`
(image-sequence decoder) — and apply across every stage. See the config comments
and `deeperfly.video` docstrings for details.
