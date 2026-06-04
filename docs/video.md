# Video I/O

`deeperfly.video` reads and writes frames through a pluggable backend registry.
The base install reads/writes via `pyav` (PyAV) — in-process libx264, with its
own FFmpeg bundled in the wheel, so no system FFmpeg is needed. Install an extra
for an alternative or faster decoder:

| Backend | Read | Write | Frames | Install |
| --- | :-: | :-: | --- | --- |
| `pyav` | ✓ | ✓ | NumPy | core (default) |
| `opencv` | ✓ | ✓ | NumPy | core |
| `video_reader_rs` | ✓ | – | NumPy | `video-reader-rs` |
| `torchcodec` | ✓ | – | `torch.Tensor` | `torchcodec` |

All decoding runs on the CPU. Image *sequences* (a directory or glob of
PNG/JPG/…) are always read via `imageio` (core), independent of the video backend
above.

```python
from deeperfly import video

frames = video.read_frames(path)                        # video file or image dir; NumPy (host)
frames = video.read_video("clip.mp4", indices=[0, 50])  # random access
frames = video.read_video("clip.mp4", backend="torchcodec")  # torch tensor (CPU)
video.write_mp4(frames, "out.mp4", fps=30)
```

`backend="auto"` (the default) picks the fastest installed decoder. `deeperfly
run` decodes on the CPU and uploads each window to the detector device (the GPU,
when present) in one shot — decode is not the bottleneck, the 2D detector forward
is. Pick the read decoder with `[pipeline.pose2d] video_backend` and the output
encoder with `[pipeline.visualization] video_backend`. See the config comments and
`deeperfly.video` docstrings for the full details.
