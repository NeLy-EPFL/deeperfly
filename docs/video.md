# Video I/O

`deeperfly.video` reads and writes video files with **PyAV** — in-process libx264,
with FFmpeg bundled in the wheel (no system FFmpeg needed). All decoding and
encoding runs on the CPU and yields `(T, H, W, 3)` uint8 RGB NumPy.

Image *sequences* (a directory or glob of PNG/JPG/…) are read by a separate,
configurable image reader:

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
video.write_mp4(frames, "out.mp4", fps=30)
```

`deeperfly run` decodes on the CPU and uploads each window to the detector device
in one shot — decode is not the bottleneck, the detector forward is. Video I/O has
no configuration; only the image-sequence decoder is selectable, via the shared
`[io.image] reader` (and `workers`), applied across every stage. See the config
comments and `deeperfly.video` docstrings for details.
