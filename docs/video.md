# Video I/O

`deeperfly.video` reads and writes frames through a pluggable backend registry.
The base install reads/writes via `pyav` (PyAV) — in-process libx264, with its
own FFmpeg bundled in the wheel, so no system FFmpeg is needed. Install an extra
for an alternative or faster decoder:

| Backend | Read | Write | Frames | Install |
| --- | :-: | :-: | --- | --- |
| `pyav` | ✓ | ✓ | NumPy (CPU) | core (default) |
| `opencv` | ✓ | ✓ | NumPy (CPU) | `opencv` |
| `imageio` | ✓ | ✓ | NumPy (CPU) | `imageio` |
| `decord` | ✓ | – | NumPy / `torch` (CPU/**CUDA**) | `decord` |
| `video_reader_rs` | ✓ | – | NumPy (CPU) | `video-reader-rs` |
| `torchcodec` | ✓ | – | `torch.Tensor` (CPU/**CUDA**) | `torchcodec` / `cuda` |
| `dali` | ✓ | – | `torch.Tensor` / NumPy (**CUDA**) | `dali` |

Image *sequences* (a directory or glob of PNG/JPG/…) are always read via
`imageio` (core), independent of the video backend above.

```python
from deeperfly import video

frames = video.read_frames(path)                        # video file or image dir; auto NumPy (host)
frames = video.read_video("clip.mp4", indices=[0, 50])  # random access
frames = video.read_video("clip.mp4", device="cuda")    # on-GPU tensor (NVDEC), zero-copy to JAX via to_jax
video.write_mp4(frames, "out.mp4", fps=30)
```

`backend="auto"` and `device="auto"` (the defaults) pick the fastest working path.
`deeperfly run` decodes on the **CPU by default** and uploads each window to the
GPU in one shot — within a few percent of GPU/NVDEC end to end, since the 2D
detector (not decode) is the bottleneck. Opt into on-device NVDEC decode with
`[detector] decode_device = "cuda"` (it falls back to CPU if no GPU decoder is
available). See the config comments and `deeperfly.video` docstrings for the full
decoder details.
