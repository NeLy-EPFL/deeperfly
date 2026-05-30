# deeperfly

A JAX rewrite of the [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) /
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) /
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment) pipeline
for markerless 3D pose estimation of tethered *Drosophila* from a multi-camera
rig: **2D pose → camera calibration (bundle adjustment) → triangulation → error
correction → visualization.**

The computer-vision core follows OpenCV's conventions (Rodrigues rotations,
`projectPoints` distortion, DLT triangulation) and is cross-checked against
OpenCV in the test suite. Everything geometric is JAX (JIT- and autodiff-
friendly); the 2D detector ships two interchangeable backends behind one
interface — a JAX (Equinox) port of DeepFly2D's stacked hourglass (the default,
and faster on GPU) and the original PyTorch network — selectable with
`--backend {jax,torch}`.

## Pipeline

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` (`backends/{jax,torch}/`) | Stacked hourglass in two backends behind one interface; JAX (Equinox) is the default, PyTorch runs the original weights directly. |
| Calibration | `pipeline.calibrate` → `bundle_adjustment/` | Fly-as-target BA: confidence weights, Huber loss, bone-length prior. |
| Triangulation | `triangulate.py` / `pipeline.reconstruct` | NaN-aware DLT + greedy reprojection-outlier rejection. |
| Correction | `correction.py` | Procrustes alignment (per side) + One-Euro / Gaussian smoothing. |
| Visualization | `viz.py`, `video.py` | matplotlib 2D overlays, 3D skeleton, MP4 export. |
| Result I/O | `io.py` | Self-contained HDF5 `PoseResult`. |
| Skeleton | `skeleton.py` + `data/skeleton_fly.toml` | 38 points, 10 limbs, 28 bones, per-camera visibility. |

## Usage

Geometry / bundle adjustment only:

```python
from deeperfly import CameraGroup, bundle_adjust

group = CameraGroup.from_config("examples/cameras.toml")
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

2D → 3D pipeline from an existing 2D detection array:

```python
from deeperfly import CameraGroup, Skeleton, run_from_points2d

cameras = CameraGroup.from_config("examples/cameras.toml")
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf, smooth="one_euro")
result.save("fly.h5")
```

End to end from images/video via the CLI:

```bash
deeperfly download-weights          # fetch original PyTorch weights (sh8)
deeperfly convert-weights           # -> native JAX checkpoint (skip if using --backend torch)
deeperfly run --in recording/ --config cameras.toml --out fly.h5 [--backend jax|torch]
deeperfly visualize --in fly.h5 --out fly_3d.mp4 --mode 3d [--bg white|black]
```

See [`examples/bundle_adjustment.ipynb`](examples/bundle_adjustment.ipynb) for the
BA walkthrough and [`examples/pipeline_demo.py`](examples/pipeline_demo.py) for a
synthetic end-to-end run (no weights required).

## 2D detector backends

The detector has two interchangeable backends behind one interface, under
`pose2d/backends/{jax,torch}/` — each exposing the same `HourglassNet` /
`load_model` / `predict_heatmaps`. Both are installed by default and selectable
with `--backend {jax,torch}`. The PyTorch backend runs the published `sh8`
weights directly; the JAX backend (the default) runs the same weights once
`convert-weights` has produced the native checkpoint, and is validated to match
the PyTorch reference numerically (see `tests/test_pose2d_torch.py`). JAX is the
faster backend on GPU — benchmark them on your own hardware:

```bash
uv run python dev/bench_pose2d.py --batch 7 --frames 8
```

## Video I/O backends

`deeperfly.video` reads and writes frames through a pluggable backend registry,
so you can pick where decoding happens and what the frames live in:

| Backend | Read | Write | Seek | Frames | Install |
| --- | :-: | :-: | :-: | --- | --- |
| `imageio` (default) | ✓ | ✓ | – | NumPy (CPU) | `viz` extra |
| `opencv` | ✓ | ✓ | ✓ | NumPy (CPU) | `opencv` extra |
| `pyav` | ✓ | ✓ | ✓ | NumPy (CPU) | `pyav` extra |
| `decord` | ✓ | – | ✓ | NumPy / `torch` (CPU/**CUDA**) | `decord` extra |
| `video_reader_rs` | ✓ | – | ✓ | NumPy (CPU) | `video-reader-rs` extra |
| `torchcodec` | ✓ | – | ✓ | `torch.Tensor` (CPU/**CUDA**) | `torchcodec` extra |
| `pynvvideocodec` | ✓ | – | ✓ | `torch.Tensor` (**CUDA**) | `pynvvideocodec` extra |
| `dali` | ✓ | – | – | `torch.Tensor` / NumPy (**CUDA**) | NVIDIA index |

```python
from deeperfly import video

frames = video.read_video("clip.mp4")                              # NumPy, CPU
frames = video.read_video("clip.mp4", backend="pyav")              # frame-accurate
frames = video.read_video("clip.mp4", indices=[0, 50, 120])        # random access
frames = video.read_video("clip.mp4", backend="torchcodec", device="cuda")  # GPU
video.write_mp4(frames, "out.mp4", fps=30, backend="opencv")
video.available_read_backends()       # what's installed here
```

`backend="auto"` picks the first installed backend (CPU: imageio → pyav →
opencv → decord → video_reader_rs; GPU: torchcodec → decord → pynvvideocodec →
dali). Pass `indices=[...]` for random access: seek-capable backends fetch just
those frames, others decode up to `max(indices)` and gather. `deeperfly run`
takes `--video-backend`. NVIDIA DALI ships from NVIDIA's package index (e.g.
`nvidia-dali-cuda120`), so it's not a pip extra but is used automatically when
importable.

### Image sequences (jpg / png / …)

A folder or glob of frames is read by `read_images`, which decodes in parallel
across threads (so throughput scales with cores), broadcasts grayscale to RGB and
supports the same `indices` / `start:stop:step` selection. `read_frames` is the
unified entry point — it routes a video file to `read_video` and an image
directory to `read_images` — so the pipeline accepts either input. With
`device="cuda"` JPEGs are decoded on the GPU (torchvision nvJPEG) straight into a
device tensor for `to_jax`:

```python
frames = video.read_images("frames/", workers=8)            # parallel CPU decode
frames = video.read_images("frames/", indices=[0, 10, 20])  # subset
frames = video.read_images("frames/", device="cuda")        # nvJPEG -> GPU tensor
frames = video.read_frames(path)                            # video file *or* image dir
```

### GPU frames → JAX (zero-copy)

The detector runs in JAX, so when you decode onto the GPU you want the frames in
a JAX array *without* a host round-trip. `video.to_jax` does this via DLPack —
on a shared GPU, JAX wraps the **same** device buffer the decoder produced:

```python
frames = video.read_video("clip.mp4", backend="torchcodec", device="cuda")  # torch, GPU
x = video.to_jax(frames)              # jax.Array on the GPU, zero-copy
# keep `frames` alive until JAX has consumed `x`
```

Notes: both sides must share the same CUDA device; modern JAX/torch handle the
DLPack stream synchronization. `video.to_jax` also accepts NumPy (host copy onto
JAX's default device) and decord/DALI GPU tensors. `video.to_numpy` is the
host-side counterpart. Since `pose2d.preprocess` already resizes/normalizes with
`jax.image.resize`, a `to_jax` array feeds straight in and stays on the GPU.

## Development

```bash
uv sync --group test                 # install with test dependencies
uv run --group test pytest           # run the suite (incl. PyTorch-equivalence tests)
```

Optional extras: `viz` (matplotlib + imageio for plotting/video), and the video
read/write backends `opencv` / `pyav` / `decord` / `video-reader-rs` /
`torchcodec` / `pynvvideocodec`. PyTorch is a core dependency (the second
detector backend), so no extra is needed for it.
