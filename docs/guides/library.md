# Using the library API

deeperfly is a Python library as well as a CLI. The public API lives at the top
level (`from deeperfly import ...`): `Config`, `Camera`, `CameraGroup`,
`Skeleton`, `PoseResult`, `Recording`, `bundle_adjust`,
`bundle_adjust_from_config`, `run_from_points2d`, `run_recording`,
`resolve_recordings`, `detect_2d`, `load_detector`, and the `geometry`,
`triangulation`, `pictorial`, `pipeline`, `recordings`, and `io` submodules. This
guide shows the common tasks; the [API reference](../reference/api.md) documents
every symbol.

Add it to your project — prefix with `UV_TORCH_BACKEND=auto` so uv picks the
right PyTorch wheel (`uv add` has no `--torch-backend` flag):

```bash
UV_TORCH_BACKEND=auto uv add git+https://github.com/NeLy-EPFL/deeperfly
```

Sections of a `config.toml` are independently usable: load it once with
`Config.from_toml`, then `CameraGroup.from_config(config)` reads only the
cameras, `Skeleton.from_config(config)` only `[skeleton]`. Foreign sections are
ignored, so a config you only use for its rig needs no detector or visualization
keys.

## Geometry and bundle adjustment

```python
from deeperfly import CameraGroup, Config, bundle_adjust

group = CameraGroup.from_config(Config.from_toml("config.toml"))
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

`bundle_adjust` returns the raw `scipy.optimize.OptimizeResult`, the refined
`CameraGroup`, and the refined 3D points. `fixed` / `shared` anchor the world
gauge with the same grammar as the config (`"*.intr"`, `"f.rvec"`, tying
`[["lf.tvec[2]", "rf.tvec[2]"]]`); extra keyword arguments
(`max_nfev`, `loss`, `f_scale`, ...) pass straight to scipy.
`bundle_adjust_from_config(config, pts2d)` drives the same call from a config's
`[bundle_adjustment]` section.

The lower-level `deeperfly.geometry` module holds the JAX projection /
triangulation / Rodrigues primitives (JIT- and grad-friendly, float64 on the
CPU), and `deeperfly.triangulation` the NaN-aware DLT and RANSAC helpers.

## The full 2D→3D pipeline from a 2D detection array

If you already have 2D detections, `run_from_points2d` runs the array pipeline
(no files): optional bundle adjustment → 3D reconstruction, returning a
`PoseResult`.

```python
from deeperfly import CameraGroup, Config, Skeleton, run_from_points2d

cameras = CameraGroup.from_config(Config.from_toml("config.toml"))
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf)
result.save("fly.h5")
```

`run_from_points2d(..., triangulation=..., do_pictorial=...)` selects the 3D
reconstruction: `triangulation` is `ransac` (default), `greedy` or `dlt`, and
`do_pictorial=True` runs pictorial-structures peak recovery first (see the
[pipeline explainer](../explanation/pipeline.md)). Unobserved points are expected
to already be NaN.

## Running a recording with caching

`run_recording` is the staged run behind `deeperfly run`: it resolves the config
against an output directory, runs the enabled `[pipeline]` stages, and reuses
cached results whose config is unchanged.

```python
from deeperfly import resolve_recordings, run_recording
from deeperfly import Config

config = Config.default()
for src, sources in resolve_recordings(["recordings/fly1"], config=config):
    run_recording(None, src / "deeperfly_outputs", sources=sources)
```

To run detection yourself, `load_detector` loads the PyTorch model and
`detect_2d` streams 2D detection over a recording given a detection plan.

## Inspecting a result

`PoseResult` is the assembled, self-contained result — the cameras, skeleton,
`pts2d` `(V, T, P, 2)`, `conf`, `pts3d` `(T, P, 3)` and `reproj_error`. It
round-trips through HDF5 and reconstructs the cameras and skeleton, so a result
is portable without the original config.

```python
from deeperfly import PoseResult

result = PoseResult.load("recording/deeperfly_outputs/results.h5")
print(result.n_views, result.n_frames)
xyz = result.pts3d              # (T, P, 3), NaN where un-triangulated
```

See the [output-format reference](../reference/output-format.md) for the on-disk
schema and the "best available" assembly rule `load` applies.

## Frame I/O

`deeperfly.io` reads and writes frames through **PyAV** (in-process FFmpeg, with
libx264 bundled in the wheel — no system FFmpeg needed). All decoding and
encoding runs on the CPU and yields `(T, H, W, 3)` uint8 RGB NumPy.

`open_reader(source)` resolves a source to a `VideoReader` (a video file) or an
`ImageSequenceReader` (a directory, glob, or explicit file list), both subclasses
of `FrameReader`. You then index it, stream it, or probe `count` / `fps`. Image
sequences are decoded by OpenCV, in parallel across threads.

```python
from deeperfly import io

reader = io.open_reader(path)                 # video file or image dir/glob/list
frames = reader[:]                            # (T, H, W, 3) uint8 NumPy (host)
clip = io.VideoReader("clip.mp4")[[0, 50]]    # random access (seeks per frame)
for block in io.open_reader(path).stream_blocks(block_size=64):  # forward, low memory
    ...

# VideoWriter encodes a frame, a batch, or any iterable -- so a long clip can be
# written as it is produced, without ever holding every frame in memory.
with io.VideoWriter("out.mp4", fps=30) as writer:
    writer.write_frames(frames)               # or write_frame() per frame
```

## Examples

Two runnable notebooks, rendered here with their outputs (the Plotly figures
stay interactive):

- [Bundle adjustment](../examples/bundle_adjustment.ipynb) — build a multi-camera
  rig, perturb it, and recover it with `bundle_adjust`.
- [Pipeline walkthrough](../examples/pipeline_walkthrough.ipynb) — the full
  2D→3D pipeline one stage at a time.
