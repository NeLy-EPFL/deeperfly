# Library usage

The public API lives at the top level (`from deeperfly import ...`): `Camera`,
`CameraGroup`, `Skeleton`, `PoseResult`, `bundle_adjust`,
`bundle_adjust_from_config`, `run_from_points2d`, and the `geometry`,
`triangulate`, `correction`, `pictorial`, `pipeline` submodules.

Sections of a `config.toml` are independently usable: `CameraGroup.from_config`
reads only the cameras, `Skeleton.from_config` only `[skeleton]`.

## Geometry / bundle adjustment only

```python
from deeperfly import CameraGroup, bundle_adjust

group = CameraGroup.from_config("examples/cameras.toml")
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

## The full 2D→3D pipeline from an existing 2D detection array

```python
from deeperfly import CameraGroup, Skeleton, run_from_points2d

cameras = CameraGroup.from_config("examples/cameras.toml")
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf, smooth="one_euro")
result.save("fly.h5")
```

`run_from_points2d(..., triangulation=..., do_pictorial=...)` selects the 3D
reconstruction: `triangulation` is `ransac` (default), `greedy` or `dlt`, and
`do_pictorial=True` runs pictorial-structures peak recovery first (see
[architecture.md](architecture.md)).

## Video I/O

`deeperfly.video` reads and writes frames through a pluggable backend registry;
see [video.md](video.md) for the backends.

```python
from deeperfly import video

frames = video.read_frames(path)                        # video file or image dir; NumPy (host)
frames = video.read_video("clip.mp4", indices=[0, 50])  # random access
frames = video.read_video("clip.mp4", backend="torchcodec")  # torch tensor (CPU)
video.write_mp4(frames, "out.mp4", fps=30)
```

## Examples

- [`examples/bundle_adjustment.ipynb`](../examples/bundle_adjustment.ipynb) — the bundle-adjustment walkthrough.
- [`examples/pipeline_walkthrough.ipynb`](../examples/pipeline_walkthrough.ipynb) — the pipeline one stage at a time.
- [`examples/pipeline_demo.py`](../examples/pipeline_demo.py) — a synthetic end-to-end run (no weights required).
