# Library usage

The public API lives at the top level (`from deeperfly import ...`): `Camera`,
`CameraGroup`, `Skeleton`, `PoseResult`, `bundle_adjust`,
`bundle_adjust_from_config`, `run_from_points2d`, and the `geometry`,
`triangulation`, `pictorial`, `pipeline` submodules.

Sections of a `config.toml` are independently usable: load it once with
`Config.from_toml`, then `CameraGroup.from_config(config)` reads only the
cameras, `Skeleton.from_config(config)` only `[skeleton]`.

## Geometry / bundle adjustment only

```python
from deeperfly import CameraGroup, Config, bundle_adjust

group = CameraGroup.from_config(Config.from_toml("examples/cameras.toml"))
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

## The full 2D→3D pipeline from an existing 2D detection array

```python
from deeperfly import CameraGroup, Config, Skeleton, run_from_points2d

cameras = CameraGroup.from_config(Config.from_toml("examples/cameras.toml"))
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf)
result.save("fly.h5")
```

`run_from_points2d(..., triangulation=..., do_pictorial=...)` selects the 3D
reconstruction: `triangulation` is `ransac` (default), `greedy` or `dlt`, and
`do_pictorial=True` runs pictorial-structures peak recovery first (see
[architecture.md](architecture.md)).

## Frame I/O

`deeperfly.io` reads and writes frames through `pyav` (in-process FFmpeg, CPU);
`open_reader(source)` returns a `VideoReader` or `ImageSequenceReader` you then
index (`reader[:]`, `reader[i]`, `reader[[0, 3, 5]]`) or stream
(`stream_frames` / `stream_blocks`). See [video.md](video.md) for details.

```python
from deeperfly import io

frames = io.open_reader(path)[:]                  # video file or image dir
frames = io.VideoReader("clip.mp4")[[0, 50]]      # random access
with io.VideoWriter("out.mp4", fps=30) as writer:
    writer.write_frames(frames)                   # or write_frame() per frame
```

## Examples

- [`examples/bundle_adjustment.ipynb`](../examples/bundle_adjustment.ipynb) — the bundle-adjustment walkthrough.
- [`examples/pipeline_walkthrough.ipynb`](../examples/pipeline_walkthrough.ipynb) — the pipeline one stage at a time.
- [`examples/pipeline_demo.py`](../examples/pipeline_demo.py) — a synthetic end-to-end run (no weights required).
