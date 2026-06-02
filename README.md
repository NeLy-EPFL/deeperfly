# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig.
Point it at a recording and it returns 3D joint positions and a rendered
skeleton video: 2D pose → camera calibration → triangulation → 3D correction →
visualization.

It is a modern rewrite of
[DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment). See [docs/comparison.md](docs/comparison.md) for how it differs
from the originals.

## Installation

deeperfly is both a command-line tool and a Python library. Install the CLI with
[uv](https://docs.astral.sh/uv/):

```bash
uv tool install deeperfly[cuda]
```

The `cuda` extra adds GPU acceleration for the 2D detector and is strongly
recommended on systems with NVIDIA GPUs; omit `[cuda]` for CPU-only use. On Apple Silicon, replace `cuda` with `mps` to use the Metal backend for the JAX-based detector.

To use deeperfly as a library, add it to your project instead:

```bash
uv add deeperfly[cuda]
```

## Quickstart

```bash
deeperfly init config.toml                                # write a config you can edit
deeperfly run recording/ -c config.toml -v                  # run the full pipeline
deeperfly info recording/deeperfly_outputs/poses.h5      # inspect the result
```

`deeperfly run` does everything in one command: detect 2D pose in every view,
calibrate the cameras, triangulate to 3D, correct and smooth the result, then
render a skeleton video. Outputs land in `recording/deeperfly_outputs/` (override
with `-o`): `poses.h5`, the rendered video, and a copy of the config used.

The config is optional — `deeperfly run recording/` runs with sensible defaults.
Generate one with `deeperfly init` when you need to point at your cameras or
tweak the pipeline.

### Resuming and partial runs

Each run reuses whatever is already cached in the output directory and computes
only what is missing, so finished work is never recomputed:

```bash
deeperfly run recording/ -o out/ --until detect   # 2D only
deeperfly run recording/ -o out/                  # resume: reuse cached 2D, finish 3D + video
deeperfly run recording/ -o out/ --overwrite      # recompute everything
```

Add `-v`/`-vv` for more logging, or `-q` to quiet it.

### The config file

A single `config.toml` (from `deeperfly init`) carries everything a run needs —
the camera rig, which file belongs to which camera, the detector, correction mode
and smoothing. The `[inputs]` section maps each camera to a filename prefix under
the recording (e.g. `rh = "camera_0"` matches `camera_0.mp4` or the image
sequence `camera_0_img_*.jpg`). The generated file is commented; edit it in place.

## Library usage

deeperfly is also a Python library — the cameras, bundle adjustment,
triangulation, the detector and video I/O are all importable:

```python
from deeperfly import CameraGroup, Skeleton, run_from_points2d

cameras = CameraGroup.from_config("examples/cameras.toml")
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf, smooth="one_euro")
result.save("fly.h5")
```

See [docs/library.md](docs/library.md) and the [`examples/`](examples) notebooks
for full walkthroughs.

## Documentation

- [docs/library.md](docs/library.md) — the Python API: bundle adjustment, the pipeline, video I/O.
- [docs/architecture.md](docs/architecture.md) — how the pipeline works: stages, 3D correction (reproject vs pictorial), detector backends.
- [docs/video.md](docs/video.md) — video decoding backends and on-GPU decode.
- [docs/comparison.md](docs/comparison.md) — what changed from DeepFly3D / DeepFly2D / PyBundleAdjustment.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development install, tests and benchmarks.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
