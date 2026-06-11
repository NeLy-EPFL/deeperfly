# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig.
It estimates camera parameters and 2D/3D keypoint locations from behavioral
recordings: 2D pose → camera calibration → triangulation → visualization.

It is a modern rewrite of
[DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment). See [docs/comparison.md](docs/comparison.md) for how it differs
from the originals.

## Installation

deeperfly is both a command-line tool and a Python library. Install the CLI with
[uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

## Checking your install

`deeperfly doctor` reports what this machine can run — inference, frame I/O
backends, detector weights, and the default config path. Run it after installing:

```bash
deeperfly doctor
```

The `GPU inference` line under `inference` tells you whether the detector will
run on the GPU:

```
  GPU inference     available (24.0 GiB memory)
```

On a CPU-only box it reads `not available -- CPU only`.

## Quickstart

```bash
deeperfly init config.toml                               # write a config you can edit
deeperfly run recording/ -c config.toml                  # run the full pipeline
deeperfly inspect recording/deeperfly_outputs/poses.h5   # inspect the result
```

`deeperfly run` does everything in one command: detect 2D pose in every view,
calibrate the cameras, triangulate to 3D, then render a skeleton video. Outputs
land in `recording/deeperfly_outputs/` (override with
`-o`): `poses.h5`, the rendered video, and a copy of the config used.

The config is optional — `deeperfly run recording/` uses sensible defaults.
Generate one with `deeperfly init` to point at your cameras or tweak the
pipeline; the generated file is commented, so edit it in place. See
[docs/configuration.md](docs/configuration.md) for what each section does — from
the `[inputs]` file-to-camera mapping you'll almost always set, through resuming
and partial runs, to the rig and skeleton you can usually leave at their
defaults.

Pass `--log-level debug` for more detail, or `--log-level warning` to quiet the
per-stage logs and progress bar.

## Library usage

deeperfly is also a Python library. Add it to your project — prefix with
`UV_TORCH_BACKEND=auto` so uv picks the right PyTorch wheel (`uv add` has no
`--torch-backend` flag):

```bash
UV_TORCH_BACKEND=auto uv add git+https://github.com/NeLy-EPFL/deeperfly
```

```python
from deeperfly import CameraGroup, Config, Skeleton, run_from_points2d

cameras = CameraGroup.from_config(Config.from_toml("config.toml"))
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf)
result.save("fly.h5")
```

See [docs/library.md](docs/library.md) and the [`examples/`](examples) notebooks
for full walkthroughs.

## Documentation

- [docs/configuration.md](docs/configuration.md) — the `config.toml`, section by section, ordered by how often you'll edit it.
- [docs/library.md](docs/library.md) — the Python API: bundle adjustment, the pipeline, video I/O.
- [docs/architecture.md](docs/architecture.md) — how the pipeline works: stages, 3D correction (triangulation ransac/greedy/dlt ± pictorial), the detector.
- [docs/video.md](docs/video.md) — frame read/write: PyAV video, OpenCV image sequences (CPU decode).
- [docs/comparison.md](docs/comparison.md) — what changed from DeepFly3D / DeepFly2D / PyBundleAdjustment.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development install, tests, linting.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
