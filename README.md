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

## Checking your install

`deeperfly doctor` reports what this machine can actually run — inference backends,
video read/write backends (in `backend="auto"` preference order), detector weights,
and the default config path. Run it after installing to check that everything is set up correctly.
```bash
deeperfly doctor
```

```
deeperfly
  version           0.1.0
  location          /home/you/.venv/lib/python3.13/site-packages/deeperfly

system
  python            3.13.1 (CPython)
  platform          Linux-7.0.0-15-generic-x86_64-with-glibc2.39

inference
  torch             2.6.0+cu124  (CUDA: NVIDIA GeForce RTX 4090)
  jax               0.4.38  (backend: gpu; devices: cuda:0)
  GPU inference     available (24.0 GiB memory)
  detectors         jax (default), torch

video backends
  read              pyav, opencv, torchcodec, imageio
  GPU decoders      torchcodec
  write             pyav, imageio, opencv
  not installed     dali, decord, video_reader_rs

weights
  cache dir         /home/you/.cache/deeperfly/weights
  PyTorch           downloaded (96.2 MiB) -- sh8_deepfly.pth
  JAX               downloaded (24.1 MiB) -- sh8_deepfly.eqx

config
  default config    /home/you/.venv/.../site-packages/deeperfly/data/default_config.toml
```

On a CPU-only box `GPU inference` reads `not available -- CPU only`, `GPU decoders`
shows `none (CPU decode only)`, and the weights show `not downloaded` until the
first `deeperfly run` fetches them.

## Quickstart

```bash
deeperfly init config.toml                                # write a config you can edit
deeperfly run recording/ -c config.toml                  # run the full pipeline
deeperfly inspect recording/deeperfly_outputs/poses.h5   # inspect the result
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

Pass `--log-level debug` for more detail, or `--log-level warning` to quiet the
per-stage logs and progress bar.

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
- [docs/architecture.md](docs/architecture.md) — how the pipeline works: stages, 3D correction (triangulation ransac/greedy/dlt ± pictorial), detector backends.
- [docs/video.md](docs/video.md) — video decoding backends and on-GPU decode.
- [docs/comparison.md](docs/comparison.md) — what changed from DeepFly3D / DeepFly2D / PyBundleAdjustment.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development install, tests and benchmarks.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
