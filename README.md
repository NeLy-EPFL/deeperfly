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
uv tool install deeperfly --python 3.14 --torch-backend=auto
```

`--torch-backend=auto` lets uv pick the PyTorch wheel for this machine — the CUDA
build on an NVIDIA GPU, the lean CPU-only wheel otherwise. The detector then uses
the GPU automatically on NVIDIA (CUDA) and Apple Silicon (Metal/MPS). deeperfly
needs Python ≥ 3.11; `--python 3.14` just pins the latest CPython for the tool.

Video decoding runs on the CPU via PyAV/OpenCV; optional extras add alternative
decoders (see [docs/video.md](docs/video.md)).

To add deeperfly as a library, set the backend for the environment (`uv add` has
no `--torch-backend` flag):

```bash
UV_TORCH_BACKEND=auto uv add deeperfly
```

## Checking your install

`deeperfly doctor` reports what this machine can run — inference backends, frame
I/O backends (in `"auto"` preference order), detector weights, and the default
config path. Run it after installing.
```bash
deeperfly doctor
```

```
deeperfly
  version           0.1.0
  location          /home/you/.venv/lib/python3.14/site-packages/deeperfly

system
  python            3.14.0 (CPython)
  platform          Linux-7.0.0-15-generic-x86_64-with-glibc2.39

inference
  torch             2.6.0+cu124  (CUDA: NVIDIA GeForce RTX 4090)
  jax               0.4.38  (backend: cpu; devices: cpu:0)
  GPU inference     available (24.0 GiB memory)
  detector          torch

frame I/O backends
  video read        pyav, opencv, torchcodec
  video write       pyav, opencv
  image read        opencv
  not installed     imageio, video_reader_rs

weights
  cache dir         /home/you/.cache/deeperfly/weights
  detector          downloaded (96.2 MiB) -- sh8_deepfly.pth

config
  default config    /home/you/.venv/.../site-packages/deeperfly/data/default_config.toml
```

On a CPU-only box `GPU inference` reads `not available -- CPU only`, and the
weights show `not downloaded` until the first `deeperfly run` fetches them.

## Quickstart

```bash
deeperfly init config.toml                                # write a config you can edit
deeperfly run recording/ -c config.toml                  # run the full pipeline
deeperfly inspect recording/deeperfly_outputs/poses.h5   # inspect the result
```

`deeperfly run` does everything in one command: detect 2D pose in every view,
calibrate the cameras, triangulate to 3D, correct and smooth, then render a
skeleton video. Outputs land in `recording/deeperfly_outputs/` (override with
`-o`): `poses.h5`, the rendered video, and a copy of the config used.

The config is optional — `deeperfly run recording/` uses sensible defaults.
Generate one with `deeperfly init` to point at your cameras or tweak the pipeline.

### Resuming and partial runs

The pipeline is a sequence of stages — `pose2d` (2D) → `bundle_adjustment` →
`pictorial_structures` → `triangulation` → `smoothing` → `visualization` — each
toggled by a `do_<stage>` boolean in `[pipeline]`, with its own `[pipeline.<stage>]`
parameter sub-table:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery (opt-in)
do_triangulation        = true   # triangulate 2D -> 3D
do_smoothing            = false  # temporal smoothing (opt-in)
do_visualization        = true   # render the videos
```

An *enabled* stage reuses its result when it is already in the output directory,
recomputing only when it's missing — so re-running a finished recording is a cheap
no-op. Force a recompute with `--overwrite`: bare redoes every stage, or name
stages to redo only those (plus the stages after them):

```bash
deeperfly run recording/ --overwrite                       # recompute everything
deeperfly run recording/ --overwrite pose2d visualization  # just these (+ what follows)
```

A *disabled* stage (`do_<stage> = false`) is dropped from the pipeline; its cached
result is read from `poses.h5` and fed to the stages still on — so `do_pose2d =
false` reconstructs 3D from a cached 2D pose without re-running detection. An
enabled stage whose input is unavailable is skipped, with the reason logged.

A run reuses the `config.toml` saved in the output directory (it owns the cached
results), so `-o out/` alone resumes; pass `-c` only for a fresh output directory,
and edit `out/config.toml` to change a run in place.

Pass `--log-level debug` for more detail, or `--log-level warning` to quiet the
per-stage logs and progress bar.

### The config file

A single `config.toml` (from `deeperfly init`) carries everything a run needs —
the camera rig, which file belongs to which camera, the detector, correction mode
and smoothing. The `[inputs]` section maps each camera to a filename prefix (e.g.
`rh = "camera_0"` matches `camera_0.mp4` or the sequence `camera_0_img_*.jpg`).
The generated file is commented; edit it in place.

## Library usage

deeperfly is also a Python library — cameras, bundle adjustment, triangulation,
the detector and video I/O are all importable:

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
- [docs/architecture.md](docs/architecture.md) — how the pipeline works: stages, 3D correction (triangulation ransac/greedy/dlt ± pictorial), the detector.
- [docs/video.md](docs/video.md) — video read/write backends (CPU decode).
- [docs/comparison.md](docs/comparison.md) — what changed from DeepFly3D / DeepFly2D / PyBundleAdjustment.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development install, tests and benchmarks.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
