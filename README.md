# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig,
in JAX: **2D pose → camera calibration (bundle adjustment) → triangulation →
3D correction → visualization.**

It is a JAX rewrite of [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment) — same
science, modern and headless. See [docs/comparison.md](docs/comparison.md) for a
detailed comparison with the originals.

The geometry follows OpenCV's conventions (Rodrigues rotations, `projectPoints`
distortion, DLT triangulation) and is cross-checked against OpenCV in the test
suite. Everything geometric is JAX (float64, JIT- and autodiff-friendly); the 2D
detector ships two interchangeable backends behind one interface — a JAX
(Equinox) port of DeepFly2D's stacked hourglass (the default, faster on GPU) and
the original PyTorch network — selected with `[detector].backend = "jax" | "torch"`.

## Installation

deeperfly is both a command-line tool and a library. Install the CLI with uv:

```bash
uv tool install deeperfly             # the `deeperfly` command, in an isolated env
uv tool install "deeperfly[cuda]"     # + NVIDIA CUDA (JAX on the GPU, NVDEC decode)
uv tool install "deeperfly[all]"      # + every cross-platform video backend
```

On Apple Silicon, `deeperfly[mps]` runs the JAX detector on Metal (the PyTorch
backend already uses Metal automatically). To use deeperfly as a library:

```bash
uv add deeperfly                      # or: pip install deeperfly
```

The base install is **batteries-included** — detector (JAX + PyTorch), geometry,
bundle adjustment, plotting and video I/O all work out of the box on CPU, and
detector weights are downloaded and converted on first use. GPU acceleration and
faster video decoders are opt-in extras (see [below](#video-io)).

> GPU/Metal acceleration can't be selected automatically at install time —
> Python packaging has no way to detect a CUDA or Metal runtime — so request it
> explicitly with the `cuda` / `mps` extra.

## Quickstart

```bash
deeperfly init config.toml                       # write a config to edit (cameras, inputs, pipeline, skeleton)
deeperfly run recording/ -c config.toml          # 2D → 3D → 3D-skeleton video, into recording/deeperfly_outputs/
deeperfly info --in recording/deeperfly_outputs/poses.h5
```

`deeperfly run` is the whole pipeline as one linear sequence of stages — `detect`
(2D) → `pose3d` (calibrate + triangulate + correct + smooth) → `visualize`. The
recording is the positional argument; `-c`/`--config` is the merged config TOML
(**defaults to the packaged default config** when omitted, so `deeperfly run
recording/` works out of the box); `-o` is an output **directory** (default
`<input>/deeperfly_outputs/`) holding `poses.h5`, the rendered video and a copy
of the config. Each run **reuses whatever is already cached** there and computes
only what is missing, so prior work is never recomputed:

```bash
deeperfly run recording/ -c config.toml -o out/ --until detect   # 2D only, cached in out/
deeperfly run recording/ -c config.toml -o out/                  # resume: reuse cached 2D, finish 3D + video
deeperfly run recording/ -c config.toml -o out/ --overwrite      # recompute everything
```

A single `config.toml` (from `deeperfly init`) carries everything a run needs.
Its `[inputs]` section maps each camera to a filename prefix under the recording (e.g.
`rh = "camera_0"` finds `camera_0.mp4` or the image sequence `camera_0_img_*.jpg`);
the detector backend, correction mode, bundle-adjustment keypoints and smoothing
all live there too. Sections are independently usable: `CameraGroup.from_config`
reads only the cameras, `Skeleton.from_config` only `[skeleton]`. Add `-v`/`-vv`
for progress logging or `-q` to quiet it.

## Library usage

Geometry / bundle adjustment only:

```python
from deeperfly import CameraGroup, bundle_adjust

group = CameraGroup.from_config("examples/cameras.toml")
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

The full 2D→3D pipeline from an existing 2D detection array:

```python
from deeperfly import CameraGroup, Skeleton, run_from_points2d

cameras = CameraGroup.from_config("examples/cameras.toml")
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf, smooth="one_euro")
result.save("fly.h5")
```

See [`examples/bundle_adjustment.ipynb`](examples/bundle_adjustment.ipynb) for the
BA walkthrough, [`examples/pipeline_walkthrough.ipynb`](examples/pipeline_walkthrough.ipynb)
for the pipeline one stage at a time, and
[`examples/pipeline_demo.py`](examples/pipeline_demo.py) for a synthetic
end-to-end run (no weights required).

## Pipeline

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` (`backends/{jax,torch}/`) | Stacked hourglass in two backends behind one interface; JAX/Equinox by default, PyTorch runs the original weights directly. |
| Calibration | `pipeline.calibrate` → `bundle_adjustment/` | Fly-as-target BA: confidence weights, Huber loss, bone-length prior. |
| Triangulation | `triangulate.py` / `pipeline.reconstruct` | NaN-aware DLT + greedy reprojection-outlier rejection. |
| 3D correction | `correction.py` / `pictorial.py` | Reprojection outlier rejection (default) or pictorial structures; Procrustes alignment + smoothing. |
| Visualization | `viz.py`, `video/` | matplotlib 2D overlays, 3D skeleton, MP4 export. |
| Result I/O | `io.py` | Self-contained HDF5 `PoseResult`. |
| Skeleton | `skeleton.py` + `data/skeleton_fly.toml` | 38 points, 10 limbs, 28 bones, per-camera visibility. |

## 3D correction: reproject vs pictorial

Each view is detected independently; the views only meet *geometrically*, and
there are two ways to do that (`[pipeline].correct`, or `run_from_points2d(..., correct=...)`):

- **`reproject`** (default) — triangulate the arg-max detections and greedily drop
  the worst-reprojecting view of each offending point. Fast; *vetoes* a bad
  per-view detection.
- **`pictorial`** — DeepFly3D-style pictorial structures over the detector's top-K
  candidate peaks (`pictorial.py`): build multi-view-consistent 3D hypotheses per
  joint, then pick one per joint by exact dynamic programming along each limb under
  bone-length priors (plus an optional temporal term). It can *recover* a joint when
  the arg-max landed on the wrong heatmap peak (occlusion, crossing legs, L/R
  confusion). It needs the full-heatmap detect path (slower) and is opt-in; on clean
  recordings it matches `reproject`.

## 2D detector backends

The detector ships two interchangeable backends under `pose2d/backends/{jax,torch}/`,
each exposing the same `HourglassNet` / `load_model` / `predict_heatmaps`, both
installed by default and selectable with `[detector].backend`. The PyTorch backend
runs the published `sh8` weights directly; the JAX backend (default) runs the same
weights from a native checkpoint that `deeperfly run` downloads and converts on
first use, validated to match the PyTorch reference numerically
(`tests/test_pose2d_torch.py`). JAX is faster on GPU — benchmark on your hardware:

```bash
uv run python dev/bench_pose2d.py --batch 7 --frames 8
```

On NVIDIA GPUs both backends use CUDA automatically (JAX via the `cuda` extra). On
Apple Silicon the PyTorch backend auto-uses Metal (MPS) with no setup; to accelerate
the *JAX* backend on macOS instead, install `deeperfly[mps]` (the experimental
[`jax-mps`](https://github.com/tillahoffmann/jax-mps) plugin) — the float32 detector
then runs on Metal while geometry and bundle adjustment stay in float64 on the CPU.

<a name="video-io"></a>
## Video I/O

`deeperfly.video` reads and writes frames through a pluggable backend registry.
The base install reads/writes via `imageio`; install an extra for a faster decoder:

| Backend | Read | Write | Frames | Install |
| --- | :-: | :-: | --- | --- |
| `imageio` | ✓ | ✓ | NumPy (CPU) | core |
| `opencv` | ✓ | ✓ | NumPy (CPU) | `opencv` |
| `pyav` | ✓ | ✓ | NumPy (CPU) | `pyav` |
| `decord` | ✓ | – | NumPy / `torch` (CPU/**CUDA**) | `decord` |
| `video_reader_rs` | ✓ | – | NumPy (CPU) | `video-reader-rs` |
| `torchcodec` | ✓ | – | `torch.Tensor` (CPU/**CUDA**) | `torchcodec` / `cuda` |
| `dali` | ✓ | – | `torch.Tensor` / NumPy (**CUDA**) | `dali` |

```python
from deeperfly import video

frames = video.read_frames(path)                        # video file or image dir; auto NumPy (host)
frames = video.read_video("clip.mp4", indices=[0, 50])  # random access
frames = video.read_video("clip.mp4", device="cuda")    # on-GPU tensor (NVDEC), zero-copy to JAX via to_jax
video.write_mp4(frames, "out.mp4", fps=30)
```

`backend="auto"` and `device="auto"` (the defaults) pick the fastest working path.
`deeperfly run` decodes on the **CPU by default** and uploads each window to the GPU
in one shot — within a few percent of GPU/NVDEC end to end, since the 2D detector
(not decode) is the bottleneck. Opt into on-device NVDEC decode with `[detector]
decode_device = "cuda"` (it falls back to CPU if no GPU decoder is available). See
the config comments and `deeperfly.video` docstrings for the full decoder details.

## Development

```bash
uv sync --group test                 # install with test dependencies
uv run --group test pytest           # run the suite (incl. PyTorch-equivalence tests)
```

Optional extras layer on GPU acceleration (`cuda`, `mps`, `dali`) and faster video
backends (`opencv`, `pyav`, `decord`, `video-reader-rs`, `torchcodec`); PyTorch is a
core dependency, so no extra is needed for it.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
