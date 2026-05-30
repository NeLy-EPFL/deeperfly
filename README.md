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
friendly); the 2D detector is a JAX (Equinox) port of DeepFly2D's stacked
hourglass, with the original PyTorch network kept alongside as an alternate
backend for benchmarking.

## Pipeline

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` | Equinox stacked hourglass (+ PyTorch backend); the original weights convert directly. |
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
deeperfly download-weights          # fetch original PyTorch weights
deeperfly convert-weights           # -> native JAX checkpoint (needs the 'torch' extra)
deeperfly run --in recording/ --config cameras.toml --out fly.h5 [--backend jax|torch]
deeperfly visualize --in fly.h5 --out fly_3d.mp4 --mode 3d
```

See [`examples/bundle_adjustment.ipynb`](examples/bundle_adjustment.ipynb) for the
BA walkthrough and [`examples/pipeline_demo.py`](examples/pipeline_demo.py) for a
synthetic end-to-end run (no weights required).

## 2D detector backends

The JAX hourglass is validated to match the original PyTorch network to within
`1e-4` (see `tests/test_pose2d_torch.py`). Both backends share the same
interface, selectable at inference; benchmark them on your GPU and pick the
faster one:

```bash
uv run --extra torch python dev/bench_pose2d.py --batch 7 --frames 8
```

## Development

```bash
uv sync --group test                 # install with test dependencies
uv run --group test pytest           # run the suite
uv run --group test --extra torch pytest   # also run the PyTorch-equivalence tests
```

Optional extras: `viz` (matplotlib + imageio for plotting/video), `torch`
(weight conversion + PyTorch detector backend).
