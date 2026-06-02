# Architecture

How the pipeline works internally. For a comparison with the upstream projects,
see [comparison.md](comparison.md).

`deeperfly run` is the whole pipeline as one linear sequence of stages — `detect`
(2D) → `pose3d` (calibrate + triangulate + correct + smooth) → `visualize` — with
per-stage caching so re-runs only compute what is missing.

## Pipeline stages

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
there are two ways to do that (`[pipeline].correct`, or
`run_from_points2d(..., correct=...)`):

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

The detector ships two interchangeable backends under
`pose2d/backends/{jax,torch}/`, each exposing the same `HourglassNet` /
`load_model` / `predict_heatmaps`, both installed by default and selectable with
`[detector].backend`. The PyTorch backend runs the published `sh8` weights
directly; the JAX backend (default) runs the same weights from a native checkpoint
that `deeperfly run` downloads and converts on first use, validated to match the
PyTorch reference numerically (`tests/test_pose2d_torch.py`). JAX is faster on GPU
— benchmark on your hardware:

```bash
uv run python dev/bench_pose2d.py --batch 7 --frames 8
```

On NVIDIA GPUs both backends use CUDA automatically (JAX via the `cuda` extra). On
Apple Silicon the PyTorch backend auto-uses Metal (MPS) with no setup; to
accelerate the *JAX* backend on macOS instead, install `deeperfly[mps]` (the
experimental [`jax-mps`](https://github.com/tillahoffmann/jax-mps) plugin) — the
float32 detector then runs on Metal while geometry and bundle adjustment stay in
float64 on the CPU.
