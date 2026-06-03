# Architecture

How the pipeline works internally. For a comparison with the upstream projects,
see [comparison.md](comparison.md).

`deeperfly run` is the pipeline as one linear sequence of stages — `pose2d`
(2D) → `bundle_adjustment` → `pictorial_structures` → `triangulation` →
`smoothing` → `visualization` — each toggled by its own `do_<stage>` boolean in the
config's `[pipeline]` table (with its own `[pipeline.<stage>]` parameter
sub-table). An enabled stage reuses its cached result when it is already in the
output directory and only recomputes when that result is missing or `--overwrite`
selects it (recomputing a stage also refreshes the stages downstream of it), so
re-running a finished recording is a no-op by default. A disabled stage is dropped
from the pipeline: its cached `poses.h5` output is read back and fed to the stages
still on, so disabling the finished stages also resumes a partial run.

## Pipeline stages

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` (`backends/{jax,torch}/`) | Stacked hourglass in two backends behind one interface; JAX/Equinox by default, PyTorch runs the original weights directly. |
| Calibration | `pipeline.calibrate` → `bundle_adjustment/` | Fly-as-target BA: confidence weights, Huber loss, bone-length prior. |
| Triangulation | `triangulate.py` / `pipeline.reconstruct{,_ransac}` | NaN-aware DLT: RANSAC consensus (default), greedy reprojection-outlier rejection, or plain DLT. |
| 3D correction | `correction.py` / `pictorial.py` | Triangulation (ransac/greedy/dlt), optionally after pictorial-structures peak recovery; temporal smoothing. |
| Visualization | `viz.py`, `video/` | matplotlib 2D overlays, 3D skeleton, MP4 export. |
| Result I/O | `io.py` | Self-contained HDF5 `PoseResult`. |
| Skeleton | `skeleton.py` + `data/skeleton_fly.toml` | 38 points, 10 limbs, 28 bones, per-camera visibility. |

## 3D correction: triangulation (± pictorial)

Each view is detected independently; the views only meet *geometrically*. The
reconstruction is two orthogonal choices — `run_from_points2d(...,
triangulation=..., do_pictorial=...)` for the library, or
`[pipeline.triangulation].method` + `[pipeline].do_pictorial_structures` for the
CLI:

**`triangulation`** — how the per-view 2D points become one 3D point:

- **`ransac`** (default) — triangulate each point from its largest set of
  mutually consistent views, *vetoing* a bad detection. Because the rig has only a
  handful of cameras it **exhaustively enumerates all `C(V,2)` two-view
  hypotheses** (the deterministic limit of RANSAC), counts inliers within
  `ransac_threshold` px, breaks ties toward the lower total reprojection error,
  and refits from the inliers. A gross outlier never enters the fit. NaN
  (unobserved) views never count as inliers.
- **`greedy`** — triangulate the arg-max detections by DLT and iteratively drop
  the single worst-reprojecting view of each offending point, re-triangulating
  from the survivors (`reproj_threshold` / `max_drops`). Cheaper, but refines an
  already-contaminated least-squares fit. (`reproject` is a legacy alias.)
- **`dlt`** — plain least-squares triangulation over all available views, no
  outlier handling. The bare baseline. (`none` is an alias.)

**`do_pictorial_structures`** (bool, default off; `do_pictorial=` in the library
call) — when on, first run DeepFly3D-style pictorial structures over the
detector's top-K candidate peaks (`pictorial.py`): build
multi-view-consistent 3D hypotheses per joint, then pick one per joint by exact
dynamic programming along each limb under bone-length priors (plus an optional
temporal term). It can *recover* a joint when the arg-max landed on the wrong
heatmap peak (occlusion, crossing legs, L/R confusion) — something the
triangulators can only *veto*. It needs the full-heatmap detect path (slower); its
committed per-view 2D then feeds the chosen `triangulation` (a plain `dlt` pass
keeps the PS estimate as-is). On clean recordings it is a no-op.

## 2D detector backends

The detector ships two interchangeable backends under
`pose2d/backends/{jax,torch}/`, each exposing the same `HourglassNet` /
`load_model` / `predict_heatmaps`, both installed by default and selectable with
`[pipeline.pose2d].backend`. The PyTorch backend runs the published `sh8` weights
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
