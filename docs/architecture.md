# Architecture

How the pipeline works internally. For a comparison with the upstream projects,
see [comparison.md](comparison.md).

`deeperfly run` is one linear sequence of stages — `pose2d` (2D) →
`bundle_adjustment` → `pictorial_structures` → `triangulation` → `smoothing` →
`visualization` — each toggled by a `do_<stage>` boolean in `[pipeline]`, with its
own `[pipeline.<stage>]` parameter sub-table. An enabled stage reuses its cached
result from the output directory and recomputes only when it's missing or
`--overwrite` selects it (which also refreshes the stages downstream). A disabled
stage is dropped: its cached `poses.h5` output is read back and fed to the stages
still on — so disabling the finished stages resumes a partial run.

## Pipeline stages

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` (`backends/torch/`) | Stacked hourglass (PyTorch) running the original DeepFly2D weights directly; CUDA / Metal automatically. |
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
  mutually consistent views, *vetoing* a bad detection. The rig has only a handful
  of cameras, so it exhaustively enumerates all `C(V,2)` two-view hypotheses (the
  deterministic limit of RANSAC), counts inliers within `ransac_threshold` px,
  breaks ties toward lower total reprojection error, and refits from the inliers.
  A gross outlier never enters the fit; NaN views never count as inliers.
- **`greedy`** — triangulate the arg-max detections by DLT and iteratively drop
  the single worst-reprojecting view of each offending point, re-triangulating
  from the survivors (`reproj_threshold` / `max_drops`). Cheaper, but refines an
  already-contaminated fit. (`reproject` is a legacy alias.)
- **`dlt`** — plain least-squares triangulation, no outlier handling. (`none` is
  an alias.)

**`do_pictorial_structures`** (default off; `do_pictorial=` in the library call) —
when on, first run DeepFly3D-style pictorial structures over the detector's top-K
candidate peaks (`pictorial.py`): build multi-view-consistent 3D hypotheses per
joint, then pick one per joint by exact dynamic programming along each limb under
bone-length priors (plus an optional temporal term). It can *recover* a joint when
the arg-max landed on the wrong heatmap peak (occlusion, crossing legs, L/R
confusion) — something the triangulators can only *veto*. It needs the
full-heatmap detect path (slower); its committed per-view 2D then feeds the chosen
`triangulation` (a plain `dlt` pass keeps the PS estimate). On clean recordings
it is a no-op.

## 2D detector

The detector is a faithful PyTorch copy of the original DeepFly2D stacked
hourglass, under `pose2d/backends/torch/` (exposed through `pose2d/backends/` as
`HourglassNet` / `load_model` / `predict_heatmaps`). It loads the published `sh8`
weights directly, with no conversion; `deeperfly run` downloads them on first use.
`pose2d/inference.py` preprocesses frames in torch, so a GPU-decoded frame is
normalized, resized and forwarded without leaving the GPU.

The detector uses CUDA automatically on NVIDIA and Metal (MPS) on Apple Silicon,
with no setup. For large CUDA batches the forward is wrapped with `torch.compile`
(see `pose2d/backends/torch/model.py`). Geometry and bundle adjustment are the
only JAX in deeperfly and run in float64 on the CPU.
