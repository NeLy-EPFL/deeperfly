# deeperfly → full Drosophila 3D pose pipeline

## Context

`deeperfly` is a clean-room JAX rewrite/improvement of three NeLy-EPFL projects:
**DeepFly2D** (PyTorch stacked-hourglass 2D pose), **DeepFly3D** (the orchestrating
3D pipeline + GUI), and **PyBundleAdjustment** (scipy-based BA). Today the repo only
covers the *geometry core* — it has no detector, no skeleton model, no pipeline, no
I/O or visualization. The goal is to grow it into an end-to-end pipeline matching the
originals' science: **images/video → 2D pose → camera calibration (bundle adjustment)
→ triangulation → error correction → visualization/export** — but modern, JAX-first,
and headless.

Already implemented and reused unchanged (well-tested, OpenCV-cross-checked):
- `geometry.py` — `project_full`/`project_full_one`, `rvec_to_rmat`/`rmat_to_rvec`
  (Rodrigues), distortion, `triangulate_dlt(pts2d, pmats)` (NaN-aware DLT, returns NaN
  for <2 views), `intr_to_kmat`. x64, JIT/grad-friendly.
- `cameras.py` — `Camera` (rvec, tvec, intr=`[fx,fy,cx,cy]`, dist) and `CameraGroup`
  (`.project()`, `.triangulate()`, `.from_config()` TOML, `.from_arrays()`).
- `bundle_adjustment/` — `bundle_adjust(cameras, pts2d, fixed=, shared=, pts3d=, **solver_kwargs)`
  (scipy `least_squares`, TRF+LSMR, analytic JAX Jacobian, sparse pattern); fixed/shared
  param grammar (`"*.intr"`, `"f.rvec"`, `[["lf.tvec[2]","rf.tvec[2]"]]`).

**Single contract with the geometry layer:** observations are `(V, N, 2)` with
**NaN = missing/invisible**. Both `triangulate_dlt` and the BA residual builder already
key off NaN, so visibility masking needs no new machinery.

### Decisions (from the user)
1. **2D pose:** reimplement the stacked-hourglass in **JAX (Equinox)** and **convert the
   original PyTorch weights**. Gated on a **GPU benchmark** — if JAX isn't faster than
   PyTorch, fall back to wrapping PyTorch behind the same interface.
2. **Headless only:** CLI + matplotlib (2D overlays, 3D skeleton) + MP4 export. No GUI.
3. **Error correction = pragmatic:** reprojection-error outlier flagging + Huber robust
   loss + bone-length priors + Procrustes alignment. No pictorial-structures/belief-prop.
4. **Modern formats only** (HDF5/npz). No legacy pickle support, no importer.

## Target package layout (new files under `src/deeperfly/`)

```
skeleton.py            # Drosophila skeleton dataclass (loaded from data/skeleton_fly.toml)
pose2d/                # JAX stacked-hourglass + inference
  model.py             #   Equinox: FrozenBatchNorm, Bottleneck, Hourglass, HourglassNet
  weights.py           #   PyTorch state_dict -> Equinox conversion; save/load checkpoint
  inference.py         #   preprocess (mirror-aware), predict_heatmaps, heatmap_to_points
  download.py          #   fetch + cache pretrained weights
triangulate.py         # apply_visibility, triangulate, reprojection_error (reuse geometry)
correction.py          # outlier flagging, Procrustes, OneEuroFilter/Gaussian smoothing
io.py                  # PoseResult HDF5 container (save/load)
viz.py                 # matplotlib 2D overlays + 3D skeleton plots
video.py               # read frames (video/dir), write MP4 (imageio-ffmpeg)
pipeline.py            # orchestrator: the df3d.Core analogue (pure functions)
cli.py                 # argparse subcommands; replaces placeholder main()
data/skeleton_fly.toml, data/template_{left,right}.npz   # packaged assets
```

The **only edit to existing code** is an additive, default-off extension to
`bundle_adjustment/core.py` (see Correction). Everything else is new modules reusing
`geometry`/`cameras`/`bundle_adjustment`.

## Component plans

### Skeleton model (`skeleton.py`)
Frozen `@dataclass Skeleton` loaded from packaged `data/skeleton_fly.toml`: 38 tracked
points (19 joint-types × 2 sides), `joint_names`, `left_idx`/`right_idx` (15 each),
`limbs` (10), `bones` (28 within-leg edges), `bones3d` (e.g. `[15,34]` antenna–antenna),
and per-camera `visibility` (right cameras see right legs, left/mirror cameras see left).
Methods: `Skeleton.fly()`, `from_config(path)`, `visibility_mask(n_cameras) -> (V,N) bool`,
`bone_index_pairs() -> (i, j)` for vectorized bone lengths. Threads into triangulation
(mask→NaN) and correction (bones, side indices).

### 2D pose in JAX (`pose2d/`)
- **Library = Equinox** — modules are plain frozen PyTrees, matching the repo's
  `@dataclass`/`NamedTuple` style; conversion is a direct leaf-set; clean `vmap` over
  the 7 cameras. Flax NNX is the fallback only if BN handling forces it.
- **Architecture** faithful to DeepFly2D: stem (7×7 s2 → BN → relu → maxpool → 2×residual),
  `num_stacks=2` hourglass modules (`depth=4`, Bottleneck blocks, `jax.image.resize` for
  upsampling), intermediate-supervision merge convs; input `3×256×512` normalized by −0.22,
  output heatmaps `(19, 64, 128)`, inference uses the last stack. **Use a custom
  `FrozenBatchNorm`** (`gamma*(x-mean)/sqrt(var+eps)+beta`) so the module is a pure PyTree
  (no stateful BN) — clean for inference and vmap.
- **Weight conversion (`weights.py`)** — Equinox `Conv2d` uses PyTorch OIHW layout, so
  conv kernels likely need **no transpose** (must be numerically verified, not assumed);
  bias maps directly; BN → FrozenBatchNorm fields; explicit ordered torch-key → Equinox-leaf
  map asserting every key consumed once. Save in Equinox native serialization so **runtime
  never needs torch**. Verify Bottleneck expansion factor against checkpoint channel dims.
- **Weights (`download.py`)** — fetch pinned URL + SHA256, cache under `platformdirs`.
  Two-stage: download original `.pth`, run `deeperfly convert-weights` (torch extra) once;
  later publish a pre-converted checkpoint so end users skip torch. *Risk: confirm
  redistribution rights before hosting.*
- **Inference (`inference.py`)** — `preprocess(image, camera_id, mirror_ids={4,5,6})`
  resizes/normalizes/CHW and **horizontally flips mirror-side cameras** (record flip to
  un-flip x after); `predict_heatmaps = vmap(model)`; `heatmap_to_points(heatmaps) ->
  (points: (V,J,2) px, conf: (V,J))` via argmax (+ optional soft-argmax). Maps the net's
  per-side 19 joints into the skeleton's `(V, 38, 2)` using visibility, **NaN where a joint
  isn't visible to that camera** — the contract handed to triangulation.

### Calibration + triangulation (`triangulate.py`, `pipeline.py`)
- `apply_visibility(pts2d, skel)` → NaN out invisible entries; `triangulate(cameras, pts2d)`
  wraps `CameraGroup.triangulate`; `reprojection_error(cameras, pts3d, pts2d)` via
  `cameras.project()`.
- **Calibration = fly-as-target BA** driving the existing `bundle_adjust` with: per-obs
  **confidence weighting** (heatmap), **bone-length prior**, and **Huber** robust loss.
  Reuse fixed/shared grammar (fix `"*.intr"`, anchor `"f.rvec"`/`"f.tvec"`, tie mirror
  pairs) via a `[bundle_adjustment]` section in the fly config.
- **Camera-order detection** opt-in (`--detect-order`): triangulate high-confidence frames
  under candidate permutations, pick min reprojection error; default to config order.

### Error correction (`correction.py` + one BA extension)
- **BA additive change (`bundle_adjustment/core.py`, default-off):** optional
  `weights: (V,N)` folded into residual/Jacobian as `*sqrt(w)`; optional bone-length term
  appending residuals `bone_weight*(||p_i−p_j|| − target)` (targets = per-bone median across
  frames) with a sparse analytic Jacobian. Strictly additive so all existing BA tests pass.
- **Huber** is pure config: `loss="huber"`, `f_scale≈reproj_threshold` flow through
  `bundle_adjust(**solver_kwargs)`.
- `flag_outliers(reproj_err, threshold≈40px)` → `drop_outliers` (NaN + re-triangulate) →
  optional re-BA. Replaces the original's pictorial-structures correction.
- `align_to_template(pts3d, skel, template_left, template_right)` — Umeyama scale+rigid
  Procrustes applied **separately to left/right** point sets (reuse `geometry.rmat_to_rvec`);
  templates in `data/template_{left,right}.npz`.
- Temporal smoothing: `OneEuroFilter` and `smooth_gaussian` on the `(T,N,3)` trajectory,
  NaN-aware.

### Data I/O (`io.py`)
`PoseResult` dataclass (cameras, skeleton, `pose2d (T,V,N,2)`, `conf (T,V,N)`,
`pose3d (T,N,3)`, diagnostics, meta) with `save(path)`/`load(path)` to **HDF5** (h5py).
`/meta` records version, skeleton name, camera names, image sizes, fps, source path,
git sha for provenance. Optional zarr backend later behind the same interface.

### Visualization / video / CLI
- `viz.py`: `plot_skeleton_2d`, `plot_skeleton_3d` (drosophila 3D), `overlay_grid`
  (7-cam montage); bones from `skel.bones`, left/right color-coded, confidence→alpha;
  `Agg` backend.
- `video.py`: `read_video`/`read_images`/`write_mp4` via `imageio[ffmpeg]` (keep cv2 test-only).
- `cli.py`: subcommands `pose2d`, `calibrate`, `pose3d`, `run`, `visualize`,
  `convert-weights` (torch extra), `download-weights`; each a thin wrapper over
  `pipeline.py`. Entry point `deeperfly = "deeperfly.cli:main"`.
- `pipeline.run(...)`: read → preprocess (mirror-aware) → predict → heatmap_to_points →
  apply_visibility → (detect_order) → calibrate → triangulate → correct (outliers →
  Procrustes → smooth) → `PoseResult` → save.

## Dependency additions (`pyproject.toml`)
Runtime: `equinox`, `h5py`, `imageio`, `imageio-ffmpeg`, `platformdirs`.
Optional groups: `viz = [matplotlib]` (promote from dev); `torch = [torch, torchvision]`
(weight conversion + PyTorch fallback only — never at normal runtime); `zarr` optional.
`test` adds torch/torchvision (equivalence tests, skipped if absent), keeps
opencv-python-headless + pytest, adds h5py/imageio. The JAX-vs-PyTorch gate decides whether
`equinox` or `torch` is the runtime detector dep; the interface is identical either way.

## Phased build order (lowest-risk first; each phase ships its own tests)

- **Phase 0 — Skeleton + I/O + triangulation (no ML).** `skeleton.py`, `io.py`,
  `triangulate.py`. Milestone: synthetic fly cloud → project (rig) → mask → triangulate →
  save/load round-trip.
- **Phase 1 — Correction + BA extension.** `correction.py`; additive `core.bundle_adjust`
  weights + bone term (finite-difference Jacobian test; all existing BA tests still pass).
  Milestone: calibrate+correct a noisy/outlier synthetic sequence to ground truth.
- **Phase 2 — Viz + video + CLI + e2e with stubbed detector.** `viz.py`, `video.py`,
  `cli.py`, `pipeline.run` with pluggable `pose2d_model`. Milestone: `deeperfly run` on
  synthetic data writes a `PoseResult` + overlay MP4 — whole pipeline minus the net validated.
- **Phase 3 — 2D net in JAX (highest risk, isolated).** `pose2d/*`; component + full-net
  equivalence tests vs PyTorch; mirror-flip round-trip.
- **Phase 3.5 — BENCHMARK GATE.** `dev/bench_pose2d.py` on target GPU. If JAX ≥ PyTorch:
  keep Equinox runtime, torch optional. Else: PyTorch-wrapper backend behind the same
  `predict_heatmaps`/`heatmap_to_points` interface — only `pose2d/__init__.py` wiring changes.
- **Phase 4 — Full integration + real data + docs.** Wire real net into `pipeline.run`;
  validate on a real 7-camera recording; tune `f_scale`/weights/smoothing; add
  `examples/pipeline.ipynb` and README pipeline section.

## Testing (matches existing OpenCV-cross-check + synthetic-rig style)
- `test_skeleton.py` — counts (38/10/28), visibility mask, left/right disjointness.
- `test_triangulate.py` — project known cloud through `rig`, mask, recover; NaN for <2 views.
- `test_correction.py` — Umeyama recovers known (s,R,t); outlier flagging; OneEuro reduces
  variance; bone residual zero at truth.
- `test_ba_weights_and_bones.py` — weights down-weight corrupted obs; bone Jacobian vs finite
  differences; **existing BA tests unchanged**.
- `test_io.py` — `PoseResult` round-trip preserves arrays (incl. NaN) + meta.
- `test_pose2d_*` (torch extra, skipped if absent) — FrozenBatchNorm/Bottleneck/Hourglass vs
  torch; full-net heatmap `allclose`; heatmap_to_points + mirror round-trip.
- `test_pipeline_e2e.py` — synthetic moving fly, **stubbed pose2d**, run `pipeline.run`,
  assert recovered 3D within tolerance + valid `PoseResult`.
- `test_viz.py`/`test_video.py` — render-without-error + MP4 write/read smoke tests.

## Verification (end-to-end)
1. `uv run pytest -q` — all phases' unit + e2e tests green; existing BA/geometry tests
   unaffected.
2. `dev/bench_pose2d.py` — record JAX-vs-PyTorch GPU throughput; document the gate outcome.
3. `python -c "from deeperfly.pose2d.weights import ...; assert allclose(jax, torch)"` —
   weight-conversion equivalence on a real input.
4. `deeperfly run --input <7-cam video/dir> --config cameras.toml --out result.h5` then
   `deeperfly visualize --in result.h5 --out overlay.mp4 --3d` — inspect overlays + 3D
   video; confirm reprojection error comparable to DeepFly3D.

## Key risks / decision points
- **JAX-vs-PyTorch GPU gate (3.5)** — isolated to `pose2d/` by the backend-agnostic interface.
- **Weight-conversion fidelity** — verify conv layout numerically; FrozenBatchNorm removes BN
  ambiguity; verify Bottleneck expansion; component + full-net equivalence tests gate it.
- **Weight licensing** — confirm redistribution before hosting a pre-converted checkpoint.
- **Camera-order detection** — opt-in; config order authoritative by default.
- **BA extension scope** — keep weights/bone term strictly additive and default-off.
