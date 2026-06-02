# deeperfly vs. DeepFly3D / DeepFly2D / PyBundleAdjustment

`deeperfly` is a clean-room, JAX-first re-implementation of the markerless
*Drosophila* 3D-pose pipeline from three NeLy-EPFL projects:

| Upstream project | Role | `deeperfly` counterpart |
| --- | --- | --- |
| [DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) | 2D pose: a PyTorch stacked-hourglass detector | [`pose2d/`](../src/deeperfly/pose2d) — the same network in two backends (JAX/Equinox + PyTorch) |
| [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) | The orchestrating 2D→3D pipeline + GUI ([Günel et al., *eLife* 2019](https://doi.org/10.7554/eLife.48571)) | [`pipeline.py`](../src/deeperfly/pipeline.py), [`triangulate.py`](../src/deeperfly/triangulate.py), [`pictorial.py`](../src/deeperfly/pictorial.py), [`correction.py`](../src/deeperfly/correction.py) |
| [PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment) | scipy-based bundle adjustment for calibration | [`bundle_adjustment/`](../src/deeperfly/bundle_adjustment) |

The science is faithful to the originals — same camera rig, same detector
weights, same fly-as-calibration-target bundle adjustment, same pictorial-
structures idea. What changed is the implementation: everything geometric is
JAX (JIT- and autodiff-friendly), the whole thing is headless and scriptable,
and the I/O is modern.

## At a glance

| | DeepFly3D / DeepFly2D / PyBundleAdjustment | deeperfly |
| --- | --- | --- |
| Numerical core | NumPy + SciPy + PyTorch | **JAX** (float64 geometry/BA), PyTorch for the detector |
| 2D detector | Stacked hourglass (PyTorch) | Same network + weights, default **JAX/Equinox** port; PyTorch kept as a second backend |
| Bundle-adjustment Jacobian | SciPy `least_squares`, sparse | SciPy `least_squares` with an **analytic JAX Jacobian** + sparsity pattern |
| 3D correction | Pictorial structures (belief propagation) | **Triangulation** — RANSAC consensus (default), greedy reprojection-outlier rejection, or plain DLT — optionally after a re-implemented pictorial-structures corrector (exact DP) |
| Interface | PyQt **GUI** | **Headless CLI + library** (one merged `config.toml`) |
| Result I/O | Pickle / custom | Self-contained **HDF5** (`PoseResult`) |
| Acceleration | CUDA (PyTorch) | CUDA (JAX + NVDEC video) and Apple **Metal/MPS** |
| Scope | Training + inference + GUI correction | **Inference only** (uses the published weights), headless |

## Component-by-component

### 2D detector

Both use the same stacked-hourglass architecture and the same published `sh8`
weights. deeperfly ports the network to **JAX/Equinox** as the default backend
(a pure PyTree, so the forward pass is `jit`/`vmap`-friendly and runs the seven
views in one batched call) and keeps a faithful **PyTorch** backend behind the
same interface. The JAX port is validated numerically against the PyTorch
reference (`tests/test_pose2d_torch.py`); the original PyTorch weights are
downloaded and converted to a native checkpoint automatically on first use.

deeperfly does **not** include training code — it consumes the released weights.
Train or fine-tune with the upstream DeepFly2D repository.

### Calibration / bundle adjustment

Like the originals, deeperfly calibrates with **no external target**: the fly's
own detected joints are the calibration points, refined by bundle adjustment
(`pipeline.calibrate`). The solver is still SciPy's `least_squares` (TRF + LSMR),
but the per-observation residual and its **Jacobian are computed analytically in
JAX** (`jax.vmap` + `jax.jacfwd` over the projection model) and assembled into a
sparse matrix from a precomputed sparsity pattern.

Beyond PyBundleAdjustment, the `bundle_adjustment/` module adds:

- a declarative **fixed/shared parameter grammar** (e.g. `"*.intr"`,
  `"f.rvec"`, tying `[["lf.tvec[2]", "rf.tvec[2]"]]`) to anchor the gauge;
- per-observation **confidence weighting** from detector heatmaps;
- a robust **Huber** loss and an optional **bone-length prior**.

### Triangulation

Both triangulate by DLT. deeperfly's is **NaN-aware** (a `NaN` observation means
"this view can't see this point", so visibility needs no separate mask array)
and returns `NaN` for points seen by fewer than two views. On top, the
`triangulation` choice offers a **RANSAC consensus** triangulator (default —
exhaustive `C(V,2)` two-view hypotheses, inlier counting with error tie-breaking),
a greedy **reprojection-outlier rejection** pass, or **plain DLT**.

### 3D correction

DeepFly3D corrects erroneous 2D detections with **pictorial structures** —
belief propagation over candidate joint locations under learned bone-length
priors and multi-view geometry. deeperfly splits this into two orthogonal knobs:
a `triangulation` strategy that *vetoes* bad views, and an optional `pictorial`
stage that *recovers* the right peak first.

- **`triangulation`** (`[pipeline].triangulation`):
  - **`ransac`** (default) — triangulate each point from its largest multi-view
    consensus set; a gross outlier never enters the fit.
  - **`greedy`** — greedily drop the worst-reprojecting view of each offending
    point. Cheaper; refines a (possibly contaminated) least-squares fit.
    (`reproject` is a legacy alias.)
  - **`dlt`** — plain least-squares triangulation, no outlier handling. (`none`
    is an alias.)
- **`do_pictorial`** (`[pipeline].do_pictorial`) — a re-implementation of the
  DeepFly3D idea over the top-K candidate peaks. Because the fly skeleton's bones
  form a forest of simple chains (each leg a 5-joint path), the MAP estimate is
  solved by **exact dynamic programming** per limb — no loopy belief propagation —
  with an optional temporal term. It can *recover* a joint when the arg-max landed
  on the wrong peak (occlusion, crossing legs, L/R confusion); its committed 2D
  then feeds the chosen `triangulation` (a plain `dlt` pass keeps the PS estimate).

`correction.py` adds template (Procrustes) alignment and NaN-aware temporal
smoothing (Gaussian or a streaming 1-Euro filter).

### Interface, I/O, reproducibility

DeepFly3D ships a PyQt **GUI** for visualization and manual correction.
deeperfly is **headless**: a single `deeperfly run` drives the whole pipeline
from one merged `config.toml` (camera rig, input→camera map, detector, pipeline,
bundle adjustment, skeleton), with per-stage caching so re-runs only compute
what changed. Results are a self-contained **HDF5** `PoseResult` (cameras,
skeleton, 2D/3D points, confidences, diagnostics, provenance) instead of pickled
objects, and visualization is matplotlib overlays + MP4 export.

### Performance

The geometry/BA core is JAX with float64 enabled, so projection, triangulation
and the BA residual/Jacobian are JIT-compiled and vectorized. Detection batches
all views through the network, **streams** frames in fixed-size windows (constant
memory for arbitrarily long recordings), and can decode video straight on the
GPU (NVDEC) and hand frames to JAX zero-copy via DLPack. CUDA acceleration is
opt-in (`deeperfly[cuda]`); on Apple Silicon the detector runs on **Metal/MPS**.

## What deeperfly intentionally drops

- **No GUI** and no manual point-by-point correction — it is built to run in a
  script or on a cluster.
- **No training** — it converts and runs the published detector weights.
- **No legacy formats** — HDF5 only, no pickle importer.

## References

- P. Günel, H. Rhodin, D. Morales, J. Campagnolo, P. Ramdya, P. Fua.
  *DeepFly3D, a deep learning-based approach for 3D limb and appendage tracking
  in tethered, adult Drosophila.* eLife 8:e48571 (2019).
- [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) ·
  [DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) ·
  [PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment)
