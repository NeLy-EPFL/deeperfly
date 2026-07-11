# Photometric camera-extrinsics refinement with Gaussian splatting

An in-depth look at whether — and how — the differentiable Gaussian-splatting
renderer can *refine* deeperfly's camera extrinsics, and why the **many
synchronized frames** we already record are the key asset. Grounded in
controlled experiments (`refine_extrinsics.py`) on `examples/data`.

## TL;DR

- gsplat's rasteriser is differentiable w.r.t. the camera view matrices, so
  camera pose can be optimised by analysis-by-synthesis. We verified gradients
  flow to `viewmats`.
- **Extrinsics are observable from the photometric signal.** With a fixed fly
  model, camera-only refinement recovers a known ~37px miscalibration back to
  the reference reprojection accuracy. *(see Experiment 1)*
- **But two axes are degenerate under this near-orthographic rig** and must be
  locked/anchored: optical-axis translation (dolly) and the global similarity
  gauge. *(Experiment 2)*
- **The hard part is that camera error and fly geometry are confounded**: when
  the geometry is co-estimated from scratch, it absorbs the pose error and the
  cameras drift. Naive multi-frame (shared cameras, *independent* per-frame
  clouds) gives only a whisper of improvement — each cloud absorbs its own
  share. *(Experiment 3, the key negative result)*
- **The right way to spend the synchronized frames is shared structure.** The
  thorax is rigid across all 64 frames; a shared rigid-body Gaussian model (or
  anchoring to the known triangulated keypoints) makes the shared cameras
  identifiable in a way no per-frame geometry can absorb. This is the
  recommended next build.

## Why this can work at all

The 7 cameras are **static across all 64 frames**; only the fly moves. So the
extrinsics are a small set of shared parameters (7 cameras × 6 DoF) constrained
by every one of the 7×64 images. Photometric refinement uses *all* the pixels,
not just the 38 keypoints deeperfly's bundle adjustment sees — orders of
magnitude more constraints, and dense over the textured cuticle.

## Parameterisation (in `CameraDeltas`)

Each camera gets a left-multiplied SE(3) delta on top of its deeperfly pose:

```
T_i = Exp([ω_i | u_i]) · T_i^init          # world→camera
```

with `ω ∈ so(3)` (rotation) and `u` the translation delta *in the camera
frame*. Two deliberate restrictions come straight from the degeneracy analysis:

- **Optical-axis translation `u_z` is locked** (only `u_x, u_y` free → 5 DoF /
  camera). Under a 2.5° FOV a dolly barely changes the image (Experiment 2).
- **One camera is fully anchored** (default the head-on `f`) to fix the global
  similarity gauge; scale is pinned by the fixed camera + fixed intrinsics.

Optimisation uses coarse-to-fine Gaussian blur (widening the tiny convergence
basin — recall `fx≈22388`, so 1° ≈ 390px), a **Cauchy robust loss** (the same
family that helped deeperfly's far-leg BA), and cosine LR decay.

## Experiments and results

Controlled perturbation-recovery: treat deeperfly's bundle-adjusted cameras as
the reference, apply a *known* extrinsic perturbation, and refine. Metrics are
mean over the 6 free cameras. `reproj RMSE` is deeperfly's native metric
(reproject the known 3D keypoints, needs no Gaussians).

### 1 — Observability (frozen geometry, camera-only)

Freeze a fitted fly model (frame 32, 306k Gaussians), perturb the cameras,
optimise only the cameras against the real images. Isolates camera
observability from geometry.

| metric (mean, 6 free cams) | reference | perturbed | **refined** |
|---|--:|--:|--:|
| reprojection RMSE (px)     | 5.84  | 36.57 | **5.84** |
| rotation error (deg)       | 0     | 0.102 | **0.045** |
| centre error, in-plane     | 0     | 0.10  | **0.076** |
| centre error, optical-axis | 0     | 0.000 | 0.000 (locked) |

**Reading:** the ~37px miscalibration is pulled all the way back to the 5.84px
reference reprojection accuracy — the photometric signal *does* localise the
cameras. (During optimisation it recovers by step ~300 and holds; the only
failure mode we hit was over-sharpening past the floor, where the fly's
repetitive leg/stripe texture aliases into false minima — hence the
coarse-to-fine blur floor.)

### 2 — Degeneracy: optical-axis translation

Perturb one component at a time and measure image sensitivity, then free `u_z`
and try to refine it back:

| perturbation (4 free cams) | image effect (reproj) | sensitivity |
|---|--:|--:|
| **in-plane** translation, 0.1 unit | 5.84 → 36.6 px | ~209 px / unit |
| **optical-axis** translation, 4.0 units | 5.84 → 8.5 px | ~0.67 px / unit |

Refining with `u_z` **freed** after a 4-unit dolly: optical-axis error only
4.0 → 2.8 units, and it *corrupts* the other axes it trades against (rotation
0 → 0.22°, in-plane 0 → 0.02).

**Reading:** optical-axis translation is ~**300× less observable** than
in-plane under this 2.5° FOV — effectively unrecoverable, and freeing it only
lets the optimiser launder error into the good axes. Lock it; and don't jointly
refine intrinsics (focal ↔ distance is the same degeneracy).

### 3 — Why many synchronized frames matter (joint geometry + cameras)

Now also estimate the fly geometry (the realistic case): warm up the geometry
with cameras frozen, then jointly refine. Same 37px perturbation, 1 frame vs 4
synchronized frames sharing the cameras, at two geometry capacities.

| refined reproj RMSE (px) | 1 frame | 4 frames |
|---|--:|--:|
| 30k Gaussians/frame | 32.5 | 31.6 |
| 8k Gaussians/frame  | 30.8 | 30.1 |

(rotation error *grew* in every joint run: 0.10° → 0.6–1.1°; in-plane unchanged
at 0.10.)

**Reading — the important negative result.** Unlike the frozen-geometry case,
joint estimation barely recovers (only ~15% of the 37px) and the cameras drift
in rotation. The reason: during warm-up the geometry fits the images *under the
wrong cameras*, becoming "perturbation-consistent", after which there is little
gradient left to move the cameras correctly. Sharing cameras across 4 frames
gives only a **whisper** of an edge (31.6 vs 32.5 px) — because each frame's
*independent* cloud absorbs its own share of the error, so the shared cameras
stay under-constrained. Lowering capacity (8k) doesn't rescue it.

The lesson is specific and actionable: **many synchronized frames only help if
something in the geometry is *shared* across them.** With independent per-frame
clouds each frame is its own ill-posed problem. The fly gives us exactly the
shared structure needed — the **thorax is rigid across all frames** (only the
legs articulate). A shared rigid-body Gaussian model (+ per-frame legs), or
anchoring Gaussians to the known triangulated keypoints, turns the 7×K images
into one over-determined constraint on the cameras that no per-frame geometry
can absorb. That is the recommended next build (below).

## Recommended protocol for real refinement

The load-bearing change over the prototype is **#2** — break the geometry
absorption with shared structure.

1. **Init** from deeperfly's BA extrinsics (already good) — this is refinement,
   not from-scratch calibration.
2. **Share geometry across frames** (the whole point of many synchronized
   frames):
   - *Simplest:* anchor Gaussians to the known triangulated 3D keypoints and
     keep them fixed/regularised, so the cloud cannot drift to absorb pose
     error — reducing this back toward a dense photometric bundle adjustment.
   - *Better:* a single **rigid thorax** Gaussian model shared across all K
     frames, with a per-frame rigid body transform and per-frame leg Gaussians.
     The shared body is seen in 7×K images and cannot be explained away frame by
     frame, so it pins the cameras.
3. **Lock the degenerate DoF**: `u_z` locked (5-DoF/camera), head-on camera
   anchored, scale fixed from known keypoint distances.
4. **Schedule**: coarse-to-fine blur with a floor (never fully sharp — the fly's
   repeated leg/stripe texture aliases), Cauchy robust loss, cosine LR decay,
   small camera LR; alternate camera and geometry steps rather than pure joint.
5. **Evaluate** on held-out signals: leave-one-view-out photometric error and a
   hold-out subset of keypoint triangulation residuals — *not* the training
   loss. Refined poses should move sub-degree; large moves (especially any
   `u_z`) mean a degenerate direction is leaking.

## Settled vs open

**Settled by these experiments:** (a) gsplat's pose gradients are usable;
(b) given good geometry, extrinsics are photometrically recoverable to
reference reprojection accuracy; (c) the near-orthographic degeneracies are
quantified and handled by locking `u_z` + anchoring the gauge; (d) naive
multi-frame with independent per-frame geometry does *not* realise the
multi-frame advantage.

**Open:**
- The shared rigid-body implementation that should realise the multi-frame
  advantage — the concrete next build.
- The headline question: does photometric refinement *beat* deeperfly's
  keypoint BA (not just match it)? Measure against held-out keypoints and,
  specifically, the far-leg / cross-side cases BA struggles with.
- Near-coplanar cameras (all ~0° elevation) leave a depth direction weakly
  constrained; more frames reduce variance but not this geometric degeneracy.
- Geometry realism sets the floor (fixed-count clouds here; MCMC would sharpen).
