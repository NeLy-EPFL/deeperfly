# Photometric cross-side refinement — findings (SHELVED)

**Status: shelved after de-risking. Do not merge into `dev`.** This package is a
working prototype kept for the record; the analysis below shows the approach does
not pay off on the target rigs and explains why. Prototype code lives in this
directory (commit `13fc48a`); the de-risk scripts + evidence are under `derisk/`.

## TL;DR

The far legs (opposite-side legs) reproject ~1–2 leg-widths off on rigs like
`examples/JSP_SCAPE_260417_IN07B001_Fly3_004`. The idea was to refine the
cross-side camera extrinsics so reprojected far-leg *bones* land on the image leg
*pixels* (a chamfer / distance-transform objective on a leg-response map).

De-risking on the example recording shows this **costs more than it gains**:

- A single rigid cross-side transform recovers only ~2px of the ~7px far-leg
  error (down to ~1 leg-width, still visibly off).
- Realizing even that unavoidably shifts the **well-pinned body by ~13px (a 5×
  regression, 2.5→13px) in the left camera views**, because the body's 3D is
  co-determined by the left cameras — you cannot move the far legs without moving
  the left cameras, and moving the left cameras drags the body.
- The rotation the optimizer favors (~4.9°) is a chamfer artifact: it barely
  helps far legs and only adds body-shift.

The good cross-side calibration is *load-bearing* for the body; the far-leg error
is largely irreducible per-leg/per-frame noise, not a global offset a rigid
transform can remove. Optic flow does not change this — the limit is
observability/entanglement, not correspondence.

## The rig and why the cross-side link is weak

7 cameras: right `{rh,rm,rf}`, left `{lf,lm,lh}`, front `f`. Within each side the
relative calibration is good; the only cross-side bridge is the front camera.

Co-visibility (fraction of frames a point is detected in **both** groups),
measured from `pose2d/conf`:

| points | Left&Right | Left&Front | Front&Right |
|---|---|---|---|
| body/midline | 0.00 | 0.12 | 0.12 |
| left legs | 0.00 | 0.40 | 0.00 |
| right legs | 0.00 | 0.00 | 0.40 |

- The body is **not** the cross-side constraint (0% direct L↔R, 12% via front).
- The real bridge is the **front camera's 100% view of both sides' front+mid
  distal legs** (femur_tibia / tibia_tarsus / claw). Front sees **zero hind**
  legs (consistent with the training data: front-camera head has 0% hind labels),
  so hind cross-side geometry is unobservable by any keypoint path.

## Approaches tried

**v1 — regressing prototype (committed, `13fc48a`).** Single rigid `T` on the left
cluster, right+front as fixed gauge, then **re-triangulate the whole pose** with
the nudged rig. Result: global scramble. On the example recording (scene radius
~1.67 world units): body midline moved 0.27u, left legs 0.19u, **right legs 0.32u
even though the right cameras never moved** — RANSAC reselecting consensus sets on
a perturbed rig. `T` overfit to 3.75°. This is what produced the broken
`pose3d.mp4`.

**v2 — redesign, de-risked in `derisk/` (never wired as a stage).** Two fixes:
1. *Rigid-move-legs-only*: copy the body + near legs straight from bundle
   adjustment; move only the far/left-leg 3D by `T`; compensate the left cameras
   for viz. The body's 3D can never be recomputed → the v1 scramble is impossible
   by construction.
2. *Hard, non-saturating anchor* on the trusted keypoints (front-bridge legs) +
   strong Tikhonov, so the chamfer can only nudge the weakly-observed DOF.

v2 does prevent the scramble — but it exposed the fundamental limit below.

## Why it cannot pay off (measured, example recording)

Current reprojection error (median px, from `triangulation/reproj_error`):

| quantity | value |
|---|---|
| body points, **left views** | **2.54px** |
| body points, right views | 2.32px |
| near legs (own-side views) | 2.1–2.6px |
| **far legs (any view)** | **NaN — the detector produces no far-leg detections** |

The far legs are literally unconstrained by keypoints; the only handle on them is
the image-based chamfer (~7px init ≈ 1.5 leg-widths).

DOF split of the v2 solve (which DOF buys the far-leg gain, and its body-shift
cost). Init mean far-leg chamfer 7.25px:

| solve | rotation | translation | far chamfer | body-shift (left views) |
|---|---|---|---|---|
| full 6-DOF | 4.89° | 0.056 | 7.25 → 5.36px | 14.97px |
| translation-only | 0° | 0.069 | 7.25 → **5.40px** | 12.94px |
| rotation-only | 4.88° | 0 | 7.25 → 6.71px | 9.52px |

- Translation alone captures essentially the entire far-leg gain (5.40 vs 5.36).
- The ~4.9° rotation is an artifact — it barely helps far legs and adds body-shift.
- **Even pure translation shifts the body ~13px in the left views.** Adding that
  to the current 2.5px body error → ~13px: a 5× regression of the *measured* body
  to marginally improve the *unmeasured* far legs (1.5 → 1 leg-width).

See `derisk/farleg_overlay.png`: corrected far legs (green) sit almost on top of
the current ones (red) — the visual gain is marginal.

### The entanglement, stated plainly

The body reprojects at ~2.5px everywhere because it is well-pinned (front bridge +
within-side views). The far legs are unmeasured. Any camera motion that improves
the far legs must move the left cameras, and the body — co-determined by those
cameras — moves with them. So "fix far legs" and "disturb body" are inseparable.
A single rigid `T` also only explains ~2px of the ~7px far-leg error; the rest is
per-leg/per-frame triangulation noise, not a global cross-side offset.

## What would actually be needed

- **Hardware**: genuine left↔right co-visibility (an extra camera / geometry that
  sees far legs from both sides). Not available on these rigs.
- **Relabel + retrain** the front-camera detector head for hind legs (0% labels
  today) to add a real cross-side hind bridge. Out of scope; large effort.
- **Reframe as a 3D-quality problem**, not calibration: the far-leg *trajectories*
  are noisy because they are triangulated from few oblique views. Robust /
  temporal smoothing of the far-leg 3D (decoupled from camera geometry, so it
  cannot touch the body) is the more promising and safer direction if the far legs
  matter enough to keep working.

## Reproduce

Against a `results.h5` that has `bundle_adjustment/cameras` + `triangulation/points3d`
(run bundle adjustment + triangulation first), on this branch:

- `derisk/objective_sweep.py` — the v2 objective, sweep of chamfer/anchor/reg weights.
- `derisk/dof_split.py` — the full vs translation-only vs rotation-only table above.
- `derisk/farleg_overlay.py` — renders `farleg_overlay.png` (BA vs corrected far legs).

Paths in the scripts point at the example recording and the session scratchpad;
adjust `REC` / `SP` for another recording.
