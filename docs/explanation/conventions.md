# Conventions & glossary

The conventions every part of deeperfly shares — array layouts, the missing-data
encoding, and the coordinate frames — plus a glossary of the terms used across
the docs and the config.

## Array layouts

Arrays are **view-leading**: the camera/view axis comes first.

| Array | Shape | Meaning |
| --- | --- | --- |
| `pts2d` | `(V, T, P, 2)` | 2D keypoints: per view, per frame, per point, `(x, y)` in raw-frame pixels. |
| `conf` | `(V, T, P)` | Detector confidence for each 2D observation. |
| `pts3d` | `(T, P, 3)` | 3D keypoints: per frame, per point, `(x, y, z)` in world units. |
| `reproj_error` | `(V, T, P)` | Per-view reprojection error of the 3D point, in pixels. |

The axes are referred to throughout by these letters:

- **`V`** — camera **views** (8 in the default rig: `rh`, `rm`, `rf`, `f`, `lf`, `lm`,
  `lh` and the axial `h`). A run [narrows itself to the footage
  present](pipeline.md#narrowing), so a recording missing a camera shortens `V` rather than
  padding it — below two views (`config.MIN_VIEWS_FOR_3D`) it refuses.
- **`T`** — **frames** (time).
- **`P`** — skeleton **points** / keypoints (38 in the default skeleton).

Single-image helpers (e.g. `CameraGroup.project`) drop the `T` axis and use
`(V, N, 2)` / `(N, 3)`, where `N` is the number of points.

## NaN means missing

There is no separate visibility mask. A keypoint that a view does not observe is
stored as `NaN`, and the same convention carries through:

- A `(camera, point)` nothing writes stays `NaN`. Every shipped detector is dense —
  channel *i* is point *i* of that camera — so every cell is written and the detector's
  own 2D has no missing entries. What each camera observes *is* the visibility, and under
  a dense detector it is total.
- Triangulation ignores `NaN` views and returns `NaN` for a point seen by fewer
  than `min_inliers` views.
- The HDF5 datasets preserve `NaN`, so it round-trips through `results.h5`. Point arrays
  are stored as `float32` (and deflated) but read back as `float64`: the values are pixel
  coordinates and the millimeter 3D fitted from them, which `float32` resolves ~4 orders
  finer than the detector can localize.

When you read `pts3d`, treat `NaN` as "not reconstructed for this frame/point".
Use `np.nanmedian` / `np.nanmax` and friends, as `deeperfly inspect` does.

## Coordinate frames

- **Pixels** are in the **raw source frame** that a view's intrinsics describe.
  Any per-pathway preprocessing (flip, crop, resize) is *inverted* before the
  points are stored, so a mirror fed to the detector never moves the stored 2D or
  the reconstructed 3D.
- **World units** are whatever the rig's `distance` / intrinsics imply (the
  default rig is metric-like but unitless). World **up** is `+z`.
- **Cameras** use the orbit (look-at) parameterization in the config:
  `look_at`, `distance`, `azimuth_deg`, `elevation_deg`, `roll_deg`. Internally a
  camera is the usual `rvec` (Rodrigues rotation), `tvec`, intrinsics
  `[fx, fy, cx, cy]`, and OpenCV-ordered distortion coefficients.

## Numerics

The geometry core — projection, triangulation, and the bundle-adjustment
residual and Jacobian — is **JAX in float64 on the CPU**; the arrays are tiny, so
a GPU never helps. The **2D detector is PyTorch** and uses the GPU (CUDA or
Metal/MPS) automatically. Detector forward precision is configurable
(`[pose2d].precision`), but everything geometric stays float64.

## Confidence

`conf` is each detector's own peak score, and the two classes do not compute it the same
way: the per-view `hrnet` arms take the raw arg-max cell value of the padded field, while
the multiview transformer sums the softmax mass in a 5×5 window around its soft-argmax
(its targets were rendered at sigma 1.25, so a correct prediction spreads over neighbors
and the single cell under-reports it). So the scale is a property of the detector, not of
the rig: compare confidences within a run, never across two classes.

`weigh_by_confidence` (in `[bundle_adjustment]` and `[triangulation]`) scales each
observation's least-squares contribution by `sqrt(confidence)`, so surer
detections pull harder; non-positive or non-finite confidences drop the
observation. For RANSAC the weighting affects the candidate fits and the final
refit but not the inlier vote, which stays a pure geometric reprojection test.

## Glossary

**Camera** — one entry of `[cameras.<name>]`: its geometry (intrinsics + orbit
extrinsics) **and** its footage. A camera's `video` pattern says which files are its
own; a source that no camera claimed was never anything but a camera without geometry,
so the two are one thing.

**Detection window** — the box a camera is detected through (`[pose2d.crops]`, one entry
per camera). Inverted on the way back, so a detection lands in the camera's raw frame,
which is what its intrinsics describe. There is no op grammar: flips and rotations had no
consumer left once detection went one-to-one.

**Detector** — the network plus its weights and input contract (`[pose2d] class` +
`weights`). One per run. Two classes ship, both dense: `class = "hrnet"` is the per-view
detector (and the loader that also runs the HGNetV2 checkpoint), `class = "mvt"` the
multiview transformer, which encodes a frame's views together. Anything else is refused
rather than defaulted — see [the dense-38 detectors](detectors.md).

**Detection plan** — *synthesized*, not declared: one source, one window and one
identity-mapped pass per camera, derived from the camera table.

**Model pack** — the mechanical model the inverse-kinematics stage fits
(`[inverse_kinematics] model`): a leg template, a baked articulation and an overlay mesh
selected as one unit. A pack names no skeleton point.

**Binding** — where each tracked point sits on a model
(`data/bindings/<skeleton>@<model>.toml`). A fact about the *pair*, so it lives in
neither half; it is the one artifact where a skeleton point name and a model body name
may appear together.

**Rig / `CameraGroup`** — the set of named cameras as one object.

**Skeleton** — the tracked points and their structure (`[skeleton]`), and exactly four
things: `points`, `edges`, `point_symmetries` (mirror pairs, which is what makes a
left/right check decidable) and the two colour tables. There is no grouping concept — no
limbs, no chains, no names for subsets of points; chains are derived from the edge graph
where anything needs them. `fly38` is the one packaged skeleton — 38 points: six 5-point
legs, two antennae, `neck`, and the 5-point dorsal-midline `abdomen0..4` chain.
`include = "fly38b"` still resolves to it.

A skeleton's `name` is a **label**, not an identity: `fly38` has meant two different point
sets. What identifies one is its ordered `point_names`, which is what every check compares,
and its `digest` — 8 hex over the points, edges and symmetry pairs — which is the printable
form of that, shown as `fly38@42da66d9`.

**Edge** — a pair of points the skeleton joins (`edges`). Used for the bone-length prior,
pictorial structures' graph, seeding a missing joint in the editor, and drawing. A point in
no edge is still tracked — the antennae and `neck` are.

**Candidates** — the detector's top-`k` heatmap peaks per joint, cached by
`pose2d` when `pictorial_structures` is enabled; the input the peak-recovery stage
reconsiders.

**Stage** — one step of the linear pipeline (`config.STAGES`: `pose2d`,
`bundle_adjustment`, `pictorial_structures`, `triangulation`, `eks`, `postprocess`,
`inverse_kinematics`, `visualization`), toggled by `[pipeline].do_<stage>` and configured
by its `[<stage>]` table. All are on by default except `pictorial_structures`.

**Fingerprint** — the result-affecting config subset recorded per stage in
`run.json`; a stage's cache is reused only while its fingerprint still matches
(see [caching](pipeline.md#caching-and-re-runs)).
