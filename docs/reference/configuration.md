# Configuration reference

Every key of the `config.toml`, by section. For a task-oriented walkthrough of
how to customize a config, start with [Writing configs](../guides/configuration.md);
this page is the exhaustive listing.

A config is one TOML file. Each stage reads its parameters through a typed
accessor whose **defaults are the single source of truth** (the frozen `*Params`
dataclasses in `src/deeperfly/config.py`); the packaged
`default_config.toml` mirrors them exactly. An unknown key in a stage table is a
hard error that names the allowed keys. Performance-only knobs (`batch_size`,
`decode_buffer`, `[io.image]`) never invalidate a stage's cache; everything else
that affects a result does.

The top-level layout:

```toml
[[sources]]            # footage globs (shared input)
[io.image]             # image-sequence decode
[skeleton]             # tracked points and limbs
[cameras.defaults]     # rig geometry: shared defaults
[cameras.<name>]       # rig geometry: per-view overrides
[pipeline]             # which stages run
[pose2d]               # 2D detection: knobs + detection plan sub-tables
[bundle_adjustment]    # camera refinement
[pictorial_structures] # opt-in peak recovery
[triangulation]        # 2D -> 3D
[inverse_kinematics]   # opt-in: 3D -> NeuroMechFly joint angles
[visualization]        # output videos
```

## `[[sources]]` — footage { #sources }

An array of tables; each names a footage glob matched inside the recording
directory. A source can feed several pathways and a visualization `imshow` panel.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | *required* | Source identifier (referenced by pathways and views). |
| `filename` | str | the `name` | Glob inside the recording dir: a named file (`camera_0.mp4`), a bare prefix (`camera_1` → `camera_1*`, a video or image sequence), or a wildcard. |

A source's footage is one video file or a naturally-sorted image sequence. A
directory is a valid recording only when every source matches footage with the
same file/frame count.

## `[io.image]` — image decode { #io }

Video files use PyAV; image sequences use OpenCV. The only knob:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `workers` | int | `0` | Image-decode threads. `0` = auto (one per CPU). |

## `[skeleton]` — tracked points { #skeleton }

The tracked points and their structure. Omit the section entirely to use the
default 38-point fly skeleton.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | `"skeleton"` | Skeleton identifier (e.g. `"fly38"`). |
| `point_names` | list[str] | *required* | Ordered tracked-point names; the length is `P`. |
| `limb_points` | table | `{}` | `[skeleton.limb_points]`: each limb name → its points in kinematic-chain order. |
| `limb_palette` | table | `{}` | `[skeleton.limb_palette]`: each limb name → a hex plotting color. Limbs without an entry fall back to a default colormap. |

```toml
[skeleton]
name = "fly38"
point_names = ["lf_thorax_coxa", "lf_coxa_trochanter", "..."]

[skeleton.limb_points]
lf_leg = ["lf_thorax_coxa", "lf_coxa_trochanter", "lf_femur_tibia", "lf_tibia_tarsus", "lf_claw"]

[skeleton.limb_palette]
lf_leg = "#0f7399"
```

Which view sees which point is **not** set here — it is the union of the
[`[pose2d.output_points]`](#output_points) tables.

## `[cameras.*]` — rig geometry { #cameras }

Each `[cameras.<name>]` is a geometric view: pure intrinsics + extrinsics, no
footage. `[cameras.defaults]` is merged into every view; per-view tables override
it (the default rig sets just `azimuth_deg` per view). A view's intrinsics
describe the raw frame of the source feeding it.

**Intrinsics:**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `focal_length_px` | float or [float, float] | *required* | `[fx, fy]` in raw-frame pixels (a scalar is allowed when `fx == fy`). |
| `principal_point_px` | [float, float] | image center `((w-1)/2, (h-1)/2)` | Principal point `[cx, cy]`. Omit to use each view's image center. |
| `distortion_coefficients` | list[float] | `[]` | OpenCV-ordered distortion coefficients; empty means no distortion. |

**Extrinsics (orbit / look-at):** the cameras orbit a target near the world
origin. World up is `+z`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `look_at` | [float, float, float] | `[0, 0, 0]` | World point the camera looks at. |
| `distance` | float | *required* | Distance from `look_at` to the camera center. |
| `azimuth_deg` | float | `0.0` | Longitude around `look_at`. |
| `elevation_deg` | float | `0.0` | Latitude above the horizon (±90 is undefined — the roll becomes ambiguous). |
| `roll_deg` | float | `0.0` | Rotation about the optical axis. |

Explicit `rvec` / `tvec` / `rotation_matrix` / `position` keys are **not**
accepted in a `[cameras.<name>]` table — an orbit is a *description* of a rig
someone built, and half-specifying it with raw extrinsics is rejected rather than
guessed at. To use raw, solved extrinsics, point at a calibration file instead
(below). The internal `CameraGroup` still uses `rvec` / `tvec`.

**A solved rig:** `[cameras].calibration`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `calibration` | str | *unset* | Path to a `calibration.toml`. Relative paths resolve **next to this config file**. |

When set it **wins** over every orbit spec, and the `[cameras.<name>]` tables are
read only for their *order* (the view axis of every points array is positional).
The run logs which of the two it used.

This is how a rig travels between recordings. Every run with bundle adjustment
enabled writes its refined rig to `<outdir>/calibration.toml`, and
`deeperfly calibration export` extracts one from any existing `results.h5` — so a
rig solved once on recording A can drive recording B:

```toml
[cameras]
calibration = "calibration.toml"
```

A calibration records the footage frame its intrinsics describe, so pointing a run
with differently-sized footage at it **fails** rather than silently misprojecting.
See [Output format](output-format.md#calibrationtoml).

```toml
[cameras.defaults]
focal_length_px = [22388.125, 22388.125]
distortion_coefficients = []
look_at = [0.0, 0.0, 0.0]
distance = 107.463
elevation_deg = 0.0
roll_deg = 0.0

[cameras.rh]
azimuth_deg = -120
```

## `[pipeline]` — which stages run { #pipeline }

One `do_<stage>` boolean per stage. Each enabled stage reads its own `[<stage>]`
table.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `do_pose2d` | bool | `true` | Detect 2D pose in every view. |
| `do_bundle_adjustment` | bool | `true` | Refine the cameras. |
| `do_pictorial_structures` | bool | `false` | DeepFly3D-style peak recovery (opt-in). |
| `do_triangulation` | bool | `true` | Triangulate 2D → 3D. |
| `do_inverse_kinematics` | bool | `false` | Fit NeuroMechFly joint angles to the 3D pose (opt-in). |
| `do_visualization` | bool | `true` | Render the videos. |

## `[pose2d]` — 2D detection { #pose2d }

The `[pose2d]` table holds the detector's performance knobs *and* (as sub-tables)
the detection plan — what to detect and how.

**Performance knobs:**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `precision` | str | `"bfloat16"` | Forward precision: `"float32"` (reference), `"float16"` (CUDA autocast, ~1.5–2× faster), `"bfloat16"` (default, wider range). Ignored on CPU/MPS. |
| `batch_size` | int | `16` | GPU forward batch (images per forward). Clamped to ≥ 1; throughput plateaus by ~16 on a fast GPU. |
| `decode_buffer` | int | `4` | Decode queue depth, in multiples of `batch_size`. Clamped to ≥ 1. Peak frames/camera ≈ `(decode_buffer + 2) * batch_size`. |

### `[[pose2d.preprocessors]]`

Named, reusable frame-op pipelines, referenced by a pathway's `preprocessor`.

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Preprocessor identifier. |
| `ops` | list[table] | Ordered frame ops (below); `[]` = identity. |

**Ops** (run in written order; flips/rotations do not commute):

| Op | Fields | Effect |
| --- | --- | --- |
| `fliplr` | — | Left–right flip. |
| `flipud` | — | Up–down flip. |
| `rot90` | `k` (int) | `k` counter-clockwise quarter-turns (any sign). |
| `crop` | `x`, `y`, `width`, `height` | Keep a window. |
| `resize` | `scale`, or `width`/`height`; optional `interpolation` (`"bilinear"`/`"nearest"`) | Rescale. |

Detections are mapped back into the raw frame by inverting these ops, so a
preprocessor never moves the stored 2D or the reconstructed 3D.

### `[[pose2d.models]]`

A detector network and its input contract.

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Model identifier (referenced by pathways). |
| `class` | str | Network registry key (`"hourglass"` = DeepFly2D). |
| `weights` | str | Checkpoint path; `""` / omitted uses the auto-provisioned cache. |
| `input_size` | [int, int] | `(height, width)` the network expects; frames are resized to it and peaks scaled back. |
| `mean` | float | Scalar subtracted after `/255` normalization. |
| `n_out_channels` | int | Output heatmap count (validated against the weights). |

### `[[pose2d.pathways]]`

A named `source → preprocessor → model` inference run. Says *what to detect on*.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | str | yes | Unique pathway identifier (referenced by `output_points`). |
| `source` | str | yes | The `[[sources]]` name to detect on. |
| `model` | str | yes | The `[[pose2d.models]]` name to use. |
| `preprocessor` | str | no | A `[[pose2d.preprocessors]]` name; omit for identity. |

### `[pose2d.output_points.<view>]` { #output_points }

For each view, where every tracked point's data comes from. A table keyed by
point name:

```toml
[pose2d.output_points.rh]
rf_thorax_coxa = { pathway = "rh", out_channel = 0 }
```

`point = { pathway, out_channel }` fills that point of the view from output
channel `out_channel` of the named pathway. Keying by `(view, point)` means each
point has exactly one source (a repeat is an error); a `(view, point)` left out
stays unobserved (`NaN`). That union is the visibility.

## `[bundle_adjustment]` — camera refinement { #bundle_adjustment }

Fly-as-target bundle adjustment over `scipy.optimize.least_squares`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `points_to_use` | list[str] or omitted | the 30 leg points | Skeleton point names that drive BA. Omit the key to use all keypoints. |
| `fixed` | list[str] | `[]` | Parameters held constant (grammar below); anchors the world gauge. |
| `shared` | list[list[str]] | `[]` | Groups of parameters tied together, e.g. `[["lf.tvec[2]", "rf.tvec[2]"]]`. |
| `weigh_by_confidence` | bool | `true` | Scale each reprojection residual by `sqrt(confidence)`; zero/non-finite confidences drop the observation (all-zero falls back to uniform). |
| `max_frames` | int or omitted | `100` | Bundle-adjust on at most this many frames (subsampled). Omit / `null` for all. |
| `frame_sampling` | str | `"even"` | Which frames to keep (below). |
| *other keys* | — | — | Any remaining flat key (`max_nfev`, `loss`, `f_scale`, `tr_solver`, …) is forwarded to `scipy.optimize.least_squares`. |

**`fixed` / `shared` grammar** — a reference is `<camera>.<param>` with optional
indexing, and `*` wildcards the camera:

- `"*.intr"` — every camera's intrinsics.
- `"f.rvec"`, `"f.tvec"` — the front camera's orientation / position.
- `"rm.tvec[2]"` — one component (the z distance) of a camera's translation.

**`frame_sampling` strategies:**

| Value | Keeps |
| --- | --- |
| `"even"` | Evenly spaced over the recording (temporal spread). |
| `"confidence"` | The highest-confidence frame in each time bin. |
| `"coverage"` | The frame in each bin with the most points seen by ≥ 2 cameras. |
| `"diversity"` | Frames whose postures are most spread apart. |

## `[pictorial_structures]` — peak recovery { #pictorial_structures }

Runs only when `do_pictorial_structures = true`. Operates on the detector's top-K
candidates (extracted and cached during detection).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `k` | int | `5` | Candidate peaks per joint. |
| `temporal` | bool | `false` | Add a temporal-consistency term. |
| `lam` | float | `1.0` | Bone-length prior weight. |

## `[triangulation]` — 2D → 3D { #triangulation }

How the per-view 2D points become one 3D point.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `method` | str | `"ransac"` | `"ransac"` (largest multi-view consensus, robust), `"greedy"` (drop the worst-reprojecting view), or `"dlt"` (plain least-squares). |
| `ransac_threshold` | float | `15.0` | Inlier reprojection cutoff (px) for `method = "ransac"`. |
| `min_inliers` | int | `2` | Minimum agreeing views to accept a point (`ransac`). |
| `reproj_threshold` | float | `40.0` | Per-view reprojection cutoff (px) for `method = "greedy"`. |
| `max_drops` | int | `5` | Max views dropped per offending point (`greedy`). |
| `weigh_by_confidence` | bool | `false` | Scale the DLT by `sqrt(confidence)` (the mirror of the BA knob, which defaults `true`). |

## `[inverse_kinematics]` — joint angles { #inverse_kinematics }

Runs only when `do_inverse_kinematics = true`. Fits a NeuroMechFly-style
articulated model to the triangulated 3D pose: the six legs (with segment lengths
**measured from the data**, so the fitted model matches this fly's proportions), plus
the **head** (yaw/pitch/roll, reaching the two antenna tips) and the **abdomen** (a
five-segment sagittal pitch chain reaching the abdomen markers). The head and abdomen
use fixed model geometry baked from the NeuroMechFly MJCF, sized to this fly by a
per-recording scale estimated from the data. Writes the joint angles **and** the fitted
model joints (which reproject onto the raw images — see the `skeleton_nmf` / `mesh_nmf`
panels and the GUI's NMF overlays) to `results.h5`.

!!! note "Needs the `ik` extra"

    The solver is [QuickIK](https://nely-epfl.github.io/quickik/), a Rust whole-body IK
    library. It is an **optional** dependency because it has no published wheels and so
    needs a Rust toolchain to install — everything else in deeperfly does not. Install it
    with `uv sync --extra ik`, or directly:

    ```
    pip install "quickik @ git+https://github.com/NeLy-EPFL/quickik#subdirectory=python"
    ```

    Without it, `do_inverse_kinematics = true` fails with an explanatory error. A result
    file that *already* holds a fit needs nothing extra: its overlays, videos and the
    GUI's static fit all work on a plain install.

deeperfly assembles a body plan for the recording and QuickIK fits the whole body at
once against every tracked keypoint, rather than solving each limb on its own. The plan
is solved in **model units**, so the two tolerances below mean the same thing whatever
scale the camera rig happens to be gauged at.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `template` | str | `"neuromechfly"` | A packaged template name, or a path to a template TOML. |
| `legs` | list[str] | all | Which legs to fit (e.g. `["rf", "lf"]`). |
| `fit_head` | bool | `true` | Fit the head chain (yaw/pitch/roll) from the antenna tips. |
| `fit_abdomen` | bool | `true` | Fit the abdomen pitch chain from the abdomen markers. |
| `n_iterations` | int | `60` | Gauss-Newton steps per frame. |
| `neutral_weight` | float | `0.001` | Weight of the pull toward each DOF's neutral value. This is what pins DOFs the keypoints do not determine (the abdomen's interior hinges, a leg's redundant thorax-coxa rotation); raise it if a sparsely-observed limb wanders, at the cost of some bias. |
| `damping` | float | `0.1` | Levenberg-Marquardt damping — see the note below; far above QuickIK's own suggested ~`1e-6`, on purpose. |
| `position_tolerance` | float | `0.001` | Early stop: largest root-position step, in model units. Inert under `fixed_body` (the root does not move). |
| `angle_tolerance` | float | `0.001` | Early stop: largest joint-angle step, in radians. |
| `fixed_body` | bool | `true` | Fix the body in the model frame — right for a **tethered** fly, whose body does not move: the leg roots sit at their measured medians and only the joint angles vary. Set `false` for a freely-moving preparation, to give QuickIK a 6-DOF root to fit per frame. |
| `weigh_by_confidence` | bool | `false` | Weigh each observation by the detector's confidence instead of treating every observed keypoint equally. |
| `parallel` | bool | `false` | Solve in overlapping segments across worker threads. Off by default: each segment restarts from the neutral pose and only warm-starts within itself, so the angle traces can step at a seam — a poor trade for a joint-angle time series unless the recording is long enough to need the speed. |
| `segment_len` | int | `200` | Frames per segment (includes the overlap); `parallel` only. |
| `overlap_len` | int | `10` | Frames shared with the next segment; `parallel` only. |

`constant_points` names skeleton points whose 3D position is physically fixed over the
recording; each is replaced by its temporal median before the fit, so it stops jittering
with per-frame detection noise (and occluded frames get filled in). Under `fixed_body`
the leg roots are already held at their measured medians **by construction**, so this
mainly matters when the body is free.

!!! warning "Joint limits, and why `damping` is large"

    QuickIK enforces joint limits by **clamping** each Gauss-Newton step, not by
    projecting the gradient the way a bounded trust-region method does. Two consequences
    worth knowing:

    - A lightly-damped step on an ill-conditioned chain overshoots into a limit, and a
      clamped angle then stays clamped while the remaining DOFs never take up the slack —
      the solve stalls at a feasible but wrong pose. The abdomen (five near-collinear
      hinges) is exactly that case: at `damping = 0.01` an exactly-*straight* synthetic
      abdomen converges to a full ventral curl. Hence the large default — which also
      cannot go much higher, since the well-conditioned head chain simply
      under-converges when over-damped.
    - Where a limit genuinely binds, it caps the achievable fit more than it used to. The
      stage logs a warning naming any DOF that sits at a limit in most frames, because
      that is a signal your `bounds` are what's limiting the fit — the packaged template
      gives the middle and hind legs the **front** leg's ranges as a placeholder, and on
      real recordings the hind legs do press against them.

    Measured on the example recording, the fitted model lands a mean 0.049 world units
    from the triangulated keypoints, against 0.037 for the pre-QuickIK solver. The gap
    is concentrated in the bound-limited hind legs: widening the pressed limits closes
    it and then some (0.032, i.e. better than the old solver).

A `[inverse_kinematics.bounds]` sub-table overrides per-DOF joint angle limits in
**degrees**. Keys are the flygym joint angle names `"<parent_body>-<child_body>-<dof>"`
(dofs `yaw`/`pitch`/`roll`) — the same names written to `results.h5` — e.g.
`"rf_trochanterfemur-rf_tibia-pitch" = [10, 160]` for a leg,
`"c_thorax-c_head-pitch" = [-30, 30]` for the head, or
`"c_abdomen12-c_abdomen3-pitch" = [-45, 20]` for the abdomen. By default each abdomen
hinge is limited to **ventral
(downward) flexion only**, up to 30° (`[-30, 0]`): the few near-midline abdomen
markers under-constrain the five-segment chain, so a symmetric range lets the solver
fold it into a non-physical zig-zag, while a downward-only range keeps the fit a
smooth ventral curl.

### Marker placement — `[inverse_kinematics.head]` / `[inverse_kinematics.abdomen]` { #ik-markers }

*Where* each head/abdomen keypoint sits relative to the NeuroMechFly model is a
**labeling-scheme choice** — e.g. the packaged abdomen markers reproduce the original
DeepFly3D annotation as small offsets from the model's abdomen joints (see
[Keypoint locations](../explanation/keypoints.md)). These tables let a different
skeleton retarget those markers **without re-running the model build**: when a table
is present it **replaces** that chain's default markers. Each entry is keyed by the
skeleton point name:

```toml
[inverse_kinematics.abdomen]
l_abdomen0 = { body = "c_abdomen3", offset = [0.0, 0.05, 0.30] }
r_abdomen0 = { body = "c_abdomen3", offset = [0.0, -0.05, 0.30] }
# ... the full set of abdomen markers you track

[inverse_kinematics.head]
l_antenna = { body = "l_pedicel", offset = [0.0, 0.0, 0.0] }
r_antenna = { body = "r_pedicel", offset = [0.0, 0.0, 0.0] }
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `body` | str | *required* | The model body the marker is rigidly attached to. The chain (head/abdomen) and the marker's chain depth follow from it. The abdomen bodies are `c_abdomen12`/`c_abdomen3`/`c_abdomen4`/`c_abdomen5`/`c_abdomen6`; the head bodies are the head subtree (`c_head`, `l_pedicel`/`r_pedicel`, eyes, …). |
| `offset` | [float, float, float] | *required* | Offset from that body's origin (the joint), in the body's frame — the model units the rest of the IK uses. The marker's neutral position is `body_frame · offset`. |
| `depth` | int | the body's chain depth | Override the chain depth (rarely needed; the body determines it). |

The joint geometry (anchors, axes) stays the model's; only the markers move. Omit
both tables to keep the packaged NeuroMechFly markers. Custom markers also flow into
the GUI's live re-fit (it reads this same config beside `results.h5`).

The `mesh_nmf` overlay (videos + GUI) renders the posed model mesh on the GPU when a
headless OpenGL (EGL) context is available — roughly 10× faster than, and with exact
depth ordering over, the pure-CPU rasterizer it falls back to where no GL device is
present. The body, head, and abdomen are placed at the recording's single
`body_scale` (registered once from the median thorax-coxa spread), so the overlay
holds a constant size and only its pose (rotation + translation) changes per frame —
the fly is rigid, so this stops the body breathing with per-frame coxa noise.

Which body parts the overlay draws is configurable, separately for the videos and the
editor: `[visualization].mesh_hide` (rendered videos) and `[gui].mesh_hide` (the
correction GUI) each list parts to drop, **defaulting to `["wings"]`**. The
vocabulary is `wings` / `halteres` / `eyes` / `antennae` / `head` / `thorax` /
`abdomen` / `legs`.

## `[visualization]` — output videos { #visualization }

Global settings plus one `[[visualization.videos]]` per output MP4.

**Global (`[visualization]`):**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `background` | str or [r, g, b] | `"black"` | Canvas fill (overridable per video / per panel). |
| `output_fps` | float | input fps | Explicit output frame rate for every video. |
| `speed` | float | `1.0` | Scale the input fps instead (`0.5` = slow motion). `output_fps` wins if both are set. |
| `mesh_hide` | list[str] | `["wings"]` | NMF overlay body parts to hide in the videos (`wings`/`halteres`/`eyes`/`antennae`/`head`/`thorax`/`abdomen`/`legs`). |

**`[visualization.kwargs]`** — draw-op defaults shared by every video, keyed by
the `plot` op name (`imshow`, `skeleton_2d`, `skeleton_3d`). Kwargs merge across
three levels — global → per-video `kwargs` → per-panel extra keys — most specific
winning.

**`[[visualization.videos]]`:**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `video_name` | str | *required* | Output filename (`<video_name>.mp4`). |
| `panels` | list[table] | *required* | Ordered panels (below); they draw in order, so a skeleton panel over an `imshow` at the same offset overlays it. |
| `width`, `height` | int | auto-size | Canvas size in pixels; omit to fit all panels. |
| `background` | str or [r, g, b] | inherits global | Per-video canvas fill. |
| `kwargs` | table | `{}` | Per-video draw-op kwargs (merges over the global). |

**Panel** — one draw op for one view at a pixel offset:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `plot` | str | *required* | `"imshow"` (the view's frame), `"skeleton_2d"` (its 2D detections), `"skeleton_3d"` (the 3D skeleton reprojected into the view), `"skeleton_nmf"` (the fitted inverse-kinematics model skeleton reprojected into the view), or `"mesh_nmf"` (the fitted NeuroMechFly *mesh*, shaded and reprojected; `alpha` sets its opacity). |
| `view` | str | *required* | Camera/view name. |
| `x0`, `y0` | int | `0` | Top-left pixel of the panel. |
| `scale` | float | `1.0` | Uniform scale. |
| `width`, `height` | int | from `scale` | Target box (priority over `scale`); one given → the other follows to keep aspect. |
| `background` | str or [r, g, b] | inherits | Per-panel fill. |
| *extra keys* | — | — | Forwarded as draw-op kwargs (`point_radius`, `line_thickness`, `palette`, …). |

A `skeleton_3d` panel needs a 3D pose, and the `skeleton_nmf` / `mesh_nmf` panels
need the inverse-kinematics model; a video that requires one is skipped (with a
logged reason) when the result has none. Videos are encoded H.264 / libx264 via
PyAV on the CPU.

The `inverse_kinematics` overlays are also available live in `deeperfly gui`: the
**NMF skeleton** toggle ghosts the fitted model joints over each view and the **NMF
mesh** toggle renders the posed NeuroMechFly mesh on the client GPU (smooth-shaded
WebGL, rendered at the view's display resolution). They can also be inspected in 3D
together with the cameras and the triangulated pose via the **3D view** button. It
opens a floating panel that overlays the editor without blocking it — drag its title
bar to move it, its corner to resize, and inside it drag to orbit, Shift/right-drag to
pan, and scroll to zoom right up to the model; because the rest of the GUI stays live,
the **main frame slider** still scrubs the 3D pose through time. Both **re-fit to the operator's 3D
corrections** — as the latent skeleton is edited, the model is re-solved for that
frame and the overlay follows. The legs skin to the corrected keypoints; the head
and abdomen are fixed model geometry, sized to this fly by a per-recording scale the
IK stage **estimates from the data** (each chain's contour length — how far its
markers reach along it — analogous to the coxa-derived body scale), so a longer
abdomen or bigger head is matched without a manual knob. The same scale is used by the `mesh_nmf` video op,
whose GPU rasterizer uploads each frame's posed geometry once and renders every
camera from it (so a multi-view mesh video renders an order of magnitude faster than
the old per-view software path).

## `[gui]` — correction editor { #gui }

Display-only settings for `deeperfly gui` (no effect on the pipeline or `results.h5`);
read from the `config.toml` snapshot beside the `results.h5`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mesh_hide` | list[str] | `["wings"]` | NMF overlay body parts to hide in the editor (`wings`/`halteres`/`eyes`/`antennae`/`head`/`thorax`/`abdomen`/`legs`). The rendered videos use `[visualization].mesh_hide`. |

## `[annotation]` — ground-truth annotation { #annotation }

How `deeperfly gui` turns your 2D labels into a live 3D estimate. Read from the
`config.toml` snapshot beside `results.h5`; no effect on the batch pipeline except
that the triangulation *method* + thresholds are shared with
[`[triangulation]`](#triangulation), so a point with no ground truth re-solves to the
run's cached 3D. Every fork is a knob with a sensible default.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `precedence` | list[str] | `["gt", "prediction", "projection"]` | How each view's *displayed* 2D is chosen. `projection` (the 3D reprojected) is display-only and never feeds the solve. |
| `solve_policy` | str | `"gt_wins"` | How ground truth (GT) and predictions combine in the live 3D solve: `gt_wins` (≥ `min_gt_for_exclusive` GT views ⇒ GT only; one GT view ⇒ GT hard-weighted, predictions fill; no GT ⇒ the configured `[triangulation]` method), `equal_weight` (GT-or-prediction per view through the configured method), or `weighted_blend` (one weighted DLT). No policy ever discards a GT observation. |
| `min_gt_for_exclusive` | int | `2` | `gt_wins`: at this many GT views, solve from GT alone. |
| `gt_weight` | float | `1000.0` | Relative weight of GT rows when GT is mixed with predictions. |
| `prediction_weight` | str \| float | `"uniform"` | Prediction row weight in a GT-present solve: `"uniform"`, `"confidence"`, or a fixed float. Defaults to uniform to match the batch (`weigh_by_confidence=false`). |
| `confirm_default` | str | `"all"` | Default suggestion set a bulk-confirm promotes to GT: `"all"` (predictions and reprojections), `"predictions"`, or `"projections"`. |
| `low_conf` | float | `0.2` | Predictions below this confidence are visually de-emphasised (not hidden). |
| `undistort_before_solve` | bool | `false` | Undistort GT/prediction pixels before the linear DLT (more accurate on distorted lenses, but a zero-GT re-solve no longer matches the batch cache, which does not undistort). |
| `equal_weight_protect_gt` | bool | `true` | Under `equal_weight`, force GT views to stay RANSAC inliers so a prediction consensus cannot vote a human label out. |
| `gt_wins_keep_stabilizers` | bool | `false` | Under `gt_wins`, keep predictions as low-weight depth stabilisers even once GT is exclusive (guards degenerate GT-view geometry). |
