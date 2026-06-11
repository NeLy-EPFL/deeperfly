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
accepted in the config (they are rejected with a pointer to the orbit keys); use
the orbit parameters. The internal `CameraGroup` still uses `rvec` / `tvec`.

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

## `[visualization]` — output videos { #visualization }

Global settings plus one `[[visualization.videos]]` per output MP4.

**Global (`[visualization]`):**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `background` | str or [r, g, b] | `"black"` | Canvas fill (overridable per video / per panel). |
| `output_fps` | float | input fps | Explicit output frame rate for every video. |
| `speed` | float | `1.0` | Scale the input fps instead (`0.5` = slow motion). `output_fps` wins if both are set. |

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
| `plot` | str | *required* | `"imshow"` (the view's frame), `"skeleton_2d"` (its 2D detections), or `"skeleton_3d"` (the 3D skeleton reprojected into the view). |
| `view` | str | *required* | Camera/view name. |
| `x0`, `y0` | int | `0` | Top-left pixel of the panel. |
| `scale` | float | `1.0` | Uniform scale. |
| `width`, `height` | int | from `scale` | Target box (priority over `scale`); one given → the other follows to keep aspect. |
| `background` | str or [r, g, b] | inherits | Per-panel fill. |
| *extra keys* | — | — | Forwarded as draw-op kwargs (`point_radius`, `line_thickness`, `palette`, …). |

A `skeleton_3d` panel needs a 3D pose; a video that requires 3D is skipped (with
a logged reason) when the result has none. Videos are encoded H.264 / libx264 via
PyAV on the CPU.
