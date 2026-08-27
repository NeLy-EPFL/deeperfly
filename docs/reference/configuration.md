# Configuration reference

Every key of the `config.toml`, by section. For a task-oriented walkthrough of
how to customize a config, start with [Writing configs](../guides/configuration.md);
this page is the exhaustive listing.

A config is one TOML file. Each stage reads its parameters through a typed
accessor whose **defaults are the single source of truth** (the frozen `*Params`
dataclasses in `src/deeperfly/config.py`). An unknown key in a stage table is a
hard error that names the allowed keys. Performance-only knobs (`batch_size`,
`decode_buffer`, `[io.image]`) never invalidate a stage's cache; everything else
that affects a result does.

**A stage table you leave out runs on its defaults**, and the packaged
`default_config.toml` deliberately omits the ones it does not change — it carries the
*structure* of a run (footage, rig, skeleton, detector) and the few values this rig has
a measured opinion about. To see a key without opening a file:

```console
$ deeperfly config show                          # every section
$ deeperfly config show eks                      # one section, with its docs
$ deeperfly config set eks.inflate_threshold 30  # validated as a run would
```

`config show` marks which values were *set* versus inherited, which is the question a
file full of defaults cannot answer.

The top-level layout:

```toml
sources = [...]        # footage globs (shared input); or [[sources]] blocks
[io.image]             # image-sequence decode
[skeleton]             # tracked points and limbs (or a preset name)
[cameras.defaults]     # rig geometry: shared defaults
[cameras.<name>]       # rig geometry: per-view overrides
[pipeline]             # which stages run
[pose2d]               # 2D detection: knobs + detection plan sub-tables
[bundle_adjustment]    # camera refinement
[pictorial_structures] # peak recovery -- the one stage off by default
[triangulation]        # 2D -> 3D
[eks]                  # ensemble Kalman smoother over the 3D pose
[postprocess]          # corrections that come from knowing the animal
[inverse_kinematics]   # 3D -> NeuroMechFly joint angles (needs the `ik` extra)
[visualization]        # output videos
```

Every array-of-tables section (`[[pose2d.models]]`, `[[pose2d.pathways]]`,
`[[pose2d.preprocessors]]`, `[cameras.<name>]`) can equivalently be written as an array
of inline tables — `models = [{ name = "m", class = "mvt", weights = "x.pth" }]` — which
is what the packaged config does. TOML parses the two identically.

!!! warning "`[[sources]]` is the exception"
    Keep the footage list as `[[sources]]` blocks. `deeperfly project` lifts whole TOML
    *tables* out of a config by their headers, and a bare top-level `sources = [...]`
    key is not a header — a project's `rig.toml` would silently come out with no footage.

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

The tracked points and their structure. Omit the section entirely to use `fly38`, the
one packaged skeleton: 38 points — six 5-point legs, two antennae, `neck`, and a
5-point dorsal-midline abdomen chain (`abdomen0`…`abdomen4`).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | `"skeleton"` | Skeleton identifier, **or a packaged preset to load** (below). |
| `file` | str | *unset* | Load a skeleton from this path (relative to *this* config), instead of a preset. |
| `point_names` | list[str] | *required* | Ordered tracked-point names; the length is `P`. |
| `symmetries` | list[[str, str]] | `[]` | Left/right mirror pairs (see below). Unordered within a pair; each point in at most one pair. |
| `limb_points` | table | `{}` | `[skeleton.limb_points]`: each limb name → its points in kinematic-chain order. |
| `limb_palette` | table | `{}` | `[skeleton.limb_palette]`: each limb name → a hex plotting color. Limbs without an entry fall back to a default colormap. |

```toml
[skeleton]
name = "fly38"
point_names = ["lf_thorax_coxa", "lf_coxa_trochanter", "..."]

symmetries = [
    ["lf_claw", "rf_claw"],
    ["l_antenna", "r_antenna"],
]

[skeleton.limb_points]
lf_leg = ["lf_thorax_coxa", "lf_coxa_trochanter", "lf_femur_tibia", "lf_tibia_tarsus", "lf_claw"]

[skeleton.limb_palette]
lf_leg = "#0f7399"
```

Which view sees which point is **not** set here — it is the union of the
[`[pose2d.output_points]`](#output_points) tables.

#### Presets — naming a skeleton instead of spelling one { #skeleton-presets }

A skeleton is ~80 lines that almost never differ between recordings of the same animal,
and when they do differ they differ completely. So a table that declares **no
`point_names`** is read as a *reference* and expanded at load:

```toml
[skeleton]
name = "fly38"           # the packaged skeleton
```

```toml
[skeleton]
file = "skeleton.toml"   # a project's own, resolved next to this config
```

Packaged presets live in `src/deeperfly/data/skeletons/`, and as of 0.2 there is exactly
one:

| Preset | Points |
| --- | --- |
| `fly38` | Six 5-point legs (`thorax_coxa` → `coxa_trochanter` → `femur_tibia` → `tibia_tarsus` → `claw`), `l_antenna` / `r_antenna`, `neck`, and a 5-point **dorsal-midline** abdomen chain `abdomen0`…`abdomen4`. 16 [symmetry pairs](#symmetries). |

!!! note "`fly38b` still resolves — it is the same 38 points under the former name"

    Until 0.2 this point set was called `fly38b`, and `fly38` named the historical
    DeepFly3D set (two 3-marker abdomen **side** chains `l_abdomen0..2` / `r_abdomen0..2`,
    no `neck`). The rename kept one alias, `config.SKELETON_ALIASES`: `name = "fly38b"`
    resolves to `fly38` and logs that it did, so old configs and old run snapshots keep
    loading with their point order unchanged.

    The alias is deliberately **not** symmetric. `fly38` now means the midline set, and an
    old config naming `fly38` means the DeepFly3D one — the same word for two different
    point orders, which no alias can disentangle. What stands in its place is a check on
    the points themselves rather than on the label: the detector refuses a checkpoint whose
    channels are another skeleton's, and a run refuses an output directory whose stored pose
    is (`deeperfly.pipeline.run._refuse_a_foreign_skeleton`) — both by the ordered names.

    The retired DeepFly3D set is no longer packaged. It survives as test data,
    `tests/data/fly38_deepfly3d.toml`, which is a complete skeleton file: point
    `file = ".../fly38_deepfly3d.toml"` at it to keep running a config written against it.

!!! warning "A skeleton the packaged articulation cannot fully reach"

    The packaged NeuroMechFly articulation targets **`fly38`**: a body plan built against
    it covers **38 of 38 points**, both the head and the abdomen chain fit, and both are
    sized from measurement. A skeleton that labels neither the head's `neck` nor the
    midline `abdomen0`…`abdomen4` — the retired DeepFly3D set is exactly that case, covering
    **32 of 38**, its six abdomen side markers having nothing on the model to attach to —
    gets **no abdomen fit at all** (all ten of that chain's DOFs come back NaN in every
    frame, since a chain with no fittable marker is left unset rather than reported at the
    neutral pose) and no measured head or abdomen size, so both chains are drawn at the
    model's own. Nothing fails, and both are warned about rather than silent.
    Retarget either chain with [`[inverse_kinematics.markers.head]` / `[inverse_kinematics.markers.abdomen]`](#ik-markers)
    if your labeling scheme differs — where a point sits on the model is a
    labeling-scheme decision, and the packaged placements are tabulated in
    [Keypoint locations](../explanation/keypoints.md).

A preset file is a complete `[skeleton]` table, in exactly the format a project's own
`skeleton.toml` is written in — which is what makes the two interchangeable, and what lets
a project be handed a skeleton by copying one file.

Keys written in the config **override the referenced ones wholesale, per key** — a
`limb_palette` here replaces the referenced palette rather than merging into it. A table
that *does* spell out `point_names` is already self-contained and is left exactly as it
is, so every config written before presets existed keeps its meaning and `name` goes on
being a free-text label for those.

The reference is what the run snapshot records, and the *resolved* point names reach the
`pose2d` fingerprint — so a preset that changes underneath a cached run invalidates it
rather than silently passing.

!!! warning "The skeleton must be the one the detector was trained on"
    A dense detector's channels **are** a skeleton, and a count check cannot tell two
    skeletons apart: `fly38` and the retired DeepFly3D set are both 38 points sharing 32
    of them in a different order, so routing one through the other's config attaches six
    points to the wrong joints and shifts the rest — a wrong limb, not a crash. The
    checkpoint's own recorded channel names are compared against `[skeleton]` on **every
    run** (`deeperfly.pose2d.stream.load_models`), which is the point: a generator only
    ever sees the moment it writes the file, and cannot see a `weights` path later
    repointed or a `[skeleton]` swapped underneath.

    A checkpoint that records **no** channel names is refused outright rather than trusted.
    Every detector class this build ships records them, so a nameless artifact is either
    not one of ours or was stripped — and the alternative to refusing is skipping the one
    check standing between a mis-stamped config and a fly with its limbs on the wrong
    joints.

### `symmetries` — left/right pairs { #symmetries }

The same relation SLEAP models as a `type 2` skeleton edge: two points that mirror
each other across the animal's sagittal plane. Which side comes first carries no
meaning (a pair is an unordered set), and a point named in no pair simply carries
no side.

Three things read the pairs, and two of them fail *silently* without them:

| Consumer | What it does with them | What its absence costs |
| --- | --- | --- |
| [`[pose2d.output_points]`](#output_points) validation | Checks that a pathway whose preprocessor **mirrors** the frame lands on the *mirrored* points | A one-word typo in one of 122 rows swaps a body side. The detector still fires and triangulation still converges — the reconstruction is just a fly with its legs crossed. |
| Flip augmentation (out of tree; see [`mirror`](#cameras)) | Permutes the point channels by `Skeleton.flip_perm()` and relabels the sample with the mirrored camera | Every left channel trains on a right joint. No error, no warning; it looks like a model that will not converge. |
| The chirality check (`deeperfly.chirality`) | Flags a pose whose left/right identities look swapped. A library call, not part of any stage — run it over a 3D pose when you want the sweep | The one labeling error that costs nothing in any point-cloud metric has nothing that can find it. |

Omitting the key switches all three off — correct for an asymmetric subject, wrong
for a fly. `deeperfly.skeleton.infer_symmetries_by_name` proposes pairs from name
tokens (`l*`/`r*`, `*_L`/`*_R`, `left_*`/`right_*`); the packaged skeleton writes out the
pairs that inference proposes (16 for `fly38`, against the retired DeepFly3D set's 19 —
the difference is the abdomen, which `fly38` puts on the midline where it has no mirror
partner) rather than relying on it, so that renaming a point cannot quietly re-pair the
skeleton.

Editing the pairs is a **non-destructive** skeleton migration: no label moves and no
sidecar is rewritten, but the change is still reported, because it changes what the
three consumers above do.

## `[cameras.*]` — rig geometry { #cameras }

Each `[cameras.<name>]` is a geometric view: pure intrinsics + extrinsics, no
footage. `[cameras.defaults]` is merged into every view; per-view tables override
it. A view's intrinsics describe the raw frame of the source feeding it.

The packaged rig declares **eight** views as of 0.2: `rh`, `rm`, `rf`, `f`, `lf`, `lm`,
`lh` — each setting just its `azimuth_deg` over the shared defaults — and `h`, the axial
**hind** camera, which sets its own focal length and distance as well.

!!! warning "The eighth view, `h`, is on a different lens — measure its numbers per rig"

    `h` is the rig's only left/right **bridge**, and the reason the shipped detector was
    trained on eight views: without it no camera sees both body sides, so a contralateral
    joint is triangulated from one side's cameras alone.

    Being on a different lens changes **both** its focal length and its distance. The
    packaged numbers are 24168.591 px at 158.9168, against the side cameras' 22388.125 at
    107.463 — the example rig's, and yours will differ. No residual will find the error for
    you: bundle adjustment holds the intrinsics fixed, so it absorbs a focal error into the
    distance and then reports a clean solve. (The foam ball in the frame is a ruler for
    exactly this.)

    A recording **without** this camera needs no edit here — see
    [narrowing](#narrowing-to-the-footage-present) below.

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

**Non-geometry keys.** A `[cameras.<name>]` table also carries per-view keys owned
by other stages, which are stripped before the rig is parsed:

!!! warning "`preprocess` was retired in 0.2 and is now refused"

    A per-camera `preprocess` list used to crop and turn a view's frames. Frame ops belong
    to the detection [pathway](#pose2d) instead, and the difference decides which of the two
    a run can have: a pathway's ops are **inverted on the way back**, so a detection reaches
    its camera in raw footage pixels however it was windowed to get to the model, and the
    camera's intrinsics go on describing the raw frame. The retired key moved the *camera*
    into cropped-pixel space. Honoring both would double-correct by exactly the crop offset
    — a fly reprojecting off with nothing in the output to point at.

    It was accepted and silently ignored for several releases, which is the worst place for
    it: a crop is exactly what a badly-framed axial camera needs, so the key that did
    nothing was the one to reach for where being wrong costs most. It is now a hard error
    naming its replacement.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `input` | str | *unset* | Footage glob for this view. |
| `mirror` | str | *unset* | The view that sees **this view's mirror image**. |

`mirror` is what makes flip augmentation honest. The animal is bilaterally
symmetric, so a horizontally-flipped right-side view is a valid left-side view —
which is what makes the augmentation legal at all, and why the flipped sample must
be relabeled with the *mirrored* camera: a metric that splits ipsilateral from
contralateral error reads the camera id, so without the remap it reports every
swapped channel under the wrong side.

The relation must be **symmetric**, and a camera on the midline names itself — on the
packaged rig both `f` and `h` do, a flipped axial view still being an axial view. A
half-declared pairing is rejected rather than tolerated — it reads as correct and behaves
as a silent side swap on one camera. A view that declares no `mirror` is left unremapped.

It is declared rather than derived from the extrinsics because which camera mirrors
which is a fact about how the rig was built, and it stays true before there is a
calibration to compute it from. (For the packaged rig it happens to be `-azimuth`,
but that is a coincidence of a symmetric layout, not something another rig
inherits.)

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

A calibration that covers only **some** of this config's cameras is not an error either: a
view the rig never measured cannot be placed, so it is dropped like a view with no footage
and the run says which and how many are left. One config routinely describes more rig than
one solve covers — a project whose earlier recordings predate a camera being added, say.
Covering **none** of them still refuses: that is the wrong-rig case, and it is the one a
subset cannot explain away. Extra cameras the run does not use are reported and ignored.

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
mirror = "lh"

[cameras.h]                       # the axial hind view, on its own lens
azimuth_deg = 180
focal_length_px = 24168.591
distance = 158.9168
mirror = "h"
```

#### A run narrows itself to the footage present { #narrowing-to-the-footage-present }

One config also routinely describes more rig than one *recording* holds, and a recording
missing a camera is not malformed. Rather than refuse it or invent the footage,
`Config.narrowed_to_sources` narrows the run, in dependency order:

- a `[[sources]]` entry that resolves **no files** is dropped, and with it every
  `[[pose2d.pathways]]` reading that source (a source may feed several, so this is not
  one-to-one);
- a `[cameras.<name>]` view **no surviving pathway feeds** leaves the rig. This is the one
  that shortens the `V` axis, and it has to: dropping a pathway alone would leave a view
  whose 2D is all-NaN, which reads as a detected-and-empty camera rather than an absent
  one — and which bundle adjustment would then export into `calibration.toml` at its
  unrefined nominal pose with nothing marking it unmeasured;
- `[pose2d.output_points.<view>]` rows naming a dropped view or pathway go with them;
- an `{ auto = true }` `[[pose2d.preprocessors]]` no surviving pathway uses is dropped,
  because an orphaned automatic crop is otherwise a hard error;
- `[[visualization.videos]]` grid cells naming a dropped view are **blanked** (`""`) rather
  than removed, so the montage keeps its shape and the remaining cameras stay in the cells
  the reader expects; explicit `panels` naming one are dropped.

Point-name sections (`[bundle_adjustment].points_to_use`, `[postprocess].ops`,
`[inverse_kinematics]`) are view-agnostic and untouched. One warning names the sources, the
pathways and the views, and the view count it is proceeding on.

Narrowing happens **before** the snapshot and the fingerprints, and only the parsed data
narrows — the snapshot goes on recording what was *asked for*, exactly as a `[skeleton]`
preset reference does. That split is what keeps the cache honest: a run that proceeded on
seven views records a **seven-view** fingerprint and recomputes when the eighth camera
turns up.

Below **two** views it refuses (`config.MIN_VIEWS_FOR_3D`). One view fails *silently*
otherwise: `triangulate` and `triangulate_ransac` both return all-NaN without raising,
RANSAC gives a single observation zero inliers and then erases it, and bundle adjustment
reports success at a cost near `1e-26`. A one-view run produces a confident-looking
nothing, so a refusal is the only honest answer.

## `[pipeline]` — which stages run { #pipeline }

One `do_<stage>` boolean per stage. Each enabled stage reads its own `[<stage>]`
table.

The **Default** column below is `deeperfly.config.STAGE_DEFAULTS` — what a config that
omits the flag gets — and as of 0.2 the packaged `default_config.toml` states every flag at
exactly that value. So the two agree: the packaged file is explicit for readability, not
because it changes anything.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `do_pose2d` | bool | `true` | Detect 2D pose in every view. |
| `do_bundle_adjustment` | bool | `true` | Refine the cameras. |
| `do_pictorial_structures` | bool | `false` | DeepFly3D-style peak recovery — the one stage off by default ([why](#pictorial_structures)). |
| `do_triangulation` | bool | `true` | Triangulate 2D → 3D. |
| `do_eks` | bool | `true` | Smooth the 3D pose with the ensemble Kalman smoother. |
| `do_postprocess` | bool | `true` | Apply the `[postprocess].ops` correction chain to the 3D pose. |
| `do_inverse_kinematics` | bool | `true` | Fit NeuroMechFly joint angles to the 3D pose. Skips with a logged reason when the optional [`ik` extra](#inverse_kinematics) is absent. |
| `do_visualization` | bool | `true` | Render the videos. |

`do_pose2d = false` reconstructs from a **cached** 2D pose without re-detecting. The 2D
cache always feeds downstream, while a derived stage's cached output is used only while
that stage is on. An enabled stage whose input is unavailable is skipped, with the reason
logged, rather than failing the run.

!!! note "Turning a stage off from the command line"

    `deeperfly config set pipeline.do_<stage> false` works as of 0.2, and did not before:
    the writer appended an override table, TOML forbids declaring a table twice, and the
    packaged config states every `[pipeline]` flag — so every stage flag was unsettable in
    exactly the file people edit. An **existing** key is now rewritten in place, which
    reorders nothing and preserves the comments.

    A genuinely *new* key under an already-declared table is still refused, and that is not
    the same bug: appending a bare `key = value` after an existing `[section]` header would
    reparent every key below it. The error says to add it by hand.

## `[pose2d]` — 2D detection { #pose2d }

The `[pose2d]` table holds the detector's performance knobs *and* (as sub-tables)
the detection plan — what to detect and how.

**A plan is DENSE.** Both remaining detector classes emit every tracked point for every
view, so channel `i` is point `i` of the pathway's view and **no mapping table is needed**:
one pathway per camera, `n_out_channels` defaulting to the skeleton's point count, and
`input_size` plus the normalization coming from the checkpoint. Selecting a detector is two
lines:

```toml
[pose2d]
model = "dense38mv"
models = [
    { name = "dense38mv", class = "mvt", weights = "mvt_r28_pad48_gray_fly38.pth" },
]
```

The [`[pose2d.output_points]`](#output_points) mapping table still exists and is still
supported — it is how one view can be fed by several pathways — but the shipped configs
declare none. (The historical 19-channel detector predicted one body side, ran each side
camera twice and needed 122 hand-written mapping rows; a contralateral point now arrives as
a prediction to correct rather than a gap to author from nothing.)

**Performance knobs:**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `precision` | str | `"float16"` | Forward precision *default*, overridable per model: `"float32"` (reference), `"float16"` (CUDA autocast, ~1.5–2× faster), `"bfloat16"` (wider range). Ignored on CPU/MPS, and ignored by a class that pins its own — `mvt` runs in float32 and refuses anything else. |
| `batch_size` | int | `16` | GPU forward batch (images per forward). Clamped to ≥ 1; throughput plateaus by ~16 on a fast GPU. |
| `decode_buffer` | int | `4` | Decode queue depth, in multiples of `batch_size`. Clamped to ≥ 1. Peak frames/camera ≈ `(decode_buffer + 2) * batch_size`. |

**`batch_size` is in images, not frames.** `detect_sequence` forwards
`batch_size // pathways` whole frames at a time, so on the packaged eight-camera rig
anything below 8 is **one frame per forward** — which is why the default is not smaller.
Measured on an RTX 4090, 8 views at 256×512, on the **r27** multiview transformer — the
packaged r28 default pads its input and is slower by about 1.9× throughout (24.4 fps against
46.3 at batch 16, measured on an RTX 4080):

| `batch_size` | 2 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| fps | 49.6 | 52.9 | 59.4 | 60.1 | 58.9 |

The transformer is compute-bound (10.0 ms/frame at batch 8 against 11.1 at batch 1), so
the knob plateaus by 32. It was worth nothing at all until host preparation stopped
blocking the GPU: before that the same sweep read 31 fps flat at every value, because the
bottleneck was a YUV→RGB conversion, a device round trip and a PIL grayscale that no batch
size touches. A batch knob buys throughput only once the thing it feeds is the bottleneck.

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
| `crop` | `x`, `y`, `width`, `height`, or `auto` | Keep a window — written down, or [searched per recording](#autocrop). |
| `resize` | `scale`, or `width`/`height`; optional `interpolation` (`"bilinear"`/`"nearest"`) | Rescale. |

Detections are mapped back into the raw frame by inverting these ops, so a
preprocessor never moves the stored 2D or the reconstructed 3D.

<a id="autocrop"></a>
### The searched crop: `{ op = "crop", auto = true }`

A detector is trained through a box, and a differently framed camera puts the animal at the
wrong apparent scale — the one thing no augmentation in the recipe undoes. On this rig the six
side cameras match training full-frame and the two **axial** ones (front and hind, 1600×1008
against the side cameras' 960×512) do not, so those are the two that usually need a box. The
packaged config gives both `f` and `h` a `{ op = "crop", auto = true }` preprocessor. `auto`
says the box should be *measured* for this recording rather than copied from the last one:

```toml
[[pose2d.preprocessors]]
name = "crop_h"
ops = [{ op = "crop", auto = true }]                       # blind: cover the whole frame

[[pose2d.preprocessors]]
name = "crop_f"
ops = [{ op = "crop", auto = true, x = 400, y = 290, width = 800, height = 400 }]  # seeded
```

The four box keys become a **seed** when `auto = true` — all four or none. A seed is not the
answer, it is where the search starts, and it narrows the search to its neighborhood.

The `pose2d` stage resolves it before detecting anything: the detector's own confidence covers
the `(center, width)` space at ~11 ms a probe, then **agreement with the other cameras' 3D** —
the target view held out of the triangulation — chooses among what confidence proposed and
refuses a box that is confidently wrong. Measured blind on a 1600×1008 hind camera: 255 px
from the other cameras' 3D at full frame, **2.9 px** after the search, against 3.4 px for a box
tuned by hand. The searched box is recorded in `<outdir>/autocrop.json` and reused by later
runs, so a resume neither re-searches nor re-detects; `deeperfly auto-crop` runs the search on
its own and prints the TOML that freezes it permanently.

**The SEARCH needs no rig; the ACCEPT GATE does.** These are worth keeping apart, because the
short version ("auto-crop needs a solved rig") is wrong and has been stated that way before.
The grid over `(center, width)` is scored by detector **confidence**, which needs nothing but
footage — so `auto = true` is worth turning on from the first run. What needs a solved rig is
the gate that *chooses* among what confidence proposed: reprojection agreement with the other
cameras' 3D.

Confidence cannot be the accept criterion on its own. Once the center is free it decouples from
accuracy: one recording went 0.338 → 0.403 confidence and 30.9 → 37.5 px **worse**, and over a
center-y × width grid on this rig's hind view the two signals are uncorrelated (`r = +0.17`).
The `r = -0.92` that would justify optimizing confidence holds only along a width ladder at a
fixed good center.

The gate self-checks rather than trusting whatever rig it is handed. Against a **nominal orbit**
rig the reference for this recording's hind view measured ~250 px out, where a solved one
measured 3 — so before using it, the rig is asked to explain the views the reference was
triangulated *from* (`autocrop.RIG_RESIDUAL_LIMIT`, 25 px). A rig that cannot reproject into the
cameras it was built from cannot be believed about a held-out one. Failing that check, the
search says so and falls back to **confidence alone**, which comes out a box ~1.7× too wide —
still far better than dropping the whole frame in, which is the 255 px collapse above.

So: run once, `deeperfly calibration export`, point [`[cameras].calibration`](#cameras) at the
result, and the gate engages on the next run. `gate = false` in
[`[pose2d.autocrop]`](#pose2d-autocrop) takes confidence's box unchecked, for a rig that will
never have a calibration.

Anything that asks an unresolved automatic crop for its geometry raises
`UnresolvedAutoCrop` rather than quietly falling back to the whole frame — including a
visualization panel with `crop = "pose2d"` in a run where `pose2d` never ran and no box was
recorded.

### `[pose2d.autocrop]` { #pose2d-autocrop }

Knobs for the search above. Its stencil (how many widths, how many centers, how many rounds)
is measured and fixed in code; these are the parts a recording can genuinely need to differ on.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `search_frames` | int | `3` | Frames the confidence objective is scored on, spread over the whole recording. Raise it when the animal's distance drifts a lot through a run. |
| `gate_frames` | int | `8` | Frames the geometry gate is scored on — disjoint from the search's. |
| `gate` | bool | `true` | `false` takes confidence's box unchecked. Measurably unsafe on its own; for rigs with no usable calibration. |
| `gate_candidates` | int | `4` | How many confidence finalists the gate scores before searching from the best. |
| `gate_evals` | int | `30` | Cap on the gate's detection passes (~350 ms each) — what bounds the wall clock. |
| `probe_batch` | int | `8` | Forward batch in probes. Performance only. |
| `agreement_warn_px` | float | `20.0` | Agreement above which a view is reported as *still* not framing the animal. |

### `[[pose2d.models]]`

A detector network and its input contract. In practice only `name`, `class` and
`weights` are written — everything else is a property of the checkpoint.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | *required* | Model identifier (referenced by pathways). |
| `class` | str | *required* | Network registry key: `"hrnet"` (= `"hrnet_timm"`) or `"mvt"` (= `"multiview_transformer"`). Both dense; see below. An unrecognized spelling is **refused** by `class_defaults`. |
| `weights` | str | *required* | A bare filename found on `$DEEPERFLY_MODELS`, or an outright path. Nothing downloads. |
| `input_size` | [int, int] | from the class | `(height, width)` of the **reported frame**: frames are resized to it and peaks scaled back against it. Not always the network's own input — an `mvt` artifact declaring `arch.hm_margin_px` is fed `input_size + 2 × margin` (352×608 for the packaged default), padded inside the class. |
| `mean` | float | from the class | Scalar subtracted after `/255` normalization. |
| `n_out_channels` | int | from the class | Output heatmap count (validated against the weights). |
| `precision` | str | from the class, then `[pose2d].precision` | `float32` / `float16` / `bfloat16`. |

#### The two classes { #model-classes }

| `class` | What it is |
| --- | --- |
| `"hrnet"` | The dense **per-view** detector: each camera is predicted on its own, 38 heatmaps out. The same loader also runs the HGNetV2-B4 checkpoint, selecting its feature maps by **stride** rather than by index — a hard-coded `out_indices` is wrong for any backbone that does not start at stride 2, and would hand the head strides 8/16/32/32 and then evaluate a plausible, wrong model. |
| `"mvt"` | The **multiview transformer**: it encodes a frame's views *together*, so a joint only one camera can see informs the cameras that cannot. Zero per-view parameters, so any `V` in any order. Pinned to float32. |

The stacked-hourglass DeepFly2D detector is **gone** in 0.2 — no `class = "hourglass"` /
`"deepfly2d"`, no `deeperfly.pose2d.model`, no `pose2d.weights`, no public
`deeperfly.load_detector`, and no auto-download of any kind. An unknown `class` is now
refused where it is named rather than left to fall back: a typo used to inherit DeepFly2D's
19 channels and 0.22 mean, then fail at load with a channel-count mismatch that said nothing
about the word that was actually wrong.

#### What the class already knows { #model-class-defaults }

The last four keys are not preferences: each is a property of the network whose loader
**already refuses to run against a disagreeing config**. Writing them was the config
restating the artifact under threat of rejection, so the class states them instead
(`deeperfly.pose2d.models.CLASS_DEFAULTS`) and a table only speaks up to override.

| class | `input_size` | `mean` | `n_out_channels` | `precision` |
| --- | --- | --- | --- | --- |
| `hrnet` | `(256, 512)` | `0.0` (the checkpoint carries its own) | the skeleton's point count | inherits `[pose2d].precision` |
| `mvt` | `(256, 512)` | `0.0` | the skeleton's point count | `float32` (bf16 moved 99.6% of cells) |

`n_out_channels` defaulting to the skeleton's point count *is* what "dense" means, so a
dense config stops restating its own skeleton's size and cannot get it wrong. A
checkpoint that records the input size it was trained at also contradicts a disagreeing
`input_size` at load, because a mis-resized fly arrives at the wrong scale rather than
crashing.

#### One grayscale input plane { #one-plane }

Every shipped detector is trained on a **single** channel — the corpus is monochrome — and
`LoadedModel.prepare` emits `(..., 1, H, W)`. A frame arrives either as `(..., H, W)` from
the decoder's gray fast path or as `(..., H, W, C)` whose planes are identical (the decoder
only takes the gray path when they are), so the first channel *is* the image and nothing is
replicated to three to feed a network that would only take one back. HRNet's `in_chans` is
`1` unconditionally, and a three-channel checkpoint therefore fails on the stem at load
rather than running.

`mvt.ARTIFACT_FORMATS` is `("deeperfly-mvt-2",)` only. The three-channel `deeperfly-mvt-1`
is retired, and dropping it costs no accuracy: a `-2` is its patch-embedding stem folded
onto one plane, which is an **exact** fold — the same function, not a retrained model. The
format version has to be the guard, because a `-1` fed one plane is a shape error PyTorch
would raise, while a `-2` whose scalar normalization was applied as if it were ImageNet's
per-channel one would *run* and be quietly wrong.

!!! note "`accepts_gray` on `_HRNetPose` was a bug fix, not a new feature"

    The gray decode path is only taken when **every** model in a plan declares it accepts
    one plane — one model that needs color forces the whole plan to decode color, since the
    models share the decoded window. `_HRNetPose` did not declare it, so grayscale was in
    effect for *no* `hrnet`/`hgnet` plan: every such run paid a YUV→RGB conversion, ~90% of
    what a decoded frame costs, to produce three identical planes and then use one.

#### Where `weights` is looked up { #weights-resolution }

**Nothing auto-provisions.** Every detector deeperfly runs is trained per project, so
`weights` is required and there is no checkpoint to fall back to. Three forms, and the
distinction is whether the value looks like a *path*:

| Value | Meaning |
| --- | --- |
| `""` / omitted | Refused, and the error *is* the setup instructions — it prints both the `$DEEPERFLY_MODELS` form and the explicit-path form. |
| `my_detector.pth` (a bare filename) | Searched along `$DEEPERFLY_MODELS` (`os.pathsep`-separated, like `PATH`), then the per-user cache directory. |
| `/path/to/x.pth`, `./x.pth`, `~/x.pth` | Used as written. Anything containing a path separator, absolute, or starting `~`. |

Prefer the bare filename: *which model a run used* is a fact about the recording and
travels with it, where `/mnt/...` is a fact about one mount on one machine and breaks the
moment the config is opened anywhere else. A failed lookup prints every directory it
searched, because "which of these did you mean" is the only question at that moment.

The per-user cache (`platformdirs.user_cache_dir("deeperfly")/weights`) is no longer written
to by anything, but it stays on the search path — it is the one place a checkpoint can be
dropped without setting an environment variable.

#### The released checkpoints { #weights }

Three checkpoints ship for 0.2. All are **one-channel**, all record the **`fly38`** point
order, all report points in a 256×512 frame and emit 38 heatmaps per view. The two
single-view detectors take that 256×512 directly; the multiview transformer pads it to
352×608 internally for its 48 px margin, which changes nothing a config can see.

The single-view pair were trained on **55 recordings / 465 moments / 138,708 label
cells**; `mvt_r28_pad48_gray_fly38` on **55 / 485 / 144,775** — the same corpus after a
later round of labeling — every recording carrying a camera rig traceable to hand labels.

| Checkpoint | `class` | Bytes | sha256 (first 8) |
| --- | --- | --- | --- |
| `mvt_r28_pad48_gray_fly38.pth` | `"mvt"` | 86,082,205 | `ae482d3a` |
| `hrnet_w32_r27_gray_fly38.pth` | `"hrnet"` | 127,076,045 | `13ceb937` |
| `hgnetv2_b4_r27_gray_fly38.pth` | `"hrnet"` | 62,452,371 | `fa427062` |

```
ae482d3a3c117dc48562394fd33b04710e8fa530fc13899c5fe0933061e7e14d  mvt_r28_pad48_gray_fly38.pth
13ceb93793655f6e974083072629a10e242608ab0d839e1f60659050c5926c9b  hrnet_w32_r27_gray_fly38.pth
fa4270628f5766461ad3ac358730d6835cd39bd86cba6b45b98303609c4e7932  hgnetv2_b4_r27_gray_fly38.pth
```

They live at `/mnt/upramdya/data/TL/deeperfly-models/260819_*`, except the MVT under
`260825_mvt_r28_pad48_gray_fly38` — one directory each, holding
the `.pth` plus a `README.md`, a `SHA256SUMS` (so `sha256sum -c SHA256SUMS` verifies the
file before you trust a whole project to it) and the `fly38.toml` the model was trained on.
That last one is byte-identical to the packaged skeleton, which is what lets the load-time
channel-name check be a real check rather than a ritual.

The packaged config names **`mvt_r28_pad48_gray_fly38.pth`**: the transformer sees a frame's
views together, which is what recovers a contralateral joint no single camera resolves. The
two `hrnet` checkpoints are per-view and interchangeable with it in a config — only the
`class` and the filename change.

**Install them by pointing at them, not by copying paths into the config:**

```console
$ export DEEPERFLY_MODELS=/mnt/upramdya/data/TL/deeperfly-models/260825_mvt_r28_pad48_gray_fly38
$ deeperfly doctor
```

`doctor` prints `$DEEPERFLY_MODELS`, every directory on the search path with how many `.pth`
files are in it (or that it is empty or absent), and whether the checkpoint the default
config names actually resolves — which is the question a stuck user has, now that "is it
downloaded" is not a question any more. Several directories can be listed, separated by
`os.pathsep`, exactly like `PATH`.

!!! note "There is no held-out accuracy number behind a ship model, on purpose"

    A ship checkpoint trains on **every** labeled recording, and the trainer selects its
    epoch on the probe — so the probe *is* the validation set and its numbers read as "did
    it train", not as accuracy. What stands between a broken checkpoint and a whole project
    is the gate each one passed, recorded in its `README.md`: probe mean 4.677 px / median
    1.808 / PCK@10 93.85% over 13,040 cells for `hrnet_w32`, 4.968 / 1.923 / 93.49% for
    `hgnetv2_b4`, and for the transformer three structural gates instead — permuting the
    views permutes the outputs and nothing else (worst 2.3e-06, 0.008% of range), a real
    view's output is bit-identical whatever the padding holds (and the mask is load-bearing:
    1.8e-03 leaks without it), and the exported artifact rebuilds the same detector.

### `[[pose2d.pathways]]`

A named `source → preprocessor → model` inference run. Says *what to detect on*.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | str | yes | Unique pathway identifier (referenced by `output_points`). |
| `source` | str | yes | The `[[sources]]` name to detect on. |
| `model` | str | no | The `[[pose2d.models]]` name to use; omit for [the default](#pathway-model). |
| `preprocessor` | str | no | A `[[pose2d.preprocessors]]` name; omit for identity. |

#### The model a pathway takes when it names none { #pathway-model }

A dense plan is one pathway per camera through **one** detector, so writing the model on
every pathway was the same string repeated once per view, existing only to point at the
single entry above it. It is hoisted instead:

```toml
[pose2d]
model = "dense38mv"                  # the default for every pathway below
models = [
    { name = "dense38mv", class = "mvt", weights = "mvt_r28_pad48_gray_fly38.pth" },
]
pathways = [
    { name = "rh", source = "vid_rh" },
    { name = "f",  source = "vid_f", preprocessor = "crop_f" },
    { name = "h",  source = "vid_h", model = "axial" },   # this one overrides it
]
```

Resolution, in order: the pathway's own `model`, then `[pose2d].model`, then — when
`models` has exactly **one** entry — that entry. So a single-detector plan need not name
it anywhere.

The last step is deliberately *not* "the first model". Adding a second entry to a plan
whose pathways are bare raises an error naming both, rather than silently deciding which
camera runs which detector:

```
[[pose2d.pathways]][0] ('rh') names no model and there is no default: this plan
declares 2 models (['axial', 'dense38mv']), so write 'model' on the pathway, or
[pose2d].model to set one for all of them
```

### `[pose2d.output_points.<view>]` { #output_points }

For each view, where every tracked point's data comes from.

**Optional, and omitted by every shipped config.** A dense plan needs no mapping — channel
`i` is point `i` of the pathway's own view — so the table's default *is* the identity. It is
still parsed and still supported, because it says one thing the identity cannot: that a
single view is fed by **several** pathways. `tests/data/fly38_sparse_config.toml` keeps a
worked example (a 19-channel one-body-side plan, 122 rows) as the fixture the per-view
visibility and mirror-check tests are written against.

A table keyed by point name:

```toml
[pose2d.output_points.rh]
rf_thorax_coxa = { pathway = "rh", out_channel = 0 }
```

`point = { pathway, out_channel }` fills that point of the view from output
channel `out_channel` of the named pathway. Keying by `(view, point)` means each
point has exactly one source (a repeat is an error); a `(view, point)` left out
stays unobserved (`NaN`). That union is the visibility.

**The mirror check.** These tables are also where the plan's left/right identities
are decided. The side cameras all feed one *side-agnostic* model and the left-side
views reach its convention through a `fliplr` preprocessor, so whether a channel
means a left or a right joint is settled here and nowhere else — 122 hand-written
rows with, historically, nothing checking them.

When the skeleton declares [`symmetries`](#symmetries) the invariant becomes
decidable and is enforced at load: for each `(model, channel)`, every point an
**un-mirrored** pathway maps it to must be the **symmetry partner** of every point
a **mirrored** pathway maps it to. A pathway counts as mirrored when its
preprocessor reverses handedness — an *odd* number of `fliplr`/`flipud` ops, so
`fliplr` + `flipud` is a half-turn and reverses nothing.

```
ValueError: [pose2d.output_points] disagrees with [skeleton].symmetries about left/right:
  - model 'sided19' channel 2: un-mirrored -> 'rf_femur_tibia' ('rf' -> view 'rf')
    but mirrored -> 'rf_femur_tibia' ('f_flip' -> view 'f'). Mirroring the frame
    mirrors the animal, so the mirrored pathway must land on 'lf_femur_tibia' --
    as declared in [skeleton].symmetries.
```

The check is skipped when the skeleton declares no pairs (there is nothing to check
against), and per channel when only one parity maps it — a one-sided rig's pathways
are all un-mirrored, which is legal. One channel feeding several points also stays
legal: `output_points` constrains where a *point's* data comes from, not a channel's
fan-out.

## `[bundle_adjustment]` — camera refinement { #bundle_adjustment }

Fly-as-target bundle adjustment over `scipy.optimize.least_squares`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `points_to_use` | list[str] or omitted | *unset* → all keypoints | Skeleton point names that drive BA. The packaged config names the 30 leg points: they sit at sharp limb corners and detect most reliably. |
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

Runs only when `do_pictorial_structures = true`, and it is the **one** stage still off by
default. Operates on the detector's top-K candidates (extracted and cached during
detection).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `k` | int | `5` | Candidate peaks per joint. |
| `temporal` | bool | `false` | Add a temporal-consistency term. |
| `lam` | float | `1.0` | Bone-length prior weight. |

**Why it stays off, and it is not for symmetry with the others.** It recovers a joint from
the top-K peaks of a detector that predicted **one body side**; a dense detector already
predicts every point in every view, so there is nothing to recover. Switching it on rewires
triangulation *and* the smoother onto its committed 2D, which is `NaN` in every view with
no candidate within 15 px (`deeperfly.pictorial.DEFAULT_INLIER_PX`) — so on a dense run it
would silently *un-densify* the result. It also adds a `candidates` key to the `pose2d`
fingerprint, re-detecting every cached tree in existence.

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

## `[eks]` — ensemble Kalman smoother { #eks }

**On by default** as of 0.2 (`do_eks = true`), after triangulation and starting from its
3D. Where triangulation solves each frame on its own, this fits **one 3D trajectory per
keypoint to the whole recording at once**, with the rig's own projection as the observation
model — the nonlinear multi-view EKS of Lightning Pose 3D (Aharon, Whiteway et al.
2026), implemented natively in deeperfly's JAX (see
[`deeperfly.eks`](api.md#ensemble-kalman-smoother)).

Two things follow. It **de-jitters**, because a random-walk prior on a 3D point costs
almost nothing to satisfy and per-frame detector noise does not survive it. And it
**repairs blown detections**, because a view whose 2D disagrees with the other views
has its observation variance inflated until the smoother stops believing it — so the
3D point stays on the animal, and that view's reported 2D becomes the trajectory
reprojected. On a synthetic seven-camera rig with 2% gross outliers, the inflation
alone cuts the worst-case 3D error by more than 30×.

Its output supersedes triangulation's for `[postprocess]`, inverse kinematics, the rendered
videos and `PoseResult.load`. Cost is roughly 20 s per 5000 frames × 38 keypoints on a CPU.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `smooth_param` | float | *(fitted)* | Process-noise scale: smaller smooths harder. Omitted, it is fitted per keypoint by maximum marginal likelihood — usually the right call, since a claw and a thorax do not move alike. |
| `inflate_vars` | bool | `true` | Test each view against the other views and down-weight the ones that disagree. Needs two **views**, not two models, so it is fully active with a single detector. This is the component that repairs outliers; leave it on. |
| `inflate_threshold` | float | `5.0` | Mahalanobis distance at which a view is called inconsistent. Lower is more suspicious — and 5 over-flags, so the packaged config raises it to `30.0`. See the calibration caveat and the measured sweep below. |
| `inflate_factor` | float | `10.0` | Variance multiplier per inflation round (the paper describes doubling; the reference CLI ships 10). |
| `ensemble` | list[str] | `[]` | Other `results.h5` files holding a **different detector's** 2D for this same recording (paths relative to the output dir). |
| `avg_mode` | str | `"median"` | How ensemble members combine: `"median"` (robust) or `"mean"`. |
| `var_mode` | str | `"confidence_weighted_var"` | `"var"` is the plain across-model variance; the default divides it by the mean confidence. |
| `fill_unobserved` | bool | `false` | Report the smoothed 3D reprojected into views that never saw the keypoint, instead of leaving those cells `NaN`. |
| `fit_frames` | int | `2000` | Leading frames the `smooth_param` fit may use (0 = all). Smoothing always covers every frame. |
| `fit_iterations` | int | `24` | Golden-section steps per keypoint. |

**With one detector, the "ensemble" half is inert.** The across-model spread is
identically zero, so the observation variance falls back to `1 / confidence` — a
prior, not a measurement. The geometric smoother and the variance inflation are
unaffected (they need views, not models), which is why the stage is still worth
running. But it does mean `inflate_threshold` is not an absolute false-alarm rate:
what fraction of observations it flags moves with the detector's confidence
calibration, so watch the logged `variance inflation down-weighted N of M` rather than
trusting the number 5. Pointing `ensemble` at a second detector's run is what turns
that term into a measurement.

**Where the temporal prior stops helping.** The latent is a *position* random walk and
the update is a single Gauss-Newton step, both as published. The accuracy gain is
therefore confined to keypoints moving no faster per frame than the detector can
localize them — a tethered fly's body and proximal joints, not a claw mid-swing. On a
target moving several times that floor the fitted parameter backs the prior off, but
about 10% of median lag survives and raising `smooth_param` does not remove it. The
de-jittering and the outlier repair are unaffected.

**What it buys and costs, measured** on a 100 fps eight-camera recording of a tethered
fly. 3D jitter — median frame-to-frame acceleration — drops **39%**, 0.0062 → 0.0038.
Against that, 779 cells (0.13%) reproject more than 100 px from the detector's 2D, and
every one of them is a **claw**: exactly the lag above, not a tuning failure. Turn the
stage off if claw timing is the measurement.

**Choosing `inflate_threshold`.** Reprojection cannot tell you: a sweep over
5 / 10 / 15 / 20 / 30 on one recording moves the flag rate from 68% to 21% while the
reprojection distribution barely shifts (median 2.90–3.01 px, p99 ≈ 26.6 throughout).
That is by design — the smoother is *supposed* to leave the 2D where it judges the 2D
unreliable — so the threshold has to be judged against ground truth instead.

Measured against hand labels on 54 recordings (63,921 labeled cells, paired per
recording), 2D accuracy improves monotonically as the threshold rises and then collapses
when the inflation is switched off entirely:

| `inflate_threshold` | mean px | p90 px | cells > 50 px | jitter | frame-to-frame jumps |
| --- | --- | --- | --- | --- | --- |
| 5 (field default) | 4.916 | 9.103 | 426 | **0.0040** | **0.0171** |
| 15 | 4.746 | 8.648 | 394 | 0.0046 | 0.0179 |
| **30** (packaged) | 4.628 | **8.343** | 376 | 0.0047 | 0.0190 |
| 60 | 4.617 | 8.546 | **368** | 0.0044 | 0.0200 |
| 100 | **4.583** | 8.416 | 373 | 0.0044 | 0.0201 |
| off | 5.848 | 10.974 | 609 | 0.0049 | 0.0258 |
| *raw detector* | 4.927 | 8.541 | 532 | — | — |

Two things follow. The inflation itself is load-bearing — turning it off is worse than
any threshold, and worse than not smoothing at all. But 5 **over-flags**: a blown
detection sits at Mahalanobis ≈ 1e4, so every threshold here still catches it, and
lowering the bar only down-weights cells the detector had right. Accuracy is flat from 30
to 100 (the 30/60/100 differences are not significant), while frame-to-frame jumps grow
monotonically — so the packaged config takes **30.0**, the point where the accuracy gain
is fully realized and the temporal cost is smallest. Raise it toward 60 if outlier repair
matters more than jump rate; lower it toward 5 if the 3D feeds inverse kinematics and
smoothness matters more than pixels.

## `[postprocess]` — corrections from knowing the animal { #postprocess }

**On by default** as of 0.2 (`do_postprocess = true`), after the smoother — and the packaged
config declares the two ops below, which assume a **tethered** animal, so turn the stage off
for a freely-moving one. Everything upstream estimates the pose from **pixels**; this stage
applies what is known about the **animal** instead — that some keypoints do not move, that
the body is bilaterally symmetric. Those are priors
rather than measurements, and keeping them out of the estimating stages is deliberate: an
estimator already told the answer cannot be checked against it.

The chain is an array of tables, applied in the order written — the same shape as
[`[[pose2d.preprocessors]]`](#pose2d):

```toml
[[postprocess.ops]]
op = "static"
method = "median"
points = ["neck",
          "lf_thorax_coxa", "lm_thorax_coxa", "lh_thorax_coxa",
          "rf_thorax_coxa", "rm_thorax_coxa", "rh_thorax_coxa"]

[[postprocess.ops]]
op = "symmetrize"
pairs = [["lf_thorax_coxa", "rf_thorax_coxa"],
         ["lm_thorax_coxa", "rm_thorax_coxa"],
         ["lh_thorax_coxa", "rh_thorax_coxa"]]
midline = ["neck"]
```

**The stage runs after EKS, and that is forced rather than chosen.** The smoother
re-derives the 3D from the 2D observations, so a correction applied before it is followed
straight back off by the fit chasing its pixels. Applied after, it sticks. The honest cost
is that these priors *overwrite* the estimate rather than informing it — a constraint
inside the smoother would use them to weigh evidence, which is better statistics and a
larger change.

**The ops do not commute**, which is why the list is ordered. `static` then `symmetrize`
(with a single fitted plane) satisfies both properties exactly: a fixed plane maps a
constant to a constant. The other order does not — a per-axis median of two mirrored
points is not itself mirrored.

Each op logs how far it moved the points it touched, and its report is stored in the
stage's `meta` under `ops`, one entry per op in order. That number is the only check on an
op's premise.

### `{ op = "static" }` — keypoints that do not move { #op-static }

Collapses each listed point to **one position for the whole recording** — in 3D, and
independently per view in 2D. On a tethered fly the six thorax-coxa joints and the neck sit
on the sclerotized thorax, so their position is a constant of the recording and everything
the estimate does over time is per-frame noise.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `points` | list[str] | `[]` | Skeleton point names to hold static. A name not in the skeleton is a hard error. |
| `method` | str | `"median"` | Which center (below). |
| `trim` | float | `0.1` | `"trimmed_mean"` only: the fraction dropped from **each** tail; must be in `[0, 0.5)`. |

Note the joint choice: the skeleton has `*_thorax_coxa` (leg to thorax, rigid) and
`*_coxa_trochanter` (the next joint out, which genuinely rotates). There is no
"thorax-trochanter", so the rigid leg set is the six `thorax_coxa` and no more.

The five estimators differ only in what they assume about the contamination:

| `method` | What it computes | When it is the right one |
| --- | --- | --- |
| `median` | Per-axis temporal median. | The default, and right until you have a reason. 50% breakdown point, and within ~2% of the mean's efficiency at a recording's sample sizes. |
| `mean` | Per-axis arithmetic mean. | Minimum-variance **if** the residual really is clean Gaussian noise — worth having in order to check exactly that. A single blown frame drags it. |
| `trimmed_mean` | Mean of the values inside `[trim, 1 - trim]`. | Tunable between the two above. Trims by *value* threshold rather than exact order statistic, so ties may keep slightly more or fewer than the nominal fraction. |
| `mode` | Half-sample mode (Robertson–Cryer). | A contaminant that is the **majority**. It follows *density* where the median follows *count*: correct detections tight, wrong ones scattered. Below 50% contamination the median is already inside the true cluster and this buys nothing, at the cost of more variance. Parameter-free, unlike a histogram or KDE mode. |
| `geometric_median` | Multivariate L1 center (Weiszfeld). | The only **rotation-equivariant** option — the other four work axis by axis, so their answer depends on the world frame's orientation, an arbitrary property of the rig. |

Within a view that observes the point, the center also fills the frames where the
detection dropped out — that is the premise of calling the point static. A `(view, point)`
pair the view *never* observes stays `NaN`, so the op does not invent an observation in a
camera that cannot see the joint. And because the 3D and the per-view 2D are each collapsed
in their own space, the frozen 2D is not exactly the projection of the frozen 3D; the
alternative (reprojecting the frozen 3D) would be exactly consistent, at the cost of moving
these points off the pixels the detector saw by the rig's residual.

### `{ op = "symmetrize" }` — impose bilateral symmetry { #op-symmetrize }

Makes each left/right pair a mirror image across the animal's **sagittal plane**, and puts
each `midline` point on that plane. Each side is triangulated from its own cameras, so a
pair drifts apart by whatever the two sides' errors differ by; this closes that gap. 3D
only — a per-view 2D detection is a measurement in that camera's pixels, and there is no
sense in which two *different cameras'* pixels mirror each other.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `pairs` | list[[str, str]] | `[]` | Left/right pairs to symmetrize. |
| `midline` | list[str] | `[]` | Points lying **on** the plane, projected onto it. |
| `per_frame` | bool | `false` | Fit a fresh plane per frame instead of one for the recording. |
| `strength` | float | `1.0` | Scales the correction; 0.5 moves each side half-way to the mirror of the other, 0.0 is a no-op. |

!!! warning "Neither list is defaulted from `[skeleton].symmetries`, and that is not an oversight"

    - That list pairs the **legs** too, and at any instant a fly's left and right legs are
      in different gait phases. That asymmetry *is* the behavior being measured;
      symmetrizing it would be a serious corruption wearing the costume of a correction.
      Only body-fixed pairs belong here.
    - "Has no mirror partner" does not imply "on the midline". A fly's abdomen bends
      laterally, so `abdomen0`…`abdomen4` are unpaired and still off-plane. On `fly38` the
      honest midline set is `neck` alone.

The plane is fitted from `pairs` **only** — each pair's midpoint lies on it, each pair's
difference vector is normal to it — so a point named in `midline` never votes on where the
plane it is about to be moved onto should go, and a laterally-bent abdomen can never drag
it. Fewer than two usable pairs is under-determined (one pair fixes a normal but no offset
along it); that is reported and the pose is left alone rather than guessed at.

Both sides move: each goes half-way toward the mirror of the other, scaled by `strength`.
Averaging into one side would import that side's error into both.

`per_frame` defaults **off**, and that default is load-bearing. A single plane is a *fixed*
map, so it leaves an already-static point static and `static` → `symmetrize` satisfies both
properties exactly. Fitted per frame the plane wobbles with the estimate, so symmetrizing
after a freeze un-freezes it. Turn it on only for a preparation whose body genuinely moves
in the world frame.

!!! note "This replaces the IK solver's old private pin"

    Before 0.2, `[inverse_kinematics]` had a `constant_points` key that collapsed the
    listed points before the fit and affected nothing else — so no stage output recorded
    it, and the stored angles could disagree with the stored 3D for exactly those points.
    `{ op = "static" }` applies the same idea to the **result**, in both 2D and 3D, which
    is what makes it checkable: the op logs how far it moved what it touched, and a point
    that had been drifting tens of pixels was moving and belongs out of the list. The key
    is **gone** in 0.2 and is an unknown-key error; if you carry it in a config, move its
    points here. `[postprocess]` is on by default now, so for the packaged point set that
    is a move into a table that already runs.

## `[inverse_kinematics]` — joint angles { #inverse_kinematics }

**On by default** as of 0.2 (`do_inverse_kinematics = true`). Fits a NeuroMechFly-style
articulated model to the **most-derived** 3D pose available — with the default stage set
that is the `postprocess` chain's output, else the smoother's, else triangulation's. It fits
the six legs (with segment lengths **measured from the data**, so the fitted model matches
this fly's proportions), plus the **head** (yaw/pitch/roll, reaching the two antenna tips)
and the **abdomen** (a
five-segment chain reaching the abdomen markers, each segment bending **vertically and
laterally** — and never twisting about the abdomen's own long axis). The head and abdomen
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

    Without it the stage **skips**, with the reason logged, rather than raising — which
    matters more than it looks now that the stage is on by default: `inverse_kinematics`
    precedes `visualization` in the stage order, so an exception here would also cost the
    videos, after detection, bundle adjustment, triangulation, the smoother and the
    correction chain had all been computed and committed.

    A result file that *already* holds a fit needs nothing extra: its overlays, videos and
    the GUI's static fit all work on a plain install.

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
| `symmetric_segments` | bool | `true` | Give each leg and its mirror image **one shared length per segment** (the mean of the two sides' measurements) instead of measuring the two sides independently. On by default, and it **costs** measured accuracy — see below. |
| `weigh_by_confidence` | bool | `false` | Weigh each observation by the detector's confidence instead of treating every observed keypoint equally. |
| `parallel` | bool | `false` | Solve in overlapping segments across worker threads. Off by default: each segment restarts from the neutral pose and only warm-starts within itself, so the angle traces can step at a seam — a poor trade for a joint-angle time series unless the recording is long enough to need the speed. |
| `segment_len` | int | `200` | Frames per segment (includes the overlap); `parallel` only. |
| `overlap_len` | int | `10` | Frames shared with the next segment; `parallel` only. |

### Left/right symmetry — `symmetric_segments` { #ik-symmetric-segments }

The leg segment lengths are **measured from the data**: each is the median, over the
recording, of the distance between two triangulated joints. That is right in principle —
it fits *this* fly rather than the generic model — but it measures each leg on its own,
and a fly's left and right femurs are the same bone measured twice. `symmetric_segments`
(**on** by default as of 0.2) gives each mirror pair the **mean of the two measurements**,
which is the combination that does not privilege a side (the same argument
[`{ op = "symmetrize" }`](#op-symmetrize) makes for the body-fixed points, and the reason
it is a mean rather than a median pooled over both sides' frames: pooling would weight
the side with more triangulated frames). A segment measured on **one** side only adopts
its mirror's length — strictly better than the alternative, which is falling back to the
model's own generic bone. One measured on neither keeps that fallback.

Two things it deliberately does **not** do:

- **It does not couple the two sides' joint angles.** The constraint is on the *animal*,
  not on its pose: a leg's left/right asymmetry at any instant *is* the behavior.
- **It does not move the coxae.** Where a leg is attached is the body registration's
  business, and imposing symmetry on the *pose* is
  [`{ op = "symmetrize" }`](#op-symmetrize)'s — which is what the example configs use it
  for, on the thorax-coxa joints and never on the legs.

!!! note "It reads the skeleton's declared `symmetries`"

    Which leg mirrors which is derived from [`[skeleton].symmetries`](#skeleton) — the
    same declared relation the training mirror augmentation and the chirality QC read —
    and not from the `l`/`r` name prefix. A skeleton that declares no pairs is stating
    that its subject is not bilaterally symmetric, so nothing is shared and the stage
    says so in a warning rather than silently doing nothing.

!!! warning "What the default costs, measured"

    This is a **prior**, and it is the one default in this table that makes every score
    slightly worse. It is on anyway, because a fly has one pair of femurs rather than two
    independent bones and most uses of a fitted model want that to be true of the model —
    but the cost is real and is worth knowing before quoting a residual.

    On the eight-view example recording the two sides genuinely disagree — the femurs come
    out **4–6% apart on all three pairs, always with the left longer**, as a stable offset
    whose 10th–90th percentile bands do not overlap:

    | pair | measured left | measured right | gap |
    | --- | --- | --- | --- |
    | `lf`/`rf` femur | 0.616 | 0.583 | 5.5% |
    | `lm`/`rm` femur | 0.741 | 0.704 | 5.2% |
    | `lh`/`rh` femur | 0.730 | 0.701 | 4.1% |

    But sharing them makes **every** measurable score worse on that recording:

    | score | free lengths | shared | |
    | --- | --- | --- | --- |
    | 3D residual, model vs its target pose (median) | 0.0120 | 0.0146 | +22% |
    | 2D reprojection vs the `pose2d` detections (median px) | 4.45 | 4.59 | +0.14 px, worse in 7 of 8 views |
    | left/right gap in each DOF's median angle (mean over 24) | 4.7° | 6.4° | +1.7° |

    The third row is the one that hurts. The hope was that a length error the solve cannot
    express as length comes out as *angle*, so sharing the bones should make the two sides'
    angle statistics more comparable — the very thing sharing is for. Measured over 2007
    frames it does the opposite. All five points of a leg are tracked, so the chain is
    over-determined and the per-leg measured lengths already *are* the best fit to the data;
    any shared length can only move the model away from it.

    Nor is the gap a per-side rig artifact, which would have been the case for imposing
    symmetry regardless. Within-side body-fixed distances are symmetric to under 2%
    (`neck`→front coxa 0.996, `neck`→antenna 0.990, front→mid coxa 1.005), as are the
    coxa and tibia segments (0.98–1.01) — it is the **femur specifically**, on all three
    pairs. A per-side scale error would have inflated all of them together.

    So: leave it on when you want one animal rather than two half-animals — comparing joint
    angles between sides, driving a simulation, reporting a morphology — and set
    `symmetric_segments = false` when the number you are quoting *is* the fit's accuracy
    against your own keypoints, or when a per-side length difference is itself the
    measurement. Note that this cuts the other way too: a 5% femur difference that is *not*
    anatomy is a detector bias you would otherwise be reporting as biology, and the
    free-length fit absorbs it silently.

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
      that is a signal your `bounds` are what's limiting the fit.

!!! note "A pinned limit is often a symptom, not the cause"

    The warning's advice — widen it — is right less often than it reads, and the legs are
    the cautionary tale. They used to pin the `±50°` thorax-coxa yaw wall in **71%** of
    frames on the example recording, with three times the residual on those frames, and
    widening the wall did cut the leg residual 30%. It was still the wrong fix: the
    template had two DOF axes swapped and the leg subtree was rotated by the wrong body
    frame, so the fit was reaching the pose through the *mirror* branch of the
    `(yaw, roll)` double cover, where the required yaw is a further 90° out. Once the frame
    was right the same NeuroMechFly ranges stopped binding on their own — the worst-pinned
    leg DOF fell from 71% of frames to **0.7%** — and the residual fell 41%, more than
    widening ever bought.

    The tell was that the fit could not reproduce the *model's own resting posture* within
    the model's own limits (it saturated yaw on four of six legs trying). That is now a
    test, `test_the_leg_parameterisation_is_flygyms`. Before widening a limit, check
    whether the pose you are asking for is one the model considers rest.

    The abdomen has the same story from the other direction — see
    [the note below](#ik-chain-size) — where a lateral wall pinned in 54% of frames stopped
    pinning entirely once the marker geometry was corrected, without being touched.

A `[inverse_kinematics.bounds]` sub-table overrides per-DOF joint angle limits in
**degrees**. Keys are the flygym joint angle names `"<parent_body>-<child_body>-<dof>"`
(dofs `yaw`/`pitch`/`roll`) — the same names written to `results.h5` — e.g.
`"rf_trochanterfemur-rf_tibia-pitch" = [10, 160]` for a leg,
`"c_thorax-c_head-pitch" = [-30, 30]` for the head, or
`"c_abdomen12-c_abdomen3-pitch" = [-45, 20]` for the abdomen.

Each abdomen segment carries **two** DOFs, and by default:

| dof | axis | default range | what it is |
| --- | --- | --- | --- |
| `…-pitch` | segment-local *y* | `[-30, 0]` | **vertical** bend, ventral (downward) only |
| `…-roll` | segment-local *z* | `[-15, 15]` | **lateral** bend, symmetric |

Read those names as flygym's labels for the *axes*, not as descriptions of the motion: on
this chain flygym's axis names are anatomically rotated, so the axis it calls `roll` is the
one you would call yaw, and its `yaw` is the axial twist. Same on the legs.

Vertical bend is one-sided because the five near-midline markers under-constrain the
sagittal chain, so a symmetric range lets the solver fold it into a non-physical zig-zag
while a downward-only range keeps the fit a smooth ventral curl. Lateral bend is symmetric
because a fly bends either way; `[-15, 15]` is where the fit's 90th-percentile marker
residual flattens and the fraction of frames spent against the bound falls to ~6%.

There is deliberately **no twist DOF** — flygym's `…-yaw`, the segment-local *x*. It is
excluded because it **aliases** the lateral bend rather than being unobservable: every
marker on this chain lies in the sagittal plane, so both DOFs swing it sideways, and the
two displacement fields agree to cos 0.87–0.97 (14–29° apart) while a 20° twist still moves
the markers 41–88% as far as the same lateral bend. Fitting both would split the motion
between them arbitrarily. Successive bends can still compose to a net axial rotation —
that is geometry rather than a degree of freedom.

Lateral bend is not a formality. Measured on three animals across two rigs, the abdomen
markers leave the midline by **30–80× the frame-to-frame noise floor**, growing
monotonically toward the tip, and that excursion is 26–51% of the vertical one. A
pitch-only chain cannot represent any of it; adding the lateral DOF cuts the abdomen's
marker residual by 24–39%.

!!! note "If the abdomen fits as a straight, static rod, the bounds are not the cause"
    A chain that reports the *same* angles in every frame — the root at exactly `-30` and
    the interior hinges at exactly `0` — is not a saturated fit but a fit whose target it
    cannot reach from any pose in the box, so the constrained optimum is the same corner
    every frame. Widening the range does not help, and the pinned-limit warning's advice
    is misleading here: with the chain's [size](#ik-chain-size) and root measured properly
    the fit lands well inside `[-30, 0]`, and removing the box entirely changes the
    residual by 0.0001. Look at `chain_scales` first — an abdomen reading substantially
    *larger* than the same fly's head is the signature.

### The head's base — `neck` { #ik-head-base }

The head's three DOFs turn about one pivot, and `neck` sits **on** that pivot. Rotation
about a point leaves that point where it was, so the neck constrains none of the three
angles — and it is deliberately not counted as evidence the head was observed, which is
why declaring *both* antennae absent leaves the head unfitted rather than reporting the
solver's neutral-biased answer.

What it does instead is say **where the pivot is**, and nothing else in the pose can. The
body registration is a similarity fit through the six thorax-coxae, which are very nearly
coplanar (singular values 0.693 / 0.325 / 0.085 on the standard rig) while the pivot sits
0.404 above their centroid — a 4.7× extrapolation along their worst-determined axis.
Measured across a 55-recording corpus that lands the pivot a median **0.125 model units
too dorsal**, 29% of the head's own radius, with the same sign in every recording; the
antennae then absorb it as roughly **11° of spurious pitch**, and the head's estimated
size reads ~12% too large because its ruler started from that displaced anchor. So the
head chain is placed on its measured median `neck` — the way each leg is placed on its
measured median thorax-coxa — and sized from the neck out to the antennae.

The shift is recorded as `chain_offsets` in the `inverse_kinematics` metadata and baked
into `body_plan`, and the mesh overlay is given the same vector, so the drawn head and
the fitted angles describe one pose.

### Chain size — `chain_scales` { #ik-chain-size }

The legs are skinned between real keypoints, so they carry this fly's own bone lengths.
The head and abdomen are fixed model geometry, so each gets **one uniform multiplier**
(x, y and z alike) on top of the body scale, recorded as `chain_scales` in the
`inverse_kinematics` metadata, baked into `body_plan` and applied to the same chain by
the mesh overlay.

It is measured by fitting the chain's **angles and its size together** against its whole
marker set, seeded from a rigid-ruler estimate (below) and reduced by the median over
sampled frames, since a chain's size is a constant of the animal. Fitting posture out as a
nuisance parameter, rather than avoiding it, is what lets every marker contribute.

The **seed** comes from **separations between the chain's own markers that the chain's own
joints cannot change** — for the head, the `neck`-to-antennae radius. The abdomen has no
such separation at all (see the note below), so its seed is simply model size. Both
restrictions are load-bearing for that seed:

- **Posture is not size.** The abdomen's five markers sit on the dorsal *surface*, which
  is the outside of a ventral bend, so a ruler drawn *along* the chain lengthens by 57%
  over the joints' 30°-per-hinge range — more than the size differences being measured. A
  curled abdomen would read as a much bigger one.
- **A left-to-right distance is not the animal.** Each side is triangulated from its own
  cameras, so a cross-midline span carries the two sides' disagreement at full strength:
  across the corpus `l_antenna`–`r_antenna` reads **1.515×** the model where either
  antenna's distance to the `neck` reads **1.196×** — 26% wider, the same sign in all 30
  recordings. Mirror-symmetric markers are therefore folded to their midpoint first,
  which a symmetric outward push leaves exactly where it was.
- **The model's own anchor is not a marker.** Starting the ruler there measures partly
  the chain and partly how well the coxa registration placed it — the extrapolation
  described above, worth ~12% on the head.

Which pairs qualify is decided by probing the chain's forward kinematics across its own
bounds, not by a hand-written list, so a chain retargeted with a
[marker table](#ik-markers) gets the right ruler with no code change. A marker set with no
qualifying pair leaves the seed at model size. That is only a *seed*, so it is not warned
about here — the whole-marker fit above still measures the size. Calling
`estimate_chain_scale` on its own does warn, because there a silent `1.0` would read as a
measurement that this fly matches the model.

!!! note "Why the abdomen has no rigid ruler, on purpose"
    A chain with **one** qualifying pair is measured only as well as that pair, and the
    abdomen used to be exactly that case: its sole ruler was `abdomen3`–`abdomen4`, the
    shortest baseline in the chain (0.234 model units) between its two most distal — and
    least reliably localised — keypoints, where a 0.05 error is a 21% size error.

    That pair qualified for a reason that was itself the problem: **both markers hung off
    the same body** (`c_abdomen6`), so no joint lay between them and the model held them
    rigidly apart. A real abdomen does not. On one recording the model's separation came
    out **18% short** of the measured one and no joint angle could make it up — the single
    largest abdomen residual, `abdomen3` and `abdomen4` sitting 0.032 / 0.024 model units
    off with 99% of that a fixed bias rather than per-frame noise.

    The ruler also over-read. On three animals across two rigs it exceeded the whole-marker
    fit by **10.8% / 14.8% / 18.5%**, always the same sign, and always making the abdomen
    come out *larger* than the same fly's head (1.34–1.49 against 1.20–1.25) — one animal
    cannot have both. The cause is a **shape** mismatch rather than a measurement error:
    the model's abdomen is not a scaled real one, reading about 1.1× at the proximal end
    and about 1.5× at the last tergite, so the pair that happened to qualify sat at the
    worst end for this purpose.

    Anchoring each stripe one segment **proximal** puts a hinge between `abdomen3` and
    `abdomen4`, which removes the residual and the ruler together: measured on the same
    recording, every abdomen segment length lands within 2% of the animal's (from −18.2%
    and +13.0%) and the chain's total residual falls **42%**. So the abdomen is now
    measured only by the whole-marker fit, and reports no ruler at all.

    The cross-check that licenses that fit is the **head**, where the ruler is a long,
    well-defined radius and is trusted: there the two agree to **+0.7…1.6%**, with an
    unchanged marker residual and identical fitted angle ranges.

    A chain with **no** measured base landmark (the abdomen: no keypoint sits on its root)
    additionally has its root position fitted here, jointly, because size and root trade
    along an exact null direction of the fit — measuring either with the other held wrong
    gives a confidently repeatable wrong answer for both. The size itself is *not* in that
    null direction, so it is well posed either way. For a **midline** chain the root's
    lateral component is deliberately not fitted: a sideways root and the chain's own
    lateral joints express the same displacement, so freeing both attributes the animal's
    lateral bend to whichever the optimizer reaches first.

### Marker placement — `[inverse_kinematics.markers.head]` / `[inverse_kinematics.markers.abdomen]` { #ik-markers }

*Where* each head/abdomen keypoint sits relative to the NeuroMechFly model is a
**labeling-scheme choice** — the packaged abdomen markers are the five dorsal-midline
tergite stripes, placed at the same body + offset the docs keypoint viewer draws them at
(both read `docs/keypoints/assets/keypoints.json`, so the labeling reference and the IK
cannot disagree; see [Keypoint locations](../explanation/keypoints.md)). These tables let a different
skeleton retarget those markers **without re-running the model build**: when a table
is present it **replaces** that chain's default markers. Each entry is keyed by the
skeleton point name:

```toml
[inverse_kinematics.markers.abdomen]
abdomen0 = { body = "c_abdomen12", offset = [-0.37, 0.0, 0.34] }
abdomen1 = { body = "c_abdomen3",  offset = [-0.22, 0.0, 0.32] }
abdomen2 = { body = "c_abdomen4",  offset = [-0.23, 0.0, 0.30] }
abdomen3 = { body = "c_abdomen5",  offset = [-0.24, 0.0, 0.28] }
abdomen4 = { body = "c_abdomen6",  offset = [-0.25, 0.0, 0.22] }

[inverse_kinematics.markers.head]
neck      = { body = "c_head",    offset = [0.0, 0.0, 0.0], base = true }
l_antenna = { body = "l_pedicel", offset = [0.0, 0.0, 0.0] }
r_antenna = { body = "r_pedicel", offset = [0.0, 0.0, 0.0] }
```

Because a table **replaces** its chain's whole marker set, a config that redeclares the
head must carry `base = true` over with it. Leave it out and the chain quietly reverts to
the registered base — the head still fits, just about the wrong pivot.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `body` | str | *required* | The model body the marker is rigidly attached to. The chain (head/abdomen) and the marker's chain depth follow from it. The abdomen bodies are `c_abdomen12`/`c_abdomen3`/`c_abdomen4`/`c_abdomen5`/`c_abdomen6`; the head bodies are the head subtree (`c_head`, `l_pedicel`/`r_pedicel`, eyes, …). |
| `offset` | [float, float, float] | *required* | Offset from that body's origin (the joint), in the body's frame — the model units the rest of the IK uses. The marker's neutral position is `body_frame · offset`. |
| `depth` | int | the body's chain depth | Override the chain depth (rarely needed; the body determines it). |
| `base` | bool | `false` | Nominate this marker as the chain's [base landmark](#ik-head-base). Its depth is forced to 0 — it is on the chain's own origin, so no chain DOF moves it. At most one per chain, and it must be one of the table's markers. |

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
| `crop` | [x, y, w, h] or str | whole frame | Default panel window (overridable per video / per panel). `"pose2d"` resolves per view — see [`crop`](#panel-crop) below. |
| `mesh_hide` | list[str] | `["wings"]` | NMF overlay body parts to hide in the videos (`wings`/`halteres`/`eyes`/`antennae`/`head`/`thorax`/`abdomen`/`legs`). |

**`[visualization.kwargs]`** — draw-op defaults shared by every video, keyed by
the `plot` op name (`imshow`, `skeleton_2d`, `skeleton_3d`). Kwargs merge across
three levels — global → per-video `kwargs` → per-panel extra keys — most specific
winning.

**`[[visualization.videos]]`:**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `video_name` | str | *required* | Output filename (`<video_name>.mp4`). |
| `grid` | list[list[str]] | *unset* | A montage, as rows of view names — expands to panels (below). |
| `plot` | str | *required with `grid`* | The overlay op drawn on each grid cell. |
| `cell` | [int, int] | from `imshow` kwargs | Grid cell size in pixels. |
| `panels` | list[table] | *unset* | Ordered panels (below); they draw in order, so a skeleton panel over an `imshow` at the same offset overlays it. Appended *after* any `grid` expansion. |
| `stage` | str | most-derived | Forwarded to every panel the `grid` expands to. |
| `width`, `height` | int | auto-size | Canvas size in pixels; omit to fit all panels. |
| `background` | str or [r, g, b] | inherits global | Per-video canvas fill. |
| `crop` | [x, y, w, h] or str | inherits global | Per-video panel window. |
| `kwargs` | table | `{}` | Per-video draw-op kwargs (merges over the global). |

#### `grid` — a montage without the arithmetic { #grid }

A montage is the overwhelmingly common video and it is pure repetition: every panel is
the same two ops at a position that is a row and a column times the cell size. `grid`
says the shape and lets the offsets be computed:

```toml
[[visualization.videos]]
video_name = "pose3d"
plot  = "skeleton_3d"
stage = "triangulation"
grid  = [["rf", "f",    "lf"],
         ["rm", "bird", "lm"],
         ["rh", "h",    "lh"]]
```

That is exactly equivalent to seventeen hand-written panels — an `imshow` plus a
`skeleton_3d` for each of the eight cameras, and a lone `skeleton_3d` for `bird`.

- **Cell size** comes from the resolved `imshow` `width`/`height` (the same
  `[visualization.kwargs]` the panels already read), or from an explicit `cell = [w, h]`.
- **A cell whose view is not a camera** gets no `imshow` — the synthetic `bird` plan view
  has no footage. With `plot = "skeleton_2d"` such a cell is skipped entirely, since
  there are no 2D detections to draw. This is derived from `[cameras.*]`, so a rig that
  gains a camera gains a drawable cell without a second list to edit.
- **`""`, `"-"` or `"."`** leaves a tile empty.
- **`panels` still works**, alone or alongside — explicit panels are appended after the
  expansion, so a one-off tile can be added to a grid without abandoning it.

**Panel** — one draw op for one view at a pixel offset:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `plot` | str | *required* | `"imshow"` (the view's frame), `"skeleton_2d"` (its 2D detections), `"skeleton_3d"` (the 3D skeleton reprojected into the view), `"skeleton_nmf"` (the fitted inverse-kinematics model skeleton reprojected into the view), or `"mesh_nmf"` (the fitted NeuroMechFly *mesh*, shaded and reprojected; `alpha` sets its opacity). |
| `view` | str | *required* | Camera/view name, or the reserved `"bird"` (below). |
| `x0`, `y0` | int | `0` | Top-left pixel of the panel. |
| `scale` | float | `1.0` | Uniform scale. |
| `width`, `height` | int | from `scale` | Target box (priority over `scale`); one given → the other follows to keep aspect. |
| `crop` | [x, y, w, h] or str | whole frame | Show only this window of the view's raw frame, and size the panel to it. A string borrows the window from `[pose2d]` instead of restating it (below). |
| `clip` | bool | `true` | Keep the op inside the panel's footprint. |
| `stage` | str | most-derived | Which stage's points to draw: `"pose2d"`, `"pictorial_structures"`, `"triangulation"`, `"eks"`, `"postprocess"`. |
| `background` | str or [r, g, b] | inherits | Per-panel fill. |
| *extra keys* | — | — | Forwarded as draw-op kwargs (`point_radius`, `line_thickness`, `line_dash`, `palette`, …). |

<a id="panel-crop"></a>
`crop` is for a camera whose animal is a small part of a wide frame — an axial view at
1600×1008 dropped whole into a 480×240 cell is a smudge with a squashed aspect. It moves
the image origin and the geometry with it (2D points and the camera's principal point),
so the overlay stays on the picture; nothing needs adjusting by hand.

Usually the window you want is the one the detector already looks through, and
`crop = "pose2d"` says so instead of repeating the numbers:

```toml
[visualization]
crop = "pose2d"    # every panel shows the window ITS OWN view's pathway detects through
```

Resolved per view from `[[pose2d.pathways]]`, so one line covers a mixed rig: a view whose
pathway runs full-frame stays uncropped, and a view no pathway writes — the derived `bird`
plan view, say — is left alone. `crop = "<preprocessor name>"`
borrows one `[[pose2d.preprocessors]]` chain by name instead, for a panel the per-view rule
cannot answer. `crop` resolves across the same three levels as the draw-op kwargs
(`[visualization]` → the video entry → the panel), most specific winning, so an explicit
box on one panel still overrides a global setting.

Prefer the reference to a copy of the box. `[[pose2d.preprocessors]]` is **per
recording** — `{ op = "crop", auto = true }` searches its own window, so the numbers differ
from one recording to the next — and a hand-copied panel box then keeps showing the
*previous* recording's window under an overlay that still looks perfectly well-formed: a
rendering bug that reads as a calibration error. The reference also travels into the
visualization fingerprint, so moving the detector's crop re-renders the videos instead of
reusing stale MP4s.

Only the *region* transfers, not the orientation: a chain that also mirrors or turns the
frame is logged, and the panel shows that region in the view's own orientation (the overlay
is projected into view pixels, and a crop cannot express a reflection). A crop that does not
fit the footage — a box from a differently-sized recording — fails loudly rather than
truncating into a plausible-looking panel. And when two pathways window one view
differently, the panel says so instead of picking one.

`stage` pins what a video means. Left unset, a panel draws whatever the result resolved
to — the most-derived stage in the file — so enabling a later stage silently changes an
existing video: turn on `do_eks` and `pose3d` becomes the smoother's output, while
`pose2d` stops showing the detector (EKS writes a corrected 2D as well). Naming the stage
also makes a before/after pair expressible, as videos differing only in this key:

```toml
[[visualization.videos]]
video_name = "pose3d"          # before
panels = [{ plot = "skeleton_3d", view = "rf", stage = "triangulation" }]

[[visualization.videos]]
video_name = "pose3d_eks"      # after the smoother
panels = [{ plot = "skeleton_3d", view = "rf", stage = "eks" }]

[[visualization.videos]]
video_name = "pose3d_post"     # ... and after the postprocess chain
panels = [{ plot = "skeleton_3d", view = "rf", stage = "postprocess" }]
```

A panel naming a stage the file does not have skips its whole video with a logged reason
rather than falling back — a fallback would render the pair as two copies of one array.

<a id="panel-two-layers"></a>
A before/after pair can also be *one* video rather than two, when what matters is the
difference and not each pose on its own. A `grid` draws exactly one overlay op per cell,
but explicit `panels` are appended after the expansion and therefore draw on top of it —
so the reference goes in the grid and the comparison layer in `panels`, and `line_dash`
says which is which:

```toml
[[visualization.videos]]
video_name = "pose3d_post_vs_ik"
plot  = "skeleton_3d"          # dashed, underneath: the pose the fit was given
stage = "postprocess"
grid  = [["rf", "f", "lf"], ["rm", "bird", "lm"], ["rh", "h", "lh"]]
panels = [                     # solid, on top: forward kinematics of the fitted angles
    { plot = "skeleton_nmf", view = "rf", x0 = 0, y0 = 0 },
    # ... one per cell, at column * cell_width, row * cell_height
]

[visualization.videos.kwargs]
skeleton_3d  = { line_dash = [4, 9], point_radius = 2 }
skeleton_nmf = { line_thickness = 2, point_radius = 3 }
```

The explicit panels carry their own `x0`/`y0` because they are outside the grid's
expansion — column times the cell width, row times its height. `line_dash` is arc length
in canvas pixels (a number for equal on/off runs, or an `[on, off]` pair), and `0` draws
solid; bias the gap, because antialiased end caps light more than the nominal duty cycle
suggests (a `[6, 6]` pattern reads as a ragged solid line at `line_thickness = 2`).

The same layering puts a skeleton over a `mesh_nmf` grid, where the order is not
optional: a skeleton drawn *under* a translucent surface is a smear.

`clip` exists because the draw ops honour `x0`/`y0` but do not stop at the panel edge, so
in a grid a limb projecting out of its cell would be painted over the neighboring
camera's picture. Turn it off for a panel that is meant to spill. A panel placed at a
negative offset is never clipped to its own footprint (only to the canvas).

`view = "bird"` is a **derived** dorsal plan view rather than a rig camera: the body axes
come from the 3D pose itself (anterior from the abdomen tip to the neck, lateral across
the thorax-coxa joints, dorsal from their cross product with its sign settled by the
claws), and the focal is solved so the animal fills the frame once for the whole clip. It
shows all six legs with no body in the way, which the rig cannot do — the tether is above
the animal. It has no footage, so give it a `skeleton_3d`/`skeleton_nmf` panel and no
`imshow`. A real camera of that name takes precedence.

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
IK stage **estimates from the data** ([one uniform multiplier per chain](#ik-chain-size),
from the marker separations its joints cannot change — analogous to the coxa-derived
body scale), so a longer abdomen or bigger head is matched without a manual knob. The same scale is used by the `mesh_nmf` video op,
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
| `solve_policy` | str | `"gt_wins"` | How ground truth (GT) and predictions combine in the live 3D solve: `gt_wins` (GT decides everything it has an opinion about and the other views supply only what it cannot — a depth along the ray at one GT view, the GT pair's ill-conditioned direction at ≥ `min_gt_for_exclusive`; no GT ⇒ the configured `[triangulation]` method), `equal_weight` (GT-or-prediction per view through the configured method), or `weighted_blend` (one weighted DLT). No policy ever discards a GT observation. |
| `min_gt_for_exclusive` | int | `2` | `gt_wins`: at this many GT views, GT becomes exclusive — the other views then only fill the direction it cannot determine (see `gt_wins_keep_stabilizers`). |
| `gt_weight` | float | `1000.0` | Relative weight of GT rows when GT is mixed with predictions. |
| `prediction_weight` | str \| float | `"uniform"` | Prediction row weight in a GT-present solve: `"uniform"`, `"confidence"`, or a fixed float. Defaults to uniform to match the batch (`weigh_by_confidence=false`). |
| `confirm_default` | str | `"all"` | Default suggestion set a bulk-confirm promotes to GT: `"all"` (predictions and reprojections), `"predictions"`, or `"projections"`. |
| `low_conf` | float | `0.2` | Predictions below this confidence are visually de-emphasised (not hidden). |
| `undistort_before_solve` | bool | `false` | Undistort GT/prediction pixels before the linear DLT (more accurate on distorted lenses, but a zero-GT re-solve no longer matches the batch cache, which does not undistort). The nonlinear `gt_wins` paths differentiate the real projection and ignore this flag. |
| `equal_weight_protect_gt` | bool | `true` | Under `equal_weight`, force GT views to stay RANSAC inliers so a prediction consensus cannot vote a human label out. |
| `gt_wins_keep_stabilizers` | bool | `true` | Under `gt_wins`, let the unlabeled views fill the direction the GT views cannot determine once GT is exclusive. A camera says nothing about distance along its own optical axis, so two GT views facing each other (`rm` and `lm` of the standard rig are exactly opposed) leave that distance nearly free: measured on the test rig, GT-only turns 0.5 px of click noise into 255 µm mean / 372 µm p90 of 3D error and lands the joint ~37 px off in every unlabeled view, against ~30 µm with the stabilizers on for +0.13 px of GT reprojection. `false` restores the older GT-only triangulation; the editor exposes both under **Skeleton ▸ Derive the 3D from**. |
| `gt_sigma_px` | float | `0.5` | The operator's click precision, in pixels — the only parameter of the stabilized solve. With the detector's scale (`[triangulation].ransac_threshold`) it sets how much more a click is trusted than a prediction; a ratio of measurable σ's rather than a unitless weight, because the best fixed weight moves by 10× with the noise regime. |
