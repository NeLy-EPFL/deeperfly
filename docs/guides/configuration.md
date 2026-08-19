# Writing configs

A run is driven by a single self-contained `config.toml`. `deeperfly init
config.toml` writes a copy to edit in place; `deeperfly run recording/` with no
`-c` falls back to the packaged defaults. A single file carries everything a run
needs — the camera rig, which file belongs to which camera, the detector, the
pipeline and the visualization.

**A section you leave out runs on its defaults.** The packaged config states only
what it changes, so it is short enough to read; to see a key it does not mention,
ask the code rather than hunting for a commented-out line:

```console
$ deeperfly config show                 # every section, every key, its default
$ deeperfly config show triangulation   # one section, with its documentation
$ deeperfly config set triangulation.method greedy
```

This guide walks through customizing a config, ordered roughly by how often
you'll touch each section: the first few you'll set for almost every recording,
the last few you can usually leave at their defaults. For an exhaustive
parameter-by-parameter listing (every key, its type and default), see the
[configuration reference](../reference/configuration.md).

## The detection plan

2D detection is described by the top-level `[[sources]]` footage list plus the
detector's own machinery under `[pose2d]` — `preprocessors`, `models` and
`pathways`. A neural network turns a preprocessed image into output channels;
the plan says which footage feeds which model (the pathways) and where each
output channel lands in the skeleton.

The packaged plan is **dense**: one pathway per camera, with the model emitting
every tracked point for every view, so channel *i* is point *i* of that pathway's
view and no mapping table is written at all. A contralateral point arrives as a
prediction to correct rather than as a gap to author from nothing.

A **19-channel** detector predicts one body side instead, which means running each
side camera twice (once mirrored) and saying, per view and per channel, which
point it landed on — a `[pose2d.output_points]` table, 132 rows for the seven-camera
rig. That plan is still supported and is documented under
[`[pose2d.output_points]`](../reference/configuration.md#output_points); it is no
longer what a new config starts from.

**Sources** name the footage, the one setting almost every recording needs. Each
`filename` is a glob matched inside the recording directory:

```toml
[[sources]]
name     = "vid_rh"
filename = "camera_0.mp4"   # a named file, used as-is
[[sources]]
name     = "vid_rm"
filename = "camera_1"       # a bare prefix -> "camera_1*": a video or an image sequence
```

A source's footage is one video file or a naturally-sorted image sequence
(`camera_1_000123.jpg ...`). A directory is a valid recording only when every
source matches footage with the same file and frame count. A source with no
`filename` defaults to its own name.

**Preprocessors** are named, reusable frame-op pipelines that a pathway
references by name (full op grammar in the *Preprocessor op grammar* section
below):

```toml
[[pose2d.preprocessors]]
name = "plain"
ops  = []
[[pose2d.preprocessors]]
name = "mirror"
ops  = [{ op = "fliplr" }]
```

**Models** select a detector network. Only two keys are really yours: `class`, the
registry key (`"mvt"` cross-view dense, `"hrnet"` per-view dense, `"hourglass"` =
19-channel DeepFly2D), and `weights`, the checkpoint.

```toml
models = [{ name = "dense38mv", class = "mvt", weights = "mvt_alt8_r27_gray_fly38.pth" }]
```

`input_size`, `mean`, `n_out_channels` and (for `mvt`) `precision` are properties of
the checkpoint whose loader already refuses a config that disagrees, so the class
states them — see
[what the class already knows](../reference/configuration.md#model-class-defaults).
Write one only to override it.

A bare `weights` filename is looked up on `$DEEPERFLY_MODELS` (then the download
cache); anything with a path separator is used as written. Prefer the bare name:
*which model a run used* travels with the recording, where `/mnt/...` is a fact about
one machine. `"hourglass"` is the only class that needs no checkpoint at all — its
published weights are downloaded and cached on first use.

**Pathways** are named `source -> preprocessor -> model` inference runs. A
pathway only says *what to detect on*; each needs a unique `name`, and in a dense
plan that name is its view:

```toml
model = "dense38mv"                                       # the default for every pathway
pathways = [
    { name = "rh", source = "vid_rh" },                   # identity preprocessor
    { name = "f",  source = "vid_f", preprocessor = "crop_f" },
]
```

A pathway takes `[pose2d].model` when it names none of its own — and with a single
`[[pose2d.models]]` entry it takes that one, so a single-detector plan need not name it
anywhere. Two models and a bare pathway is an error naming both, never a silent pick:
see [the model a pathway takes](../reference/configuration.md#pathway-model).

**Where the outputs land** needs no table in a dense plan: channel *i* is point *i*
of the view the pathway is named after. When a detector is *not* dense — the
19-channel one predicts one body side, so its channels mean different points in
different views — a `[pose2d.output_points.<view>]` table says so explicitly, keyed
by point name:

```toml
[pose2d.output_points.f]                  # one view fed by two pathways, disjoint points
rf_femur_tibia = { pathway = "f",      out_channel = 2 }   # right, un-flipped
lf_femur_tibia = { pathway = "f_flip", out_channel = 2 }   # left, mirrored
```

Keying on `(view, point)` makes every point's data come from exactly one place (a
duplicate is a config error); a `(view, point)` no entry names is left unobserved
(NaN) — that union *is* the visibility, with no separate table. See the
[reference](../reference/configuration.md#output_points) for the full form and the
left/right check that guards it.

This modularity supports a range of setups: a single front model predicting both
legs, per-view or per-side specialized models, or a different `model` per
pathway.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off `do_<stage>` switch:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every camera view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery (opt-in)
do_triangulation        = true   # triangulate 2D -> 3D
do_eks                  = true   # ensemble Kalman smoother over the 3D
do_postprocess          = true   # corrections from knowing the animal (assumes TETHERED)
do_inverse_kinematics   = false  # fit NeuroMechFly joint angles (opt-in)
do_visualization        = true   # render the videos
```

Each enabled stage has its own top-level `[<stage>]` parameter table (below).
Pictorial structures is the opt-in stage most commonly flipped on.

Two of these are on in the packaged config because they are worth having on a *tethered*
preparation, and both are worth understanding before you keep them.
[`[eks]`](../reference/configuration.md#eks) fits one 3D trajectory per keypoint to the
whole recording, which de-jitters the pose (measured: **39% less** frame-to-frame
acceleration) and repairs blown detections — at the cost of lagging a keypoint that moves
faster per frame than the detector localizes it, which at 100 fps means a claw in swing
(0.13% of cells reproject >100 px, and every one is a claw). Turn it off if claw timing is
the measurement. [`[postprocess]`](../reference/configuration.md#postprocess) then applies
the corrections that come from knowing the *animal* rather than the pixels — an ordered
chain, so a new correction costs a block rather than a new stage. **It assumes the animal
is tethered**; turn it off for a freely-moving one:

```toml
[[postprocess.ops]]              # these keypoints do not move (a tethered thorax)
op = "static"
points = ["neck",
          "lf_thorax_coxa", "lm_thorax_coxa", "lh_thorax_coxa",
          "rf_thorax_coxa", "rm_thorax_coxa", "rh_thorax_coxa"]

[[postprocess.ops]]              # ... and the body is bilaterally symmetric
op = "symmetrize"
pairs = [["lf_thorax_coxa", "rf_thorax_coxa"],
         ["lm_thorax_coxa", "rm_thorax_coxa"],
         ["lh_thorax_coxa", "rh_thorax_coxa"]]
midline = ["neck"]
```

Never put the **legs** in `symmetrize.pairs`: left and right are in different gait phases
at any instant, and that asymmetry is the behavior being measured.

Editing the config and re-running recomputes exactly the stages you changed (and
the ones after them); the slow `pose2d` cache is reused untouched. That
resume/recompute behavior — and `--overwrite` — is covered in the
[CLI guide](cli.md#resuming-and-recomputing).

## Tune the opt-in stage — pictorial structures

This runs only when its `do_pictorial_structures` switch is on.

```toml
[pictorial_structures]   # peak recovery before triangulation
k        = 5       # candidate peaks per joint
temporal = false   # add a temporal-consistency term
lam      = 1.0     # bone-length prior weight
```

Candidate peaks are extracted during detection and cached in `results.h5` when this
stage is enabled. Enabling it on an existing output directory therefore re-runs
`pose2d` once (announced loudly); after that, tweaking `temporal` / `lam` re-runs
only the recovery from the cached candidates. Resuming with `do_pose2d = false`
from a 2D result that stored no candidates skips the stage with a notice.

## Output videos — `[visualization]`

Each `[[visualization.videos]]` is one output MP4, composited from an ordered
list of `panels`; each panel draws one op (`imshow`, `skeleton_2d`,
`skeleton_3d`, `skeleton_nmf`, `mesh_nmf`) for one camera view at a pixel offset.
Common edits:

```toml
[visualization]
background  = "black"
# output_fps = 30    # explicit output fps for every video
# speed      = 0.5   # or scale the input fps instead (0.5 = slow motion)
# crop       = "pose2d"   # every panel shows the window its view's detector looked through

[visualization.kwargs]   # draw-op defaults shared by every video
imshow      = { width = 480, height = 240 }
skeleton_2d = { line_thickness = 2, width = 480, height = 240 }
skeleton_3d = { line_thickness = 2, width = 480, height = 240 }
```

The generated config ships two montage videos (`pose2d`, `pose3d`) wired to the
7-camera rig; reorder, drop, or add `panels` to change the layout. Draw-op kwargs
merge across three levels (global → per-video → per-panel), most specific
winning. Video frames are read and written with PyAV.

`crop = "pose2d"` is worth reaching for on any rig with a camera the detector crops
(an axial view, typically): it resolves *per view* from `[[pose2d.pathways]]`, so a
single line frames each panel the way its own detector saw it and leaves the
full-frame views alone. That keeps the box in one place — the `[pose2d]` crop is
searched per recording, and a copy of the numbers under `[visualization]` would silently
keep showing the previous recording's window. See the
[configuration reference](../reference/configuration.md#visualization) for the
full panel and kwargs schema.

## Triangulation — `[triangulation]`

How the per-view 2D points become one 3D point:

```toml
[triangulation]
method              = "ransac"   # ransac (default, robust) | greedy | dlt
ransac_threshold    = 15.0       # inlier reprojection cutoff (px), method = ransac
min_inliers         = 2          # min agreeing views to accept a point (ransac)
# reproj_threshold  = 40.0       # method = greedy: per-view reprojection cutoff (px)
# max_drops         = 5          # method = greedy: max views dropped per point
weigh_by_confidence = false      # weight the DLT by detector confidence
```

`ransac` keeps the largest multi-view consensus; `greedy` drops the
worst-reprojecting view; `dlt` is plain least-squares with no outlier handling
(the [pipeline explainer](../explanation/pipeline.md#3d-reconstruction-triangulation-pictorial)
compares them).

`weigh_by_confidence` scales each view's contribution to the DLT by
`sqrt(confidence)`, so surer detections pull the 3D point harder (non-positive or
non-finite confidences drop the view). For `ransac` it weights the candidate fits
and the final refit but not the inlier vote, which stays a geometric reprojection
test so a confidently-wrong detection cannot vote itself into the consensus.

Changing `method` (or any `[triangulation]` key) and re-running reuses the cached
2D pose untouched and recomputes only triangulation and the videos — see
[change one stage, reuse the rest](cli.md#example-change-one-stage-reuse-the-rest)
for the exact commands.

## Detector precision and memory — `[pose2d]`

```toml
[pose2d]
precision     = "bfloat16"  # the default: as fast as float16 under CUDA autocast
                            # (~1.5-2x over float32) with a wider range (no overflow).
                            # "float32" is the reference; ignored on CPU/MPS
batch_size    = 16          # GPU forward batch (images/forward); throughput plateaus
                            # by ~16 on a fast GPU
decode_buffer = 4           # decode queue depth, in multiples of batch_size
```

These are the `[pose2d]` table's performance knobs; *what* to detect (sources,
models, pathways — including per-model `weights`) is the detection plan, which
shares the same `[pose2d]` table (and the top-level `[[sources]]`) and is
documented above. `batch_size` is the GPU forward batch; `decode_buffer` is a
*memory* knob (peak frames per camera is `~(decode_buffer + 2) * batch_size`) —
raise it to keep the GPU fed when decode is jittery, lower it to shave memory.
These knobs never invalidate a cache.

## Frame I/O — `[io]`

Video files are read and written with PyAV (in-process FFmpeg, on the CPU); image
sequences are decoded with OpenCV. The only knob is the image-decode thread
count:

```toml
[io.image]
workers = 0   # decode threads (0 = one per CPU)
```

The reader/writer API is in the [library guide](library.md#frame-io).

## Preprocessor op grammar — `[[pose2d.preprocessors]]` `ops`

A preprocessor is an ordered list of frame ops applied to a pathway's frames
before the model — to feed the detector a mirrored/cropped/rotated view. Steps
run in the order written (flips and rotations do not commute, so the order is
yours):

```toml
[[pose2d.preprocessors]]
name = "corrected"
ops  = [
    { op = "rot90", k = 1 },                                  # k CCW quarter-turns (any sign)
    { op = "fliplr" },                                         # left-right flip; also: flipud
    { op = "crop", x = 10, y = 10, width = 80, height = 80 },  # keep a window
    { op = "resize", scale = 0.5 },                            # or width/height; optional
]                                                              # interpolation = "bilinear"|"nearest"
```

A pathway's detections are mapped back into its view frame by inverting these ops
(plus the model's resize to its `input_size`), so the points always land in the
raw source frame the view's intrinsics describe. The flip is therefore a
detector-input concern only — it never reflects the reconstructed 3D skeleton.

### Letting the crop be measured — `auto = true`

The crop is the one op whose right value is a property of *this* recording. A detector is
trained through a box, and a differently framed camera puts the animal at the wrong
apparent scale — which no augmentation in the recipe undoes. On this rig the six side
cameras match training full-frame and the two axial ones (front and hind, 1600×1008 against
the side cameras' 960×512) do not, so those are the two that usually need one.

Rather than copy last recording's numbers, ask for it to be measured:

```toml
[[pose2d.preprocessors]]
name = "crop_h"
ops  = [{ op = "crop", auto = true }]                      # blind: search the whole frame

[[pose2d.preprocessors]]
name = "crop_f"
ops  = [{ op = "crop", auto = true,                        # seeded: search near a box you
          x = 400, y = 290, width = 800, height = 400 }]   # already trust (fewer probes)
```

The `pose2d` stage resolves it before detecting: the detector's own confidence covers the
`(centre, width)` space cheaply, then agreement with the **other cameras' 3D** — the target
view held out of the triangulation — chooses among what confidence proposed, and refuses a
box that is confidently wrong. Measured blind on a 1600×1008 hind camera: 255 px from the
other cameras' 3D at full frame, **2.9 px** after the search, against 3.4 px for a box a
person tuned by hand; about 15 s a view.

Two things to know. **It needs a solved rig** — against the nominal orbit rig the reference
is ~120 px out, the gate detects that and refuses to run, and the confidence-only fallback
comes out ~1.7× too wide. Run once with bundle adjustment and point
`[cameras].calibration` at the exported `calibration.toml`. And **the box is recorded**, in
`<outdir>/autocrop.json`, so a resume neither re-searches nor re-detects, and the panels
that borrow it (`crop = "pose2d"`) keep working in a later process.

To measure without committing to a run — or to freeze the result as plain numbers:

```console
$ deeperfly auto-crop RECORDING -c config.toml
```

It prints each view's incumbent, the searched box, the confidence, the agreement in pixels,
and the TOML to paste back if you would rather the box never be searched again.

## Bundle adjustment — `[bundle_adjustment]`

Bundle adjustment uses the fly itself as the target, solved with
`scipy.optimize.least_squares` — its kwargs (`max_nfev`, `loss`, ...) sit
directly in the table. The defaults suit the standard rig; you rarely need to
change them.

```toml
[bundle_adjustment]
points_to_use       = [ "..." ]   # skeleton point names that drive bundle adjustment (default: the 30 leg points)
fixed               = ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"]   # held constant; fixes the world gauge
shared              = []          # e.g. [["lf.tvec[2]", "rf.tvec[2]"]] to tie cameras' z distances
weigh_by_confidence = false       # scale each reprojection residual by sqrt(confidence)
max_frames          = 200         # bundle-adjust on at most this many frames (subsampled)
frame_sampling      = "even"      # even | confidence | coverage | diversity
max_nfev            = 2000         # forwarded to scipy.optimize.least_squares
loss                = "linear"
```

The `fixed` / `shared` grammar (`"*.intr"`, `"f.rvec"`, `"rm.tvec[2]"`, tying
`[["lf.tvec[2]", "rf.tvec[2]"]]`) anchors the world gauge and ties parameters
between cameras; the [reference](../reference/configuration.md#bundle_adjustment)
gives the full grammar and the `frame_sampling` strategies. See the
[library guide](library.md#geometry-and-bundle-adjustment) for calling the bundle
adjuster directly.

## Camera rig geometry — `[cameras.defaults]` and `[cameras.*]`

A `[cameras.<name>]` is a geometric **view** that a pathway maps its points back
into — pure geometry (intrinsics + orbit extrinsics), no footage or
preprocessing. The cameras orbit an object near the world origin;
`[cameras.defaults]` is merged into every view, and each `[cameras.<name>]`
overrides it (the default rig sets just `azimuth_deg` per view). A view's
intrinsics describe the raw frame of the source feeding it. The shipped values
describe the standard DeepFly3D 7-camera rig — leave them unless your rig differs.

```toml
[cameras.defaults]
focal_length_px = [22388.125, 22388.125]
distance        = 107.463
elevation_deg   = 0.0
# principal_point_px = [479.5, 239.5]   # omit to use each view's image center

[cameras.f]
azimuth_deg = 0
```

The orbit parameters (`look_at`, `distance`, `azimuth_deg`, `elevation_deg`,
`roll_deg`) and intrinsics are detailed in the
[reference](../reference/configuration.md#cameras).

## Skeleton — `[skeleton]`

The tracked points and their structure (38-point, 7-camera *Drosophila* rig):
`point_names`, `limb_points` kinematic chains (each a list of point names), and
the plotting `limb_palette`. Naming a packaged skeleton (`name = "fly38b"`) loads all
three; write a key to override it. Which view sees which point is not set here — it is
the union of the pathway maps, which a dense plan makes total. Edit this only to track a
different animal — see the
[reference](../reference/configuration.md#skeleton) and the
[pipeline explainer](../explanation/pipeline.md).
