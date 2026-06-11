# Configuring a run

A run is driven by a single self-contained `config.toml`. `deeperfly init
config.toml` writes a fully commented copy to edit in place; `deeperfly run
recording/` with no `-c` falls back to the packaged defaults. A single file
carries everything a run needs — the camera rig, which file belongs to which
camera, the detector, the pipeline and the visualization.

The sections below are ordered roughly by how often you'll touch them: the first
few you'll set for almost every recording, the last few you can usually leave at
their defaults.

## The detection plan

2D detection is described by the top-level `[[sources]]` footage list plus the
detector's own machinery under `[pose2d]` — `[[pose2d.preprocessors]]`,
`[[pose2d.models]]`, `[[pose2d.pathways]]` — and a `[pose2d.output_points]` mapping
table. A neural network turns a preprocessed image into output channels; the plan
says which footage feeds which model (the pathways) and where each output channel
lands in the skeleton (`[pose2d.output_points]`). The default fly rig is **7
sources → 8 pathways → 7 views** (the front camera is read twice, once mirrored).

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

**Models** select a detector network: `class` is the registry key
(`"hourglass"` = DeepFly2D), `weights` a checkpoint (`""`/omitted uses the cached
download), `input_size` the `(height, width)` it expects, `mean` the scalar
subtracted after `/255`, and `n_out_channels` the output heatmap count.

```toml
[[pose2d.models]]
name = "deepfly2d"
class = "hourglass"
weights = ""
input_size = [256, 512]
mean = 0.22
n_out_channels = 19
```

**Pathways** are named `source -> preprocessor -> model` inference runs. A pathway
only says *what to detect on*; each needs a unique `name`:

```toml
[[pose2d.pathways]]
name = "rh_noflip"; source = "vid_rh"; preprocessor = "noflip"; model = "deepfly2d"
[[pose2d.pathways]]        # the front source, mirrored pass
name = "f_fliplr";  source = "vid_f";  preprocessor = "fliplr"; model = "deepfly2d"
```

**`[pose2d.output_points.<view>]`** says *where the outputs land*: for each view, a table
keyed by point name where `point = { pathway, out_channel }` fills that point from
output channel `out_channel` of the named pathway. Keying on `(view, point)` makes
every point's data come from exactly one place (a duplicate is a config error); a
`(view, point)` no entry names is left unobserved (NaN) — that union *is* the
visibility, with no separate table.

```toml
[pose2d.output_points.rh]                 # right-side view: 19 channels of one pathway
rf_thorax_coxa = { pathway = "rh_noflip", out_channel = 0 }
# ... through ...
r_abdomen2     = { pathway = "rh_noflip", out_channel = 18 }

[pose2d.output_points.f]                  # one view fed by two pathways, disjoint points
rf_femur_tibia = { pathway = "f_noflip", out_channel = 2 }   # right, un-flipped
lf_femur_tibia = { pathway = "f_fliplr", out_channel = 2 }   # left, mirrored
```

This modularity supports a range of setups: a single front model predicting both
legs, per-view or per-side specialized models (… → 14 pathways → 7 views), or a
different `model` per pathway.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off `do_<stage>` switch:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every camera view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery (opt-in)
do_triangulation        = true   # triangulate 2D -> 3D
do_visualization        = true   # render the videos
```

Each enabled stage has its own top-level `[<stage>]` parameter table (below).
Pictorial structures is the opt-in stage most commonly flipped on.

## Resume and recompute — fingerprints and `--overwrite`

An *enabled* stage reuses its result while its config is unchanged and its
output is in the output directory — so re-running a finished recording is a
cheap no-op, and **editing the config recomputes exactly the affected stages**.
Tweak `[triangulation]` or the videos and re-run: the slow 2D
detection is reused, only triangulation/visualization recompute (each stage's
parameters are recorded in `<outdir>/run.json` when it completes).
Performance-only knobs (`batch_size`, `decode_buffer`, `[io.image]`) never
trigger a recompute; a change that invalidates the slow `pose2d` stage is
announced loudly with exactly what changed. `--overwrite` forces a recompute
even when nothing changed: bare redoes every stage, or name stages to redo only
those (plus the stages after them):

```bash
deeperfly run recording/ --overwrite                       # recompute everything
deeperfly run recording/ --overwrite pose2d visualization  # just these (+ what follows)
```

The `pose2d` cache always feeds the stages downstream — `do_pose2d = false`
reconstructs 3D from a cached 2D pose without re-running detection. A *derived*
stage's cached output (bundle adjustment, pictorial structures, triangulation)
feeds downstream only while that stage is enabled: turning
`do_pictorial_structures` off re-triangulates from the raw detections. An
enabled stage whose input is unavailable is skipped, with the reason logged.

The run's config is snapshotted to `<outdir>/config.toml`. On a re-run `-c`
wins when given (and refreshes the snapshot); without `-c` the snapshot is
reused — so both workflows work: edit `out/config.toml` and re-run with
`-o out/` alone, or keep your own config and pass `-c` each time.

## Tune the opt-in stage — pictorial structures

This runs only when its `do_pictorial_structures` switch is on.

```toml
[pictorial_structures]   # peak recovery before triangulation
k        = 5       # candidate peaks per joint
temporal = false   # add a temporal-consistency term
lam      = 1.0     # bone-length prior weight
```

Candidate peaks are extracted during detection and cached in `poses.h5` when
this stage is enabled. Enabling it on an existing output directory therefore
re-runs `pose2d` once (announced loudly); after that, tweaking `temporal` /
`lam` re-runs only the recovery from the cached candidates. Resuming with
`do_pose2d = false` from a 2D result that stored no candidates skips the stage
with a notice.

## Output videos — `[visualization]`

Each `[[visualization.videos]]` is one output MP4, composited from an
ordered list of `panels`; each panel draws one op (`imshow`, `skeleton_2d`,
`skeleton_3d`) for one camera view at a pixel offset. Common edits:

```toml
[visualization]
background  = "black"
# output_fps = 30    # explicit output fps for every video
# speed      = 0.5   # or scale the input fps instead (0.5 = slow motion)

[visualization.kwargs]   # draw-op defaults shared by every video
imshow      = { width = 480, height = 240 }
skeleton_2d = { line_thickness = 2, width = 480, height = 240 }
skeleton_3d = { line_thickness = 2, width = 480, height = 240 }
```

The generated config ships two montage videos (`pose2d`, `pose3d`) wired to the
7-camera rig; reorder, drop, or add `panels` to change the layout. Draw-op
kwargs merge across three levels (global → per-video → per-panel), most specific
winning. Video frames are read and written with PyAV.

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
worst-reprojecting view; `dlt` is plain least-squares with no outlier handling.

`weigh_by_confidence` scales each view's contribution to the DLT by
`sqrt(confidence)`, so surer detections pull the 3D point harder (non-positive or
non-finite confidences drop the view). For `ransac` it weights the candidate fits
and the final refit but not the inlier vote, which stays a geometric reprojection
test so a confidently-wrong detection cannot vote itself into the consensus.

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
*memory* knob (peak
frames per camera is `~(decode_buffer + 2) * batch_size`) — raise it to keep the
GPU fed when decode is jittery, lower it to shave memory.

## Frame I/O — `[io]`

Video files are read and written with PyAV (in-process FFmpeg, on the CPU);
image sequences are decoded with OpenCV. The only knob is the image-decode
thread count:

```toml
[io.image]
# workers = 0   # decode threads (0 = one per CPU)
```

See [video.md](video.md) for the reader API.

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

A pathway's detections are mapped back into its view frame by inverting these
ops (plus the model's resize to its `input_size`), so the points always land in
the raw source frame the view's intrinsics describe. The flip is therefore a
detector-input concern only — it never reflects the reconstructed 3D skeleton.

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
weigh_by_confidence = true        # scale each reprojection residual by sqrt(confidence)
max_nfev            = 2000        # forwarded to scipy.optimize.least_squares
loss                = "linear"
```

`weigh_by_confidence` (default `true`) makes surer detections pull the
bundle adjustment harder, scaling each reprojection residual by `sqrt(confidence)`;
non-positive or non-finite confidences drop the observation, and if *every*
weight is zero it falls back to uniform weighting. Set it `false` to weight all
observations equally. (This is the mirror of
`[triangulation].weigh_by_confidence`, which defaults `false`.)

See [library.md](library.md) for calling the bundle adjuster directly.

## Camera rig geometry — `[cameras.defaults]` and `[cameras.*]`

A `[cameras.<name>]` is a geometric **view** that a pathway maps its points back
into — pure geometry now (intrinsics + orbit extrinsics), no footage or
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

## Skeleton — `[skeleton]`

The tracked points and their structure (38-point, 7-camera *Drosophila* rig):
`point_names`, `limb_points` kinematic chains (each a list of point names), and
the plotting `limb_palette`. Which view sees which point is not set here — it is
the union of the `[pose2d.output_points]` tables. Edit this only to track a different
animal — see [library.md](library.md) and [architecture.md](architecture.md).
