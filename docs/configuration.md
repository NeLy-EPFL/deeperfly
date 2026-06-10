# Configuring a run

A run is driven by a single self-contained `config.toml`. `deeperfly init
config.toml` writes a fully commented copy to edit in place; `deeperfly run
recording/` with no `-c` falls back to the packaged defaults. A single file
carries everything a run needs — the camera rig, which file belongs to which
camera, the detector, the pipeline and the visualization.

The sections below are ordered roughly by how often you'll touch them: the first
few you'll set for almost every recording, the last few you can usually leave at
their defaults.

## Map input files to cameras — `input` under `[cameras.*]`

The one setting almost every recording needs. Each camera's footage is given by
its `input` key — a filename glob matched inside the recording directory — right
beside that camera's geometry:

```toml
[cameras.rh]
azimuth_deg = -120
input = "camera_0.mp4"   # a named file, used as-is
[cameras.rm]
azimuth_deg = -90
input = "camera_1"       # a bare prefix -> "camera_1*": a video or an image sequence
[cameras.lf]
azimuth_deg = 45
input = "cam*/*"         # your own wildcard, used as-is
```

A camera's footage is one video file or a naturally-sorted image sequence
(`camera_1_000123.jpg ...`). A directory is a valid recording only when every
camera matches footage with the same file and frame count. A camera with no
`input` defaults to its own name.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off `do_<stage>` switch:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every camera view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery (opt-in)
do_triangulation        = true   # triangulate 2D -> 3D
do_visualization        = true   # render the videos
# fps = 100.0   # optional; omit to detect from the input videos
```

Each enabled stage has its own `[pipeline.<stage>]` parameter table (below).
Pictorial structures is the opt-in stage most commonly flipped on.

## Resume and recompute — fingerprints and `--overwrite`

An *enabled* stage reuses its result while its config is unchanged and its
output is in the output directory — so re-running a finished recording is a
cheap no-op, and **editing the config recomputes exactly the affected stages**.
Tweak `[pipeline.triangulation]` or the videos and re-run: the slow 2D
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
[pipeline.pictorial_structures]   # peak recovery before triangulation
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

## Output videos — `[pipeline.visualization]`

Each `[[pipeline.visualization.videos]]` is one output MP4, composited from an
ordered list of `panels`; each panel draws one op (`imshow`, `skeleton_2d`,
`skeleton_3d`) for one camera view at a pixel offset. Common edits:

```toml
[pipeline.visualization]
background  = "black"
# output_fps = 30    # explicit output fps for every video
# speed      = 0.5   # or scale the input fps instead (0.5 = slow motion)

[pipeline.visualization.kwargs]   # draw-op defaults shared by every video
imshow      = { width = 480, height = 240 }
skeleton_2d = { line_thickness = 2, width = 480, height = 240 }
skeleton_3d = { line_thickness = 2, width = 480, height = 240 }
```

The generated config ships two montage videos (`pose2d`, `pose3d`) wired to the
7-camera rig; reorder, drop, or add `panels` to change the layout. Draw-op
kwargs merge across three levels (global → per-video → per-panel), most specific
winning. Video frames are read and written with PyAV.

## Triangulation — `[pipeline.triangulation]`

How the per-view 2D points become one 3D point:

```toml
[pipeline.triangulation]
method           = "ransac"   # ransac (default, robust) | greedy | dlt
ransac_threshold = 15.0       # inlier reprojection cutoff (px), method = ransac
min_inliers      = 2          # min agreeing views to accept a point (ransac)
# reproj_threshold = 40.0     # method = greedy: per-view reprojection cutoff (px)
# max_drops        = 5        # method = greedy: max views dropped per point
```

`ransac` keeps the largest multi-view consensus; `greedy` drops the
worst-reprojecting view; `dlt` is plain least-squares with no outlier handling.

## Detector precision and memory — `[pipeline.pose2d]`

```toml
[pipeline.pose2d]
precision     = "bfloat16"  # the default: as fast as float16 under CUDA autocast
                            # (~1.5-2x over float32) with a wider range (no overflow).
                            # "float32" is the reference; ignored on CPU/MPS
batch_size    = 16          # GPU forward batch (images/forward); throughput plateaus
                            # by ~16 on a fast GPU
decode_buffer = 4           # decode queue depth, in multiples of batch_size
# checkpoint = "/path/to/weights"   # defaults to the cached weights
```

`batch_size` is the GPU forward batch; `decode_buffer` is a *memory* knob (peak
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

## Per-camera preprocessing — `[cameras.<camera>.preprocess]`

Optional per-camera geometric correction applied right after decoding — for a
camera mounted sideways/upside-down or with a mirrored sensor. The transformed
frame becomes canonical for the whole run; nothing maps back to the raw footage.

```toml
[cameras.rh.preprocess]
fliplr = false   # left-right flip
flipud = false   # up-down flip
rot90  = 0       # counter-clockwise 90-degree turns (0-3)
```

Applied in that order. Cameras with no table are left untouched.

## Bundle adjustment — `[pipeline.bundle_adjustment]`

Calibration uses the fly itself as the target, solved with
`scipy.optimize.least_squares` — its kwargs (`max_nfev`, `loss`, ...) sit
directly in the table. The defaults suit the standard rig; you rarely need to
change them.

```toml
[pipeline.bundle_adjustment]
keypoints = [ "..." ]   # skeleton points that drive calibration (default: the 30 leg points)
fixed     = ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"]   # held constant; fixes the world gauge
shared    = []          # e.g. [["lf.tvec[2]", "rf.tvec[2]"]] to tie cameras' z distances
max_nfev  = 2000        # forwarded to scipy.optimize.least_squares
loss      = "linear"
```

See [library.md](library.md) for calling the bundle adjuster directly.

## Camera rig geometry — `[cameras.defaults]` and `[cameras.*]`

The cameras orbit an object near the world origin. `[cameras.defaults]` is merged
into every camera; each `[cameras.<name>]` overrides it (the default rig sets
just `azimuth_deg` and `input` per view). The shipped values describe the
standard DeepFly3D 7-camera rig — leave them unless your rig differs.

```toml
[cameras.defaults]
focal_length_px = [22388.125, 22388.125]
distance        = 107.463
elevation_deg   = 0.0
# principal_point_px = [479.5, 239.5]   # omit to use each view's image center

[cameras.f]
azimuth_deg = 0
input = "camera_3.mp4"
```

## Skeleton — `[skeleton]`

The tracked points and their structure (38-point, 7-camera *Drosophila* rig):
joint names, `limb_joints` kinematic chains, plotting `palette`, and per-camera
`visibility`. Edit this only to track a different animal — see
[library.md](library.md) and [architecture.md](architecture.md).
