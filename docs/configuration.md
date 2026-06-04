# Configuring a run

A run is driven by a single self-contained `config.toml`. `deeperfly init
config.toml` writes a fully commented copy to edit in place; `deeperfly run
recording/` with no `-c` falls back to the packaged defaults. A single file
carries everything a run needs — the camera rig, which file belongs to which
camera, the detector, the pipeline, correction and smoothing.

The sections below are ordered roughly by how often you'll touch them: the first
few you'll set for almost every recording, the last few you can usually leave at
their defaults.

## Map input files to cameras — `[inputs]`

The one section almost every recording needs. Each key is a camera name (from
`[cameras.*]`); each value is a filename glob matched inside the recording
directory:

```toml
[inputs]
rh = "camera_0.mp4"   # a named file, used as-is
rm = "camera_1"       # a bare prefix -> "camera_1*": a video or an image sequence
lf = "cam*/*"         # your own wildcard, used as-is
```

A camera's footage is one video file or a naturally-sorted image sequence
(`camera_1_000123.jpg ...`). A directory is a valid recording only when every
camera matches footage with the same file and frame count. A camera with no
entry defaults to its own name.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off `do_<stage>` switch:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every camera view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery (opt-in)
do_triangulation        = true   # triangulate 2D -> 3D
do_smoothing            = false  # temporal smoothing of the 3D track (opt-in)
do_visualization        = true   # render the videos
# fps = 100.0   # optional; omit to detect from the input videos
```

Each enabled stage has its own `[pipeline.<stage>]` parameter table (below). The
two opt-in stages — pictorial structures and smoothing — are the most common
toggles to flip on.

## Resume and recompute — caching and `--overwrite`

An *enabled* stage reuses its result when it is already in the output directory,
recomputing only when it's missing — so re-running a finished recording is a
cheap no-op. Force a recompute with `--overwrite`: bare redoes every stage, or
name stages to redo only those (plus the stages after them):

```bash
deeperfly run recording/ --overwrite                       # recompute everything
deeperfly run recording/ --overwrite pose2d visualization  # just these (+ what follows)
```

A *disabled* stage (`do_<stage> = false`) is dropped from the pipeline; its
cached result is read from `poses.h5` and fed to the stages still on — so
`do_pose2d = false` reconstructs 3D from a cached 2D pose without re-running
detection. An enabled stage whose input is unavailable is skipped, with the
reason logged.

A run reuses the `config.toml` saved in the output directory (it owns the cached
results), so `-o out/` alone resumes; pass `-c` only for a fresh output
directory, and edit `out/config.toml` to change a run in place.

## Tune the opt-in stages — pictorial structures and smoothing

These run only when their `do_<stage>` switch is on.

```toml
[pipeline.pictorial_structures]   # peak recovery before triangulation
k        = 5       # candidate peaks per joint
temporal = false   # add a temporal-consistency term
lam      = 1.0     # bone-length prior weight

[pipeline.smoothing]
method = "gaussian"   # gaussian | one_euro
```

Pictorial structures needs `do_pose2d` in the same run (candidate peaks aren't
cached), so it is skipped when resuming from a stored 2D result.

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
winning. The output encoder and input decoder come from `[io.video]`.

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
precision    = "float32"   # "float16" runs under CUDA autocast: ~1.5-2x faster,
                           # negligible drift. "bfloat16" is as fast with a wider
                           # range (no overflow); ignored on CPU/MPS
chunk_frames = 64          # frames decoded + detected at a time, per camera
# checkpoint = "/path/to/weights"   # defaults to the cached weights
```

`chunk_frames` is a *memory* knob, not a speed one: lower it for high-res
footage, many cameras, or a small GPU. The frame decoder lives in `[io.video]`.

## Frame I/O backends — `[io]`

Decoder/encoder choices shared across every stage (all decode/encode is CPU):

```toml
[io.video]
reader = "auto"   # auto | pyav | opencv | torchcodec | video_reader_rs
writer = "auto"   # auto | pyav | opencv

[io.image]
reader = "auto"   # auto/opencv (core) | imageio (optional, broader formats)
# workers = 0     # decode threads (0 = one per CPU)
```

`"auto"` picks the fastest installed backend. See [video.md](video.md) for what
each backend supports and how to install the optional ones.

## Per-camera preprocessing — `[preprocess]`

Optional per-camera geometric correction applied right after decoding — for a
camera mounted sideways/upside-down or with a mirrored sensor. The transformed
frame becomes canonical for the whole run; nothing maps back to the raw footage.

```toml
[preprocess.rh]
fliplr = false   # left-right flip
flipud = false   # up-down flip
rot90  = 0       # counter-clockwise 90-degree turns (0-3)
```

Applied in that order. Cameras with no table are left untouched.

## Bundle adjustment — `[pipeline.bundle_adjustment]`

Calibration uses the fly itself as the target. The defaults suit the standard
rig; you rarely need to change them.

```toml
[pipeline.bundle_adjustment]
solver    = "least_squares_scipy"
keypoints = [ "..." ]   # skeleton points that drive calibration (default: the 30 leg points)
fixed     = ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"]   # held constant; fixes the world gauge
shared    = []          # e.g. [["lf.tvec[2]", "rf.tvec[2]"]] to tie cameras' z distances

[pipeline.bundle_adjustment.least_squares_scipy]   # forwarded to scipy.optimize.least_squares
max_nfev = 2000
loss     = "linear"
```

See [library.md](library.md) for calling the bundle adjuster directly.

## Camera rig geometry — `[camera_defaults]` and `[cameras.*]`

The cameras orbit an object near the world origin. `[camera_defaults]` is merged
into every camera; each `[cameras.<name>]` overrides it (the default rig sets
just `azimuth_deg` per view). The shipped values describe the standard DeepFly3D
7-camera rig — leave them unless your rig differs.

```toml
[camera_defaults]
focal_length_px = [22388.125, 22388.125]
distance        = 107.463
elevation_deg   = 0.0
# principal_point_px = [479.5, 239.5]   # omit to use each view's image center

[cameras.f]
azimuth_deg = 0
```

## Skeleton — `[skeleton]`

The tracked points and their structure (38-point, 7-camera *Drosophila* rig):
joint names, `limb_joints` kinematic chains, plotting `palette`, and per-camera
`visibility`. Edit this only to track a different animal — see
[library.md](library.md) and [architecture.md](architecture.md).
