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

## Footage: which file is which camera

**A camera declares its own footage.** There is no footage section, and nothing is
inferred from a camera's index — a source that no camera claimed was never anything but a
camera without geometry, so the two live together:

```toml
[cameras.rh]
video = 'camera_RH.mp4'          # a glob, matched inside the recording directory
azimuth_deg = -120

[cameras.rm]
video = '/camera_(RM|1)\.mp4/'    # a REGEX: slashes make it one, case-insensitive
azimuth_deg = -90

[cameras.rf]
video = ['rf_part1.mp4', 'rf_part2.mp4']   # a LIST is CONCATENATED, in this order
azimuth_deg = -45
```

A pattern is a **glob** by default and a **regex** when wrapped in slashes. Everything one
pattern matches is **one stream**, in natural order — which is what makes a split
recording and an image sequence (`camera_1_000123.jpg …`) the same rule. The matches have
to be parts of one series (identical once digit runs are masked), which is what stops
`/camera_(RH|0)/` from concatenating two naming schemes in a directory that holds both.

So alternates go **inside** the pattern and a list means concatenation. That is the one key
whose v1 shape still parses and now means something else, so it gets an explicit check:
a v1 `video = ["a.mp4", "b.mp4"]` meant *either*, and now means *both, end to end*.

A camera that writes no `video` at all falls back to its own name as the pattern, which is
convenient for `camera_rh/` style layouts and wrong for most others — write the pattern.

**A partial recording is a recording.** A camera whose pattern matches no files no longer
makes the directory malformed: the footage that *is* there is reported with a warning
naming what is absent, and the run [narrows itself](cli.md#inputs-single-batch-recursive),
dropping that camera from the rig and blanking its cell in the montages. What still warns
and skips is a directory that cannot be read coherently: several footage extensions in one
folder, or an unequal file or frame count across the cameras that are present.

## The detector — `[pose2d]`

**Detection is dense and one-to-one**, and the plan is synthesized from the camera table:
one detector, run once per camera, channel *i* → point *i*. There is nothing to declare
but the detector itself:

```toml
[pose2d]
class   = "mvt"
weights = "mvt_r28_pad48_gray_fly38.pth"
```

Two classes, both dense — every tracked point in every view:

| `class` (aliases) | what it is |
| --- | --- |
| `"mvt"` (`"multiview_transformer"`) | encodes a frame's views **together**, so a joint only one camera can see informs the ones that cannot. Pinned to float32. The default. |
| `"hrnet"` (`"hrnet_timm"`) | the dense **per-view** detector; also runs the HGNetV2-B4 checkpoint, whose feature maps it selects by **stride** rather than by index |

An unrecognized `class` is **refused**, naming the classes this build has. It used to fall
through to DeepFly2D's defaults — 19 channels and a mean of 0.22 — and then fail at load
with a channel-count mismatch, which says nothing about the word that was misspelled.

`input_size`, `mean`, `n_out_channels` and `precision` are properties of the checkpoint
whose loader already refuses a config that disagrees, so the class states them — see
[what the class already knows](../reference/configuration.md#model-class-defaults). Write
one only to override it. `n_out_channels` defaults to **the skeleton's point count**, which
is what dense means, so a config stops restating its own skeleton's size and cannot get it
wrong.

**Nothing downloads.** Every detector deeperfly runs is trained per project, so `weights` is
required and a run that cannot resolve it stops with the search path printed. A bare
filename is looked up on `$DEEPERFLY_MODELS` (`os.pathsep`-separated, like `PATH`, then the
local cache directory); anything with a path separator, a leading `~`, or an absolute path is
used as written. Prefer the bare name: *which model a run used* is a fact about the recording
and travels with it, where `/mnt/...` is a fact about one mount on one machine.
`deeperfly doctor` prints the variable, every directory searched with what is in it, and
whether the default config's checkpoint resolves. Three checkpoints ship with 0.2 — all
one-channel, all recording the `fly38` point order, all trained on the same 55 recordings
(465 moments / 138,708 label cells for the two `hrnet` arms, 485 / 144,775 for the MVT);
the packaged config names
`mvt_r28_pad48_gray_fly38.pth`. Their sizes, checksums and location are in
[the released checkpoints](../reference/configuration.md#weights).

**One input plane.** Every shipped detector takes a single grayscale channel, so
`LoadedModel.prepare` emits `(…, 1, H, W)` and the decoder can hand over the luma plane
without the YUV→RGB conversion that is most of what reading a frame costs.

**The channel order is checked against the checkpoint on every run**, by the ordered point
*names* — not by a count. A count cannot tell two 38-point skeletons apart: `fly38` and the
retired DeepFly3D set share 32 points in a different order, so routing one through the
other's config attaches six points to the wrong joints and shifts the rest. That is a wrong
limb, not a crash. A checkpoint that records **no** channel names at all is refused outright.

### The detection window — `[pose2d.crops]`

A detector is trained through a box, and a differently framed camera puts the animal at
the wrong apparent scale. One entry per camera, always a box:

```toml
[pose2d.crops]
f = { x = 400, y = 290, width = 800, height = 400 }
```

There is no op grammar. `fliplr`, `flipud`, `rot90` and `resize` had no consumer left once
detection went dense and one-to-one — the side-agnostic detector that needed a mirror is
gone — and a window is the one thing a camera genuinely needs. The window is inverted on
the way back, so a detection reaches its camera in raw footage pixels and the intrinsics go
on describing the raw frame.

**Let the box be measured.** `auto_crops` names the cameras whose window should be searched
for this recording rather than copied from the last one:

```toml
[pose2d]
auto_crops = ["f", "h"]        # the two axial cameras
```

A `[pose2d.crops]` entry for a camera listed there becomes a **seed** — where the search
starts — rather than the answer; a camera with no entry searches blind. See
[the searched crop](../reference/configuration.md#auto-crop) for what the search optimizes
and why it needs a solved rig to *accept* a box but not to propose one.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off switch named after the stage:

```toml
[pipeline]
pose2d               = true   # detect 2D pose in every camera view
bundle_adjustment    = true   # refine the cameras (bundle adjustment)
pictorial_structures = false  # DeepFly3D-style peak recovery -- the one stage off
triangulation        = true   # triangulate 2D -> 3D
eks                  = true   # ensemble Kalman smoother over the 3D
postprocess          = true   # corrections from knowing the animal (assumes TETHERED)
inverse_kinematics   = true   # fit the model's joint angles
visualization        = true   # render the videos
```

The `do_` prefix is gone — it was a prefix on a key inside a table already called
`pipeline`. A `do_<stage>` key is **refused by name** rather than ignored, because an
ignored one leaves the stage at its default and reads as "the flag did nothing".

**Every stage is on by default except `pictorial_structures`**, and the packaged config
states each flag at exactly the value it would inherit — explicit for readability, not
because it changes anything. Each enabled stage has its own top-level `[<stage>]` parameter
table (below).

To turn one off without opening the file:

```bash
deeperfly config set pipeline.eks false -c config.toml
```

That works as of 0.2 and did not before — see
[turning a stage off](cli.md#turning-a-stage-off) for why.

Three of these deserve a sentence before you keep them.

[`[eks]`](../reference/configuration.md#eks) fits one 3D trajectory per keypoint to the
whole recording, which de-jitters the pose (measured: **39% less** frame-to-frame
acceleration) and repairs blown detections — at the cost of lagging a keypoint that moves
faster per frame than the detector localizes it, which at 100 fps means a pretarsus in swing
(0.13% of cells reproject >100 px, and every one is a pretarsus). Turn it off if pretarsus timing is
the measurement.

[`[inverse_kinematics]`](../reference/configuration.md#inverse_kinematics) fits a
mechanical model's joint angles to the 3D pose. It needs the optional `ik` extra, whose
one dependency is [QuickIK](https://nely-epfl.github.io/quickik/) — a Rust library with no
published wheels, so installing it builds the extension and needs a Rust toolchain, which
nothing else in deeperfly does:

```bash
uv sync --extra ik                                                     # from a checkout
pip install "quickik @ git+https://github.com/NeLy-EPFL/quickik#subdirectory=python"
```

**Without it the stage skips with the reason logged** rather than failing the run. That
matters more than it looks now the stage is on by default: `inverse_kinematics` runs
*before* `visualization`, so an exception there would also cost the videos — after
detection, bundle adjustment, triangulation, the smoother and the correction chain had all
been computed and committed. A result file that already holds a fit renders its overlays
(videos and GUI alike) without the extra. Its `symmetric_segments` is
now on by default; see [below](#inverse-kinematics).

[`[postprocess]`](../reference/configuration.md#postprocess) applies the corrections that
come from knowing the *animal* rather than the pixels — an ordered chain, so a new
correction costs a block rather than a new stage. **It assumes the animal is tethered**;
turn it off for a freely-moving one:

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
at any instant, and that asymmetry is the behavior being measured. `abdomen*` is
deliberately absent from `static` for the same kind of reason — it pitches, telescopes and
bends, which is worth measuring.

**The order of the ops matters.** With one fitted plane (the default) `symmetrize` is a
fixed map, so it leaves an already-static point static and static-then-symmetrize satisfies
both exactly; the other order would not. Each op logs how far it moved what it touched — a
point drifting tens of pixels was *moving*, and belongs out of the list rather than nailed
down.

The chain runs **after** the smoother, and has to: the smoother re-derives 3D from the 2D,
so anything corrected before it is followed straight back off. Keeping these priors out of
the estimating stages is the same deliberate choice — an estimator already told the answer
cannot be checked against it.

Editing the config and re-running recomputes exactly the stages you changed (and
the ones after them); the slow `pose2d` cache is reused untouched. That
resume/recompute behavior — and `--overwrite` — is covered in the
[CLI guide](cli.md#resuming-and-recomputing).

## Inverse kinematics — `[inverse_kinematics]` { #inverse-kinematics }

Every knob is at its default in the packaged config, so no table is written
(`deeperfly config show inverse_kinematics`). Three things about it are worth knowing here;
the rest is in the [reference](../reference/configuration.md#inverse_kinematics).

**The model is a pack, and the placement is a binding.** `model = "neuromechfly"` selects
the whole model — leg template, articulation and overlay mesh — as one unit, and *where*
each tracked point sits on it lives in a third file keyed on the pair,
`bindings/fly38@neuromechfly.toml`. That is the axis the fact belongs on: `abdomen0`
sitting at a particular offset on `c_abdomen12` is a statement about how fly38 was
*labelled against* NeuroMechFly, true of neither half alone. The packaged pair binds all
38 points, so one body plan covers them and the [keypoint
viewer](../explanation/keypoints.md) is generated from the same file — what the labeling
reference shows and what the IK fits are the same rows.

Binding a differently-labelled skeleton is one new file
(`deeperfly ik bind <skeleton> <model>` generates it for review); an **unbound pair is a
load error** naming both halves, rather than a fit against all-NaN observations.
`[inverse_kinematics.markers.<chain>]` stays a per-run *patch* over the binding, and
**replaces** that chain's rows, so list all of them.

**Which non-leg chains are fit is `chains`**, a list of names rather than a boolean per
chain: `chains = []` fits the legs only, and omitting it fits every chain the pack
defines. `fit_head` / `fit_abdomen` are refused by name.

**`symmetric_segments` is on by default**, and it costs something measurable, so it is worth
knowing which way you want it. The leg segment lengths are otherwise measured per leg, which
fits this fly rather than a generic one but measures each side on its own; sharing gives each
mirror pair the mean of the two — one animal rather than two half-animals. It constrains the
*animal* only: the two sides' joint angles stay independent, because a leg's left/right
asymmetry at any instant is the behavior. What it costs, on the 8-view example recording: 3D
residual **+22%**, 2D reprojection **+0.14 px** in 7 of 8 views, and the left/right gap in
each DOF's median angle **4.7 → 6.4 deg** — the opposite of what sharing was supposed to fix.
All five points of a leg are tracked, so the chain is over-determined and the per-leg lengths
already *are* the best fit to the data; the femurs measure 4–6% apart on all three pairs
(always left-longer, a stable offset), and that gap is femur-specific rather than a per-side
rig error. Keep it on when you want one animal (comparing angles across sides, driving a
simulation), and turn it off when you want the best fit to *these* pixels:

```bash
deeperfly config set inverse_kinematics.symmetric_segments false -c config.toml
```

How the sharing is done, and the per-DOF breakdown behind those numbers, is in
[left/right symmetry](../reference/configuration.md#ik-symmetric-segments).

!!! warning "`constant_points` is gone"

    `[inverse_kinematics].constant_points` held named keypoints at their temporal median
    before the fit. It is superseded by [`{ op = "static" }`](../reference/configuration.md#op-static)
    in `[postprocess]`, which is on by default now and does the same thing where it belongs —
    to the finished 3D, logging how far it moved what it touched. A config still carrying the
    key is **refused** as an unknown key rather than ignored.

## Tune the opt-in stage — pictorial structures

`pictorial_structures` is the one stage off by default, and not for symmetry. It recovers a
joint from the **top-K peaks** of a detector that predicted one body side; a dense detector
already predicts every point in every view. Switching it on rewires triangulation *and* the
smoother onto its committed 2D, which is NaN in every view with no candidate within 15 px —
so on a dense run it would un-densify the result — and it adds a `candidates` key to the
`pose2d` fingerprint, re-detecting every cached tree in existence.

```toml
[pictorial_structures]   # peak recovery before triangulation
k        = 5       # candidate peaks per joint
temporal = false   # add a temporal-consistency term
lam      = 1.0     # bone-length prior weight
```

Candidate peaks are extracted during detection and cached in `results.h5` when this
stage is enabled. Enabling it on an existing output directory therefore re-runs
`pose2d` once (announced loudly); after that, tweaking `temporal` / `lam` re-runs
only the recovery from the cached candidates. Resuming with `pose2d = false`
from a 2D result that stored no candidates skips the stage with a notice.

## Output videos — `[visualization]`

Each `[visualization.videos.<name>]` is one output MP4 — **keyed by name**, so the table
key *is* the filename and a duplicate is a TOML error rather than two videos overwriting
each other. A video is a `grid` of camera cells with `layers` over them:

```toml
# What every video is unless it says otherwise.
[visualization.default_video]
background = "black"
crop       = "pose2d"   # every cell shows the window its own camera detects through
cell       = [480, 240]
# output_fps = 30       # explicit output fps
# speed      = 0.5      # or scale the input fps instead (0.5 = slow motion)

# What every layer is unless it says otherwise.
[visualization.default_layer]
line_thickness = 2
```

Resolution is exactly **default → explicit**, once per collection. The old three-level
merge (`[visualization.kwargs]` → per-video `kwargs` → per-panel keys) is gone, and with it
the namespace collision where `width` meant either a draw argument or a panel size
depending on where it was written. `[visualization]` itself now holds nothing.

The packaged config ships **four** montage videos — `pose2d` (the raw detections), `pose3d`
(the triangulated 3D reprojected back into each view), and `pose_model` / `mesh_model`
(the fitted model's skeleton and mesh, which need `[pipeline] inverse_kinematics`):

```toml
[visualization.videos.pose3d]
grid   = [["rf", "f", "lf"], ["rm", "bird", "lm"], ["rh", "h", "lh"]]
layers = [{ draw = "skeleton_3d", stage = "triangulation" }]
```

**Footage is implicit.** A camera cell gets its own frame underneath; `footage = false`
drops it. That removes the commonest mistake in the old shape — a grid whose `imshow`
panel and overlay panel disagreed about `crop`. `""` leaves a gap, and a cell whose name is
not a camera gets no footage: `"bird"` is a synthetic dorsal plan view fitted to the
animal's own body axes from the 3D, the one viewpoint showing all six legs with no body in
the way, which the rig cannot have because the tether is up there.

**Layers draw in order**, which is how a before/after goes in one video:

```toml
[visualization.videos.pose_model]
grid   = [["rf", "f", "lf"], ["rm", "bird", "lm"], ["rh", "h", "lh"]]
layers = [
    { draw = "skeleton_3d", stage = "postprocess",      # dashed, underneath: the target
      line_thickness = 1, line_dash = [4, 9], point_radius = 2 },
    { draw = "skeleton_model", point_radius = 3 },      # solid, on top: the fit
]
```

The same ordering puts a skeleton over a `mesh_model` grid, where it is not optional: a
skeleton drawn *under* a translucent surface is a smear. `edge_color` draws every edge in
one colour with the joints keeping theirs, which reads better under a mesh.

**Name the `stage`.** Left out it means "the most-derived stage present" — so with `eks`
on, a video named `pose3d` silently becomes the smoother's output and `pose2d` stops
showing the detector at all. Naming it is how each video keeps meaning one thing, and the
only way to render a before/after pair.

`crop = "pose2d"` is worth reaching for on any rig with a camera the detector crops (an
axial view, typically): it resolves *per camera* from `[pose2d.crops]`, so a single line
frames each cell the way its own detector saw it and leaves the full-frame cameras alone.
That keeps the box in one place — a searched crop differs per recording, and a copy of the
numbers here would silently keep showing the previous recording's window. See the
[configuration reference](../reference/configuration.md#visualization) for the full schema.
Video frames are read and written with PyAV.

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
precision     = "float16"   # the default: ~1.5-2x over float32 under CUDA autocast.
                            # "bfloat16" trades that speed for a wider range;
                            # "float32" is the reference. Ignored on CPU/MPS
batch_size    = 16          # GPU forward batch (images/forward); throughput plateaus
                            # by ~16 on a fast GPU
decode_buffer = 4           # decode queue depth, in multiples of batch_size
```

`precision` is a *default*: an explicit `[pose2d] precision` overrides the class's, and a
class that pins its own ignores both — `mvt` runs in float32 and refuses anything else. It
is result-affecting, so the resolved value is fingerprinted and changing it re-detects.

`batch_size` is in **images, not frames**: a forward takes `batch_size // cameras` whole
frames, so on the packaged eight-camera rig anything below 8 is one frame per forward, which
is why the default is not smaller. It plateaus by 32 (measured on an RTX 4090, 8 views at
256×512, on the r27 transformer: 52.9 fps at 8, 59.4 at 16, 60.1 at 32, 58.9 at 64 — the
packaged r28 default pads its input and is about 1.9× slower at every value). `decode_buffer` is a *memory*
knob (peak frames per camera is `~(decode_buffer + 2) * batch_size`) — raise it to keep the
GPU fed when decode is jittery, lower it to shave memory. Neither ever invalidates a cache.

These are the `[pose2d]` table's performance knobs; *what* to detect — the detector's
`class` and `weights`, and the per-camera `[pose2d.crops]` — is documented above, and the
footage is each camera's own `video`.

## Frame I/O — `[io]`

Video files are read and written with PyAV (in-process FFmpeg, on the CPU); image
sequences are decoded with OpenCV. The only knob is the image-decode thread
count:

```toml
[io.image]
workers = 0   # decode threads (0 = one per CPU)
```

The reader/writer API is in the [library guide](library.md#frame-io).

## Letting the crop be measured — `auto_crops` { #auto-crop }

A crop's right value is a property of *this* recording. A detector is trained through a
box, and a differently framed camera puts the animal at the wrong
apparent scale — which no augmentation in the recipe undoes. On this rig the six side
cameras match training full-frame and the two axial ones (front and hind, 1600×1008 against
the side cameras' 960×512) do not, so those are the two that usually need one.

Rather than copy last recording's numbers, ask for them to be measured:

```toml
[pose2d]
auto_crops = ["f", "h"]        # search these two; `h` has no seed, so it searches blind

[pose2d.crops]
f = { x = 400, y = 290, width = 800, height = 400 }   # a SEED: search near a box you
                                                      # already trust (fewer probes)
```

The `pose2d` stage resolves it before detecting: the detector's own confidence covers the
`(center, width)` space cheaply (~11 ms a probe), then agreement with the **other cameras'
3D** — the target view held out of the triangulation — chooses among what confidence
proposed, and refuses a box that is confidently wrong. Measured blind on a 1600×1008 hind
camera: 255 px from the other cameras' 3D at full frame, **2.9 px** after the search,
against 3.4 px for a box a person tuned by hand.

**The search needs no rig; the accept gate does.** These are two different steps and it is
worth keeping them apart, because the first is what makes this usable on a recording you
have never calibrated. The search is confidence-scored and needs nothing. The gate is the
reprojection check, and confidence cannot stand in for it: confidence decouples from
accuracy once the center is free — one recording that went 0.338 → 0.403 confidence went
30.9 → 37.5 px **worse**, and on this rig's hind view the two signals correlate at
r = +0.17.

So the gate self-checks. Against a nominal orbit rig it measures its own reference about
**250 px** out, says so, and refuses rather than choose with a broken ruler; the run then
falls back to confidence alone and lands a box roughly **1.7× too wide** — which is still
far better than handing the detector the whole frame. Run once, `deeperfly calibration
export`, point `[calibration] path` at the result, and the gate engages.

**The box is recorded**, in `<outdir>/autocrop.json`, so a resume neither re-searches nor
re-detects, and the video cells that borrow it (`crop = "pose2d"`) keep working in a later
process. Anything that asks an *unresolved* automatic crop for its geometry raises rather
than quietly falling back to the whole frame.

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
points              = ["l?_*", "r?_*"]   # point selectors driving BA (default: all)
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

## Camera rig geometry — `[default_camera]` and `[cameras.*]` { #cameras }

A `[cameras.<name>]` is one camera: its geometry (intrinsics + orbit extrinsics) **and its
footage**. The cameras orbit an object near the world origin; `[default_camera]` is merged
into every one of them, and each `[cameras.<name>]` overrides it. A camera's intrinsics
describe the **raw** frame of its own footage, since detections are mapped back to raw
pixels before they meet a camera.

The rig table split three ways so each half says what it is: `[calibration]` is a solved
rig, `[default_camera]` is what every camera shares, and `[cameras.<name>]` is a camera. A
camera called `defaults` used to be indistinguishable from the shared table.

The packaged rig declares **eight** cameras — `rh`, `rm`, `rf`, `f`, `lf`, `lm`, `lh`, each
setting just its footage and `azimuth_deg`, and `h`:

```toml
[default_camera]
focal_length_px = 22388.125               # scalar when fx == fy, else [fx, fy]
distance        = 107.463
# principal_point_px = [479.5, 239.5]     # omit: inferred per view from its own frame

[cameras.f]
video       = 'camera_F.mp4'
azimuth_deg = 0

[cameras.h]                                # the axial hind view, on its own lens
video           = 'camera_H.mp4'
azimuth_deg     = 180
focal_length_px = 24168.591
distance        = 158.9168
```

**`h` is the rig's only left/right bridge**, and the reason the shipped detector was trained
on eight views: without it no camera sees both body sides, so a contralateral joint is
triangulated from one side's cameras alone.

**Measure `h`'s two numbers on your own rig.** It sits on a different lens, which changes
**both** the focal length and the distance, and no residual will find the error for you:
bundle adjustment holds the intrinsics fixed, so it absorbs a focal error into the distance
and then reports a clean solve. The shipped values are the example rig's. A recording that
has no such camera needs no edit here — the run
[narrows itself](../reference/configuration.md#narrowing-to-the-footage-present) to the
footage it finds and says so.

`mirror` is **gone** and refused by name. It named the camera seeing this one's mirror
image, for an out-of-tree flip augmentation; nothing in the package has read it since 0.2,
and it is derivable from `azimuth_deg` on any rig where it holds at all.

`[calibration] path` points at a **solved** rig — a `calibration.toml` from a previous
run (`deeperfly calibration export`) or a board — and **wins** over the orbits, which are
then read only for their order:

```toml
[calibration]
path = "calibration.toml"   # resolved relative to THIS file
```

That is how a rig travels between recordings: an orbit is a human's description of it, a
calibration a solver's measurement, and raw extrinsics cannot be written in a view table at
all. A calibration covering only *some* of the config's cameras subsets with a warning;
covering none of them, it belongs to a different rig and is refused.

The orbit parameters (`look_at`, `distance`, `azimuth_deg`, `elevation_deg`,
`roll_deg`) and intrinsics are detailed in the
[reference](../reference/configuration.md#cameras).

!!! warning "`[cameras.<name>].preprocess` is gone"

    A detection window is `[pose2d.crops]`, where the transform is **inverted on the way
    back** — so a detection reaches its camera in raw footage pixels however it was
    windowed to get to the model, and the camera's intrinsics go on describing the raw
    frame. The retired key instead moved the *camera* into cropped-pixel space, and both at
    once is a silent double correction: a fly reprojecting off by exactly the crop offset,
    with nothing to point at.

    It had been accepted-and-silently-ignored for several releases, which is the worst place
    for it to be, because a crop is exactly what a wrongly-framed axial camera needs. It is
    now a **hard error** naming its replacement, [`[pose2d.crops]`](#the-detection-window--pose2dcrops).

## Skeleton — `[skeleton]`

`fly38` is the **one** packaged skeleton: 38 points — six 5-point legs
(`thorax_coxa` → `coxa_trochanter` → `femur_tibia` → `tibia_tarsus` → `pretarsus`),
`l_antenna` / `r_antenna`, `neck`, and a 5-point **dorsal-midline** abdomen chain
`abdomen0`…`abdomen4`. 16 left/right symmetry pairs.

**A config normally says nothing at all here.** A skeleton is four things — `points`,
`edges`, `point_symmetries`, colours — in a version-controlled file of its own, and the run
resolves it from the detector's recorded skeleton name. Write the table only to override
that:

```toml
[skeleton]
include = "skeleton.toml"   # a project's own, resolved next to this config
```

Keys written alongside `include` replace that skeleton's **per key**. There are no limbs
and no groups: where a group name used to say "these five points", the answer is a **point
selector** — a point name or a `*` pattern, with an exact name beating a pattern:

```toml
[skeleton.point_colors]
"lf_*"     = "#0f7399"   # the five points of the left front leg, and their edges
l_antenna  = "#0a4f6b"
```

An edge nothing names takes the average of its two endpoints' colours, so that table
colours the whole drawing. `[skeleton.edge_colors]` overrides one, keyed by an
`"<a>--<b>"` endpoint pattern (`"abdomen*--abdomen*" = "#404040"`).

The same grammar drives `[bundle_adjustment] points` and the `static` / `symmetrize` ops.
A pattern matching nothing is a hard error (always a typo) and the resolved set is logged,
because over-matching is the one failure a selector cannot catch itself.

Which camera sees which point is not set here either — a dense plan makes every camera
see every point.

**It must be the skeleton the detector was trained on.** A dense detector's channels *are* a
skeleton, so the wrong one attaches points to the wrong joints. That is checked against the
checkpoint every run by the ordered point **names**, not by this label — which is what makes
it a real check: these points were called `fly38b` before 0.2, and the DeepFly3D set that
`fly38` named until then was *also* 38 points, sharing 32 of these in a different order. A
count could not tell them apart. deeperfly prints a skeleton as `fly38@42da66d9` — name
plus a digest of the points, edges and symmetry pairs — so two that share a name still read
apart.

!!! note "`fly38b` still resolves, and the DeepFly3D set is retired"

    `include = "fly38b"` loads exactly these 38 points (`config.SKELETON_ALIASES`) and logs
    that it did, so old configs and old run snapshots keep loading. The alias is deliberately
    **not** symmetric: an old config naming `fly38` means the *other* point order, and no
    alias can disentangle one word meaning two things — which is why the real guard is the
    name-list check above and not the label.

    The historical DeepFly3D set — two 3-marker abdomen *side* chains `l_abdomen0…2` /
    `r_abdomen0…2`, no `neck` — is no longer packaged. It survives as a complete skeleton
    file in the test data, so `include = ".../tests/data/fly38_deepfly3d.toml"` keeps a
    config written against it running. It has **no binding** to the packaged model, so the
    inverse-kinematics stage refuses the pair by name rather than fitting against all-NaN
    observations; `deeperfly ik bind` generates one. The
    [reference](../reference/configuration.md#ik-binding) has the detail.

Edit this only to track a different animal — see the
[reference](../reference/configuration.md#skeleton) and the
[pipeline explainer](../explanation/pipeline.md).
