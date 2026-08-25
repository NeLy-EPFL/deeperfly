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

**Dense is the only plan shape now.** One pathway per camera, the model emitting every
tracked point for every view, so channel *i* is point *i* of that pathway's view and no
mapping table is written at all. Selecting a detector is two lines. A contralateral point
arrives as a prediction to correct rather than as a gap to author from nothing.

The [`[pose2d.output_points]`](../reference/configuration.md#output_points) mapping table
still exists and is still supported — it is how one view can be fed by *several* pathways —
but the shipped configs declare none. (The historical 19-channel detector predicted one
body side, ran each side camera twice and needed 122 hand-written mapping rows. It is gone:
there is no `class = "hourglass"` / `"deepfly2d"`, no `deeperfly.pose2d.model`, no
`deeperfly.load_detector`, and nothing auto-downloads any more.)

**Sources** name the footage, the one setting almost every recording needs. Each
`filename` is a glob matched **case-insensitively** inside the recording directory, or a
list of alternates tried in order — which is how the packaged config accepts both the
anatomical file name and the DeepFly3D index:

```toml
[[sources]]
name     = "vid_rh"
filename = ["camera_RH.mp4", "camera_0.mp4"]   # RH = right hind; falls back to the index
[[sources]]
name     = "vid_h"
filename = ["camera_H.mp4", "camera_7.mp4"]    # the axial hind camera
[[sources]]
name     = "vid_rm"
filename = "camera_1"       # a bare prefix -> "camera_1*": a video or an image sequence
```

A source's footage is one video file or a naturally-sorted image sequence
(`camera_1_000123.jpg ...`), decoded once however many views read it. A source with no
`filename` defaults to its own name. The packaged config declares **eight**: one per view
of the [eight-camera rig](#cameras).

**A partial recording is a recording.** A source that matches no files no longer makes the
directory malformed: the footage that *is* there is reported with a warning naming what is
absent, and the run [narrows itself](cli.md#inputs-single-batch-recursive) — dropping that
source, the pathways reading it and the views they fed. What still warns and skips is a
directory that cannot be read coherently: several footage extensions in one folder, or an
unequal file or frame count across the sources that are present.

`[[sources]]` blocks, not one inline array under a bare top-level key: `deeperfly project`
lifts whole TOML *tables* by their headers, and a bare key is not a header.

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

**Models** select a detector network. Only two keys are really yours: `class` and
`weights`.

```toml
models = [{ name = "dense38mv", class = "mvt", weights = "mvt_r28_pad48_gray_fly38.pth" }]
```

Two classes, both dense — every tracked point in every view:

| `class` (aliases) | what it is |
| --- | --- |
| `"mvt"` (`"multiview_transformer"`) | encodes a frame's views **together**, so a joint only one camera can see informs the ones that cannot. Pinned to float32. What the packaged config names. |
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
without the YUV→RGB conversion that is most of what reading a frame costs. The gray decode
needs *every* model in a plan to declare it accepts gray; `hrnet` did not, which silently
cost every hrnet/hgnet plan the fast path until 0.2.

**The channel order is checked against the checkpoint on every run**, by the ordered point
*names* — not by a count. A count cannot tell two 38-point skeletons apart: `fly38` and the
retired DeepFly3D set share 32 points in a different order, so routing one through the
other's config attaches six points to the wrong joints and shifts the rest. That is a wrong
limb, not a crash. A checkpoint that records **no** channel names at all is refused outright.

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

**Where the outputs land** needs no table in a dense plan: channel *i* is point *i* of the
view the pathway is named after. Declare a `[pose2d.output_points.<view>]` table only when
one view is fed by **several** pathways and you have to say which of them owns which point:

```toml
[pose2d.output_points.f]                  # one view fed by two pathways, disjoint points
rf_femur_tibia = { pathway = "f",      out_channel = 2 }   # un-flipped
lf_femur_tibia = { pathway = "f_flip", out_channel = 2 }   # mirrored
```

Keying on `(view, point)` makes every point's data come from exactly one place (a
duplicate is a config error); a `(view, point)` no entry names is left unobserved
(NaN) — that union *is* the visibility, with no separate table. See the
[reference](../reference/configuration.md#output_points) for the full form and the
left/right check that guards it.

This modularity still supports a range of setups: per-view or per-side specialized
models, or a different `model` per pathway.

## Choose which stages run — `[pipeline]`

The pipeline is a linear sequence of stages, each an on/off `do_<stage>` switch:

```toml
[pipeline]
do_pose2d               = true   # detect 2D pose in every camera view
do_bundle_adjustment    = true   # refine the cameras (bundle adjustment)
do_pictorial_structures = false  # DeepFly3D-style peak recovery -- the one stage off
do_triangulation        = true   # triangulate 2D -> 3D
do_eks                  = true   # ensemble Kalman smoother over the 3D
do_postprocess          = true   # corrections from knowing the animal (assumes TETHERED)
do_inverse_kinematics   = true   # fit NeuroMechFly joint angles
do_visualization        = true   # render the videos
```

**Every stage is on by default except `pictorial_structures`**, and the packaged config
states each flag at exactly the value it would inherit — explicit for readability, not
because it changes anything. Each enabled stage has its own top-level `[<stage>]` parameter
table (below).

To turn one off without opening the file:

```bash
deeperfly config set pipeline.do_eks false -c config.toml
```

That works as of 0.2 and did not before — see
[turning a stage off](cli.md#turning-a-stage-off) for why.

Three of these deserve a sentence before you keep them.

[`[eks]`](../reference/configuration.md#eks) fits one 3D trajectory per keypoint to the
whole recording, which de-jitters the pose (measured: **39% less** frame-to-frame
acceleration) and repairs blown detections — at the cost of lagging a keypoint that moves
faster per frame than the detector localizes it, which at 100 fps means a claw in swing
(0.13% of cells reproject >100 px, and every one is a claw). Turn it off if claw timing is
the measurement.

[`[inverse_kinematics]`](../reference/configuration.md#inverse_kinematics) fits a
NeuroMechFly model's joint angles to the 3D pose. It needs the optional `ik` extra, whose
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
(`deeperfly config show inverse_kinematics`). Two things about it are worth knowing here;
the rest is in the [reference](../reference/configuration.md#inverse_kinematics).

**Both chains are targeted at `fly38`**, so one body plan covers all 38 points. The
abdomen's five markers are the dorsal-midline tergite stripes, placed on the model at the
same body + offset the [keypoint viewer](../explanation/keypoints.md) draws them at — read
from the same file, so what the labeling reference shows and what the IK fits are the same
choice. A different labeling scheme retargets a chain with a marker table
([`[inverse_kinematics.head]`](../reference/configuration.md#ik-markers) /
`[inverse_kinematics.abdomen]`), which **replaces** that chain's markers, so list all of
them.

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
crop        = "pose2d"   # every panel shows the window its view's detector looked through
# output_fps = 30    # explicit output fps for every video
# speed      = 0.5   # or scale the input fps instead (0.5 = slow motion)

[visualization.kwargs]   # draw-op defaults shared by every video
imshow      = { width = 480, height = 240 }
skeleton_2d = { line_thickness = 2, width = 480, height = 240 }
skeleton_3d = { line_thickness = 2, width = 480, height = 240 }
```

The generated config ships **four** montage videos — `pose2d` (the raw detections),
`pose3d` (the triangulated 3D reprojected back into each view), and `pose_nmf` /
`pose_mesh` (the fitted NeuroMechFly skeleton and mesh, which need
`do_inverse_kinematics`). Each is laid out with `grid` rather than explicit `panels`, on the
eight-camera rig, right-side cameras in the left column so the montage reads as the animal
from above:

```toml
[[visualization.videos]]
video_name = "pose3d"
plot  = "skeleton_3d"
stage = "triangulation"
grid  = [["rf", "f", "lf"], ["rm", "bird", "lm"], ["rh", "h", "lh"]]
```

`grid` expands to an `imshow` plus the named `plot` per cell with the offsets computed; `""`
leaves a gap, and a cell whose view is not a camera gets no imshow — `"bird"` is a synthetic
dorsal plan view fitted to the animal's own body axes from the 3D, the one viewpoint showing
all six legs with no body in the way, which the rig cannot have because the tether is up
there. Add an explicit `panels` list for a layout a grid cannot say; it draws on top.
Draw-op kwargs merge across three levels (global → per-video → per-panel), most specific
winning. Video frames are read and written with PyAV.

**Name the `stage`.** Left out it means "the most-derived stage present" — so with
`do_eks` on, a video named `pose3d` silently becomes the smoother's output and `pose2d`
stops showing the detector at all. Naming it is how each video keeps meaning one thing, and
the only way to render a before/after pair.

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
precision     = "float16"   # the default: ~1.5-2x over float32 under CUDA autocast.
                            # "bfloat16" trades that speed for a wider range;
                            # "float32" is the reference. Ignored on CPU/MPS
batch_size    = 16          # GPU forward batch (images/forward); throughput plateaus
                            # by ~16 on a fast GPU
decode_buffer = 4           # decode queue depth, in multiples of batch_size
```

`precision` is a *default*: a per-model `[[pose2d.models]].precision` overrides it, and a
class that pins its own ignores both — `mvt` runs in float32 and refuses anything else. It
is result-affecting, so the resolved per-model value is fingerprinted and changing it
re-detects.

`batch_size` is in **images, not frames**: a forward takes `batch_size // pathways` whole
frames, so on the packaged eight-camera rig anything below 8 is one frame per forward, which
is why the default is not smaller. It plateaus by 32 (measured on an RTX 4090, 8 views at
256×512, on the r27 transformer: 52.9 fps at 8, 59.4 at 16, 60.1 at 32, 58.9 at 64 — the
packaged r28 default pads its input and is about 1.9× slower at every value). `decode_buffer` is a *memory*
knob (peak frames per camera is `~(decode_buffer + 2) * batch_size`) — raise it to keep the
GPU fed when decode is jittery, lower it to shave memory. Neither ever invalidates a cache.

These are the `[pose2d]` table's performance knobs; *what* to detect (sources, models,
pathways — including per-model `weights`) is the detection plan, which shares the same
`[pose2d]` table (and the top-level `[[sources]]`) and is documented above.

## Frame I/O — `[io]`

Video files are read and written with PyAV (in-process FFmpeg, on the CPU); image
sequences are decoded with OpenCV. The only knob is the image-decode thread
count:

```toml
[io.image]
workers = 0   # decode threads (0 = one per CPU)
```

The reader/writer API is in the [library guide](library.md#frame-io).

## Preprocessor op grammar — `[[pose2d.preprocessors]]` `ops` { #preprocessors }

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

### Letting the crop be measured — `auto = true` { #auto-crop }

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
export`, point `[cameras].calibration` at the result, and the gate engages.

**The box is recorded**, in `<outdir>/autocrop.json`, so a resume neither re-searches nor
re-detects, and the panels that borrow it (`crop = "pose2d"`) keep working in a later
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

## Camera rig geometry — `[cameras.defaults]` and `[cameras.*]` { #cameras }

A `[cameras.<name>]` is a geometric **view** that a pathway maps its points back
into — pure geometry (intrinsics + orbit extrinsics), no footage or
preprocessing. The cameras orbit an object near the world origin;
`[cameras.defaults]` is merged into every view, and each `[cameras.<name>]`
overrides it. A view's intrinsics describe the **raw** frame of the source feeding it,
since detections are mapped back to raw pixels before they meet a camera.

The packaged rig declares **eight** views — `rh`, `rm`, `rf`, `f`, `lf`, `lm`, `lh`, each
setting just its `azimuth_deg`, and `h`:

```toml
[cameras.defaults]
focal_length_px = 22388.125               # scalar when fx == fy, else [fx, fy]
distance        = 107.463
# principal_point_px = [479.5, 239.5]     # omit: inferred per view from its own frame

[cameras.f]
azimuth_deg = 0
mirror      = "f"                          # a midline camera is its own mirror image

[cameras.h]                                # the axial hind view, on its own lens
azimuth_deg     = 180
focal_length_px = 24168.591
distance        = 158.9168
mirror          = "h"
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

`mirror` names the view seeing this one's mirror image. It is a **training** key only —
flip augmentation relabels a flipped sample with the mirrored camera — and a run drops it.
It must be an involution, and it is written out rather than computed because that it is
`-azimuth` here is a coincidence of a symmetric layout.

`[cameras].calibration` points at a **solved** rig — a `calibration.toml` from a previous
run (`deeperfly calibration export`) or a board — and **wins** over the orbits, which are
then read only for their order:

```toml
[cameras]
calibration = "calibration.toml"   # resolved relative to THIS file
```

That is how a rig travels between recordings: an orbit is a human's description of it, a
calibration a solver's measurement, and raw extrinsics cannot be written in a view table at
all. A calibration covering only *some* of the config's cameras subsets with a warning;
covering none of them, it belongs to a different rig and is refused.

The orbit parameters (`look_at`, `distance`, `azimuth_deg`, `elevation_deg`,
`roll_deg`) and intrinsics are detailed in the
[reference](../reference/configuration.md#cameras).

!!! warning "`[cameras.<name>].preprocess` is gone"

    Frame ops moved to the detection pathway, where the transform is **inverted on the way
    back** — so a detection reaches its camera in raw footage pixels however it was
    windowed to get to the model, and the camera's intrinsics go on describing the raw
    frame. The retired key instead moved the *camera* into cropped-pixel space, and both at
    once is a silent double correction: a fly reprojecting off by exactly the crop offset,
    with nothing to point at.

    It had been accepted-and-silently-ignored for several releases, which is the worst place
    for it to be, because a crop is exactly what a wrongly-framed axial camera needs. It is
    now a **hard error** naming its replacement:
    [`[pose2d].preprocessors`](#preprocessors) plus a
    pathway's `preprocessor`.

## Skeleton — `[skeleton]`

`fly38` is the **one** packaged skeleton: 38 points — six 5-point legs
(`thorax_coxa` → `coxa_trochanter` → `femur_tibia` → `tibia_tarsus` → `claw`),
`l_antenna` / `r_antenna`, `neck`, and a 5-point **dorsal-midline** abdomen chain
`abdomen0`…`abdomen4`. 16 left/right symmetry pairs.

```toml
[skeleton]
name = "fly38"
```

A table that declares no `point_names` is read as a *reference* and expanded at load, so
naming a preset loads its `point_names`, its `limb_points` kinematic chains, its
`limb_palette` and its symmetry pairs; write any key to override one, or `file =
"skeleton.toml"` for a project's own. Which view sees which point is not set here — it is
the union of the pathway maps, which a dense plan makes total.

**It must be the skeleton the detector was trained on.** A dense detector's channels *are* a
skeleton, so the wrong one attaches points to the wrong joints. That is checked against the
checkpoint every run by the ordered point **names**, not by this label — which is what makes
it a real check: these points were called `fly38b` before 0.2, and the DeepFly3D set that
`fly38` named until then was *also* 38 points, sharing 32 of these in a different order. A
count could not tell them apart.

!!! note "`fly38b` still resolves, and the DeepFly3D set is retired"

    `name = "fly38b"` loads exactly these 38 points (`config.SKELETON_ALIASES`) and logs
    that it did, so old configs and old run snapshots keep loading. The alias is deliberately
    **not** symmetric: an old config naming `fly38` means the *other* point order, and no
    alias can disentangle one word meaning two things — which is why the real guard is the
    name-list check above and not the label.

    The historical DeepFly3D set — two 3-marker abdomen *side* chains `l_abdomen0…2` /
    `r_abdomen0…2`, no `neck` — is no longer packaged. It survives as a complete skeleton
    file in the test data, so `file = ".../tests/data/fly38_deepfly3d.toml"` keeps a config
    written against it running. Expect less of the IK there: it covers 32 of the 38 points
    the packaged articulation targets, and gets no abdomen fit and no measured head or
    abdomen size. Nothing fails, and both are warned about — the
    [reference](../reference/configuration.md#skeleton-presets) has the detail.

Edit this only to track a different animal — see the
[reference](../reference/configuration.md#skeleton) and the
[pipeline explainer](../explanation/pipeline.md).
