# Output format

A run writes everything to its output directory (`<input>/deeperfly_outputs/` by
default, or `-o`):

```
deeperfly_outputs/
├── results.h5           # the result: cameras, skeleton, per-stage 2D/3D data
├── config.toml          # byte-for-byte snapshot of the config this run used
├── calibration.toml     # the bundle-adjusted rig, portable to other recordings
├── autocrop.json        # the window each automatic crop resolved to, and the evidence
├── run.json             # per-stage fingerprints (drives cache reuse)
├── labels.h5            # ground-truth annotations from 'deeperfly gui' (if any)
├── labels_suggest.json  # frames to label next, from 'deeperfly labels-suggest'
└── *.mp4                # one per [[visualization.videos]] entry
```

## `results.h5`

A self-contained HDF5 file (schema **version 4**). Each pipeline stage writes its
own group, so a stage never overwrites another's data and any downstream stage
can be re-run later from pristine upstream outputs. The file fully reconstructs
the cameras and skeleton, so results are portable without the original config.

Arrays use the [view-leading layout](../explanation/conventions.md#array-layouts);
`NaN` encodes missing observations / un-triangulated points. Point arrays are *stored*
as float32 and deflated, and every reader hands them back as float64, so no consumer's
arithmetic silently changes precision because of which build wrote the file — see
[what a stage stores](#what-a-stage-stores) for why the narrower dtype costs nothing.

```text
attrs["meta"]               json: {deeperfly_format_version: 4, created_utc, ...}
skeleton/                   attrs["name"], attrs["digest"]; point_names, edges,
                            point_symmetries, point_colors, edge_colors
animal/                     absent (P,) bool + attrs["subject_id"] -- written only when
                            something is declared; NOT a stage, so no recompute drops it
pose2d/
    points                  (V, T, P, 2)  arg-max 2D detections (NaN for a (view, point)
                                          no pathway writes -- with a dense plan, none)
    conf                    (V, T, P)     detection confidences
    cameras/                the config rig as built at detect time
    attrs["image_sizes"]    json {camera_name: [h, w]} of the raw footage frames
    attrs["footage"]        json {camera_name: {abs, rel, names, bytes}}, so a viewer
                            handed only this file can find the videos
    candidates/             top-K peaks (xy, score) -- only if pictorial_structures
                            was enabled at detect time (it is off by default)
bundle_adjustment/
    cameras/                the BA-refined rig, with its units / scale_source /
                            provenance / quality
pictorial_structures/
    points                  (V, T, P, 2)  PS-corrected 2D
    points3d                (T, P, 3)     initial 3D estimate
    reproj_error            (V, T, P)
triangulation/
    points                  (V, T, P, 2)  cleaned 2D (outlier-rejecting methods)
    points3d                (T, P, 3)
    reproj_error            (V, T, P)     against its own cleaned 2D
eks/
    points3d                (T, P, 3)     the smoothed 3D trajectory
    posterior_var           (T, P, 3)     per-axis posterior variance (world units^2)
    smooth_param            (P,)          the fitted process-noise scale per keypoint
    attrs["meta"]           json {method, n_members, n_inflated, n_testable, ...}
postprocess/
    points3d                (T, P, 3)     3D after the correction chain
    points2d_override       (V, R, 2)     the 2D the ops froze in pixel space
    points2d_override_cols  (R,)          which skeleton columns those are
    attrs["meta"]           json {pose_from, ops: [{op, ...what each op measured}]}
inverse_kinematics/
    angles                  (T, D)        fitted joint angles (radians)
    angle_names             (D,)          the angle names, in column order
    points3d                (T, P, 3)     fitted model joints (world; skeleton order)
    body_plan               scalar str    the QuickIK body plan that was solved (JSON)
    attrs["meta"]           json {template, solver, alignment, chain_scales,
                                  chain_offsets, body_scale}
```

Which groups are present depends on which stages ran. A group exists only once its
stage completed; only the stages that were enabled (and whose inputs were
available) appear. Every stage but `pictorial_structures` is on by default, so a
default run writes all of the above bar `pictorial_structures/` and `candidates/`.

### What a stage stores, and what it rebuilds { #what-a-stage-stores }

Every stage used to keep a full `(V, T, P, 2)` 2D array and a full `(V, T, P)` error,
which on an eight-view 2007-frame recording is 14.6 MB per stage before any of them
says anything new. Since **v3** a stage stores only what cannot be rebuilt from what
its neighbors already store, and the classification is *measured* per write rather than
declared per stage — so a new stage, or an op that starts moving pixels per frame, gets
the right answer with no schema change. Each points group records which case it was in,
as `attrs["points2d_storage"]`:

| `points2d_storage` | means | what lands on disk |
| --- | --- | --- |
| `derived` | the 2D **is** `cameras.project(points3d)` | nothing (the smoother's is exactly that) |
| `override` | that projection except on a few columns held constant over time | just those constants |
| `full` | an independent pixel measurement | the whole array |

The correction chain is the `override` case: its `static` op freezes the neck and the
six thorax-coxae in *pixel* space, and projection is nonlinear, so those seven columns no
longer agree with it — 112 numbers on an eight-view rig instead of the 9.8 MB the whole
array took in v2. The detections, the pictorial candidate selection and triangulation's
outlier-cleaned observations are `full`. So is a 2D whose `NaN` pattern differs from the
projection's: "this view did not see it" is not interchangeable with a reprojected
coordinate, so such an array is kept whole. That is why the smoother's 2D is dropped on a
dense run, where every cell is observed, but stored whole on a run with gaps.

`reproj_error` is dropped only when a recomputation reproduces it **and** the stage's 2D
was not stored whole. The second half is a safeguard rather than an optimization: the
stored error is the only witness that an outside tool overwrote a stage's 2D with
something other than the detections (`deeperfly.labels_suggest` reads it for exactly that),
and a recomputed error agrees with the stored 3D by construction, so it could never
disagree with itself.

None of this reaches a reader. Every reader prefers a stored array and reconstructs only
what is missing, so `StageStore.read_points("eks")` hands back the array the smoother
produced whether or not the file holds it — which it has to, since the correction chain
asks for exactly that. The one case that cannot be served is a *dropped* 2D in a file
whose rig is missing: the reader returns `None` rather than a guess.

What it buys, measured on that eight-view 2007-frame recording: 65.9 MB as v2 against
15.6 MB now, 4.2x. Most of that — 3.4x of the 4.2x — is the narrower dtype and the
dropped arrays, which cost almost nothing to read; deflate buys the last 24% and is what
makes `PoseResult.load` 64 ms where v2 was 7. The trade is taken deliberately: 64 ms to
open a recording is nothing next to the 157–252 ms a single video frame costs to decode,
and nothing reads these arrays in a hot loop (turning the filter off is the one knob to
reach for if something ever does — it keeps 3.4x). And 28 of those 64 ms are not
decompression at all but the reprojection rebuilding the dropped 2D. float32 resolves
a coordinate to ~6e-05 px against the ~1 px the detector can actually localize, while the
small arrays whose precision *is* load-bearing — a rig's rvecs/tvecs/intrs, the skeleton,
the 112 frozen numbers — stay float64 and uncompressed.

### Schema versions and `deeperfly repack` { #schema-versions }

Writing is always v4; **v2 and v3 are still read**, so an existing corpus opens
unchanged. A v2 file stores every array, so it simply never takes the reconstruction path;
a v3 file differs from v4 only in two dataset names inside `skeleton/`, which the reader
takes either way. That is why the readers need no version branch, and why an old file and
a repacked copy of it hand back the same 2D and error to within the float32 storage step.

What a v2 file cannot do is be *extended*. `StageStore.has()` reports every stage
incomplete on one, so a run recomputes from `pose2d` (whose write truncates the file and
makes the whole thing current) rather than appending current groups beside v2 ones in a
file whose recorded version names only one of them. Reading an old file is safe; extending one
is not.

[`deeperfly repack`](../guides/cli.md#deeperfly-repack) is the migration, and the way to
keep an old file's contents without recomputing them: it reads whatever this build can
read and writes the current version, narrowing the point arrays and leaving out every 2D
and error a reader can rebuild. **Nothing is recomputed** — the pose in the file is the pose that comes out.
Groups this schema knows nothing about are copied through rather than dropped: a
`dfpose_predict/` group is somebody else's record of what they did to the file, and losing
it during a *space* optimization would be the same silent loss of provenance the
reprojection-error rule above exists to prevent.

!!! warning "A file from a newer deeperfly is refused, not ignored"
    Everything that opens a result — `PoseResult.load`, the staged run, `repack` — raises
    on a file whose recorded version is *higher* than this build's. Treating it as absent
    would be silent destruction: every `has(stage)` would report false, the run would
    recompute `pose2d`, and that truncating write would take the newer file with it,
    reporting success. Regenerable is not the same as disposable, and only the older
    direction is the former. A pre-v2 (or non-deeperfly) file *is* treated as absent by
    the staged run, which recomputes it; `load` refuses it naming the version it found.

### The joint angles

`angle_names` are the flygym joint names `<parent_body>-<child_body>-<dof>`: e.g.
`c_thorax-rf_coxa-{yaw,pitch,roll}` / `rf_coxa-rf_trochanterfemur-{pitch,roll}` /
`rf_trochanterfemur-rf_tibia-pitch` / `rf_tibia-rf_tarsus1-pitch` for a leg,
`c_thorax-c_head-{yaw,pitch,roll}` for the head, and `c_thorax-c_abdomen12-{pitch,roll}`
… `c_abdomen5-c_abdomen6-{pitch,roll}` for the abdomen (the head/abdomen columns are
present only when `[inverse_kinematics] chains` selects them). With `fly38` and both on
that is 55 columns: seven per leg, three for the head, two for each of the abdomen's five
hinges.

The values **are flygym's own joint angles**, in flygym's sign convention on both sides —
feed a frame's leg columns to NeuroMechFly and you get the pose back. That holds because
the model's axes are used verbatim, including the sign flip it applies to the right legs;
the check is `test_the_leg_parameterisation_is_flygyms`, which fits the model's own neutral
keypoints and requires the angles to come back as the model's own resting angles. Results
written before that fix carry a different convention: the leg subtree was rotated by the
coxa-derived body frame (≈23° of pitch away from the model's), two axes were swapped, and
the mid and hind legs were reported in the mirror branch of the `(yaw, roll)` double cover.

Read the dof names as flygym's labels for the *axes* rather than as descriptions of the
motion — on both the legs and the abdomen they are anatomically rotated: `yaw` is the
segment-local *x*, `pitch` its *y*, `roll` its *z*. So on the abdomen `pitch` is the
sagittal ventral curl and `roll` the lateral swing, and the twist (`yaw`) is deliberately
not fitted at all, because it aliases the lateral bend on midline markers.

A branch whose observations cannot support a fit is `NaN` for that frame rather than
filled with the solver's neutral-biased guess, so "not fitted" stays distinguishable from
"fitted straight". The bar is two observed keypoints in the frame, with two deliberate
adjustments: a keypoint declared **absent** (an amputation, an ablation) lowers it to what
the branch can still deliver — a unilateral antenna ablation keeps the head fitted, where
a flat threshold of two would NaN all three head DOFs forever — but only as far as
`3 × n_fittable ≥ n_dofs`, so a leg amputated down to its coxa stays unfitted; and a
missing *detection* never lowers it, because a leg the detector merely lost is a tracking
failure and reporting neutral-biased angles for it would be a fabricated measurement. A
chain's base landmark is not counted at all: the head's `neck` sits on the very center the
three head DOFs turn about, so no angle can move it.

`points3d` carries the model's prediction for every fitted keypoint in skeleton order, so
it reprojects with the skeleton's own edges. With `fly38` that is every column — 30 leg
joints, two antennae, the `neck` and the five abdomen markers are all markers of the plan
— so a `NaN` there is a frame that could not be fitted, never a point the model has no
opinion about.

`body_plan` is the kinematic tree the fit was solved on, with this recording's
**measured** segment lengths baked into it — both a record of what was fitted and what
the editor's live re-fit re-solves on, so it cannot drift from the stored result. It is a
dataset rather than a meta key because it runs to tens of kilobytes (about 20 KB for
`fly38`), close enough to the 64 KB an HDF5 attribute allows to be worth keeping out.

The meta's `alignment` is what was measured from this recording: the body frame, each
leg's median thorax-coxa origin and its segment lengths. With
[`symmetric_segments`](configuration.md#inverse_kinematics) on — the default — a leg and
its mirror image carry one shared row of `seglens`, so the fitted model is one animal
rather than two half-animals.

`chain_scales` (`{"head": …, "abdomen": …}`) is the head/abdomen size relative to the
model — one uniform multiplier per chain, fitted against that chain's whole marker set
together with its posture and reduced by the median over sampled frames (see
[Chain size](configuration.md#ik-chain-size)). `chain_offsets`
(`{"head": [x, y, z], "abdomen": […]}`) is the companion translation: where that chain's
base really sits, in model units. A chain that names a base landmark is *measured* — the
head is placed on its median `neck`, the way each leg is placed on its median thorax-coxa
— and one that names none (the abdomen: no keypoint sits on its root) has its root
**fitted** from the whole marker set, jointly with the size, because the two trade along
an exact null direction of the fit and measuring either with the other held wrong gives a
confidently repeatable wrong answer for both. Both are baked into `body_plan` **and**
handed to the overlay, because the fit and the drawing have to describe one pose; a chain
drawn about the model's anchor while its angles were fitted about the measured one is off
by exactly this vector. A file written before chains had base landmarks carries no
`chain_offsets`, which correctly reads as no shift.

`body_scale` is the recording's
body size relative to the model, from the single coxa registration that also places the
body plan; the mesh overlay holds the body, head, and abdomen at this fixed size and
varies only rotation + translation per frame, so the body does not breathe.

### The smoother's and the chain's residuals

The `eks/` group's `reproj_error` means something different from the others', and
deliberately. Its 2D **is** `points3d` reprojected, so measuring one against the
other would be identically zero; the residual is taken against the `pose2d`
observations instead, and so reads "how far the smoother moved from the raw
detection". That is also why the column need not be stored: it is exactly what a reader
recomputes from the stored 3D and the `pose2d` observations, so v3 leaves it out and
rebuilds it unless the two would differ (triangulation's, measured against its own cleaned
2D, is stored).
`posterior_var` is the smoother's own uncertainty — it grows through
stretches no view could pin the keypoint down and shrinks where the views agree — and
is what makes this an *uncertainty-aware* output rather than just a smoothed one.
The reconstructed 2D keeps `NaN` wherever the detector observed nothing, unless
[`[eks].fill_unobserved`](configuration.md#eks) is on.

The `postprocess/` group is the pose after the
[`[postprocess].ops`](configuration.md#postprocess) chain — the corrections that come from
knowing the animal rather than the pixels. Its `reproj_error` is taken against the `pose2d`
observations for the same reason the smoother's is. The `meta` records `pose_from` (the
stage the chain was applied to: `eks`, else `triangulation`, else `pictorial_structures`,
so the un-corrected pose is still on file in its own group) and `ops` — **one entry per op,
in order**, since the same op may appear twice and what the second measured depends on what
the first did. Each entry carries that op's own configuration plus what it measurably did:
for `static`, the median and p90 drift removed in each space, `moved_2d_median_px_per_point`
and `worst_point`; for `symmetrize`, the fitted `plane_normal` / `plane_offset` and each
pair's `asymmetry_before`. Those numbers are the only check on an op's premise — a point
that had been drifting tens of pixels was moving, and does not belong in a `static` list.

### The rig and the skeleton

A `cameras/` group stores `names`, `rvecs`, `tvecs`, `intrs` (`[fx, fy, cx, cy]`),
`dists`, `dist_lengths` and `image_sizes`. `dist_lengths` is each camera's *true*
distortion length: `CameraGroup` zero-pads every camera to the group-wide maximum by
contract, which is harmless numerically (all-zero coefficients are the identity) and a
textual round-trip failure, since a camera authored `dist = []` would read back as five
zeros. The bundle-adjusted rig's group additionally carries `units`, `scale_source`,
`provenance` and `quality` as attributes — the same record
[`calibration.toml`](#calibrationtoml) holds, so an exporter passes a rig's real
provenance through instead of inventing one.

The `skeleton/` group stores the skeleton's `name` and `digest` (attributes) plus
`point_names`, `edges`, `point_symmetries` (`(S, 2)` left/right mirror pairs — see
[`point_symmetries`](configuration.md#point_symmetries)), `point_colors` (one hex colour
per point) and `edge_colors` (one per edge). The name is whatever the config resolved at
detect time, so a file written before the skeleton was renamed records `fly38b` and loads
unchanged (`fly38b` is an alias of `fly38`).

`digest` is written for a human reading the file or an error quoting it, and is **never
compared** — a stored digest disagreeing with the recomputed one would mean a corrupt
file, not another skeleton. What is compared is `point_names`.

`edge_colors` is stored rather than re-derived because the endpoint average is only the
*default*: an edge the skeleton coloured explicitly would come back a different colour if
it were reconstructed.

`limb_names`, `limb_id` and the `palette/` subgroup are **no longer written**: the limb
concept went with the declaration that created it. A file that has them reads back on its
`point_names` and edges as before, and one that lacks `point_colors` reads back on a
colormap, which is acceptable because colours are cosmetic.

!!! note "The stored format still says `limb_id` in one place"

    `_write_skeleton` / `_read_skeleton` keep the name for the on-disk dataset while the
    config and the code say `chain`. One documented mismatch, and this is where a
    format-versus-code naming difference belongs — but it is a mismatch.

That record is what gives the `P` axis its meaning: every array here is `(..., P, ...)`
with no names beside it. So a run whose config resolves a **different ordered point set**
is refused before anything is read or detected, naming both sets and what is only in each.
A count check cannot see this — two 38-point skeletons load each other's files perfectly
happily — and neither can the skeleton's name, which is why the check compares the ordered
`point_names`. The same choice is why renaming a preset costs nothing: the name is in no
stage fingerprint either.

`point_symmetries` is additive: a file written before it existed simply has no such
dataset and loads as a skeleton with no declared pairs, so a pair-driven consumer falls
back to inferring pairs from the point names rather than refusing to open the file. So is
`edge_colors`, which a file without takes from its points.

v4 **renamed** two datasets inside `skeleton/` (`bones` → `edges`, `symmetries` →
`point_symmetries`). Nothing needs converting — the arrays are identical — so the reader
takes either spelling and `repack` is not required. The version bump is for the other
direction: an older build reading a v4 file would find no `skeleton/bones` and fail with a
bare `KeyError`, and refusing on the version is exactly what the version is for.

### What the library reads back

`PoseResult.load(path)` assembles the **most-derived data present**, so you get
the best result without knowing which stages ran. A stage counts as having a 2D when one
can be *rebuilt*, not only when one is stored, so the preference order is unaffected by
what v3 chose to leave out:

| Field | Preference order |
| --- | --- |
| `pts2d` | `postprocess` → `eks` → `triangulation` → `pictorial_structures` → `pose2d` |
| `pts3d` | `postprocess` → `eks` → `triangulation` → `pictorial_structures` |
| `reproj_error` | `postprocess` → `eks` → `triangulation` → `pictorial_structures` |
| `cameras` | `bundle_adjustment` → `pose2d` (config rig) |
| `conf` | `pose2d` |
| `model_pts3d`, `model_angles`, `model_angle_names`, `model_body_plan`, `model_chain_scales`, `model_chain_offsets`, `model_body_scale` | `inverse_kinematics` |
| `absent` (the whole-recording `(P,)` declaration), `subject_id` | `animal/` (both `None` when nothing is declared) |

`PoseResult.save(path)` is the library one-shot (no staged groups): it writes
`pts2d`/`conf` to `pose2d/` and, when a 3D pose is present, the 2D/3D/error to
`triangulation/`, so `load` round-trips the assembled view. Its 2D is stored whole
in both groups — an assembled result's 2D is whatever stage produced it, and the one-shot
writer has no upstream group to reconstruct it from. It also rewrites `animal/`, because
it is a whole-file rewrite and a load → mutate → save round trip would otherwise drop the
declaration. An *uncalibrated* result (no rig — the editing scaffold a from-scratch
annotation session starts from) raises instead of being written: a `results.h5` without
cameras would claim a pipeline ran.

## `config.toml` (snapshot)

The exact config text that drove the run, copied byte-for-byte for
reproducibility. On a later run, `-c` wins when given (and refreshes this
snapshot); without `-c`, this snapshot is reused — so you can edit it in place and
re-run with just `-o`.

It records what was **asked for**, not what the run narrowed itself to. A recording
missing a camera narrows the run's parsed config (dropping that source, its pathways, the
views nothing feeds, and the grid cells naming them), but the snapshot still describes the
rig the operator configured — so an eight-camera config over a seven-camera recording
leaves a snapshot naming eight cameras beside a `results.h5` holding seven views. The
narrowed plan is what reaches the fingerprints, which is what makes the eighth camera
turning up later recompute rather than validate.

## `calibration.toml`

The rig the `bundle_adjustment` stage solved, written as a standalone file (schema
**version 1**). `results.h5` already holds these cameras — but only as arrays
inside *one* recording, where nothing can point a second recording at them, diff
them against a later solve, or read their residuals without opening HDF5.

```toml
[calibration]
format_version = 1
name        = "IN07B001_260417_Fly4_004"
created_utc = "2026-08-03T11:22:00+00:00"
units        = "config"       # "arbitrary" | "mm" | "config"
scale_source = "orbit_prior"  # "none" | "orbit_prior" | "bone_prior" | "known_distance" | "board"

[calibration.provenance]      # how this rig was produced
method     = "labels_ba"      # "orbit_prior" | "labels_ba" | "board" | "imported"
intrinsics = "config"
frames     = 100
source     = ".../deeperfly_outputs/results.h5"
# plus `solver` (the least-squares settings that produced it) and, when this rig
# refined one from [cameras].calibration, `refined_from = { name, path }`

[calibration.quality]         # how well it fits — read this before trusting it
rms_reproj_px    = 1.84
median_reproj_px = 1.33
p90_reproj_px    = 3.16
max_reproj_px    = 7.80
n_observations   = 12040

[calibration.quality.per_camera_rms_px]
rh = 1.62
# ...

[calibration.cameras.rh]      # world -> camera is R(rvec) @ X + tvec
rvec       = [0.0, 0.0, 0.0]
tvec       = [0.0, 0.0, 107.463]
intr       = [22388.125, 22388.125, 479.5, 255.5]   # [fx, fy, cx, cy]
dist       = []
image_size = [512, 960]       # [height, width] the intrinsics describe
```

Three things make it safe to reuse, and each exists because of a specific way a
shared rig goes wrong:

- **`image_size`** — intrinsics are *pixel* quantities, so a rig applied to
  rescaled or cropped footage would silently misproject every point. Loading a
  calibration whose frame size disagrees with the footage in hand is refused.
- **`provenance`** — `method = "orbit_prior"` means the rig was never solved, only
  written down. A file that did not say so would be indistinguishable from one
  that was bundle-adjusted.
- **`quality`** — a rig without its residuals is a number you cannot refuse. The
  block is *empty* rather than zero when nothing was observed, because zeros read
  as a perfect fit.

**`units` is deliberately not assumed.** Bundle adjustment started from a config
orbit inherits that orbit's scale, but deeperfly was never told what `distance`
measures — so the unit is recorded as `"config"` rather than guessed to be `"mm"`.
A rig solved from correspondences alone with nothing to fix the scale is
`"arbitrary"`: fine for angles, meaningless for velocities. `units` and `scale_source`
are *inherited* when the run refined a calibration it was pointed at, so refining a
millimeter board solve does not relabel it as an arbitrary-scale orbit guess.

A rig covering only some of the config's cameras is **subset** to the ones it does
cover, naming the missing ones in a warning — a view the rig never measured cannot be
placed, so it is dropped like a view with no footage rather than refused, since one config
routinely describes more rig than one solve covers. A rig covering *none* of them is
refused: that is the wrong-rig case, and a subset cannot explain it away.

Point a config at one with [`[cameras].calibration`](configuration.md#cameras);
extract one from any existing result with `deeperfly calibration export`.

## `autocrop.json`

Which window each `{ op = "crop", auto = true }` preprocessor resolved to, and the
evidence for it. The packaged config gives the front (`crop_f`) and axial-hind (`crop_h`)
views one, so a default run writes this file; the search behind it, and how to read the
numbers, is [`deeperfly auto-crop`](../guides/cli.md#deeperfly-auto-crop).

```json
{
 "version": 1,
 "boxes": { "crop_f": [0, 258, 1232, 616], "crop_h": [547, 26, 699, 350] },
 "detail": {
  "crop_f": {
   "view": "f", "box": [0, 258, 1232, 616], "incumbent": [0, 247, 1286, 643],
   "seeded": false, "conf": 0.98, "conf_incumbent": 0.9785,
   "accepted": true, "gated": false, "probes": 462, "seconds": 5.5,
   "search_frames": [0, 1003, 2006],
   "notes": ["the geometry gate could not run: ...", "so this box was chosen on confidence alone -- check the pose2d overlay"]
  }
 }
}
```

`boxes` is keyed by **preprocessor** name (a preprocessor may be shared, and it is what
the config declares) and holds `[x, y, width, height]` in raw footage pixels. `detail`
adds, per target, the incumbent the search started from, the confidence of both boxes,
whether the search moved off the incumbent (`accepted`), how many probes it cost, which
frames it looked at, and — when the reprojection **accept gate** ran (`gated`) — the
agreement in pixels of both boxes and the rig residual the gate measured. `notes` is where
a gate that could *not* run says so and says what to do about it (solve the rig, then point
[`[cameras].calibration`](configuration.md#cameras) at the exported `calibration.toml`);
the example above is a run with no solved rig, whose box was therefore chosen on
confidence alone and is worth checking against the `pose2d` overlay.

The window is deliberately **not** part of the `pose2d` fingerprint: it is a
deterministic function of the footage, the weights and the seed, all of which are
fingerprinted already or fixed by the output directory, so recording it there would make
every search look like a config change and re-detect forever. This file is instead how a
later process learns what the detector looked through — a re-render that reuses the cached
2D loads it at startup, which is what lets a visualization panel crop the same way the
detection did. An unreadable or wrong-version file is ignored with a warning and the crops
are searched again.

## `run.json`

A small JSON sidecar recording, per stage, the **fingerprint** (the
result-affecting config subset) and the completion time. It is outdir-local run
state, kept out of the portable `results.h5`; deleting it merely recomputes
everything.

```json
{
  "format_version": 1,
  "stages": {
    "pose2d":        { "fingerprint": { "...": "..." }, "completed_utc": "2026-..." },
    "triangulation": { "fingerprint": { "...": "..." }, "completed_utc": "2026-..." }
  }
}
```

On a re-run a stage is reused only when its recorded fingerprint still matches the
current config **and** its output is present. Comparison is *subset* semantics: a
key dropping out of the expected fingerprint (e.g. `candidates` when
`pictorial_structures` is disabled again) does not invalidate the cache, while a
changed or newly-appearing key does. Performance-only knobs (`batch_size`,
`decode_buffer`, `[io.image]`) are deliberately excluded. Fingerprints are stored
verbatim (not hashed) so a mismatch can be reported as a readable diff. See
[caching and re-runs](../explanation/pipeline.md#caching-and-re-runs) for the full
model.

Three deliberate choices about what a fingerprint sees:

- **The skeleton's name is not in it** — the ordered `point_names` and the `edges` are.
  Two skeletons agreeing on both compute the same result whatever they are called, so
  renaming a preset must not buy a full re-detection. (`Skeleton.digest` hashes the same
  two, plus the symmetry pairs, and is the printable form of that same identity.)
- **The rig is, including the calibration's *contents*** — not just the path, because
  re-solving a calibration rewrites it under the same name, which is the common case and
  the one a path alone cannot see.
- **The views are the narrowed ones.** Narrowing happens before anything is fingerprinted,
  so a run that proceeded on seven views records a seven-view fingerprint and recomputes
  when the eighth camera turns up. Fingerprinting the full rig against a short result
  would let the next complete run reuse it.

Because subset semantics mean a *dropped* key cannot invalidate a cache, a behavior change
with no config key of its own has to announce itself with one: the `eks` and
`inverse_kinematics` fingerprints carry hand-bumped revision numbers for exactly that (the
IK one is at 2, for the leg re-parameterization — revision-1 angles are a different
quantity, not a worse fit of the same one).

## `labels.h5`

Ground-truth annotations authored in [`deeperfly gui`](../guides/gui.md), written
next to `results.h5` and **never** modifying it (labels schema **version 8**; every older
version is read and migrated on load, and a file from a newer build is refused rather than
misread). Only what the operator actually authored is stored — sparsely (COO), so the file
is tiny and carries no copy of the predictions. Group by group, with the version each
arrived in:

```
attrs["meta"]   json: { deeperfly_labels_format_version, created_utc, identity,
                        subject_id }        # subject_id added in v3; may be null
gt/                              the affirmed labels -- LIVE rows only
    index       (N, 3) int32     [view, frame, point]
    xy          (N, 2) float64   affirmed 2D pixel (footage space)
seeds/                           v8: the instance's evidence-backed start positions
    index       (S, 3) int32
    xy          (S, 2) float64
instance/                        v8: frames carrying an annotation skeleton
    index       (J,)   int32
occluded/                        the HIDDEN flag -- cells held out of the training loss
    index       (M, 3) int32     [view, frame, point]
reviewed/                        v2: frames the operator ticked reviewed
    index       (K,)   int32
absent/                          v3; missing in a v1/v2 file -> nothing absent
    spans       (Q, 3) int32     [point, t0, t1) run-length runs of absence (v4);
                                 the authoritative form
    index       (Q',)  int32     the points absent in EVERY frame -- v3's whole
                                 representation, still written so a v3 reader sees
                                 the whole-recording declarations
    void_gt/                     rows the declaration vetoes, kept so it can be lifted
        index   (N', 3) int32
        xy      (N', 2) float64
    void_occluded/
        index   (M', 3) int32
```

There is no `landmarks/` group. The calibration-landmark namespace was removed with the
schema: the animal is the calibration target, which is what every rig here was solved
from, and no recording ever carried one.

**Absence** (`absent/`) is "this joint is not on this animal" — categorically different
from `occluded` ("it exists but no camera here can see it") and from unlabeled ("nobody
has looked yet"). It is view-independent by construction (an amputated joint is missing
from every camera at once), which is why it is not stored per cell; it *is* per frame,
since a limb can be lost part-way through a recording. `spans` is the authoritative
run-length form, so a whole-recording declaration — the common case — costs one row
however long the recording is, while `index` keeps the whole-recording subset that a v3
reader understands.

`gt/` and `occluded/` hold **live rows only**: rows an absence declaration vetoes are
moved to `absent/void_*`. So a consumer that reads `gt/index` straight out of HDF5 sees a
self-consistent file with no labels on keypoints that do not exist, while nothing authored
is lost — un-declaring a point restores its rows on the next load.

A GT row carries **no provenance**: ground truth is created by dragging a point from a
proposed position, and which proposal it started from is not a property of the label that
results. The proposal layers are still readable separately (`seeds/`, and the result file's
own detections and reprojections), which is where a consumer wanting that precedence
should look. Files written by v5/v6 did carry a per-row `gt/provenance`, and it is read on
load for one purpose only: dropping the invented placeholder-seed rows, which were never
exportable.

`identity` fingerprints the recording (skeleton points, camera names, frame count,
image sizes, footage basenames) so a sidecar from a *different* recording is refused;
it excludes the predictions and `created_utc`, so re-running detection/triangulation
on the same recording keeps the labels valid (ground truth is absolute, not relative
to what the network predicted). It deliberately excludes `absent`, which is
point-indexed and so already domain-checked by the `point_names` match — which is also
what lets one animal's declaration be copied across all of its recordings. Export the
labels as a training/eval `.npz` with
[`deeperfly labels-export`](../guides/cli.md#deeperfly-labels-export).

## `labels_suggest.json`

The active-learning queue written by
[`deeperfly labels-suggest`](../guides/cli.md#deeperfly-labels-suggest): which frames
a human should correct next, ranked by the multi-view disagreement of the detector's
own 2D. JSON rather than HDF5 because it is a handful of nested, human-facing records
(kilobytes) that should be readable with `less` — and because a JSON sidecar can
never be mistaken for a pipeline stage group in `results.h5`. Written atomically
(`tmp` + `os.replace`); `results.h5` and `labels.h5` are only ever read.

```
deeperfly_suggestions_format_version   1  (an unknown version reads as *absent*)
created_utc, deeperfly_version
params      every knob, plus min_gap_frames and a one-line statement of the score
source      what was scored: results md5 + size + mtime_ns, cameras_from
            (bundle_adjustment | pose2d), scored_array (always "pose2d/points") and
            what is never scored, the camera names, n_views / n_frames / n_points, the
            recording `identity`, and -- when the directory came from `dfpose.predict`
            -- `reseed`, including what the stored reproj_error *would* have said on
            the substituted cells
labels      the labels.h5 md5 and its labeled / reviewed frame sets
coverage    scorable cell + joint fractions, median observing views, the global
            residual level and the score percentiles
shortfall   requested vs selected, most_wrong vs diversity, and why it came short
excluded    the frames skipped as already-labeled, and how many were unscorable
frames[]    rank, frame, t_s, score, percentile, kind (most-wrong | diversity) and
            a `reason`: the driving joints with their worst camera, pixel error,
            how many views disagree, and a geometric near/far side flag
```

Every input's fingerprint travels with the queue, so a reader (the GUI) can tell a
stale queue from a fresh one without recomputing anything: a differing `identity`
means a different recording, a differing `results_md5` means superseded predictions,
and a grown labeled-frame set is just normal progress.
