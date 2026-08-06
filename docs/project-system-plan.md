# Projects, GUI-first operation, from-scratch rigs, and the road to multi-animal

**Status:** design / plan (pre-implementation).
**Scope:** a new `deeperfly.project` layer; `config.py`, `cameras.py`, `recordings.py`,
`results.py`, `gui/**`, `web/**`, `cli/**`; a new `calibration` stage.
**Date:** 2026-08-03.

This document was written from a full read of the current package (24 k LOC, 786 tests) and a
survey of how SLEAP, DeepLabCut, Lightning Pose and Anipose structure the same problem. It
answers four asks:

1. a **project system** — related work in one place, several recordings per project, projects
   that merge (today the labels are scattered across `~/fly-pose-data`);
2. **everything doable from the GUI** — the config is comprehensive but overwhelming;
3. **projects from scratch** — add videos, label per-view with no calibration, then run bundle
   adjustment *from the GUI* once there are enough correspondences, optionally using dedicated
   non-skeleton calibration landmarks;
4. **multi-animal**, planned for now and built later.

Sections 1–3 are the audit and the survey. Sections 4–10 are the design. Section 11 is the
phasing. Section 12 is the adversarial review — the places this plan is most likely to be
wrong. Section 13 lists the forks that need your call.

---

## 1. Guiding principles

1. **A project is an index and a set of shared artifacts — not a re-home of the data.**
   `results.h5` and `labels.h5` stay exactly where they are, beside each recording. The project
   adds a manifest, project-scoped artifacts (skeleton, rig, calibrations, landmarks), and the
   ability to *adopt* a recording that already exists. This is the single most important
   decision in the document, because it means **no migration of the precious hand labels** —
   the 9,175 GT points in `~/fly-pose-data` are adopted by reference on day one.

2. **Every existing command keeps working on a bare recording.** `deeperfly run rec/ -c cfg.toml`
   and `deeperfly gui rec/deeperfly_outputs` must not change behavior. The project is a *layer
   above*, never a precondition. This is what makes the change safe to land incrementally.

3. **The GUI's job is not to render 800 config keys.** It is to make the ~20 decisions that
   actually vary *nameable and safe*, and to give everything else a validated editor. The
   `*Params` dataclasses are already a schema with defaults and prose — generate the forms from
   them so the GUI cannot drift from the code.

4. **Calibration is an artifact, not a config section.** It is produced by a solve, has
   provenance, has a quality report, is shared across recordings on one rig, and is versioned.
   Nothing else in this plan works until this is true.

5. **2D observations remain the only source of truth; 3D stays derived.** The existing
   annotation model ([keypoint-editor-redesign.md](keypoint-editor-redesign.md)) is right and is
   extended, not replaced. An uncalibrated project is simply the degenerate case where the
   derivation is unavailable.

---

## 2. What the other packages actually do

### 2.1 SLEAP — the closest model, and the one to steal from

SLEAP keeps a whole project in **one HDF5 file** (`.slp`), with a documented schema:

```text
/metadata          attrs: format_id (1.4), json {skeletons, nodes, provenance}
/videos_json       JSON array of video backend configs  (referenced OR embedded)
/video{N}/video    embedded frames (PNG/JPEG-encoded or raw), when packaged
/frames            labeled-frame records (video, frame_idx, instance range)
/instances         instance records (instance_id, type user|predicted, frame_id,
                   skeleton, track, point_id_start/end, score)
/points            user points (x, y, visible, complete)
/pred_points       predicted points (+ score)
/tracks_json       [spawn_frame_idx, track_name]
/suggestions_json  {"video": "0", "frame_idx": 42, "group": 0}
/sessions_json     multi-camera calibration + synchronization
/negative_frames   frames explicitly containing no animals
```

Six ideas worth taking:

| SLEAP idea | Why it matters here |
|---|---|
| **`.pkg.slp` — self-contained package with embedded frames** | A project shareable as one file. Only 50 of 4,073 frames are labeled, so embedding labeled frames is cheap and makes merge/transfer trivial. |
| **`embed` policy: `"all"` / `"user"` / `"source"`** | Explicit control of "portable copy" vs "pointer to the original video", with `source_video` preserving lineage. |
| **`Labels.merge` + GUI "Merge Data From…"** | Merge is a *first-class, GUI-reachable* operation. SLEAP's rationale is exactly yours: *you can only train on one dataset*, so import before training. |
| **Suggestions with a `group` field** | Batches of suggested frames tagged by generation method. `labels_suggest.json` already exists here ([acquisition.py](../src/deeperfly/acquisition.py)); the `group` field is what makes several rounds coexist. |
| **`RecordingSession` / `CameraGroup` / `FrameGroup` / `InstanceGroup`** | The multi-view multi-animal data model, worked out. `FrameGroup` = one time point across views; `InstanceGroup` = the same animal across views; `Identity` = an animal across *sessions*, distinct from an in-video `Track`. |
| **`format_id` on the file, with documented migrations** | `labels.h5` is already at v5 with real migrations. Keep that discipline at project scope. |

Two SLEAP choices **not** to take:

- **One monolithic file for everything.** SLEAP embeds video frames because its unit of work is a
  frame. Here the unit of work is a *recording with seven synchronized videos*, `results.h5` is
  already a per-recording container, and footage lives on a lab share. A single file would force
  either gigantic files or a fragile embedded-path scheme. → **directory-with-manifest, plus a
  packaged single-file export.**
- **Skeleton edges typed by pickle-style enums.** A legacy wart they now recommend YAML over.

### 2.2 DeepLabCut — the directory convention

```text
project-name-experimenter-date/
    config.yaml               bodyparts, skeleton, videos, train/test split, iteration
    videos/                   symlinks by default (copy_videos=True to copy)
    labeled-data/<video>/     extracted frames, filename encodes source frame index
    training-datasets/iteration-N/
    dlc-models/
```

Take: **symlink/reference by default** (the videos are big and belong to the acquisition
system); **frame filenames that encode provenance** (traceable back to `(video, frame_idx)`);
and an **`iteration` counter** bumped on every merge, so a training set is always attributable to
a project state. Do not take: bodyparts as a flat YAML list with no identity — renaming a
bodypart in DLC silently invalidates data, which is precisely the failure mode §9 guards
against.

### 2.3 Lightning Pose — the two-directory split

A project is a **Data directory** (a copy of everything needed to train) plus a **Model
directory** (weights, inference metadata, prediction outputs). Two interfaces: `litpose` (CLI)
and an App (`litpose run_app`) for labeling, training and evaluation. The clean data/model split
is worth taking: a project's *labels* and a project's *models* have different lifetimes,
different sizes, and different sharing rules.

### 2.4 Anipose — the calibration file

`calibration.toml` holds per-camera intrinsics and extrinsics as a standalone artifact, produced
by a ChArUco/checkerboard solve with iterative bundle adjustment, and consumed by triangulation.
**This is exactly the artifact deeperfly is missing** (§5). Anipose's board flow is also the
right *fallback* for intrinsics when a from-scratch rig has no lens datasheet (§6.2).

---

## 3. Where deeperfly stands today — the audit

### 3.1 What already exists and is good

| Capability | Where | Note |
|---|---|---|
| Sparse, versioned GT sidecar with provenance, occlusion, absence, quarantine | [gui/labels.py](../src/deeperfly/gui/labels.py) (v5) | Genuinely excellent. The COO layout widens to multi-animal cheaply (§10). |
| Per-stage cached results with fingerprint-driven recompute | [results.py](../src/deeperfly/results.py), [pipeline/fingerprint.py](../src/deeperfly/pipeline/fingerprint.py) | The job model of §7.4 maps straight onto it. |
| Bundle adjustment with analytic Jacobians, robust IRLS losses, per-parameter fix/share, **bone-length priors** | [bundle_adjustment/](../src/deeperfly/bundle_adjustment/) | `build_state(fixed=["f.rvec", "rm.tvec[2]"])` is exactly the gauge fixing §6.4 needs; `bone_pairs`/`bone_targets` is exactly the scale bar. |
| Active-learning frame ranking with staleness tiers | [acquisition.py](../src/deeperfly/acquisition.py) | Already sidecar-based; needs only a `group` field and project scope. |
| A browser editor with writer-lock, undo, WebSocket edits, WebGL 3D | [gui/](../src/deeperfly/gui/) | The chassis for everything in §7. |
| Recording discovery, batch globbing, outdir planning | [recordings.py](../src/deeperfly/recordings.py) | `Recording` is already the right unit; it just has no home. |

### 3.2 The seven structural blockers

**B1 — Extrinsics cannot be expressed.**
[cameras.py:55](../src/deeperfly/cameras.py#L55) lists `rvec`, `tvec`, `rotation_matrix`,
`position`, `forward`, `up`, `center`, `eye` as **unsupported**, raising rather than ignoring
them; [`resolve_extrinsics`](../src/deeperfly/cameras.py#L146) requires an orbit `distance`.
`CameraGroup.from_arrays` exists but is reachable only from BA output and HDF5. **A calibration
produced from labels has no file it can be written to.** Blocks ask 3 entirely.

**B2 — The GUI cannot open anything but a finished result.**
[`build_session`](../src/deeperfly/gui/__init__.py#L75) does `PoseResult.load` → requires
`pose2d/points` ([results.py:272](../src/deeperfly/results.py#L272) raises without it) →
requires a trained detector and a rig. A from-scratch project has neither.
`EditorState.from_result` takes a `PoseResult`; `Session.build` derives `n_frames` from it.
Blocks ask 3.

**B3 — One config, four concerns, no composition.**
706 lines, 50 top-level tables, 132 `output_points` rows, snapshotted per outdir. There is no
way to say "same rig and skeleton, different triangulation" without copying the whole file.
Blocks ask 2 (and is the actual cause of the overwhelm).

**B4 — Label identity is exact-match and index-based.**
[`_check_identity`](../src/deeperfly/gui/labels.py#L434) refuses a sidecar whose `point_names`,
`camera_names` or `n_frames` differ, with the comment *"name-based remap of reordered
points/cameras is a future enhancement"*. `image_sizes` and footage basenames are also compared.
**That future enhancement is the merge machinery** (§8) and the skeleton-migration machinery
(§9). Blocks ask 1.

**B5 — No stable recording identity.**
A recording is identified by its directory path. `~/fly-pose-data/recordings/<rid>/` is stable by
convention only; `_backups/` already contains 20+ divergent copies of the same recordings. Merge
cannot deduplicate without a content-derived id. Blocks ask 1.

**B6 — Everything is shaped for one animal.**
`pts2d (V,T,P,2)`, `conf (V,T,P)`, `pts3d (T,P,3)`, `Labels.gt (V,T,P,2)`, `absent (T,P)`,
`DetectionPlan` mapping `channel -> (view, point)`. There is no instance axis anywhere. Blocks
ask 4.

**B7 — Training lives in another repo.**
`~/dfpose` owns the `predict → correct → retrain` loop and reads `labels.h5` directly
(`dfpose.datasets.labels_h5.ingest_labels`), aggregating snapshots through a hand-generated
`data/labels/MANIFEST.toml`. **That MANIFEST is a proto-project file** — it already tracks
per-recording GT counts, md5s, live-vs-snapshot paths and format versions. Per **F3** the
train/eval/dataset core is absorbed into `deeperfly.training` at Phase 5b, behind a
`deeperfly[train]` extra.

### 3.3 The scattering, concretely

```text
~/fly-pose-data/
    recordings/<rid>/{camera_*.mp4, recording_config.toml, deeperfly_outputs/{results,labels}.h5}
    predicted/<rid>_label/                 # a second, divergent copy per recording
    predicted/_label_snapshots/
    _backups/2026-07-29_round2b/<rid>__labels.h5
    _backups/refresh_<rid>_<timestamp>/{results,labels}.h5     # 20+ of these
    corpora/df3d_keypoints_clean/
~/dfpose/data/labels/{MANIFEST.toml, <rid>/labels.h5}          # committed snapshots
```

Four copies of some labels, three naming conventions, and a generated manifest to keep them
straight. The failure this invites is not "hard to find" — it is **editing the wrong copy**.

---

## 4. The project model

### 4.1 On-disk layout

```text
myproject/                              # a directory; the manifest names it
    project.toml                        # the manifest: identity, index, settings (§4.2)
    skeleton.toml                       # the project's skeleton — one per project
    rig.toml                            # source/view topology, per-camera preprocessing
    calibrations/
        2026-08-03_from-labels.toml     # a calibration artifact (§5)
        2026-08-03_from-labels.report.json
        current -> 2026-08-03_from-labels.toml      # symlink or manifest key
    landmarks.toml                      # non-skeleton calibration landmarks (§6.3)
    profiles/
        default.toml                    # algorithm knobs, layered over the packaged defaults
        fast-preview.toml
    recordings/
        <rid>/                          # one dir per registered recording
            recording.toml              # footage pointers, calibration ref, frame count, notes
            deeperfly_outputs/          # UNCHANGED: results.h5, labels.h5, config.toml, …
    models/                             # optional; detector checkpoints + their metadata
    exports/                            # training packages, npz, videos
    project.log
```

**Adoption, not migration.** `recordings/<rid>/deeperfly_outputs/` may be a **symlink** to an
existing outputs directory (the default for `deeperfly project add --link`), or a real directory
(`--copy`, or a fresh recording). Everything downstream reads the same paths it reads today, so
`dfpose.datasets.labels_h5` keeps working unmodified against a project.

**Footage is referenced, never copied** (DLC's default), recorded per camera as both an absolute
path and a path relative to the project root, matching what
[`write_pose2d`](../src/deeperfly/results.py#L410) already does for `results.h5`.

### 4.2 `project.toml`

```toml
[project]
format_version = 1
name          = "new-rig-38kp"
created_utc   = "2026-08-03T09:00:00Z"
id            = "prj_7f3a1c92"          # random, stable; survives renames and moves
description   = "Tethered Drosophila, 7-camera rig, all-keypoint labeling"
iteration     = 3                        # bumped by every merge/import (DLC's idea)

[project.skeleton]
path = "skeleton.toml"
# Content hash of the *resolved* skeleton (names + bones + limbs). Every label file in the
# project is checked against this; a mismatch triggers the §9 migration flow, never a
# silent load.
digest = "sha256:1a2b…"

[project.rig]
path = "rig.toml"
digest = "sha256:9c8d…"

[project.calibration]
current = "calibrations/2026-08-03_from-labels.toml"
# `null` is a first-class state: an uncalibrated project (§6.1).

[project.defaults]
profile = "profiles/default.toml"

[[recordings]]
id       = "rec_a91f4e07"                # content-derived, stable (§4.3)
slug     = "IN07B001_260417_Fly4_004"    # human-facing; may be renamed freely
path     = "recordings/IN07B001_260417_Fly4_004"
subject  = "IN07B001_260417_Fly4"        # animal identity across clips (already in labels.h5)
calibration = "calibrations/2026-08-03_from-labels.toml"   # may differ per recording
n_frames = 4009
fps      = 100.0
added_utc = "2026-08-03T09:04:11Z"
origin   = { kind = "adopted", from = "/home/tlam/fly-pose-data/recordings/IN07B001_…" }
```

### 4.3 Recording identity (fixes B5)

`id` is derived from content, not path, so the same recording adopted twice — or arriving through
a merge — is recognized as one thing:

```python
rec_id = sha256(
    b"\0".join(
        f"{camera}:{basename}:{size_bytes}:{n_frames}".encode()
        for camera, path in sorted(footage.items())
    )
).hexdigest()[:16]
```

Deliberately **not** a full content hash of the video bytes (minutes per recording over a network
share) and deliberately **not** including mtime (rsync churns it). Basename + byte size + frame
count is enough to distinguish real recordings and cheap enough to run on adoption. Collisions
are reported, never silently merged.

### 4.4 What the project *owns* versus what it *indexes*

| Owned by the project (single copy, project-scoped) | Indexed by the project (stays per-recording) |
|---|---|
| skeleton | `results.h5` |
| rig topology (sources, views, preprocessing) | `labels.h5` |
| calibrations + their reports | `labels_suggest.json` |
| calibration landmarks | rendered videos |
| algorithm profiles | the config snapshot each run wrote |
| suggestion rounds (`group`) | footage (referenced) |
| models + their training metadata | |

This split is what makes merge tractable: **only the owned column can conflict**, and it is
small.

### 4.5 The portable package (`.dfpkg`)

For sharing and merging, `deeperfly project export` writes **one HDF5 file** carrying the owned
column plus, per recording, the labels and *the labeled frames themselves* (SLEAP's `.pkg.slp`
insight):

```text
/meta                     attrs: format_version, project.toml verbatim, provenance
/skeleton                 (the existing _write_skeleton layout)
/rig, /calibrations/…, /landmarks
/recordings/<rec_id>/
    meta                  footage basenames, sizes, n_frames, fps, subject
    labels                the existing labels.h5 groups, verbatim
    frames/<camera>       (F, H, W[, 3]) uint8 — only the labeled/suggested frames
    frame_index           (F,) int32 — source frame index per row (DLC's provenance idea)
```

`--embed {all,user,none}` mirrors SLEAP's policy. `none` produces a small index-only package for
collaborators who share the lab filesystem; `user` embeds only frames carrying human labels — for
this project, **50 frames × 7 views ≈ 40 MB against 5.5 GB of footage**.

---

## 5. Calibration as a first-class artifact (fixes B1)

### 5.1 The file

```toml
[calibration]
format_version = 1
name        = "2026-08-03_from-labels"
created_utc = "2026-08-03T11:22:00Z"
units       = "arbitrary"        # "mm" | "arbitrary"  — see §6.4 on scale
scale_source = "none"            # "none" | "bone_prior" | "known_distance" | "board"

[calibration.provenance]
method      = "labels_ba"        # "orbit_prior" | "labels_ba" | "board" | "imported"
recordings  = ["rec_a91f4e07"]
frames      = 42
landmarks   = { skeleton = 380, static = 96 }
intrinsics  = "optics"           # "optics" | "board" | "guess" | "fixed"
solver      = { loss = "cauchy", f_scale = 4.0, max_nfev = 800, nfev = 431 }

[calibration.quality]
rms_reproj_px       = 1.84
per_camera_rms_px   = { rh = 1.62, rm = 1.71, rf = 2.05, f = 2.44, lf = 1.98, lm = 1.66, lh = 1.59 }
p90_reproj_px       = 3.9
min_pair_baseline_deg = 21.4
gauge               = { fixed = ["rh.rvec", "rh.tvec"], scale_fixed_by = "bone_prior" }

# Raw, absolute extrinsics — the thing cameras.py cannot express today.
[calibration.cameras.rh]
rvec  = [0.0, 0.0, 0.0]
tvec  = [0.0, 0.0, 0.0]
intr  = [1580.2, 1580.2, 479.5, 255.5]     # [fx, fy, cx, cy], RAW footage pixels
dist  = [-0.021, 0.004, 0.0, 0.0, 0.0]
image_size = [512, 960]                     # [h, w] the intrinsics describe
```

### 5.2 The code change

Add a **third construction path** alongside the orbit spec, rather than loosening the orbit
parser:

```python
# cameras.py
@classmethod
def from_calibration(cls, path_or_dict, *, image_sizes=None) -> CameraGroup: ...

def to_calibration(self, path, *, provenance: dict, quality: dict) -> None: ...
```

`resolve_extrinsics`'s rejection of `rvec`/`tvec` **stays**, and its error message gains a third
sentence: *"…or load a solved rig with `CameraGroup.from_calibration` / `[project.calibration]`."*
The rejection is a good guard — it stops a hand-edited orbit config from silently carrying
half-specified extrinsics; it just needed the other door to exist.

`Config.camera_group()` learns to defer: when the project (or `[cameras].calibration = "…"`)
names a calibration, it is loaded and the orbit specs are ignored **with a log line naming which
won**. The orbit spec is then re-read as what it always really was: a *prior*, used to initialize
BA (§6.4) and to render the rig diagram before a solve exists.

### 5.3 Why an artifact and not a config section

- One rig, many recordings: a calibration must be shareable across `[[recordings]]` without
  copying 706 lines.
- A calibration has **quality**, and a number without its residuals is a trap. `rms_reproj_px`
  and the per-camera breakdown belong *with* the parameters, so a downstream consumer can refuse
  a bad rig.
- Calibrations must be **comparable and revertible**: keeping `2026-08-03_from-labels.toml` next
  to `2026-08-10_board.toml` and switching `current` is how you find out that the far-leg
  reprojection error is co-visibility and not calibration — which is a question this lab has
  already had to answer once.

---

## 6. Projects from scratch: label first, calibrate second (fixes B2)

This is ask 3, and it is the hardest part of the plan because it inverts the pipeline's
dependency order. Today: *cameras → detect → triangulate → label*. From scratch:
*label → calibrate → (train) → detect → triangulate → label more*.

### 6.1 Uncalibrated mode

The editor must run with **no cameras, no predictions, no 3D**. Concretely:

- **`Session` gains a project-backed constructor** that does not require `results.h5`:

  ```python
  Session.for_recording(project, rec_id)   # results.h5 optional
  ```

- **`EditorState` gains a `cameras=None` path.** Everything geometric already funnels through a
  handful of methods — `_solve_point`, `_ensure_pts3d`, `display_pts3d`,
  `display_pts3d_projected`, `placeholder_pts2d`, `display_nmf_projected` — and each returns
  `None`/NaN when there is no 3D. The work is to make `has_3d = False` a *supported* state
  rather than an accident, and to hoist the `result.cameras` accesses behind a
  `state.cameras -> CameraGroup | None`.

- **`PoseResult` becomes optional in the session**, replaced by a small `Predictions | None`.
  With no result, `pts2d` is all-NaN, `conf` is `None`, and every view shows a bare frame.

- **The front-end drops the 3D panel and the reprojection layer** and shows an
  **"Uncalibrated"** banner with a live readiness meter (§6.5). The 2D editing surface —
  select→act, drag, occlude, absent, reviewed, undo — is *unchanged*, which is the point: the
  operator learns one editor.

**This is the largest single piece of work in the plan** and it is worth doing first, because it
is also what makes the editor usable on a recording whose detection failed.

### 6.2 Intrinsics — the honest part

Extrinsics can be recovered from correspondences. **Focal length essentially cannot**, not from a
few hundred hand-labeled points on a 3 mm deforming animal. Offer three sources, in this order,
and *name which one was used* in `[calibration.provenance].intrinsics`:

| Source | How | When |
|---|---|---|
| **`board`** | `deeperfly calibrate-intrinsics` over a ChArUco/checkerboard video per camera (OpenCV `calibrateCamera`), one-off per lens. GUI wizard with live corner detection. | The right answer. Anipose's flow. |
| **`optics`** | GUI form: lens focal length (mm) + sensor width (mm) *or* pixel pitch (µm) → `fx = f_mm · W_px / W_mm`. Two numbers off a datasheet. | Good enough to initialize; refine `f` in BA with a tight bound. |
| **`guess`** | `fx = fy = 1.2·max(W,H)` (≈45° horizontal FOV), `(cx, cy)` at the image center. | Last resort. Badged **estimated** everywhere it surfaces. |

Distortion: default all-zero; allow BA to free **`k1` only**. Freeing `k2`/`p1`/`p2` against
sparse hand labels is unidentifiable and will absorb real extrinsic error — the GUI should not
offer it without a board calibration present.

### 6.3 Calibration landmarks — your idea, and it is the load-bearing one

You proposed optional dedicated non-skeleton points for calibration. **This is not a
nice-to-have; it is what makes from-scratch BA converge**, for a reason worth stating plainly:

> A skeleton keypoint at frame *t* is a **different 3D point** from the same keypoint at frame
> *t+1* — the animal moves. So *N* labeled frames of *P* keypoints give `3·N·P` new unknowns
> alongside `6·(V−1)` camera unknowns. A **static** landmark — a scratch on the coverslip, the
> tether tip, a fiducial dot — is **one** 3D point observed in `V·N` images. It contributes
> `2·V·N` equations against 3 unknowns, and it is spread through the *scene volume* rather than
> concentrated in a 3 mm blob at the field center.

So landmarks carry a `static` flag, and it changes the solve, not just the bookkeeping:

```toml
# landmarks.toml — project-scoped, shared by every recording on this rig
[[landmark]]
name   = "tether_tip"
static = true            # ONE 3D point across all frames of a recording
color  = "#e8a33d"
note   = "the sharpened tip of the tungsten pin"

[[landmark]]
name   = "coverslip_corner_ne"
static = true
scope  = "rig"           # "rig": also shared ACROSS recordings that share the rig
                         # "recording": one 3D point per recording (default for static)

[[landmark]]
name   = "dust_speck_A"
static = true
scope  = "recording"

[[landmark]]
name   = "ball_top"
static = false           # per-frame, like a skeleton point (a moving fiducial)
```

`scope = "rig"` is the strongest constraint available: a landmark that does not move *between*
recordings ties every recording on the rig into one solve. It is also the most dangerous, so it
must be opt-in and the report must show its per-recording residual spread — a drifting rig shows
up there first.

**Storage.** A new `landmarks/` group in `labels.h5`, in its own namespace so the fingerprinted
`point_names` identity is untouched and every existing sidecar stays loadable:

```text
landmarks/                                    (labels.h5 schema v6)
    names       (L,)   utf-8   the landmark names, in order
    static      (L,)   bool
    index       (N, 3) int32   [view, frame, landmark]
    xy          (N, 2) float64 footage pixels
    provenance  (N,)   uint8   reuses Provenance (dragged / confirmed_*)
```

Keeping landmarks out of `Skeleton` is deliberate: they must never reach the detector's
`output_points`, the IK body plan, the bone-length priors, the training export, or the rendered
videos. A separate namespace enforces that by construction, where a "landmark limb" inside the
skeleton would need a filter at every one of those call sites.

### 6.4 The calibration solve

A new pipeline stage, `deeperfly.calibration`, sitting *before* `pose2d` and reachable
independently. Five steps.

**Step 1 — assemble observations.** From the project's labels, per the operator's choice of
`--points {skeleton,landmarks,both}` (your requirement):

- static landmarks → one 3D unknown per `(recording, landmark)`, or per `(rig, landmark)` at
  `scope = "rig"`;
- per-frame landmarks and skeleton points → one 3D unknown per `(recording, frame, point)`;
- **only frames the operator marked `reviewed`** are eligible by default. A half-labeled frame
  contributes a systematically-biased point and there is no way to detect it from residuals
  alone. `--include-unreviewed` exists and warns.

**Step 2 — gate on conditioning, before solving.** Refuse early and say why, rather than return a
plausible wrong rig:

```text
unknowns  = 6·(V−1) [+ V·f if free] [+ V·k1 if free] + 3·N_points
equations = 2·N_observations
```

Hard gates: every camera in ≥1 frame with ≥6 points shared with an already-registered camera;
every point in ≥2 views; `equations ≥ 1.5 · unknowns`; the pairwise co-visibility graph
**connected**. The last is the one that bites a 7-camera ring: left and right cameras may share
*no* points at all, with the front camera the sole bridge — a known property of this rig. The
gate must report the graph, not just a boolean.

**Step 3 — initialize.** Two paths, and the existing rigs take the cheap one:

- **Orbit prior present** (every current deeperfly rig): initialize straight from
  `resolve_extrinsics` and go to step 4. **This is the continuity guarantee** — a from-scratch
  code path must not change what today's rigs do.
- **No prior** (genuinely from scratch): incremental SfM.
  1. Choose the seed pair by `n_shared_points` weighted by median normalized disparity (a
     baseline proxy — a pair with 200 shared points and a 2° baseline is worthless).
  2. `cv2.findEssentialMat(..., method=RANSAC)` on normalized correspondences →
     `cv2.recoverPose` (chirality) → relative `R, t` with `‖t‖ = 1`.
  3. Triangulate the shared points with the existing `CameraGroup.triangulate`.
  4. Register the remaining cameras by `cv2.solvePnPRansac` against the growing cloud, in
     descending order of 2D–3D correspondence count; re-triangulate; repeat.
  5. Log the registration order and each camera's inlier count — when this fails it fails at a
     nameable camera.

**Step 4 — global bundle adjustment.** `deeperfly.bundle_adjustment.bundle_adjust` **verbatim**.
The gauge (7 free DOF: 3 rotation, 3 translation, 1 scale) is fixed through existing machinery:

```python
build_state(..., fixed=["rh.rvec", "rh.tvec"])   # 6 DOF: camera rh IS the world frame
```

and scale by one of three, in order of preference:

1. **`known_distance`** — the operator names two landmarks and types a distance in mm. This maps
   *exactly* onto the existing `bone_pairs` / `bone_targets` / `bone_weight` arguments
   ([bundle_adjustment/core.py:102](../src/deeperfly/bundle_adjustment/core.py#L102)). Sets
   `units = "mm"`. **No new solver code.**
2. **`bone_prior`** — a skeleton bone with a known length (a fly's femur), same mechanism, same
   caveat that a deforming animal makes it noisier.
3. **`none`** — additionally fix `cam1.tvec[0]` (or its norm); `units = "arbitrary"`, badged
   everywhere downstream. Triangulated output is then valid up to scale, which is fine for
   angles and useless for velocities. The GUI must say so.

Robust loss defaults to `cauchy, f_scale=4.0` — the setting this lab already measured as the one
that helps far-leg reprojection.

**Step 5 — report, then accept or discard.** Write `<name>.report.json` and render it in the
GUI: per-camera RMS and p90, the co-visibility matrix as a heatmap, the residual distribution,
the pairwise baseline angles, the 3D rig plot (reuse `_cameras_3d` + `scene3d.js` — it already
draws camera frusta), and a **before/after** on the same frames if a previous calibration exists.
**Nothing is written to `[project.calibration].current` until the operator accepts.** A
calibration that silently replaced a good one would be the single most destructive bug this
feature could ship.

### 6.5 The readiness meter

The GUI's answer to *"have I labeled enough to calibrate?"*, live in the sidebar while labeling —
this is the difference between a feature that gets used and one that gets abandoned:

```text
Calibration readiness                      [ Solve ]  (enabled)

  views registered        7 / 7        ✓
  co-visibility graph     connected    ✓   (f bridges {rh,rm,rf} ↔ {lf,lm,lh})
  static landmarks        3            ✓   tether_tip, coverslip_ne, dust_A
  reviewed frames         12           ✓   (≥ 8 recommended)
  observations / unknowns 2.3×         ✓   (≥ 1.5× required)
  weakest pair            rf↔f  8 pts  ⚠   label 4+ more shared points in rf and f
  scale reference         none         ⚠   units will be arbitrary
```

Every ⚠ is a link that navigates the editor to a frame and view where labeling helps most. This
is `labels-suggest` retargeted from *detector disagreement* to *calibration conditioning* — the
same acquisition idea against a different objective, and it reuses `select_frames`'s spacing
logic.

### 6.6 The full from-scratch walkthrough

```text
1. New Project…            name, skeleton (preset | blank | import), units
2. Skeleton editor         add points, draw bones, group into limbs, pick colors   (§7.2)
3. Add Recording…          pick a folder → cameras auto-detected from the files;
                           confirm the view↔file mapping and per-view preprocessing
4. Intrinsics              board wizard | optics form | guess                       (§6.2)
5. Landmarks…              declare the static fiducials you can see                (§6.3)
6. Label                   uncalibrated: 7 independent 2D canvases, no 3D           (§6.1)
7. Readiness meter fills   the GUI tells you exactly what is still weak            (§6.5)
8. Solve Calibration…      choose skeleton / landmarks / both; review the report;
                           Accept → project.calibration.current is set             (§6.4)
9. The 3D panel appears    reprojection overlays, derived 3D, the existing editor
10. Train…                 first detector from the labels                          (§7.5, F3)
11. Predict → correct → retrain                                                     (the SLEAP loop)
```

Steps 1–8 are new. Step 9 onward is what exists today.

---

## 7. Everything through the GUI (fixes B3)

### 7.1 The three tiers

Rendering 800 config keys as a form is a worse TOML editor. Split by *how often the value
actually varies*:

| Tier | What | Surface |
|---|---|---|
| **1 — Project** | skeleton, rig topology, calibration, landmarks, recordings, subjects | Bespoke GUI. These are structural, they have real invariants, and they are what a new user must get right. |
| **2 — Knobs** | `[triangulation]`, `[bundle_adjustment]`, `[annotation]`, `[inverse_kinematics]`, `[pipeline].do_*`, `[gui]` | **Generated forms** (§7.3). ~60 fields, all typed, all documented. |
| **3 — Plan** | `[pose2d]` preprocessors/models/pathways/`output_points`, `[visualization]` videos | Validated TOML editor in-browser, with a schema-aware linter and a *visual pathway graph* (read-only first, editable later). 132 `output_points` rows will never be a good form. |

### 7.2 The skeleton editor (tier 1)

Canvas-based, on a chosen frame: add/rename/delete points, drag bones between them, assign limbs,
pick palette colors, edit symmetry pairs. Presets ship: **fly38** (today's default), **fly19**
(ipsilateral, the legacy DeepFly3D layout), **blank**.

> **Symmetry pairs: done, ahead of the editor.** `Skeleton` now models them
> (`[skeleton].symmetries`, the same relation SLEAP calls edge `type 2`), with
> `flip_perm()` as the channel permutation a mirror implies and `infer_symmetries_by_name`
> to propose pairs for a skeleton that declares none. They landed early because two
> consumers wanted them before the editor did: `[pose2d.output_points]` validation (a
> mirrored pathway must land on the mirrored points — previously 132 unchecked config rows,
> where one typo swapped a body side silently) and `deeperfly.training.mirror` (flip
> augmentation, per fork **F3**). The editor's chirality check reads them too. So this
> section's remaining work is the *canvas*, not the model — editing pairs is already a
> non-destructive migration (`SkeletonChange("symmetries", …)`).

Every edit is a **migration**, not a mutation — see §9.

### 7.3 Schema-derived forms — the cheap win

The `*Params` dataclasses are *already* a complete schema: field names, types, defaults, and
prose. Generate the forms from `dataclasses.fields()` + the parsed docstring:

```python
# config_schema.py  (new)
def describe(cls) -> list[FieldSpec]:
    """Name, type, default, doc, and (where declared) choices/range per field."""
```

served at `GET /api/schema/{section}` and rendered generically by the front-end. Three
consequences worth naming:

- **The GUI cannot drift from the code.** A new field in `TriangulationParams` appears in the
  GUI on the next reload with its docstring as help text. No parallel schema to maintain.
- **The prose is already excellent.** `InverseKinematicsParams`' explanation of why `damping`
  defaults to 0.1 (the abdomen's five near-collinear hinges) is better help text than anyone
  would write for a form.
- **Validation is already written.** `_params()` raises on unknown keys with the allowed list;
  the same path validates the GUI's writes, so the two cannot disagree.

The only additions needed are `choices` and `range` metadata, via
`field(metadata={"choices": [...]})` — a small, local change to the dataclasses.

### 7.4 Profiles instead of one file

A project's `profiles/*.toml` hold **only the keys that differ** from the packaged defaults; the
effective config is `packaged ← profile ← per-recording override`. Two consequences:

- `deeperfly init` stops emitting 706 lines. It emits the ~15 a user actually sets, with a
  pointer to `deeperfly config show --all` for the rest.
- The GUI's forms write *deltas*, so a saved profile is readable and diffable. Today's
  save-the-whole-file behavior would make every GUI edit a 706-line diff.

The existing `Config.read_for_run` snapshot mechanism is preserved: a run still snapshots its
**fully-resolved** config into the outdir, byte-for-byte reproducible. Layering is a *authoring*
convenience, never a *runtime* ambiguity.

### 7.5 The job runner

The GUI must run things: detection, calibration, triangulation, IK, rendering, export, training.
Add `deeperfly.jobs`:

```python
@dataclass
class Job:
    id: str; kind: str; args: dict; recording: str | None
    state: Literal["queued","running","done","failed","cancelled"]
    progress: float; log_path: Path; started/finished: datetime | None
    result: dict | None
```

- A single background worker per project (a subprocess, so a segfault in torch cannot take the
  editor with it), FIFO queue, cancellable.
- Progress and log lines pushed over the **existing** `/ws` socket as a new message type; the
  front-end already has the plumbing.
- `POST /api/jobs`, `GET /api/jobs`, `DELETE /api/jobs/{id}`. Every job is *also* the exact CLI
  command it corresponds to, shown in the UI and copyable — so the GUI teaches the CLI rather
  than hiding it, and a failed GUI job can be reproduced in a terminal.
- Jobs map onto the existing stage granularity and fingerprints, so "run triangulation" already
  means "recompute exactly this and everything after it".

Training (`kind = "train"`) is a project-scoped job rather than a per-recording one, and per
**F3** it runs an absorbed, in-tree `deeperfly.training` — still in a subprocess, so a CUDA OOM
cannot take the editor down with it.

### 7.6 Project-scoped GUI navigation

`deeperfly gui myproject/` opens a **project view**: recordings table (frames, labeled frames,
reviewed, GT count, calibration, last run), suggestion rounds, calibration status, job queue.
Clicking a recording enters the editor already built. `deeperfly gui rec/deeperfly_outputs` keeps
working and simply shows a single-recording project with no owned artifacts.

**Concurrency.** The current single-writer lock is per-*session*. A project with several
recordings needs a **per-recording** lock plus a project-level lock for owned artifacts
(skeleton, calibration). Two annotators on two recordings of one project must both be able to
write; neither may edit the skeleton while the other is labeling. Advisory lock files under
`recordings/<rid>/.lock` carrying `{host, pid, user, since}`, with a stale-lock override that
names who holds it.

---

## 8. Merge (fixes B4)

SLEAP's *"Merge Data From…"* is the UX model. The operation is
`merge(src: Project | .dfpkg, dst: Project) -> MergeReport`, and it is **dry-run by default**.

### 8.1 The four reconciliations

**1. Skeleton.** Match by name, not index. Produce a mapping report:

```text
skeleton: 38 → 38 points
  matched by name          36
  renamed (fuzzy, confirm)  1   src "l_ant" → dst "l_antenna"
  only in source            1   src "l_haltere"        → [add to dst | drop]
  only in destination       0
  bones                    28 matched, 1 added
  ORDER DIFFERS: labels will be remapped by name, not copied by index
```

The last line is the whole point. Index-based copying between two 38-point skeletons in different
orders is a silent, catastrophic corruption — and today's exact-match `_check_identity` is what
prevents it. Merge replaces that refusal with a **name-based remap**, which is the "future
enhancement" the code comment already anticipates.

**2. Cameras.** Same treatment: matched by name, with an explicit rename table. A mismatch in
`image_sizes` for a same-named camera is **fatal** — the stored GT pixels would be
misinterpreted, which is exactly what `_check_identity` refuses today, correctly.

**3. Recordings.** Deduplicate by content `id` (§4.3). For a recording present on both sides,
merge labels (below). For one present only in the source, adopt it, rewriting footage paths and
reporting any that no longer resolve.

**4. Labels**, per `(view, frame, point)` cell, with an explicit policy:

| Situation | Default | Alternatives |
|---|---|---|
| only source has a value | take it | — |
| only destination has a value | keep it | — |
| both, identical | keep | — |
| both, differ, one is `dragged` and the other `confirmed_projection` | **prefer `dragged`** | `--on-conflict {ours,theirs,newest,manual}` |
| both, differ, same provenance class | **`manual`** — queue for review | `ours` / `theirs` / `newest` |
| `occluded` vs `gt` | **`manual`** | — |
| `absent` spans differ | **union**, reported | `ours` / `theirs` |
| `reviewed` differs | logical OR, reported | — |

Provenance-aware defaults matter here: a human drag beating a bulk-confirmed reprojection is
almost always right, and the existing `Provenance` enum already encodes exactly that distinction.

Conflicts routed to `manual` are written to `exports/merge_<ts>_conflicts.json` and surfaced as a
**review queue in the editor** — navigate cell by cell, see both candidates overlaid on the
frame, pick one. This is the piece that makes merge trustworthy rather than merely automatic.

### 8.2 Safety

- `--dry-run` is the default; `--apply` is required to write.
- `--apply` first snapshots `dst` to `exports/premerge_<timestamp>.dfpkg`.
- `iteration` is bumped; the full `MergeReport` is appended to `project.log` and stored under
  `exports/`.
- The merge never touches source.

---

## 9. Skeleton evolution — the invalidation problem

**This is the sharpest hazard in the whole plan** and it deserves its own section. Today the
skeleton's `point_names` is fingerprinted into every `labels.h5`
([labels.py:389](../src/deeperfly/gui/labels.py#L389)) *and* written into every `results.h5`. A
GUI that lets a user edit the skeleton can invalidate every label in the project with one click.

Treat every skeleton edit as a **typed migration** with a declared effect on existing labels:

| Edit | Effect on labels | Confirmation |
|---|---|---|
| add a point | none (new column, all-unset) | silent |
| rename a point | remap by identity; names in `labels.h5` are rewritten | one-line notice |
| reorder points | remap by name; on-disk COO indices rewritten | one-line notice |
| add/remove a bone | none — bones are display + BA prior only | silent |
| change limb/palette | none | silent |
| **delete a point** | its labels are **quarantined**, not deleted (the `absent/void_*` pattern already in the format) | modal: *"N GT labels across M recordings will be quarantined"* |

Requirements this imposes:

1. **Points get stable ids.** `point_names` stays the wire format, but `skeleton.toml` gains a
   per-point `id` (a short random token) so a rename is unambiguous and a rename-plus-reorder in
   one edit is still resolvable. `Skeleton` grows an optional `point_ids` tuple; absent ⇒ names
   are the ids (every existing file keeps loading).
2. **A dry run before every destructive edit**, counting affected labels across the project.
3. **A project-wide migration job** that rewrites each `labels.h5` and stamps the new skeleton
   digest — one job, one log, resumable, with a pre-migration `.dfpkg` snapshot.
4. **`results.h5` is left alone.** It is a derived cache; a skeleton change simply invalidates
   `pose2d` onward through the existing fingerprint mechanism.

---

## 10. Multi-animal — what to decide now, what to build later (B6)

You asked to plan with this in mind and focus on UX first. The right amount of work *now* is to
make sure the storage decisions above do not have to be redone. Three commitments:

### 10.1 Reserve the instance axis in the label schema

`labels.h5` is already sparse COO with `index (N, 3) int32 = [view, frame, point]`. Widening to
`[view, frame, instance, point]` is a **v6 bump costing one column**. Do it in the same version
that adds `landmarks/` (§6.3), so there is one migration rather than two:

```text
gt/index        (N, 4) int32   [view, frame, instance, point]
occluded/index  (M, 4) int32
absent/spans    (S, 4) int32   [instance, point, t0, t1)
instances/                                   (new)
    ids         (I,)  utf-8    per-recording instance ids
    identity    (I,)  int32    index into the project's identities table, or −1
```

In memory the dense arrays become `(V, T, I, P, …)`. **Every current call site is `I = 1`**, and
a `labels.dense(instance=0)` accessor keeps them byte-identical. The v5→v6 migration writes
`instance = 0` everywhere. Cost now: small. Cost later, if skipped: a second migration of the
one dataset in this project that is genuinely irreplaceable.

### 10.2 Adopt SLEAP's identity vocabulary

Distinguish, in the project schema, three things that get conflated:

- **Instance** — one animal's keypoints in one `(view, frame)`.
- **Track** — a within-recording temporal chain of instances (ephemeral, algorithmic).
- **Identity** — an animal across recordings and sessions (durable, human-assigned).

`labels.h5` already has `subject_id` — that is an Identity, and it should be promoted to a
project-level `identities` table that `[[recordings]].subject` references, so an absence
declaration ("this fly is missing its left front leg") can be authored **once per animal** rather
than once per clip. That is a genuine UX win *today*, single-animal, and it is the correct
foundation for later.

### 10.3 Reserve the cross-view correspondence

SLEAP's `FrameGroup` (one time point across views) and `InstanceGroup` (one animal across views)
are the right model. For multi-view multi-animal, `InstanceGroup` is **the** hard problem: which
blob in camera `lf` is the same fly as which blob in camera `rh`. Reserve the structure in
`project.toml` and in the `.dfpkg` schema; build nothing yet.

### 10.4 What is explicitly deferred

Multi-instance detection (top-down centroid+centered-instance, or bottom-up PAFs), tracking,
identity models, cross-view assignment, and the `DetectionPlan` changes needed to express a
variable instance count. Note that the current plan maps `channel -> (view, point)`, which has no
room for an instance — that rework is real and belongs to its own document.

---

## 11. Phasing

Each phase is independently shippable and leaves the tree green.

### Phase 0 — Calibration artifact *(unblocks everything; ~1 week)*
`CameraGroup.from_calibration` / `to_calibration`; `calibration.toml` schema + tests; BA output
written to a calibration file; `Config.camera_group()` prefers a named calibration; `deeperfly
calibration show/export`. **No GUI, no project.** Independently useful today: the current rigs
gain a way to persist and share their BA result.

### Phase 1 — The project layer *(~2 weeks)*
**1a — SHIPPED.** `deeperfly.project` (manifest, content ids, adoption, resolution);
`deeperfly project new/add/ls/status/rm`; the skeleton extracted into its own file;
`~/fly-pose-data` adopted as the acceptance test (see §14). No GUI changes.

**1b — moved to Phase 4.** Extracting the *rig* into `rig.toml` and layering profiles is
config composition, which is the same work as the tier-2/tier-3 split in §7 — doing it twice
would mean two competing notions of "the effective config". A project therefore owns its
skeleton today and points at a calibration; the rig topology (`[[sources]]`, per-camera
preprocessing) stays in the run config until Phase 4.

Two deviations from the design as written, both deliberate:

- **The recording fingerprint drops `n_frames` from the footage basis.** The plan had
  `camera:basename:size:n_frames`; the frame count costs a video decode per camera on every
  adoption, and per-camera byte size across seven files is already far more discriminative
  than a frame count. `n_frames` survives in the *fallback* basis (`result`), where it is the
  only recording-specific quantity available and there is a `results.h5` open anyway.
- **`gt_points` is reported alongside `gt_trainable`.** See §14.

### Phase 2 — Uncalibrated editing *(~2 weeks)*
`Session`/`EditorState` without a `PoseResult`; `has_3d = False` as a supported state; front-end
uncalibrated mode; `deeperfly gui <project>` with the project view and per-recording locks.
**This is the highest-risk phase** — it touches the editor core that 786 tests cover, and its
value is immediate regardless of the rest.

### Phase 3 — Calibration from labels *(~2 weeks)*
Landmarks (`labels.h5` v6, project `landmarks.toml`, editor gestures); the conditioning gate;
SfM initialization; the BA solve reusing existing machinery; the report; the readiness meter;
`deeperfly calibrate`. **Ask 3 lands here.**

### Phase 4 — GUI-first configuration *(~2 weeks)*
`config_schema.describe`; `/api/schema`; generated tier-2 forms; the TOML editor with linting for
tier 3; the skeleton editor with §9 migrations; `deeperfly init` presets.

### Phase 5 — Jobs, then the trainer *(~1.5 weeks + ~3 weeks)*
**5a — Jobs.** `deeperfly.jobs`; subprocess worker; WebSocket progress; run/cancel/logs for
pose2d, BA, triangulation, IK, render, export; the "copy this CLI command" affordance.
**5b — Training absorbed** (fork F3c): `deeperfly.training` migrated from `~/dfpose` behind a
`deeperfly[train]` extra, as a project-scoped stage with its own fingerprint. 5a ships without
5b; the `train` job kind simply reports "trainer not installed" until 5b lands.

### Phase 6 — Merge & packaging *(~2 weeks)*
`.dfpkg` export/import with `--embed`; the four reconciliations; the conflict review queue in the
editor; `project merge --dry-run/--apply`; pre-merge snapshots. **Ask 1 fully lands here.**

### Phase 7 — Multi-animal foundations *(scoped with Phase 3's v6 bump)*
The instance axis in the schema (shipped early, in Phase 3's migration); the identities table;
the reserved correspondence structures. Detection/tracking is a separate document.

**Ordering note.** Phases 0 and 2 are the two that pay for themselves immediately and
independently. If the plan has to be cut, cut from the back.

---

## 12. Adversarial review — where this is most likely to be wrong

**R1 — BA from a deforming animal is not SfM.** Classical SfM assumes a rigid scene. A fly's
keypoints move every frame, so skeleton points give `3·N·P` unknowns rather than `3·P`, and they
occupy a ~3 mm blob near the field center — poor conditioning for the essential-matrix
initialization. *Mitigation:* static landmarks (§6.3) are the primary constraint and skeleton
points the secondary; the conditioning gate refuses rather than returns a plausible-wrong rig;
`--points skeleton` alone is offered but warned against. **If from-scratch calibration fails in
practice, it will fail here.** Prototype step 3 on synthetic data with a known rig before
building the GUI around it.

**R2 — This rig's co-visibility graph is nearly disconnected.** Left and right cameras share no
keypoints; the front camera is the sole bridge — a property this lab has already characterized.
A from-scratch solve on such a rig hinges entirely on the front camera's labels. *Mitigation:*
the gate reports the graph explicitly; the readiness meter names the weakest pair; static
landmarks visible from both sides (a coverslip corner, the tether) are worth far more than any
number of keypoints and the GUI should say so.

**R3 — Focal length is not recoverable here.** §6.2 is deliberately conservative. The risk is a
user picking `guess`, getting an rms that *looks* fine (BA will happily trade focal error against
depth), and producing 3D that is systematically wrong in scale and mildly wrong in shape.
*Mitigation:* `intrinsics = "guess"` is stamped into the calibration and badged in every consumer;
the board wizard is offered first, not buried.

**R4 — Editing the skeleton can destroy every label in a project.** §9 exists entirely for this.
The residual risk is a *bulk* edit (import a different skeleton) bypassing the per-edit
migrations. *Mitigation:* skeleton import goes through the same migration path as an edit, with
the same dry-run counts; a pre-migration `.dfpkg` snapshot is mandatory, not optional.

**R5 — Adoption by symlink creates two writers.** If `~/fly-pose-data/recordings/<rid>/…/labels.h5`
is also opened directly by an old `deeperfly gui` invocation, two processes can write the same
file, and `save_labels` is a whole-file rewrite. This failure mode **already exists** — it is why
`labels-absent`'s help says to close any running GUI first. *Mitigation:* the advisory lock lives
beside `labels.h5`, not inside the project, so both entry points honor it; `save_labels` checks
the file's mtime against what it loaded and refuses a clobber.

**R6 — Merge can silently corrupt via index-based copying.** The nightmare is two 38-point
skeletons in different orders. *Mitigation:* remap by name is the *only* implemented path; index
copying is not written at all; the dry-run report prints the mapping and refuses on any unmatched
name without an explicit decision.

**R7 — The `.dfpkg` frame embedding can explode.** `--embed all` on a project with 20 recordings
× 200 suggested frames × 7 views is ~28,000 JPEGs. *Mitigation:* `user` is the default; the export
reports the projected size before writing and asks.

**R8 — `has_3d = False` is currently an accident, not a contract.** Phase 2 changes an
assumption that runs through `EditorState`, `server.py`'s payload builders, and ~120 KB of
`app.js`. *Mitigation:* land it behind the existing browser test
([tests/test_gui_browser.py](../tests/test_gui_browser.py)) — the JS throws it catches are
exactly this class of bug — and add a synthetic no-cameras session fixture before touching the
editor.

**R9 — Scope.** Seven phases is roughly a quarter of work. The plan is written so Phases 0 and 2
stand alone; nothing later is a prerequisite for anything earlier.

---

## 13. Forks — **decided 2026-08-03**

| Fork | Decision | Consequence |
|---|---|---|
| **F1 — Project shape** | **Directory + `project.toml`**, with a single-file `.dfpkg` export for sharing/merging | §4.1 as written |
| **F2 — Adoption default** | **`--link`** (symlink `deeperfly_outputs/`) — no label byte moves on day one. `--copy` available; `--move` deliberately absent | §4.1 as written |
| **F3 — Training** | **(c) Absorb the trainer.** deeperfly grows a real training stage rather than shelling out to `dfpose` or defining a plugin seam | **Scope change — see below** |
| **F4 — Multi-animal schema** | **Now.** The instance column rides along with Phase 3's unavoidable `labels.h5` v6 bump | §10.1 as written |
| **F5 — Uncalibrated 3D** | **Show nothing 3D.** No weak-perspective guess — an approximate overlay in the one phase where the operator cannot check it would teach the wrong thing | §6.1 as written; no depth-free overlay |
| **F6 — Landmark `scope = "rig"`** | **Implement, default off**, always reporting per-recording residual spread | §6.3 as written |

### F3's consequences — absorbing the trainer

Taking (c) makes deeperfly a standalone SLEAP-class tool and changes the plan in four places.
Recording them here so Phase 5 is not re-litigated:

1. **Phase 5 grows a `deeperfly.training` package**, migrated from `~/dfpose`: heatmap targets,
   augmentation/transforms, the model zoo, the `LabelsH5Dataset` ingestion, the train loop, and
   metrics. `dfpose.predict` / `refit` / `crop` are *research* code and are **not** absorbed —
   only the train/eval/dataset core is.

   *Landed so far:* `training.heatmaps` (the numeric contract) and `training.mirror` (the
   left-right flip). The mirror went in ahead of the dataset because it is the piece whose
   failure is silent — a wrong channel permutation trains every left channel on a right
   joint with no error and no warning — and because it carries three details that are not
   obvious from dfpose's code and would not survive a re-derivation: the flipped sample must
   be relabeled with the **mirrored camera** (`[cameras.*].mirror`), the flip decision
   belongs to the **frame group** rather than the sample, and the per-point masks riding
   alongside the coordinates must be permuted too.
2. **Dependency surface.** `timm` (and whatever the backbone needs) joins the dependency set,
   but as a `deeperfly[train]` **extra**, not core — mirroring how `quickik` is handled for the
   IK stage. A plain install must stay able to run `gui`, `run` and `calibrate` with no trainer
   present, and a `results.h5` that a trained model produced must render without it.
3. **`training` becomes a real pipeline stage** with a fingerprint, so "retrain because the
   labels changed" is the same mechanism as "re-triangulate because the method changed".
   It sits *outside* `STAGES` (which is the per-recording linear pipeline) because training is
   **project-scoped**, spanning many recordings — a distinction that must be explicit in the code
   or the fingerprint logic will be quietly wrong.
4. **The labels→samples seam is the contract to preserve.** `dfpose.datasets.labels_h5`'s
   footage-pixel → model-space transform, its left-first channel-order assertion, and its
   frame-cache design are the parts that took the longest to get right; they migrate
   substantially as-is, behind their existing tests.

**Sequencing is unchanged.** Absorption still happens at Phase 5. Phases 0–4 neither depend on it
nor are blocked by it — and shipping them first is what makes the training stage worth having,
since it needs a project with a calibration and labels to train from.

---

## Appendix A — CLI surface

```bash
# projects
deeperfly project new  DIR [--skeleton fly38|fly19|blank|PATH] [--rig PATH]
deeperfly project add  PROJECT RECORDING... [--link|--copy] [--subject ID]
deeperfly project ls   PROJECT
deeperfly project status PROJECT              # labels/reviewed/calibration/jobs per recording
deeperfly project merge SRC --into DST [--dry-run|--apply] [--on-conflict ours|theirs|newest|manual]
deeperfly project export PROJECT -o pkg.dfpkg [--embed all|user|none]
deeperfly project import PKG --into PROJECT

# calibration
deeperfly calibrate-intrinsics PROJECT --camera NAME --board BOARD.toml VIDEO
deeperfly calibrate PROJECT [--points skeleton|landmarks|both] [--recordings ...]
                            [--scale-from LANDMARK_A,LANDMARK_B=1.8mm] [--dry-run]
deeperfly calibration show|ls|use PROJECT [NAME]

# config
deeperfly config show PROJECT [--all] [--effective] [--section triangulation]
deeperfly config set  PROJECT triangulation.method ransac

# unchanged
deeperfly run|inspect|doctor|gui|labels-export|labels-absent|labels-suggest
```

Every command accepts a bare recording where it makes sense, so nothing in the existing
docs stops being true.

## Appendix B — File/format version bumps

| Format | Now | After | Change |
|---|---|---|---|
| `project.toml` | — | 1 | new |
| `calibration.toml` | — | 1 | new |
| `labels.h5` | 5 | 6 | `landmarks/` group; instance column in every COO index; `instances/` group |
| `results.h5` | 2 | 2 | unchanged — `absent`/`subject_id` already live outside the stages |
| `labels_suggest.json` | 1 | 2 | `group` field for suggestion rounds; optional `objective: "detector"｜"calibration"` |
| `.dfpkg` | — | 1 | new |

---

## Sources

- [SLP Format — sleap-io](https://io.sleap.ai/v0.6.4/formats/slp/)
- [3D data model — sleap-io](https://io.sleap.ai/v0.7.0/model/3d/)
- [Using the GUI — SLEAP](https://docs.sleap.ai/v1.5.2/learnings/gui/)
- [Configuring models — SLEAP](https://docs.sleap.ai/dev/learnings/configuring-models/)
- [SLEAP: A deep learning system for multi-animal pose tracking — Nature Methods](https://www.nature.com/articles/s41592-022-01426-1)
- [DeepLabCut standard user guide](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md)
- [Lightning Pose — core concepts](https://lightning-pose.readthedocs.io/en/latest/source/core_concepts.html)
- [Lightning Pose — create your first project](https://lightning-pose.readthedocs.io/en/latest/source/create_first_project.html)
- [Aniposelib tutorial — Anipose](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html)
- [Anipose: a toolkit for robust markerless 3D pose estimation](https://www.sciencedirect.com/science/article/pii/S2211124721011797)
- [Multi-Camera Self-Calibration in Sports Motion Capture: Leveraging Human and Stick Poses](https://arxiv.org/pdf/2604.17567)

---

## 14. What adopting the real corpus taught (2026-08-03)

Phase 1's acceptance test was to adopt `~/fly-pose-data` into a project. It surfaced two
things that changed the code, and one that changes the plan's priorities.

### 14.1 There is ~3× more hand labeling on disk than anything tracks

`deeperfly project add` over `recordings/*` and `predicted/*_label/` found **21
non-backup `labels.h5` files**, across three parallel trees:

| tree | files | note |
|---|---|---|
| `recordings/<rid>/deeperfly_outputs/` | 2 | where the pipeline writes by default |
| `predicted/<rid>_label/deeperfly_outputs/` | 15 | the `predict → correct` round outputs |
| `predicted/_superseded/`, `predicted/_work/.../pass1/` | 4 | earlier passes, kept |

Their provenance breakdown is the surprise: **essentially all of it is `dragged`** — 47,330
of 47,354 stored GT rows are human-placed pixels, not machine seeds (one file has 24
`confirmed_prediction`; nothing anywhere is `confirmed_projection` or `placeholder_seed`).
Discounting the one recording whose 4,461 labels appear in *five* separate copies, that is
roughly **29,500 unique human-placed points across ~17 recordings**.

`~/dfpose/data/labels/MANIFEST.toml` tracks **3 files and 9,175 points**. The other ~20,000
are real labels that no training set has ever seen.

Two consequences:

1. **`label_stats` agrees exactly with the manifest** wherever they overlap (1,759/12 and
   4,461/28 GT/occlusions), which is the strongest available check that the project layer
   reads these files correctly.
2. **Phase 6 (merge) is worth more than its position suggests.** The blocker on using that
   corpus is not labeling effort — it is reconciliation. Consider promoting the
   label-merge half of Phase 6 ahead of Phase 4.

### 14.2 One recording, several label sets

Content-based identity worked exactly as designed: 13 of the 15 `predicted/*_label/`
directories were recognized as recordings *already adopted* from `recordings/` (same footage
bytes → same id), so adopting them was a no-op. Correct — but it means their labels are not
counted, and a silent zero is indistinguishable from "the labels are gone".

`add_recording` therefore **warns** when a de-duplicated source carries labels the indexed
entry cannot see, naming both counts and both paths. The real fix is merge (§8): a recording
having several label sets is a first-class situation here, not a mistake.

### 14.3 "Labeled" is not "trainable"

The first status table reported raw GT rows. But `export_gt` **drops**
`confirmed_projection` (a bulk-accepted triangulation guess — the model's own output
promoted to ground truth) and `placeholder_seed` (a coordinate the editor invented at the
image edge so the operator had something to grab). A status number that counted those would
overstate the training set by exactly the amount of geometry someone bulk-confirmed, and it
would *disagree with the export it is supposed to predict*.

So `label_stats` reports both, and the status table shows `trainable` with anything dropped
counted and named. A test pins `gt_trainable` against `export_gt`'s own mask, so the two
cannot drift.

This corpus happens to be almost entirely `dragged`, so the numbers barely move — which is
precisely why it was worth fixing now rather than after a round of bulk confirmation made
the discrepancy load-bearing.

---

## 15. Implementation status — 2026-08-03 (final)

Branch `feature/project-system`, off `dev`. **1,207 tests green** (from 904), ruff clean, 19
of them driving a real headless chromium. Sixteen commits.

### Every phase

| Phase | State | What landed |
|---|---|---|
| **0** Calibration artifact | **done** | `calibration.toml` (v1) + guards + quality; `CameraGroup.from_calibration`/`to_calibration`; `[cameras].calibration`; every BA run emits one; `deeperfly calibration show/export` |
| **1a** Project layer | **done** | `project.py`; content-derived ids; adoption by symlink; `deeperfly project new/add/ls/status/rm`; `~/fly-pose-data` adopted as the acceptance test |
| **1b** Rig + profiles | **done** | `rig.toml`, `profiles/`, `deeperfly project rig` / `project config` — composition by concatenating disjoint fragments |
| **2** Uncalibrated editing | **done** | `PoseResult.uncalibrated`; `has_cameras`; `deeperfly gui <project>` (opens the first recording; the picker switches); the banner; browser tests |
| **3** Calibration from labels | **done** | `landmarks.py`; `labels.h5` **v6**; `calibration_solve.py`; `deeperfly calibrate` + readiness meter; **in-GUI landmark placement** |
| **4** GUI-first config | **done** | `config_schema.py`; `deeperfly config show/set`; `GET /api/schema` + `/api/config`; a **generated Settings panel**; **skeleton migrations** + `deeperfly project skeleton` |
| **5a** Jobs | **done** | `jobs.py` (subprocess queue, allow-listed kinds); `/api/jobs`; a Jobs panel; `python -m deeperfly` |
| **5b** Trainer | **part** | `deeperfly.training.heatmaps` — the numeric contract, tested. **The dataset, model zoo and train loop are not written.** See below. |
| **6** Merge + packaging | **done** | `merge.py` + `deeperfly labels-merge`; `package.py` + `project export/import` (`.dfpkg`) |
| **7** Multi-animal | **schema** | The instance column is reserved in every `labels.h5` COO index (F4). Detection/tracking is a separate document. |

### Verified against reality

- **A known rig recovered cold**, no prior: camera centres to **0.01%** (landmarks), **0.02%**
  (keypoints), **0.00%** (both) of the rig radius. End-to-end from real videos: **0.004%**.
- **`labels.h5` v6 reads all 21 real label files losslessly**, quarantine round trip included.
- **`label_stats` matches the dfpose manifest exactly** where they overlap.
- **Merge on the real corpus**: 1,687 stranded labels + 1,505 occlusions, 0 conflicts.
- **`.dfpkg` of a real project**: 8.3 MB with 84 embedded frames.
- **A destructive skeleton change on real labels**: 84 at-risk labels named, nothing written.
- **Uncalibrated editor, landmark placement, jobs and settings all drive in real chromium
  with zero JS errors.**

### Five bugs the validation caught

Each would have shipped as plausible-looking wrong output rather than a crash:

1. **The Rodrigues singularity.** `∂R/∂rvec` carries `sin θ/θ` → 0/0 at exactly zero, so a
   *free* camera at identity gives a NaN Jacobian column. No orbit camera has an identity
   rotation, which is why 24 k lines of working BA never hit it; a cold-start SfM reference
   view sits exactly there.
2. **A numpy aliasing bug** in the frame rebase — `t0 = tvecs[0]` is a *view*, and the loop
   zeroed row 0 first. A perfect rig came out at 112,915 px rms.
3. **Scale normalization applied to orbit priors**, which would have written a calibration
   claiming a rig 209× too large while `units = "config"` asserted it was the config's scale.
4. **`extract_section` swallowing the next section's comment banner**, so every project's
   `skeleton.toml` would have shipped with a paragraph about camera rigs.
5. **`label_stats` counting untrainable rows as ground truth** — `export_gt` drops
   `confirmed_projection` and `placeholder_seed`, so the progress number disagreed with the
   export it was supposed to predict.

### What remains: Phase 5b, and why it was not a copy

The plan's F3c said "absorb the trainer". The **numeric contract** is absorbed and tested.
The dataset, model zoo and train loop are not, and a wholesale copy of `~/dfpose` would have
been the wrong move for three measured reasons:

1. **6,700 of its 11,332 lines are in scope, and its `footage.py` is lab policy, not library
   code** — a hardcoded `(camera, (H, W)) -> crop` table for three specific rig geometries,
   carrying measurement notes about named recordings, that *refuses* an unlisted pair on
   purpose. deeperfly already has the general form: per-camera `preprocess` in `rig.toml`.
2. **It is 37 commits of actively-iterating "Round 0" work** whose `data/labels/` snapshots
   are git-tracked and described in its own README as "the precious bit". Copying discards
   that history and freezes a moving target.
3. **A trainer that has never completed a real training run is a liability, not an asset.**
   The parts already landed are testable without a GPU; a loop is not.

The remaining work, in order: a `TrainingSet` that turns a *project's* labels into
model-space samples through the rig's own `FrameTransform` (~400 lines, GPU-free to test);
the model zoo behind a `deeperfly[train]` extra; the loop; and `deeperfly train` as a `jobs`
kind — `JOB_KINDS` already reserves it and reports "trainer not installed" until then.

Two smaller items are also open: the **skeleton *editor* UI** (its migrations, the risky
half, are done and tested — what is missing is a canvas to draw bones on), and **`[[sources]]`
/ detection-plan editing**, which stays in the config file by design (§7 tier 3).

