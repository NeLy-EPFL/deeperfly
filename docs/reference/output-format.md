# Output format

A run writes everything to its output directory (`<input>/deeperfly_outputs/` by
default, or `-o`):

```
deeperfly_outputs/
├── results.h5      # the result: cameras, skeleton, per-stage 2D/3D data
├── config.toml     # byte-for-byte snapshot of the config this run used
├── run.json        # per-stage fingerprints (drives cache reuse)
├── labels.h5       # ground-truth annotations from 'deeperfly gui' (if any)
├── labels_suggest.json  # frames to label next, from 'deeperfly labels-suggest'
└── *.mp4           # one per [[visualization.videos]] entry
```

## `results.h5`

A self-contained HDF5 file (schema **version 2**). Each pipeline stage writes its
own group, so a stage never overwrites another's data and any downstream stage
can be re-run later from pristine upstream outputs. The file fully reconstructs
the cameras and skeleton, so results are portable without the original config.

Arrays use the [view-leading layout](../explanation/conventions.md#array-layouts)
in float64; `NaN` encodes missing observations / un-triangulated points.

```text
attrs["meta"]               json: {deeperfly_format_version: 2, created_utc, ...}
skeleton/                   point_names, limb_names, limb_id, bones, palette/
pose2d/
    points                  (V, T, P, 2)  arg-max 2D detections (visibility-masked)
    conf                    (V, T, P)     detection confidences
    cameras/                the config rig as built at detect time
    attrs["image_sizes"]    json {camera_name: [h, w]} of the raw footage frames
    candidates/             top-K peaks (xy, score) -- only if pictorial_structures
                            was enabled at detect time
bundle_adjustment/
    cameras/                the BA-refined rig
pictorial_structures/
    points                  (V, T, P, 2)  PS-corrected 2D
    points3d                (T, P, 3)     initial 3D estimate
    reproj_error            (V, T, P)
triangulation/
    points                  (V, T, P, 2)  cleaned 2D (outlier-rejecting methods)
    points3d                (T, P, 3)
    reproj_error            (V, T, P)
inverse_kinematics/
    angles                  (T, D)        fitted joint angles (radians)
    angle_names             (D,)          the angle names, in column order
    points3d                (T, P, 3)     fitted model joints (world; skeleton order)
    body_plan               scalar str    the QuickIK body plan that was solved (JSON)
    attrs["meta"]           json {template, solver, alignment, chain_scales, body_scale}
```

`angle_names` are the flygym joint names `<parent_body>-<child_body>-<dof>`: e.g.
`c_thorax-rf_coxa-{yaw,pitch,roll}` / `rf_coxa-rf_trochanterfemur-{pitch,roll}` /
`rf_trochanterfemur-rf_tibia-pitch` / `rf_tibia-rf_tarsus1-pitch` for a leg,
`c_thorax-c_head-{yaw,pitch,roll}` for the head, and the `c_thorax-c_abdomen12-pitch`
… `c_abdomen5-c_abdomen6-pitch` chain for the abdomen (the head/abdomen columns are
present only when `fit_head` / `fit_abdomen` are on). A limb with too few observed
keypoints in a frame — fewer than two — is `NaN` for that frame rather than filled with
the solver's neutral-biased guess, so "not fitted" stays distinguishable from "fitted
straight". `points3d` carries the model's prediction for every fitted
keypoint in skeleton order (leg joints, antenna tips, abdomen markers; other points
`NaN`), so it reprojects with the skeleton's own bones.

`body_plan` is the kinematic tree the fit was solved on, with this recording's
**measured** segment lengths baked into it — both a record of what was fitted and what
the editor's live re-fit re-solves on, so it cannot drift from the stored result. It is a
dataset rather than a meta key because it runs to tens of kilobytes, close enough to the
64 KB an HDF5 attribute allows to be worth keeping out.

The meta's `chain_scales`
(`{"head": …, "abdomen": …}`) are the head/abdomen size relative to the model that
the stage estimates from each chain's contour length (its markers' reach along the
chain), applied to the fit and the mesh overlay alike. `body_scale` is the recording's
body size relative to the model, from the single coxa registration that also places the
body plan; the mesh overlay holds the body, head, and abdomen at this fixed size and
varies only rotation + translation per frame, so the body does not breathe.

A `cameras/` group stores `names`, `rvecs`, `tvecs`, `intrs` (`[fx, fy, cx, cy]`),
and `dists`. The `skeleton/` group stores `point_names`, `limb_names`, `limb_id`,
`bones`, and a `palette/` subgroup of limb → hex color.

Which groups are present depends on which stages ran. A group exists only once its
stage completed; only the stages that were enabled (and whose inputs were
available) appear.

### What the library reads back

`PoseResult.load(path)` assembles the **most-derived data present**, so you get
the best result without knowing which stages ran:

| Field | Preference order |
| --- | --- |
| `pts2d` | `triangulation` → `pictorial_structures` → `pose2d` |
| `pts3d` | `triangulation` → `pictorial_structures` |
| `reproj_error` | `triangulation` → `pictorial_structures` |
| `cameras` | `bundle_adjustment` → `pose2d` (config rig) |
| `conf` | `pose2d` |
| `nmf_pts3d` | `inverse_kinematics` (the fitted model joints) |

`PoseResult.save(path)` is the library one-shot (no staged groups): it writes
`pts2d`/`conf` to `pose2d/` and, when a 3D pose is present, the 2D/3D/error to
`triangulation/`, so `load` round-trips the assembled view.

A file in an older schema version is rejected on `load` (re-run to regenerate)
and simply read as empty by the staged run (so it recomputes).

## `config.toml` (snapshot)

The exact config text that drove the run, copied byte-for-byte for
reproducibility. On a later run, `-c` wins when given (and refreshes this
snapshot); without `-c`, this snapshot is reused — so you can edit it in place and
re-run with just `-o`.

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

## `labels.h5`

Ground-truth annotations authored in [`deeperfly gui`](../guides/gui.md), written
next to `results.h5` and **never** modifying it. Only what the operator actually
authored is stored — sparsely (COO), so the file is tiny and carries no copy of the
predictions:

```
attrs["meta"]   json: { deeperfly_labels_format_version, created_utc, identity }
gt/
    index       (N, 3) int32    [view, frame, point]
    xy          (N, 2) float64  affirmed 2D pixel (footage space)
    provenance  (N,)   uint8     1=dragged, 2=confirmed-prediction, 3=confirmed-projection
occluded/
    index       (M, 3) int32    [view, frame, point]  views a human flagged unusable
```

`identity` fingerprints the recording (skeleton points, camera names, frame count,
image sizes, footage basenames) so a sidecar from a *different* recording is refused;
it excludes the predictions and `created_utc`, so re-running detection/triangulation
on the same recording keeps the labels valid (ground truth is absolute, not relative
to what the network predicted). A legacy `corrections.h5` is migrated to this schema
on open. Export the labels as a training/eval `.npz` with
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
            (bundle_adjustment | pose2d), scored_array (always "pose2d/points"),
            n_views / n_frames / n_points, the recording `identity`, and -- when the
            directory came from `dfpose.predict` -- `reseed`, including what the
            stored reproj_error *would* have said on the substituted cells
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
