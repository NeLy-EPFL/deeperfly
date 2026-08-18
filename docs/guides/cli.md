# CLI usage

`deeperfly` has seven commands: `init` (write a config), `run` (the pipeline),
`gui` (annotate a result), `labels-suggest` (rank the frames worth labeling
next), `labels-export` (export ground truth), `inspect` (summarize a result),
and `doctor` (report the install). Every command takes
`--log-level` (`debug` / `info` / `warning` /
`error` / `critical`; `warning` or higher hides the per-stage logs and the progress
bar) and `-h` / `--help`.

```bash
deeperfly --help            # the command list
deeperfly run --help        # a command's options
```

## `deeperfly init` — write a config

```bash
deeperfly init [OUTPUT] [--overwrite]
```

Writes the packaged, fully commented default config so you can edit it in place.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `OUTPUT` | `config.toml` | Destination path. |
| `--overwrite` | off | Overwrite an existing file (otherwise it warns and leaves it untouched). |

```bash
deeperfly init                       # -> config.toml
deeperfly init rig.toml --overwrite  # overwrite rig.toml
```

See [Writing configs](configuration.md) for what to edit.

## `deeperfly run` — the pipeline

```bash
deeperfly run INPUT... [-r] [-c CONFIG] [-o OUTPUT_DIR] [--overwrite [STAGE...]]
```

Detects 2D pose → bundle-adjusts the cameras → triangulates to 3D → renders the
videos, running only the [enabled stages](configuration.md#choose-which-stages-run-pipeline)
and reusing any cached results whose config is unchanged.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `INPUT...` | — | One or more recording directories and/or wildcard patterns. |
| `-r`, `--recursive` | off | Treat each `INPUT` as a parent directory and run every recording nested under it. |
| `-c`, `--config` | snapshot, else packaged default | The merged config TOML (from `deeperfly init`). |
| `-o`, `--output-dir` | `<input>/deeperfly_outputs/` | Where outputs go (created if missing). |
| `--overwrite [STAGE...]` | off | Force a recompute even when nothing changed (see [below](#resuming-and-recomputing)). |

### Inputs: single, batch, recursive

`INPUT` is a recording directory holding the configured per-camera footage, or a
wildcard matching several. Several inputs or a wildcard run as a **batch**;
recordings that don't match the config are skipped.

```bash
deeperfly run recording/                     # one recording
deeperfly run 'fly*'                          # batch: fly1/, fly2/, ... (quote the glob)
deeperfly run -r experiments/                 # every recording nested under experiments/
```

In a batch each recording runs independently: a failure is logged and the batch
continues, then `run` exits non-zero listing the failures. A single recording
fails fast.

### Output directory

By default each recording's outputs go to `<input>/deeperfly_outputs/`. With
`-o`:

- a path ending in `/` collects one subdirectory per recording under it (on a
  name collision it falls back to mirroring the input tree, after confirming);
- a relative name without `/` creates that directory inside each recording;
- for a single recording, `-o` is just that recording's output directory.

Each output directory holds `results.h5`, the rendered MP4s, a `config.toml`
snapshot, and `run.json` — see the
[output-format reference](../reference/output-format.md).

### Which config is used

`-c` wins when given (and refreshes the snapshot). Without `-c`, a run reuses the
`config.toml` already in the output directory; with neither, the packaged default
is used. So two workflows both work: keep your own config and pass `-c` each
time, or edit the snapshot in the output dir and re-run with just `-o`.

### Resuming and recomputing

An enabled stage **reuses its cached result while its config is unchanged and its
output is present** — so re-running a finished recording is a cheap no-op, and
editing the config recomputes exactly the affected stages (and the ones after
them). Tweak `[triangulation]` or the videos and re-run: the slow 2D detection is
reused, only triangulation/visualization recompute. Each stage records its
parameters in `run.json` when it completes; performance-only knobs (`batch_size`,
`decode_buffer`, `[io.image]`) never trigger a recompute.

`--overwrite` forces a recompute even when nothing changed — bare redoes every
stage, or name stages to redo only those (plus the stages after them):

```bash
deeperfly run recording/ --overwrite                       # recompute everything
deeperfly run recording/ --overwrite pose2d visualization  # just these (+ what follows)
```

The cached 2D pose always feeds the stages downstream, so `do_pose2d = false`
reconstructs 3D from a stored 2D pose without re-detecting. A *derived* stage's
cached output (bundle adjustment, pictorial structures, triangulation) feeds
downstream only while that stage is enabled. An enabled stage whose input is
unavailable is skipped, with the reason logged. The caching model is explained in
the [pipeline explainer](../explanation/pipeline.md#caching-and-re-runs).

### Example: change one stage, reuse the rest

A common edit is switching the triangulation method (or retuning the videos)
*without* re-detecting the slow 2D pose. Change `[triangulation]` `method` in
**one** config and re-run the same recording — no stage-skipping flag needed:

```bash
# A — keep your own config and pass it each time (-c wins, and refreshes the snapshot)
deeperfly run recording/ -c my_config.toml

# B — edit the snapshot the last run left in the output dir, re-run with no -c
$EDITOR recording/deeperfly_outputs/config.toml   # e.g. method = "dlt" -> "ransac"
deeperfly run recording/                          # add -o DIR if you used a custom output dir
```

Only triangulation and the videos after it recompute; the `pose2d` and bundle
adjustment caches are reused automatically. Do **not** edit the output-dir
snapshot *and* pass `-c` at the same time — `-c` wins and overwrites the
snapshot, silently discarding your edit. Pick one config and stick with it.

## `deeperfly auto-crop` — measure a view's detector crop

```bash
deeperfly auto-crop INPUT... [-c CONFIG] [-o DIR] [-r] [--no-write] [--no-gate]
```

Runs the crop search on its own, for every view whose config says
`{ op = "crop", auto = true }`, and prints what it found: the incumbent box, the searched
box, the detector's confidence on each, their agreement with the other cameras' 3D, and
whether the searched box was accepted. The `pose2d` stage does this by itself, so the
command is for the two cases where you want it separately — **seeing** the numbers before
trusting them, and **freezing** the result, which it prints as pasteable TOML so the box
becomes a plain window that never searches again.

```console
$ deeperfly auto-crop recording/ -c config.toml
preprocessor  view  incumbent            searched             conf            agree px      
crop_f        f     (0, 104, 1600, 800)  (395, 304, 955, 478)  0.62 -> 0.95   255.3 -> 4.9   accepted
crop_h        h     (0, 104, 1600, 800)  (521, 57, 544, 272)   0.48 -> 0.96   229.9 -> 2.9   accepted
```

The box is recorded in `<outdir>/autocrop.json`, which the next `deeperfly run` reuses
instead of searching again (`--no-write` measures without recording). `--no-gate` takes
whatever the detector's confidence preferred, without checking it against the other
cameras' 3D — measurably unsafe on its own, and only for a rig with no usable calibration.
Which views need this, and why the gate matters, is in the
[configuration guide](configuration.md#letting-the-crop-be-measured--auto--true).

## `deeperfly gui` — annotate a result

```bash
deeperfly gui PATH [--footage-dir DIR] [--host HOST] [--port PORT] [--no-browser] [--keep-alive]
```

Opens the interactive web viewer for a result and lets you author the ground-truth
2D pose (with the run's prediction as a starting point): every camera view with its
2D skeleton overlay, drag-to-place / confirm keypoints, mark cells **Hidden** (held out of
the training loss), and the live NeuroMechFly overlays. Labels go to a `labels.h5` sidecar and never modify
`results.h5` (an older `corrections.h5` is migrated on open). See the
[annotation GUI guide](gui.md) for the editor itself.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, or a directory containing one (e.g. `<recording>/deeperfly_outputs`). |
| `--footage-dir` | recorded paths | Directory to search for the footage when the paths recorded in `results.h5` no longer resolve. |
| `--host` | `127.0.0.1` | Address to bind. The loopback default keeps the editor private; it is **unauthenticated**, so bind a routable address only behind a trusted network — prefer an `ssh -L` tunnel. |
| `--port` | `8000` | TCP port to serve on (`0` picks a free one). |
| `--no-browser` | off | Do not open a browser on startup (e.g. when tunnelling). |
| `--keep-alive` | off | Keep the server running after the last tab closes (by default it stops a few seconds later; a refresh reconnects). |

```bash
deeperfly gui recording/deeperfly_outputs           # open the editor
deeperfly gui results.h5 --port 0                    # any free port
deeperfly gui results.h5 --no-browser                # headless / over a tunnel
```

## `deeperfly labels-suggest` — rank the frames worth labeling next

```bash
deeperfly labels-suggest PATH [-n N] [--min-gap-s S] [--reserve-diversity F] [-o OUT.json]
```

Active learning: ranks a recording's frames by the **multi-view disagreement** of
the detector's own 2D and writes the ranked list to `labels_suggest.json` beside
`results.h5`, for [`deeperfly gui`](gui.md) to navigate. Every joint is
RANSAC-triangulated from the pristine `pose2d/points` and each view's detection is
compared to the reprojection; views cannot conspire, so a large residual means the
model is probably wrong. Detector *confidence* is deliberately **not** used — it is
confidently wrong exactly where it is wrong.

The list is never the raw top-N: at 100 fps neighbouring frames are the same pose,
so `--min-gap-s` is a *hard* constraint (seeded with the frames already labeled, so
a suggestion can never land beside existing work), and `--reserve-diversity` spends
part of the list on a uniform temporal grid so the round still sees typical poses.
`results.h5` and `labels.h5` are only ever **read**.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, or a directory containing one. |
| `-n`, `--count` | `20` | How many frames to suggest. |
| `--min-gap-s` | `2.0` | Hard minimum spacing between suggestions (and from already-labeled frames), in seconds. |
| `--fps` | from `results.h5`, else `100` | Capture rate `--min-gap-s` is converted with. |
| `--reserve-diversity` | `0.25` | Fraction of `-n` taken on a uniform temporal grid instead of by score. |
| `--threshold` | `15.0` | px; the RANSAC inlier gate and the "this cell disagrees" gate. A *ranking* knob, not an accuracy claim. |
| `--cap` | `60.0` | px; per-cell saturation, so one blown view cannot make the ranking a single-outlier lottery. |
| `--top-k` | `8` | How many of the worst joints are averaged into a frame's score. |
| `--min-views` | `3` | Observing views a joint needs to be scorable (a 2-view joint reprojects onto both by construction). |
| `--points` | all | Glob(s) over skeleton point names to score (repeatable). |
| `--cameras` | all | Glob(s) over camera names to score (repeatable); triangulation still uses every view. |
| `--exclude-labeled` / `--no-exclude-labeled` | on | Skip frames that already carry human work, read from `labels.h5`. |
| `-o`, `--output` | `labels_suggest.json` beside `results.h5` | Destination `.json`. |
| `--dry-run` | off | Print the ranking, write nothing. |

```bash
deeperfly labels-suggest recording/deeperfly_outputs              # 20 frames, >= 2 s apart
deeperfly labels-suggest results.h5 -n 10 --min-gap-s 1 --dry-run # just look
deeperfly labels-suggest rec/ --points '*claw' --cameras 'l*'     # score a subset
```

The printed report is the whole feature without the GUI: each pick comes with its
score, its within-recording percentile, whether it is `most-wrong` or `diversity`,
and the joints/views that drove it. Two facts it always surfaces — because both
mislead silently otherwise — are whether the directory's *displayed* triangulation
layer was reseeded (its stored residual is 0 by construction and is never scored),
and any shortfall against `-n` caused by the spacing constraint. Scores rank
**within one recording only**; the absolute level tracks how many cells the detector
fired, so never compare them across files.

See [`labels_suggest.json`](../reference/output-format.md#labels_suggestjson) for the
sidecar schema.

## `deeperfly labels-export` — export ground truth

```bash
deeperfly labels-export PATH [-o OUT.npz] [--include-projection]
```

Exports the labels authored in [`deeperfly gui`](gui.md) (the `labels.h5` beside the
result) as a training/eval dataset: the ground-truth pixels and the **Hidden** mask, in
footage pixel space.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, or a directory containing one (its `labels.h5` is exported). |
| `-o`, `--output` | `labels_gt.npz` beside `results.h5` | Destination `.npz`. |
| `--include-projection` | off | Also export GT confirmed from the 3D reprojection (the model's own guess); excluded by default so the export is human-placed pixels only. |

The `.npz` holds `gt_xy` (V,T,P,2), `gt_mask` (V,T,P), `occluded` (V,T,P), `absent` (T,P),
and `point_names` / `camera_names`. Coordinates are footage-space; a detector-training
pipeline maps them into model-input space by inverting each pathway's preprocessing.

`occluded` is the editor's **Hidden** flag — *"do not include this cell in the training
loss"* — keeping its old array name. It is a **separate axis** from `gt_mask`, not a filter
already applied to it: a cell can carry a hand-placed pixel and still be held out, which is
exactly the pairing the flag exists to record. So the supervision mask is

```python
supervise = gt_mask & ~occluded
```

and dropping the second term trains on everything, which is a valid choice the export
deliberately leaves to you.

`absent` marks keypoints that are **not on this animal** (see
[`deeperfly labels-absent`](#deeperfly-labels-absent)). Those channels are excluded from
`gt_mask` *and* from `occluded` — an amputated joint is not ground truth, and a hold-out mark
on something already unsupervised is not a decision anyone made. **Mask** them in the loss.

## `deeperfly labels-absent` — mark keypoints that are not on this animal

```bash
deeperfly labels-absent PATH... --points 'lf_femur_tibia,lf_tibia_tarsus,lf_claw' \
    [--subject ID] [--clear]
```

For an amputated leg or an ablated antenna: the keypoint does not exist. That is
different from *Hidden* (it exists and keeps its position; only the loss skips that cell)
and from *unlabeled*, and
getting it wrong costs real accuracy — the detector's argmax decode always emits a peak,
so a phantom limb is confidently localized onto whatever looks leg-like nearby and that
peak feeds triangulation, bundle adjustment and the bone-length prior.

The declaration is per keypoint and, by default, covers the whole recording — one command
replaces marking every frame and every view by hand, and because it carries no view
indices it means the same thing in every clip of the same animal, hence `PATH...`. Use
`--frames` for a limb lost part-way through (`--frames 900:` = from frame 900 to the end).

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH...` | — | One or more `results.h5` files or directories. Pass every clip of the same animal. |
| `--points` | — | Comma-separated keypoint names or `fnmatch` globs (`lf_*`). An unmatched name is an error, so a typo cannot silently declare nothing. |
| `--frames` | whole recording | A frame or half-open range: `900`, `900:`, `0:900`, `:900`. For a limb lost mid-recording. |
| `--subject` | none | Animal identifier stamped into the sidecar, so one animal's recordings can be grouped. |
| `--clear` | off | Un-declare instead of declare. |

Nothing is destroyed either way: labels an absence declaration hides are quarantined in
the sidecar and restored if it is lifted. The editor has the same gesture — select a
joint and press `x`. **Close any running `deeperfly gui` on these directories first**:
saving is a whole-file rewrite, so an open session would overwrite what this writes.

## `deeperfly project` — group related recordings

A project gathers related recordings under one skeleton and one place to see what is
labeled. It **indexes** rather than re-homes: each recording's `results.h5` and
`labels.h5` stay where they already are and are adopted by **symlink**, so the file the
editor writes is the very file a training set reads. Creating a project copies nothing
and can lose nothing.

```bash
deeperfly project new  DIR [--skeleton fly38b|fly38|blank|PATH] [--name NAME]
deeperfly project add  PROJECT RECORDING... [--copy] [--subject ID] [-c CONFIG]
deeperfly project ls     [PROJECT]
deeperfly project status [PROJECT]
deeperfly project rm     RECORDING [PROJECT] [--delete]
```

`PROJECT` may be omitted on `ls` / `status` / `rm`: the nearest enclosing project is
used, the way `git` finds its repository.

### `new`

Writes a `project.toml` (the index) and a `skeleton.toml` (what is tracked). Nothing
else — a camera rig is either pointed at with a
[calibration](#deeperfly-calibration-reuse-a-solved-camera-rig) or solved later from
labels.

`--skeleton fly38b` (the default) seeds the packaged 38-point *Drosophila* skeleton --
six legs, two antennae, the neck and a midline abdomen chain; `fly38` is the historical
DeepFly3D set with per-side abdominal markers; `blank` gives you an empty one to define
yourself; a path copies the `[skeleton]` table out of any config. A config that only
*names* its skeleton is resolved on the way in, so the project always records its own.

### `add`

`RECORDING` is whichever path you have: a recording directory, its
`deeperfly_outputs/`, or a `results.h5`. A recording with **no outputs at all** — just
videos — is adopted too; that is the from-scratch starting point.

Recordings are identified by **content**, not path, so adopting the same one twice is a
no-op and a backup copy is recognized as the same recording. Because a recording is one
entry, a *second* label set for it (an earlier round, another annotator's directory) is
not counted — and `add` says so loudly rather than reporting zero labels.

`--copy` copies the outputs instead of linking them. The copy is a **snapshot**: labels
authored in the original will not appear in the project, and vice versa.

### `status`

```console
$ deeperfly project status
project:  new-rig-38kp  (.)
skeleton: fly38b  (38 points)
calib:    none (uncalibrated -- no 3D until a rig is solved)
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━┓
┃ slug        ┃ frames ┃ labeled ┃ reviewed ┃ trainable ┃ dropped ┃ occl ┃ state ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━┩
│ oldrig_demo │     64 │      10 │       10 │     1,759 │       — │   12 │ link  │
│ scape_Fly3  │  4,009 │      23 │       23 │     4,461 │       — │   28 │ link  │
└─────────────┴────────┴─────────┴──────────┴───────────┴─────────┴──────┴───────┘
```

**`trainable`, not "GT points".** It is what `deeperfly labels-export` would actually
yield: rows a human placed or confirmed, with `confirmed_projection` (a bulk-accepted
triangulation guess — the model's own output) and `placeholder_seed` (a drag handle the
editor invented at the image edge) excluded, exactly as the export excludes them.
Anything dropped is counted separately and named, because a progress number that
disagrees with the export is worse than no progress number.

Counts are the **live** rows: a keypoint declared absent has its labels quarantined, and
those are not reported as ground truth.

### `rm`

Drops the entry. The adopted files are left untouched — a symlinked project must not be
able to delete the originals. `--delete` also removes the project's own directory for the
recording (for a linked one, just the link), and **refuses** when the outputs are a real
directory, since that would be the only copy of the labels.

## `deeperfly calibration` — reuse a solved camera rig

A run with bundle adjustment enabled writes its refined rig to
`<outdir>/calibration.toml` — a portable file, unlike the copy inside `results.h5`.
Point another recording's config at it and that recording skips solving its own:

```toml
[cameras]
calibration = "/path/to/calibration.toml"
```

### `show`

```bash
deeperfly calibration show CALIBRATION
```

Prints the cameras, how the rig was produced, and its reprojection residuals.
`CALIBRATION` is a `calibration.toml` or a directory holding one.

**Read the residuals before reusing a rig.** Nothing downstream will warn you that
a rig reprojecting at 15 px is triangulating your fly.

```console
$ deeperfly calibration show recording/deeperfly_outputs
calibration: IN07B001_260417_Fly4_004
  created:      2026-08-03T11:22:00+00:00
  cameras:      7  (rh, rm, rf, f, lf, lm, lh)
  units:        config  (scale from orbit_prior, in whatever unit the config's [cameras] distance was written in)
  method:       labels_ba
  reprojection:
    rms         1.842 px
    median      1.331 px
    p90         3.155 px
    rh          1.618 px rms
    ...
```

### `export`

```bash
deeperfly calibration export RESULT [-o OUT.toml]
```

Extracts a `calibration.toml` from a result's stored rig — the migration path for
every recording processed before calibrations existed. `RESULT` is a `results.h5`
or a directory containing one; the default output is `calibration.toml` beside it.

The bundle-adjusted rig is preferred. A result with **no** bundle adjustment
exports the un-refined config rig it detected with, warning you and recording
`method = "orbit_prior"` — an unsolved rig is still worth having, but the file must
not claim it was measured.

Residuals are re-measured against the exported rig rather than copied from the
stored `reproj_error`, which can describe a different rig or a substituted 2D
layer — which is how a bad rig ships with someone else's good numbers attached.

## `deeperfly inspect` — summarize a result

```bash
deeperfly inspect RESULT.h5
```

Prints the file path, the views (count and camera names), the frame count, the
skeleton (name and point count), whether a 3D pose is present, and the median /
max reprojection error.

```bash
deeperfly inspect recording/deeperfly_outputs/results.h5
```

## `deeperfly doctor` — report the install

```bash
deeperfly doctor
```

Reports, each guarded so a missing piece is shown rather than crashing:

- **deeperfly** — version and install location.
- **system** — Python version/implementation and platform.
- **inference** — PyTorch version, CUDA/MPS availability, GPU memory.
- **frame I/O** — PyAV for video, OpenCV for images.
- **weights** — the detector-weight cache directory and whether they're
  downloaded.
- **config** — the packaged default config path.

Run it right after installing to confirm the GPU and frame-I/O backends are
available.

## `deeperfly calibrate` — solve a rig from hand labels

The from-scratch path: label 2D with no calibration at all, then recover the camera rig
from those labels.

```bash
deeperfly calibrate [PROJECT] --dry-run                     # readiness only
deeperfly calibrate [PROJECT] --points both --focal-px 22388 --accept
```

### Run it with `--dry-run` while you label

```console
$ deeperfly calibrate --dry-run
                             Calibration readiness
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    ┃ check                  ┃ value             ┃ what would help            ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ OK │ views with labels      │ 7 / 7             │                            │
│ OK │ co-visibility          │ connected         │                            │
│ !  │ static landmarks       │ 0                 │ one static point is worth   │
│    │                        │                   │ more than many keypoint     │
│    │                        │                   │ frames                      │
│ OK │ observations/unknowns  │ 4.21x             │                            │
│ !  │ scale reference        │ none              │ angles yes, lengths no      │
└────┴────────────────────────┴───────────────────┴────────────────────────────┘
```

Each shortfall is phrased as the labeling that would fix it, so "enough" is never a guess.

### `--points landmarks | keypoints | both`

Your choice of what drives the solve, and it matters more than it looks:

> A skeleton keypoint at frame *t* is a **different 3D point** from the same keypoint at
> *t+1* — the animal moved. So *N* frames of *P* points add `3·N·P` unknowns, all inside a
> 3 mm blob near the field centre. A **static** landmark — a coverslip scratch, the tether
> tip, a dust speck — is **one** 3D point observed in `V·T` images, spread through the
> scene volume. That is what conditions the solve.

Declare landmarks in the project's `landmarks.toml`; `scope = "rig"` shares one 3D point
across every recording on the rig (the strongest constraint available, and the easiest to
get wrong if the rig is bumped — the report always breaks its residual down per recording).

### Intrinsics are never guessed

Extrinsics are recoverable from correspondences. Focal length essentially is not, from a
few hundred hand labels on a 3 mm deforming animal — and a solve permitted to guess it
trades focal error against depth and reports a *beautiful* residual for a wrong rig. So one
of these is required, and which was used is recorded in the calibration's provenance:

| flag | when |
| --- | --- |
| `--from-calibration PATH` | reuse a board solve or a previous run — **best** |
| `--lens-mm F --sensor-mm W` | compute it: `f_px = f_mm · W_px / W_mm`, two datasheet numbers |
| `--focal-px F` | state it directly |

A badly wrong focal makes the solve **fail loudly** (views cannot be placed) rather than
converge to a mis-scaled rig.

### Nothing is accepted without `--accept`

The solve always writes `calibrations/<name>.toml` and a `.report.json` beside it, but the
project's *current* rig only changes with `--accept`. A calibration that silently replaced a
good one would be the most destructive thing this command could do.

Only frames marked **reviewed** are used, unless `--include-unreviewed`: a half-labeled
frame contributes a systematically biased 3D point, and no residual reveals that afterwards.

Scale: `--scale-from A,B=1.8` pins it with a known distance between two landmarks (which
reuses the bundle adjuster's existing bone-length prior). Without one the rig is valid *up
to scale* — angles are meaningful, lengths and velocities are not, and the calibration
records `units = "arbitrary"` so nothing downstream can forget.

## `deeperfly labels-merge` — reconcile a second label set

For ground truth that ended up in two places: an earlier round, another annotator's
directory, the same recording under a different tree. A project indexes a recording **once**
(by content), so the other copy's labels are otherwise invisible — `project add` warns about
exactly this, and this is the fix.

```bash
deeperfly labels-merge RECORDING SOURCE [PROJECT] [--on-conflict POLICY] [--apply]
```

**Dry run by default.** `--apply` always snapshots the destination `labels.h5` first, so
trying it is safe.

### Points and cameras are matched by name, never by index

Two same-sized skeletons in a different order are the one case where an index-based copy
corrupts every label while leaving each cell looking perfectly plausible. That path is not
implemented, so it cannot be reached by accident; a reorder is reported and remapped.

A same-named camera whose **footage size** differs is **refused outright** — ground truth is
stored in footage pixels, so merging across resolutions would reinterpret every coordinate.

### Conflicts are settled by provenance first

| situation | outcome |
| --- | --- |
| only the source has a value | taken |
| only the destination has one | kept |
| identical | kept, counted |
| a human **drag** vs a bulk-confirmed **reprojection** | **the drag wins**, whatever the policy |
| both the same provenance, different pixels | `--on-conflict` (default `manual`) |
| a **Hidden** mark vs a pixel | **both are kept** — they are independent axes, not rival claims |
| absence declarations | **unioned** (declaring is non-destructive by construction) |
| `reviewed` flags | OR'd |

The provenance rule is not a preference: a `confirmed_projection` is the *model's own output*
promoted to ground truth, so a human's pixel beating it is the difference between evidence
and a guess. `--on-conflict manual` leaves genuinely ambiguous cells as the destination had
them and writes them to `exports/merge_conflicts_<recording>.json` for review.

### Note on linked recordings

A recording adopted with `--link` (the default) has its `deeperfly_outputs/` symlinked to the
original, so `--apply` writes **into that original**. That is the point of adopting by
reference — the file the editor writes is the file a training set reads — but it means a
merge is not confined to the project directory. Use `--copy` at adoption time if you want it
to be.

## `deeperfly config` — find and set one key without reading the file

The packaged config is 706 lines across 50 tables. Changing one triangulation knob should
not mean scrolling past 132 detector channel mappings, and should not mean guessing what the
knob does.

```bash
deeperfly config show [SECTION] [-c CONFIG] [-v]
deeperfly config set SECTION.KEY VALUE -c CONFIG
```

```console
$ deeperfly config show triangulation
                    [triangulation]
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━━━┓
┃ key                 ┃ value  ┃     ┃ type  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━━━┩
│ method              │ ransac │ set │ str   │
│ reproj_threshold    │ 40.0   │     │ float │
└─────────────────────┴────────┴─────┴───────┘
```

The **`set` marker** is the part a config file cannot give you: a 706-line file where 690
lines are defaults reads as 706 decisions. `-v` adds what each key *means*.

**Everything here is derived from the code**, not from a parallel description — the keys,
types, defaults and documentation all come from the `*Params` dataclasses that already define
them, so a new option appears here with nothing else to update, and its help text is the
prose already written for it (`inverse_kinematics.damping` explains the abdomen's five
near-collinear hinges, which is better than any form label). The same data is served at
`GET /api/schema` for the editor.

`set` **appends** rather than rewriting, so comments survive, and validates the result
through the same strict loader a run uses — it cannot write a key a run would reject. It
refuses when the target table already exists, because appending a bare key after an existing
header silently reparents it.

Four things are **not** described: `[cameras]`, `[skeleton]`, `[[sources]]` and the
`[pose2d]` detection plan. They are structural or open-ended, and a half-schema for them
would be a fiction — they belong in the file, or (for the skeleton and rig) in the project.
