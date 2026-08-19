# CLI usage

`deeperfly --help` is the authority; this page is the same list with the reasons
attached. Twelve commands, plus three groups that carry subcommands:

| | |
| --- | --- |
| **Run the pipeline** | [`init`](#deeperfly-init) (write a config), [`run`](#deeperfly-run), [`auto-crop`](#deeperfly-auto-crop) (measure a view's detector crop) |
| **Annotate** | [`gui`](#deeperfly-gui), [`labels-suggest`](#deeperfly-labels-suggest), [`labels-export`](#deeperfly-labels-export), [`labels-absent`](#deeperfly-labels-absent), [`labels-merge`](#deeperfly-labels-merge) |
| **Solve a rig** | [`calibrate`](#deeperfly-calibrate), [`calibration`](#deeperfly-calibration) (`show` / `export`) |
| **Organize** | [`project`](#deeperfly-project) (`new` / `add` / `ls` / `status` / `rm` / `config` / `rig` / `export` / `import` / `import-outputs` / `skeleton`), [`config`](#deeperfly-config) (`show` / `set`) |
| **Inspect** | [`inspect`](#deeperfly-inspect), [`repack`](#deeperfly-repack), [`doctor`](#deeperfly-doctor) |

Every command takes `--log-level` (`debug` / `info` / `warning` / `error` /
`critical`; `warning` or higher hides the per-stage logs and the progress bar) and
`-h` / `--help`.

```bash
deeperfly --help            # the command list
deeperfly run --help        # a command's options
```

!!! warning "0.2 removed `deeperfly dense-config`"

    It existed to generate a dense detection plan when the default plan was the
    19-channel one. A dense plan is what a plan *is* now — two lines under `[pose2d]`,
    no mapping table — so there is nothing left to generate. See
    [the detection plan](configuration.md#the-detection-plan).

## `deeperfly init` — write a config { #deeperfly-init }

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

## `deeperfly run` — the pipeline { #deeperfly-run }

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

`INPUT` is a recording directory holding footage for at least one configured source, or a
wildcard matching several. Several inputs or a wildcard run as a **batch**; a directory
matching *no* source is not a recording and is dropped from a batch silently (that is how an
intermediate or output directory is recognized), while a single literal path is kept even
then, so a resume from its cached result still works.

```bash
deeperfly run recording/                     # one recording
deeperfly run 'fly*'                          # batch: fly1/, fly2/, ... (quote the glob)
deeperfly run -r experiments/                 # every recording nested under experiments/
```

In a batch each recording runs independently: a failure is logged and the batch
continues, then `run` exits non-zero listing the failures. A single recording
fails fast.

**A recording missing a camera is not skipped.** One config routinely describes more
rig than one recording holds — a session that predates the eighth camera being added,
say — so the run *narrows itself* to the footage it finds: a source that resolves no
files is dropped, with it every pathway reading that source, and then every view no
surviving pathway feeds. Grid cells naming a dropped view are blanked rather than
removed, so the montage keeps its shape. One warning names the sources, the pathways,
the views and the count it is proceeding on. Below **two** views it refuses, because one
view triangulates to all-NaN without raising anything.

Narrowing happens *before* the config snapshot and the fingerprints, so a seven-view run
records a seven-view fingerprint and recomputes when the eighth camera turns up — see
[narrowing](../reference/configuration.md#narrowing-to-the-footage-present) for the exact
order. A `[cameras].calibration` covering only some of the config's cameras subsets the
same way, with a warning; one covering *none* of them is a different rig and is refused.

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

**A resume onto a different skeleton is refused.** Every array in `results.h5` is
`(…, P, …)` with no names beside it, and a resume that does not recompute `pose2d`
never rewrites the skeleton record — so a config that now resolves a different point
set would read one skeleton's names against the other's columns. The check is on the
ordered point *names*, not the skeleton's label: two 38-point skeletons load each
other's files perfectly happily, and a preset can be renamed without a coordinate
changing. `--overwrite` does not get past it — the check runs before any stage is
considered, precisely so it cannot cost you a detection pass to discover. The error names
both skeletons and what differs, and the three ways out: point `[skeleton]` at the skeleton
the pose was detected on, run into a fresh output directory, or migrate the labels onto the
new point order with `deeperfly project skeleton`.

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

## `deeperfly auto-crop` — measure a view's detector crop { #deeperfly-auto-crop }

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
                                                   auto-crop
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ preprocessor ┃ view ┃ incumbent           ┃ searched             ┃ conf           ┃ agree px     ┃          ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ crop_f       │ f    │ (0, 104, 1600, 800) │ (395, 304, 955, 478) │ 0.620 -> 0.951 │ 255.3 -> 4.9 │ accepted │
│ crop_h       │ h    │ (0, 104, 1600, 800) │ (521, 57, 544, 272)  │ 0.481 -> 0.964 │ 229.9 -> 2.9 │ accepted │
└──────────────┴──────┴─────────────────────┴──────────────────────┴────────────────┴──────────────┴──────────┘

recorded in recording/deeperfly_outputs/autocrop.json -- the next run reuses it

To freeze a searched window as an explicit box (so nothing re-searches), replace that
preprocessor's op with:
  [[pose2d.preprocessors]]  name = "crop_f"
  ops = [{ op = "crop", x = 395, y = 304, width = 955, height = 478 }]
```

The last column is `accepted` when the searched box replaced the incumbent and `kept` when
the incumbent stood. The box is recorded in `<outdir>/autocrop.json`, which the next
`deeperfly run` reuses instead of searching again (`--no-write` measures without recording).

**The search itself needs no camera rig.** It is a grid over (center, width) scored by the
detector's own confidence, which is why it works on a recording you have never calibrated.
What needs a solved rig is the **accept gate** — reprojection agreement with the other
cameras' 3D, which is the `agree px` column. Against a nominal orbit rig the gate measures
its own reference about 250 px out and refuses rather than choose with a broken ruler: that
column then reads `not gated`, two notes under the table name why and say to check the
`pose2d` overlay, and the box comes from confidence alone — roughly 1.7× too wide, which is
still far better than handing the detector the whole frame (255 px, a collapse). Export a
calibration, point `[cameras].calibration` at it, and the gate engages.

`--no-gate` skips the check deliberately and takes whatever confidence preferred.
Measurably unsafe on its own — a box can get more confident *and* less accurate — so it is
for a rig with no usable calibration and nothing else. Which views need a crop at all, and
why, is in the [configuration guide](configuration.md#auto-crop).

## `deeperfly gui` — annotate a result { #deeperfly-gui }

```bash
deeperfly gui PATH [--footage-dir DIR] [--host HOST] [--port PORT] [--no-browser] [--keep-alive]
```

Opens the interactive web viewer for a result and lets you author the ground-truth
2D pose (with the run's prediction as a starting point): every camera view with its
2D skeleton overlay, drag-to-place / confirm keypoints, mark cells **Hidden** (held out of
the training loss), and the live NeuroMechFly overlays. Labels go to a `labels.h5` sidecar and never modify
`results.h5`. See the
[annotation GUI guide](gui.md) for the editor itself.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, a directory containing one (e.g. `<recording>/deeperfly_outputs`), or a [project](#deeperfly-project) directory — which is what opens the recording tabs, the Jobs pane and the Bundle-adjust tab. |
| `--footage-dir` | recorded paths | Directory to search for the footage when the paths recorded in `results.h5` no longer resolve. |
| `--host` | `127.0.0.1` | Address to bind. The loopback default keeps the editor private; it is **unauthenticated**, so bind a routable address only behind a trusted network — prefer an `ssh -L` tunnel. |
| `--port` | `8000` | TCP port to serve on (`0` picks a free one). |
| `--no-browser` | off | Do not open a browser on startup (e.g. when tunnelling). |
| `--keep-alive` | off | Keep the server running after the last tab closes (by default it stops a few seconds later; a refresh reconnects). |

```bash
deeperfly gui recording/deeperfly_outputs           # one recording
deeperfly gui myproject/                             # a project: every recording, plus Jobs
deeperfly gui results.h5 --port 0                    # any free port
deeperfly gui results.h5 --no-browser                # headless / over a tunnel
```

## `deeperfly labels-suggest` — rank the frames worth labeling next { #deeperfly-labels-suggest }

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

The list is never the raw top-N: at 100 fps neighboring frames are the same pose,
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

## `deeperfly labels-export` — export ground truth { #deeperfly-labels-export }

```bash
deeperfly labels-export PATH [-o OUT.npz]
```

Exports the labels authored in [`deeperfly gui`](gui.md) (the `labels.h5` beside the
result) as a training/eval dataset: the ground-truth pixels and the **Hidden** mask, in
footage pixel space.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, or a directory containing one (its `labels.h5` is exported). |
| `-o`, `--output` | `labels_gt.npz` beside `results.h5` | Destination `.npz`. |

There is nothing to filter and so no flag to filter with: the label store keeps **one** kind
of ground truth — a pixel the operator created — and stopped recording which layer a
proposal had been copied out of, because the layer a suggestion came from is not a property
of the label the operator accepted.

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

## `deeperfly labels-absent` — mark keypoints that are not on this animal { #deeperfly-labels-absent }

```bash
deeperfly labels-absent PATH... --points 'lf_femur_tibia,lf_tibia_tarsus,lf_claw' \
    [--frames RANGE] [--subject ID] [--clear]
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

## `deeperfly project` — group related recordings { #deeperfly-project }

A project gathers related recordings under one skeleton and one place to see what is
labeled. It **indexes** rather than re-homes: each recording's `results.h5` and
`labels.h5` stay where they already are and are adopted by **symlink**, so the file the
editor writes is the very file a training set reads. Creating a project copies nothing
and can lose nothing.

```bash
deeperfly project new  DIR [--skeleton fly38|blank|PATH] [--name NAME] [--description TEXT]
deeperfly project add  PROJECT RECORDING... [--copy] [--slug NAME] [--subject ID] [-c CONFIG]
deeperfly project ls     [PROJECT]
deeperfly project status [PROJECT]
deeperfly project rm     RECORDING [PROJECT] [--delete]
```

`PROJECT` may be omitted on `ls` / `status` / `rm`: the nearest enclosing project is
used, the way `git` finds its repository.

Five more subcommands are not covered here — `project config` (compose the resolved run
config from the project's parts), `project rig` (lift a config's camera rig in as
`rig.toml`), `project export` / `project import` (a whole project as one shareable
`.dfpkg`), `project import-outputs` (adopt a stray `deeperfly_outputs/`) and `project
skeleton` (change the skeleton, migrating every label onto the new point order).
`deeperfly project <name> --help` documents each.

### `new`

Writes a `project.toml` (the index) and a `skeleton.toml` (what is tracked). Nothing
else — a camera rig is either pointed at with a
[calibration](#deeperfly-calibration) or solved later from
labels.

`--skeleton fly38` (the default) seeds the one packaged 38-point *Drosophila* skeleton --
six 5-point legs, two antennae, `neck`, and a 5-point dorsal-midline abdomen chain
(`abdomen0`…`abdomen4`); `blank` gives you an empty one to define yourself; a path copies
the `[skeleton]` table out of any config. A config that only *names* its skeleton is
resolved on the way in, so the project always records its own.

(`fly38b`, what these points were called before 0.2, still resolves as a config's
`[skeleton] name` so old configs and stored results keep loading — but it is not a
`--skeleton` value here, which accepts only the live preset names.)

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
id:       prj_697f48bb
iteration:0
skeleton: fly38  (38 points)
calib:    none (uncalibrated -- no 3D until a rig is solved)
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ slug        ┃ frames ┃ labeled ┃ reviewed ┃ GT pts ┃ occl ┃ absent ┃ state ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ oldrig_demo │     64 │      10 │       10 │  1,759 │   12 │      — │ link  │
│ scape_Fly3  │  4,009 │      23 │       23 │  4,461 │   28 │      3 │ link  │
├─────────────┼────────┼─────────┼──────────┼────────┼──────┼────────┼───────┤
│ total       │        │      33 │       33 │  6,220 │   40 │        │       │
└─────────────┴────────┴─────────┴──────────┴────────┴──────┴────────┴───────┘
2 of 2 recording(s) carry labels
```

**`GT pts` is what `deeperfly labels-export` would yield**, not a count of rows in the
file. The two used to differ, and the numbers are read from each `labels.h5`'s sparse
indices for exactly that reason: a progress number that disagrees with the export is worse
than no progress number.

Counts are the **live** rows. A keypoint declared absent has its labels quarantined — the
`absent` column says how many keypoints that is — and quarantined labels are not reported
as ground truth. `occl` is the **Hidden** flag (held out of the training loss), counted
beside `GT pts` rather than subtracted from it, because a cell can carry a hand-placed
pixel *and* be held out.

`state` says how the outputs are attached — `link` (adopted by reference, the default) or
`dir` (`--copy`), each qualified `no results` for a recording that is still just videos, and
`missing` when the symlink's target has gone (an unmounted share is a normal Monday, not an
error).

### `rm`

Drops the entry. The adopted files are left untouched — a symlinked project must not be
able to delete the originals. `--delete` also removes the project's own directory for the
recording (for a linked one, just the link), and **refuses** when the outputs are a real
directory, since that would be the only copy of the labels.

## `deeperfly calibration` — reuse a solved camera rig { #deeperfly-calibration }

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
  cameras:      8  (rh, rm, rf, f, lf, lm, lh, h)
  units:        config  (scale from orbit_prior, in whatever unit the config's [cameras] distance was written in)
  method:       labels_ba
  reprojection:
    rms         1.842 px
    median      1.331 px
    p90         3.155 px
    rh          1.618 px rms
    ...
  rh     pos (-53.731,-93.066, -0.000)  f (22388.1, 22388.1)  c (479.5, 255.5)  frame 960x512
  ...
  h      pos (-158.917,  0.000,  0.000)  f (24168.6, 24168.6)  c (799.5, 503.5)  frame 1600x1008
```

A rig covering only **some** of the cameras a config declares is not refused: the run
warns, drops the ones it never measured (the same way it drops a view with no footage) and
proceeds on the rest — one config routinely describes more rig than one solve covers. A
rig covering *none* of them belongs to a different rig entirely, and that is refused.

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

## `deeperfly inspect` — summarize a result { #deeperfly-inspect }

```bash
deeperfly inspect RESULT.h5
```

Prints the file path, the views (count and camera names), the frame count, the
skeleton (name and point count), whether a 3D pose is present, and the median /
max reprojection error.

```bash
deeperfly inspect recording/deeperfly_outputs/results.h5
```

## `deeperfly repack` — rewrite results in the current schema { #deeperfly-repack }

```bash
deeperfly repack PATHS... [--dry-run]
```

Rewrites `results.h5` files in the current on-disk schema (**v3**) without recomputing
anything: point arrays are narrowed to float32 and deflated, and any 2D or reprojection
error a reader can rebuild is left out. The pose in the file is the pose that comes out, to
within the float32 storage step. Each `PATH` is a `results.h5` or a directory to search
recursively for one, so repacking a whole corpus is one command rather than a shell loop.
Files already in v3 are skipped (there would be nothing to take out of them), a file from a
*newer* build is refused rather than downgraded, and a failure on one file is reported while
the rest continue. The rewrite goes to a sibling temporary file and is moved into place only
once complete, so an interrupted repack leaves the original intact.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATHS...` | — | `results.h5` files, or directories to search for them. |
| `--dry-run` | off | Report what each file *would* shrink to, and replace nothing. |

```bash
deeperfly repack recordings/ --dry-run     # what would it save?
deeperfly repack recordings/               # do it
```

`--dry-run` still does the whole rewrite — into a temporary file it then throws away —
because that is the only honest way to report the size. The report gives per-file and total
before → after with the ratio. What the schema stores and what it rebuilds is in the
[output-format reference](../reference/output-format.md).

## `deeperfly doctor` — report the install { #deeperfly-doctor }

```bash
deeperfly doctor
```

Reports, each guarded so a missing piece is shown rather than crashing:

- **deeperfly** — version and install location.
- **system** — Python version/implementation and platform.
- **inference** — PyTorch version, CUDA/MPS availability, GPU memory.
- **frame I/O** — PyAV for video, OpenCV for images.
- **gui** — whether FastAPI + uvicorn are importable.
- **weights** — `$DEEPERFLY_MODELS`, every directory on the search path with what is in
  it, and whether the checkpoint the *default config* names actually resolves.
- **config** — the packaged default config path.

Run it right after installing to confirm the GPU and frame-I/O backends are
available.

```console
$ deeperfly doctor
...
weights
  DEEPERFLY_MODELS  unset -- set it to the directory holding the checkpoints
  searched [0]      /home/you/.cache/deeperfly/weights  (absent)
  default wants     mvt_alt8_r27_gray_fly38.pth  --  NOT FOUND on the search path above
```

**Nothing downloads.** Every detector deeperfly ships is trained per project, so there is
no published checkpoint to fetch and the weights section is not "are they cached yet" but
"where do you look, and is the one my config names there". Point `$DEEPERFLY_MODELS` at the
directory holding the `.pth` files (`os.pathsep`-separated, like `PATH`) and name the
checkpoint as a bare filename; a run that cannot resolve one stops with the search path
printed. See
[where `weights` is looked up](../reference/configuration.md#weights-resolution), and
[the released checkpoints](../reference/configuration.md#weights) for the three that ship
with 0.2 — their sizes, their sha256s and where they live.

## `deeperfly calibrate` — solve a rig from hand labels { #deeperfly-calibrate }

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
> 3 mm blob near the field center. A **static** landmark — a coverslip scratch, the tether
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

## `deeperfly labels-merge` — reconcile a second label set { #deeperfly-labels-merge }

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

### A disagreement is two operators disagreeing, so the default is to ask

Everything a `labels.h5` holds travels, not just the pixels — `gt`, `occluded` (the
**Hidden** flag), `seeds`, `instance`, `reviewed` and `absent`:

| situation | outcome |
| --- | --- |
| only the source has a value | taken |
| only the destination has one | kept |
| identical | kept, counted |
| both authored, different pixels | `--on-conflict`: `manual` (default), `ours`, `theirs`, `newest` |
| a **Hidden** mark vs a pixel | **both are kept** — they are independent axes, not rival claims |
| an instance **seed** the destination lacks | taken; an existing seed is **never** overwritten |
| the `instance` flag | unioned, *and implied* by any GT or seed this merge added |
| absence declarations | **unioned** (declaring is non-destructive by construction) |
| `reviewed` flags | OR'd |

There is exactly **one** kind of ground truth — a pixel an operator created — so nothing in
the data ranks one side's pixel above the other's, and the honest default is a review queue
rather than a coin flip. `--on-conflict manual` leaves those cells as the destination had
them and writes them to `exports/merge_conflicts_<recording>.json`. (An earlier build stored
a per-row *provenance* code and let a hand drag beat a bulk-confirmed reprojection; the code
is gone, because the layer a proposal came from is not a property of the label the operator
accepted.)

A **seed** is where an instance's keypoint started — evidence, not a claim — so two sides
cannot conflict over one; but overwriting a seed *is* reseeding, and reseeding is an
explicit operator gesture that a merge must not perform as a side effect. The `instance`
flag has to be implied rather than merely unioned: a frame this merge put a row into but
that carries no flag would drop back to the pre-v8 display layer, which looks exactly like
the labels going missing.

### Note on linked recordings

A recording adopted by reference — the default; `--copy` is the opt-out — has its
`deeperfly_outputs/` symlinked to the original, so `--apply` writes **into that original**. That is the point of adopting by
reference — the file the editor writes is the file a training set reads — but it means a
merge is not confined to the project directory. Use `--copy` at adoption time if you want it
to be.

## `deeperfly config` — find and set one key without reading the file { #deeperfly-config }

The packaged config is 417 lines across 30 tables — 136 of TOML and 252 of comment, because
it states only what it *changes* and explains why. The keys it says nothing about are
therefore exactly the ones you cannot find by reading it, and guessing what they do is not
much better.

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
│ method              │ ransac │     │ str   │
│ ransac_threshold    │ 15.0   │     │ float │
│ min_inliers         │ 2      │     │ int   │
│ reproj_threshold    │ 40.0   │     │ float │
│ max_drops           │ 5      │     │ int   │
│ weigh_by_confidence │ false  │     │ bool  │
└─────────────────────┴────────┴─────┴───────┘
  [triangulation] -- method + per-method thresholds.
```

The third column is the **`set` marker**, and it is the part a config file cannot give you:
it says which of these values your file actually chose and which it merely inherited. Above
it is empty for every row, because the packaged config declares no `[triangulation]` table
at all — every one of those numbers is a default. `-v` adds a column saying what each key
*means*.

**Everything here is derived from the code**, not from a parallel description — the keys,
types, defaults and documentation all come from the `*Params` dataclasses that already define
them, so a new option appears here with nothing else to update, and its help text is the
prose already written for it (`inverse_kinematics.damping` explains the abdomen's five
near-collinear hinges, which is better than any form label). The same data is served at
`GET /api/schema` for the editor.

### Turning a stage off

```bash
deeperfly config set pipeline.do_eks false -c config.toml
```

This works as of 0.2 and did not before — which mattered, because the packaged config states
every `[pipeline]` flag and 0.2 turns all of them on but `do_pictorial_structures`. `set`
used to only ever *append* an override table; TOML forbids declaring a table twice, so a key
in an already-declared table was unsettable, and `[pipeline]` is always declared.

Three cases now, and the middle one is the fix:

| the file | what `set` does |
| --- | --- |
| the table is absent | appends `[section]` with the key, under a comment saying which command wrote it — comments elsewhere survive |
| the table and the key are both there | rewrites the value **in place**, reordering nothing |
| the table is there without the key | **refused**, naming the key — appending a bare `key = value` after an existing `[section]` header would reparent every key below it |

Either way the result is validated by loading it through the same strict loader a run uses,
so `set` cannot write a key or a value a run would then reject.

Four things are **not** described: `[cameras]`, `[skeleton]`, `[[sources]]` and the
`[pose2d]` detection plan. They are structural or open-ended, and a half-schema for them
would be a fiction — they belong in the file, or (for the skeleton and rig) in the project.
