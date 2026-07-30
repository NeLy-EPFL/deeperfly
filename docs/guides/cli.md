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

## `deeperfly gui` — annotate a result

```bash
deeperfly gui PATH [--footage-dir DIR] [--host HOST] [--port PORT] [--no-browser] [--keep-alive]
```

Opens the interactive web viewer for a result and lets you author the ground-truth
2D pose (with the run's prediction as a starting point): every camera view with its
2D skeleton overlay, drag-to-place / confirm keypoints, occlude unusable views, and
the live NeuroMechFly overlays. Labels go to a `labels.h5` sidecar and never modify
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
result) as a training/eval dataset: the provenance-filtered ground-truth pixels and
the occluded mask, in footage pixel space.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `PATH` | — | A `results.h5`, or a directory containing one (its `labels.h5` is exported). |
| `-o`, `--output` | `labels_gt.npz` beside `results.h5` | Destination `.npz`. |
| `--include-projection` | off | Also export GT confirmed from the 3D reprojection (the model's own guess); excluded by default so the export is human-placed pixels only. |

The `.npz` holds `gt_xy` (V,T,P,2), `gt_mask` (V,T,P), `occluded` (V,T,P), and
`point_names` / `camera_names`. Coordinates are footage-space; a detector-training
pipeline maps them into model-input space by inverting each pathway's preprocessing.

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
