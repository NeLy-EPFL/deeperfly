# CLI usage

`deeperfly` has four commands: `init` (write a config), `run` (the pipeline),
`inspect` (summarize a result), and `doctor` (report the install). Every command
takes `--log-level` (`debug` / `info` / `warning` / `error` / `critical`;
`warning` or higher hides the per-stage logs and the progress bar) and `-h` /
`--help`.

```bash
deeperfly --help            # the command list
deeperfly run --help        # a command's options
```

## `deeperfly init` — write a config

```bash
deeperfly init [OUTPUT] [--force]
```

Writes the packaged, fully commented default config so you can edit it in place.

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `OUTPUT` | `config.toml` | Destination path. |
| `--force` | off | Overwrite an existing file (otherwise it errors). |

```bash
deeperfly init                       # -> config.toml
deeperfly init rig.toml --force      # overwrite rig.toml
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

Each output directory holds `poses.h5`, the rendered MP4s, a `config.toml`
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

## `deeperfly inspect` — summarize a result

```bash
deeperfly inspect RESULT.h5
```

Prints the file path, the views (count and camera names), the frame count, the
skeleton (name and point count), whether a 3D pose is present, and the median /
max reprojection error.

```bash
deeperfly inspect recording/deeperfly_outputs/poses.h5
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
